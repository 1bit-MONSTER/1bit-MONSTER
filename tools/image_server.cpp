// image_server.cpp — OpenAI-compatible image generation server
//
// Routes:
//   POST /v1/images/generations  — text-to-image
//   POST /v1/images/edits        — image-to-image (inpainting/outpainting)
//   POST /v1/images/variations   — create variations of an image
//   POST /v1/video/generations   — text-to-video (Wan, LTX, Hunyuan)
//   GET  /v1/models              — list available models
//   GET  /health                 — health check
//
// Built on stable-diffusion.cpp via the diffusion_bridge.
// Pure C++23, zero Python at runtime.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <csignal>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <array>

#include "httplib.h"
#include "nlohmann/json.hpp"
#include "diffusion_bridge.h"
#include <filesystem>

using json = nlohmann::json;
namespace fs = std::filesystem;

// Simple base64 encode (no external dependency needed)
static std::string base64_encode(const uint8_t* data, size_t len) {
    static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        int b = (data[i] << 16) | (i+1 < len ? data[i+1] << 8 : 0) | (i+2 < len ? data[i+2] : 0);
        out += tbl[(b >> 18) & 0x3F];
        out += tbl[(b >> 12) & 0x3F];
        out += i+1 < len ? tbl[(b >> 6) & 0x3F] : '=';
        out += i+2 < len ? tbl[b & 0x3F] : '=';
    }
    return out;
}

// Minimal base64 decode (std::string in, bytes out)
static bool base64_decode(const std::string& in, std::string& out) {
    static const auto tbl = []() {
        std::array<int8_t, 256> t{};
        t.fill(-1);
        const char* abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; abc[i]; i++) t[(uint8_t)abc[i]] = (int8_t)i;
        return t;
    }();
    out.clear();
    out.reserve(in.size() / 4 * 3);
    uint32_t acc = 0;
    int bits = 0;
    for (char c : in) {
        if (c == '=' || c == '\n' || c == '\r') continue;
        int8_t v = tbl[(uint8_t)c];
        if (v < 0) return false;
        acc = (acc << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((char)((acc >> bits) & 0xFF));
        }
    }
    return true;
}

// Decode an uploaded base64 image into a temp file; caller unlinks.
static bool b64_image_to_temp(const std::string& b64, std::string& out_path) {
    std::string bytes;
    if (!base64_decode(b64, bytes) || bytes.empty()) return false;
    static std::atomic<int> seq{0};
    out_path = "/tmp/onebit_img_" + std::to_string(getpid()) + "_" +
               std::to_string(seq++) + ".png";
    std::ofstream f(out_path, std::ios::binary);
    f.write(bytes.data(), (std::streamsize)bytes.size());
    f.close();
    return true;
}

// ─── Global config ────────────────────────────────────────────────
static std::string g_models_dir = "./models";
static int g_port = 8089;
static bool g_verbose = false;
static std::string g_backend_str = "auto";
// Bind address (--host ADDR). Loopback-only by default — previously this
// server hardcoded 0.0.0.0 and was unconditionally network-reachable; that
// was a security hole (no auth existed). Exposing it now requires an
// explicit --host 0.0.0.0 and is loudly warned about at startup.
static std::string g_bind_host = "127.0.0.1";
// Optional bearer token (--api-token / IMAGE_API_TOKEN env). Empty = off.
static std::string g_api_token = []() -> std::string {
    const char* env = getenv("IMAGE_API_TOKEN");
    return (env && env[0]) ? std::string(env) : std::string();
}();

// ─── Model registry ───────────────────────────────────────────────
struct DiffusionModelEntry {
    std::string id;
    std::string path;
    std::string vae_path;
    std::string t5xxl_path;
    std::string clip_vision_path;
    std::string description;
};

static std::vector<DiffusionModelEntry> g_models;
static std::mutex g_model_mtx;
static std::string g_active_model_id;
static std::unique_ptr<DiffusionEngine> g_engine;

// ─── Discover models in directory ─────────────────────────────────
static void discover_models(const std::string& dir) {
    g_models.clear();
    
    try {
        for (auto& entry : fs::directory_iterator(dir)) {
            auto path = entry.path();
            auto ext = path.extension().string();
            auto stem = path.stem().string();
            
            // Supported extensions: .gguf, .safetensors, .ckpt, .pth
            if (ext != ".gguf" && ext != ".safetensors" &&
                ext != ".ckpt" && ext != ".pth" && ext != ".pt") continue;
            
            DiffusionModelEntry model;
            model.id = stem;
            model.path = path.string();
            
            // Detect model type from filename conventions
            std::string lower = stem;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            
            if (lower.find("flux") != std::string::npos) {
                model.description = "FLUX";
            } else if (lower.find("sdxl") != std::string::npos) {
                model.description = "SDXL";
            } else if (lower.find("sd3") != std::string::npos || lower.find("sd_3") != std::string::npos) {
                model.description = "SD3";
            } else if (lower.find("qwen") != std::string::npos && lower.find("image") != std::string::npos) {
                model.description = "Qwen-Image";
            } else if (lower.find("wan") != std::string::npos) {
                model.description = "Wan Video";
            } else if (lower.find("ltx") != std::string::npos) {
                model.description = "LTX Video";
            } else if (lower.find("hunyuan") != std::string::npos) {
                model.description = "Hunyuan Video";
            } else if (lower.find("chroma") != std::string::npos) {
                model.description = "Chroma";
            } else if (lower.find("lens") != std::string::npos) {
                model.description = "Lens";
            } else if (lower.find("z-image") != std::string::npos || lower.find("zimage") != std::string::npos) {
                model.description = "Z-Image";
            } else {
                model.description = "SD1.x / default";
            }
            
            // Look for companion files: <stem>.vae, <stem>.t5, <stem>.clip_vision
            // (Wan video needs .t5 = umt5-xxl; i2v also needs .clip_vision).
            // Companions may use a different container than the model, so
            // probe the common extensions.
            auto dot = path.string().find_last_of('.');
            std::string base = dot == std::string::npos ? path.string()
                            : path.string().substr(0, dot);
            auto companion = [&](const char* tag) {
                for (const char* e : {".safetensors", ".gguf", ".ckpt", ".pth"}) {
                    std::string p = base + "." + tag + e;
                    if (fs::exists(p)) return p;
                }
                return std::string();
            };
            model.vae_path = companion("vae");
            model.t5xxl_path = companion("t5");
            model.clip_vision_path = companion("clip_vision");
            
            g_models.push_back(model);
            
            if (g_verbose) {
                printf("  [%zu] %s (%s)\n", g_models.size() - 1,
                       model.id.c_str(), model.description.c_str());
            }
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Error scanning models dir: %s\n", e.what());
    }
}

// ─── HTTP Handlers ────────────────────────────────────────────────

static void handle_health(const httplib::Request& req, httplib::Response& res) {
    json resp = {
        {"status", "ok"},
        {"engine", "1bit-systems image server"},
        {"model_loaded", g_engine && g_engine->is_loaded()},
        {"models_available", g_models.size()},
    };
    res.set_content(resp.dump(), "application/json");
}

static void handle_list_models(const httplib::Request& req, httplib::Response& res) {
    json data = json::array();
    for (auto& m : g_models) {
        data.push_back({
            {"id", m.id},
            {"object", "model"},
            {"created", 0},
            {"owned_by", "1bit-systems"},
            {"description", m.description},
        });
    }
    json resp = {{"object", "list"}, {"data", data}};
    res.set_content(resp.dump(), "application/json");
}

static bool ensure_model_loaded(const std::string& model_id) {
    if (g_engine && g_engine->is_loaded() && g_active_model_id == model_id) {
        return true;  // Already loaded
    }
    
    // Find model in registry
    DiffusionModelEntry* found = nullptr;
    for (auto& m : g_models) {
        if (m.id == model_id) { found = &m; break; }
    }
    if (!found) {
        fprintf(stderr, "Model '%s' not found in registry\n", model_id.c_str());
        return false;
    }
    
    if (!g_engine) g_engine = std::make_unique<DiffusionEngine>();
    printf("Loading model: %s (backend=%s)\n", model_id.c_str(), g_backend_str.c_str());
    if (!g_engine->load_model(found->path, found->vae_path,
                              found->t5xxl_path, found->clip_vision_path)) {
        fprintf(stderr, "Failed to load model '%s'\n", model_id.c_str());
        return false;
    }
    
    g_active_model_id = model_id;
    printf("Loaded model: %s\n", model_id.c_str());
    return true;
}

static void handle_image_generation(const httplib::Request& req, httplib::Response& res) {
    try {
        json body = json::parse(req.body);
        
        std::string model_id = body.value("model", g_models.empty() ? "" : g_models[0].id);
        std::string prompt = body.value("prompt", "");
        std::string negative_prompt = body.value("negative_prompt", "");
        int width = body.value("width", 512);
        int height = body.value("height", 512);
        int steps = body.value("steps", 20);
        float cfg_scale = body.value("cfg_scale", 7.0f);
        int seed = body.value("seed", -1);
        int n = body.value("n", 1);
        
        // Optional LoRA
        std::vector<std::string> lora_paths;
        std::vector<float> lora_strengths;
        if (body.contains("lora_paths")) {
            for (auto& lp : body["lora_paths"]) lora_paths.push_back(lp);
            for (auto& ls : body["lora_strengths"]) lora_strengths.push_back(ls);
        }
        
        if (prompt.empty()) {
            json err = {{"error", "prompt is required"}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }
        
        if (!ensure_model_loaded(model_id)) {
            json err = {{"error", "Failed to load model: " + model_id}};
            res.status = 500;
            res.set_content(err.dump(), "application/json");
            return;
        }
        
        DiffusionParams params;
        params.prompt = prompt;
        params.negative_prompt = negative_prompt;
        params.width = width;
        params.height = height;
        params.steps = steps;
        params.cfg_scale = cfg_scale;
        params.seed = seed;
        params.lora_paths = lora_paths;
        params.lora_weights = lora_strengths;
        
        auto result = g_engine->txt2img(params);
        
        if (result.data.empty()) {
            json err = {{"error", "Generation failed (null result)"}};
            res.status = 500;
            res.set_content(err.dump(), "application/json");
            return;
        }
        
        // Build OpenAI-compatible response
        std::string b64 = base64_encode(result.data.data(), result.data.size());
        json resp = {
            {"created", std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()},
            {"data", json::array({
                {
                    {"b64_json", b64},
                    {"seed", result.seed_used},
                    {"width", result.width},
                    {"height", result.height},
                    {"mime_type", result.mime_type},
                }
            })}
        };
        
        res.set_content(resp.dump(), "application/json");
        
    } catch (const std::exception& e) {
        json err = {{"error", std::string("Internal error: ") + e.what()}};
        res.status = 500;
        res.set_content(err.dump(), "application/json");
    }
}

static void handle_image_edits(const httplib::Request& req, httplib::Response& res) {
    try {
        json body = json::parse(req.body);
        
        std::string model_id = body.value("model", g_models.empty() ? "" : g_models[0].id);
        std::string prompt = body.value("prompt", "");
        std::string negative_prompt = body.value("negative_prompt", "");
        int width = body.value("width", 512);
        int height = body.value("height", 512);
        int steps = body.value("steps", 20);
        float cfg_scale = body.value("cfg_scale", 7.0f);
        int seed = body.value("seed", -1);
        float strength = body.value("strength", 0.75f);
        
        std::vector<std::string> lora_paths;
        std::vector<float> lora_strengths;
        if (body.contains("lora_paths")) {
            for (auto& lp : body["lora_paths"]) lora_paths.push_back(lp);
            for (auto& ls : body["lora_strengths"]) lora_strengths.push_back(ls);
        }
        
        // Init image: server path, or base64 PNG uploaded by the client.
        std::string init_path = body.value("init_image", "");
        std::string temp_file;
        if (init_path.empty() && body.contains("init_image_b64")) {
            if (!b64_image_to_temp(body["init_image_b64"].get<std::string>(), temp_file)) {
                json err = {{"error", "failed to decode init_image_b64"}};
                res.status = 400;
                res.set_content(err.dump(), "application/json");
                return;
            }
            init_path = temp_file;
        }
        if (init_path.empty()) {
            json err = {{"error", "init_image or init_image_b64 is required"}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }
        
        // Optional mask (inpainting): path or base64.
        std::string mask_path = body.value("mask_image", "");
        std::string temp_mask;
        if (mask_path.empty() && body.contains("mask_image_b64")) {
            if (!b64_image_to_temp(body["mask_image_b64"].get<std::string>(), temp_mask)) {
                json err = {{"error", "failed to decode mask_image_b64"}};
                res.status = 400;
                res.set_content(err.dump(), "application/json");
                return;
            }
            mask_path = temp_mask;
        }
        
        if (prompt.empty()) {
            json err = {{"error", "prompt is required"}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }
        
        if (!ensure_model_loaded(model_id)) {
            json err = {{"error", "Failed to load model: " + model_id}};
            res.status = 500;
            res.set_content(err.dump(), "application/json");
            return;
        }
        
        DiffusionParams params;
        params.prompt = prompt;
        params.negative_prompt = negative_prompt;
        params.width = width;
        params.height = height;
        params.steps = steps;
        params.cfg_scale = cfg_scale;
        params.seed = seed;
        params.strength = strength;
        params.init_image_path = init_path;
        params.mask_image_path = mask_path;
        params.lora_paths = lora_paths;
        params.lora_weights = lora_strengths;
        
        auto result = g_engine->img2img(params);
        
        if (!temp_file.empty()) std::remove(temp_file.c_str());
        if (!temp_mask.empty()) std::remove(temp_mask.c_str());
        
        if (result.data.empty()) {
            json err = {{"error", "Generation failed (null result)"}};
            res.status = 500;
            res.set_content(err.dump(), "application/json");
            return;
        }
        
        std::string b64 = base64_encode(result.data.data(), result.data.size());
        json resp = {
            {"created", std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()},
            {"data", json::array({
                {
                    {"b64_json", b64},
                    {"seed", result.seed_used},
                    {"width", result.width},
                    {"height", result.height},
                    {"mime_type", result.mime_type},
                }
            })}
        };
        
        res.set_content(resp.dump(), "application/json");
        
    } catch (const std::exception& e) {
        json err = {{"error", std::string("Internal error: ") + e.what()}};
        res.status = 500;
        res.set_content(err.dump(), "application/json");
    }
}

static void handle_video_generation(const httplib::Request& req, httplib::Response& res) {
    try {
        json body = json::parse(req.body);
        
        std::string model_id = body.value("model", g_models.empty() ? "" : g_models[0].id);
        std::string prompt = body.value("prompt", "");
        std::string negative_prompt = body.value("negative_prompt", "");
        int width = body.value("width", 832);
        int height = body.value("height", 480);
        int steps = body.value("steps", 20);
        float cfg_scale = body.value("cfg_scale", 7.0f);
        int seed = body.value("seed", -1);
        int frames = body.value("frames", 81);
        int fps = body.value("fps", 16);
        std::string output_format = body.value("output_format", "avi");
        float strength = body.value("strength", 0.75f);
        
        // Optional first-frame image (image-to-video): path or base64.
        std::string init_path = body.value("init_image", "");
        std::string temp_file;
        if (init_path.empty() && body.contains("init_image_b64")) {
            if (!b64_image_to_temp(body["init_image_b64"].get<std::string>(), temp_file)) {
                json err = {{"error", "failed to decode init_image_b64"}};
                res.status = 400;
                res.set_content(err.dump(), "application/json");
                return;
            }
            init_path = temp_file;
        }
        
        // Optional LoRA
        std::vector<std::string> lora_paths;
        std::vector<float> lora_strengths;
        if (body.contains("lora_paths")) {
            for (auto& lp : body["lora_paths"]) lora_paths.push_back(lp);
            for (auto& ls : body["lora_strengths"]) lora_strengths.push_back(ls);
        }
        
        if (prompt.empty()) {
            json err = {{"error", "prompt is required"}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }
        
        if (!ensure_model_loaded(model_id)) {
            json err = {{"error", "Failed to load model: " + model_id}};
            res.status = 500;
            res.set_content(err.dump(), "application/json");
            return;
        }
        
        DiffusionParams params;
        params.prompt = prompt;
        params.negative_prompt = negative_prompt;
        params.width = width;
        params.height = height;
        params.steps = steps;
        params.cfg_scale = cfg_scale;
        params.seed = seed;
        params.video_frames = frames;
        params.video_fps = fps;
        params.video_output_format = output_format;
        params.init_image_path = init_path;
        params.strength = strength;
        params.lora_paths = lora_paths;
        params.lora_weights = lora_strengths;
        
        // Image-to-video when a first frame is supplied, text-to-video otherwise.
        auto result = init_path.empty() ? g_engine->txt2vid(params)
                                        : g_engine->img2vid(params);
        
        if (!temp_file.empty()) std::remove(temp_file.c_str());
        
        if (result.data.empty()) {
            json err = {{"error", "Video generation failed"}};
            res.status = 500;
            res.set_content(err.dump(), "application/json");
            return;
        }
        
        // Build OpenAI-compatible response
        std::string b64 = base64_encode(result.data.data(), result.data.size());
        json resp = {
            {"created", std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()},
            {"data", json::array({
                {
                    {"b64_json", b64},
                    {"seed", result.seed_used},
                    {"width", result.width},
                    {"height", result.height},
                    {"frames", result.frames},
                    {"fps", fps},
                    {"output_format", output_format},
                    {"mime_type", result.mime_type},
                }
            })}
        };
        
        res.set_content(resp.dump(), "application/json");
        
    } catch (const std::exception& e) {
        json err = {{"error", std::string("Internal error: ") + e.what()}};
        res.status = 500;
        res.set_content(err.dump(), "application/json");
    }
}

// ─── Main ─────────────────────────────────────────────────────────

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  -w, --models-dir DIR    Model directory (default: ./models)\n"
        "  -p, --port PORT         Server port  (default: 8089)\n"
        "      --host ADDR         Bind address (default: 127.0.0.1; 0.0.0.0 = network)\n"
        "      --api-token TOKEN   Require 'Authorization: Bearer <token>' (env: IMAGE_API_TOKEN)\n"
        "  --backend BACKEND       Backend: auto|cpu|cuda|vulkan|metal (default: auto)\n"
        "  -v, --verbose           Verbose output\n"
        "  -h, --help              Show this help\n",
        prog);
}

int main(int argc, char** argv) {
    // Parse args
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-w" || arg == "--models-dir") {
            if (i + 1 < argc) g_models_dir = argv[++i];
        } else if (arg == "-p" || arg == "--port") {
            if (i + 1 < argc) g_port = std::stoi(argv[++i]);
        } else if (arg == "--host") {
            if (i + 1 < argc) g_bind_host = argv[++i];
        } else if (arg == "--api-token") {
            if (i + 1 < argc) g_api_token = argv[++i];
        } else if (arg == "--backend") {
            if (i + 1 < argc) g_backend_str = argv[++i];
        } else if (arg == "-v" || arg == "--verbose") {
            g_verbose = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    // Discover models
    printf("Scanning models in %s...\n", g_models_dir.c_str());
    discover_models(g_models_dir);
    printf("Found %zu diffusion model(s)\n", g_models.size());
    
    // Create HTTP server
    httplib::Server svr;

    // Optional bearer-token auth (--api-token / IMAGE_API_TOKEN). When set,
    // every request must carry "Authorization: Bearer <token>".
    svr.set_pre_routing_handler([&](const httplib::Request& req, httplib::Response& res) {
        if (req.method == "OPTIONS") return httplib::Server::HandlerResponse::Unhandled;
        if (!g_api_token.empty()) {
            const std::string expected = "Bearer " + g_api_token;
            if (req.get_header_value("Authorization") != expected) {
                res.status = 401;
                res.set_header("WWW-Authenticate", "Bearer");
                res.set_content("{\"error\":\"unauthorized: missing or invalid API token\"}",
                                "application/json");
                return httplib::Server::HandlerResponse::Handled;
            }
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    // Routes
    svr.Get("/health", handle_health);
    svr.Get("/v1/models", handle_list_models);
    svr.Post("/v1/images/generations", handle_image_generation);
    svr.Post("/v1/images/edits", handle_image_edits);
    svr.Post("/v1/video/generations", handle_video_generation);
    
    // Error handler
    svr.set_exception_handler([](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
        json err = {{"error", "Internal server error"}};
        res.status = 500;
        res.set_content(err.dump(), "application/json");
    });
    
    printf("Starting image server on port %d...\n", g_port);
    printf("  Host:    %s\n", g_bind_host.c_str());
    printf("  Endpoints:\n");
    printf("    GET  /health\n");
    printf("    GET  /v1/models\n");
    printf("    POST /v1/images/generations\n");
    printf("    POST /v1/images/edits\n");
    printf("    POST /v1/video/generations\n");
    if (!g_api_token.empty())
        printf("  Auth:    bearer token required (Authorization: Bearer <token>)\n");
    else
        printf("  Auth:    NONE — requests are unauthenticated\n");

    if (g_bind_host != "127.0.0.1" && g_bind_host != "localhost" && g_bind_host != "::1") {
        printf("\n  *** WARNING: binding to %s — server is network-reachable. ***\n",
               g_bind_host.c_str());
        if (g_api_token.empty())
            printf("  *** No API token set (--api-token) — requests are UNAUTHENTICATED. ***\n"
                   "  *** Use a reverse proxy / firewall, or set IMAGE_API_TOKEN. ***\n");
        else
            printf("  *** Requests must send 'Authorization: Bearer <token>'. ***\n");
    }

    svr.listen(g_bind_host.c_str(), g_port);
    
    return 0;
}
