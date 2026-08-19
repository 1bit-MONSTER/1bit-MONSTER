// vision_server.cpp — OpenAI-compatible VL inference server.
//
// Extends the unified backend server with image_url support:
//   POST /v1/chat/completions
//   Accepts messages with image_url parts (data:image/...;base64 or http(s)://)
//   Returns text descriptions of images.
//
// Build:
//   cmake --build . --target vision_server -j8
//
// Run:
//   ./build/vision_server --port 8089 --mmproj /path/to/mmproj.gguf \
//                         --model /path/to/text.gguf
//
// API: OpenAI-compatible /v1/chat/completions
//   {
//     "model": "zaya-vl",
//     "messages": [{
//       "role": "user",
//       "content": [
//         {"type": "text", "text": "Describe this image:"},
//         {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
//       ]
//     }],
//     "max_tokens": 256
//   }
//
// Upstream tracking: additive file, no existing file modified.
// Cherry-pick: this + include/vl_preprocess.h + include/vl_processor.h +
//   src/vl_processor.cpp + kernels/vl_resize_norm.hip + CMakeLists.txt edits.

#include "backend.h"
#include "model_discovery.h"
#include "vision_encoder.h"
#include "vl_processor.h"
#include "onebp_loader.h"
#include "rocm_cpp/tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <signal.h>
#include <getopt.h>

#include <httplib.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ── Constants for Qwen2-VL ──
static const int VL_INPUT_SIZE  = 224;   // 16x16 patches
static const int VL_VISION_START = 151652;
static const int VL_VISION_END   = 151653;
static const int VL_EOS_ID       = 151645; // Qwen2 <|im_end|>

// Qwen3/Mage-VL chat template (applied when the .htok tokenizer is loaded,
// i.e. the special tokens exist in vocab) — matches chat_template.jinja:
//   <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
//   <|im_start|>user\n<|vision_start|>...<|vision_end|>{text}<|im_end|>\n
//   <|im_start|>assistant\n
static const char* VL_TMPL_SYS       = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
static const char* VL_TMPL_USER_OPEN = "<|im_start|>user\n";
static const char* VL_TMPL_ASSIST    = "<|im_end|>\n<|im_start|>assistant\n";

// ── Globals ──
static std::atomic<bool> keep_running{true};
static std::mutex g_inference_mutex;  // serialize backend access (httplib thread pool, issue #1276)
static int g_port = 8089;
// Bind address (--host ADDR). Loopback-only by default — no auth unless
// --api-token is set, so network exposure is opt-in and warned about.
static std::string g_bind_host = "127.0.0.1";
// Optional bearer token (--api-token / VISION_API_TOKEN env). Empty = off.
static std::string g_api_token = []() -> std::string {
    const char* env = getenv("VISION_API_TOKEN");
    return (env && env[0]) ? std::string(env) : std::string();
}();
static std::string g_mmproj_path;
static std::string g_model_path;
static std::string g_tokenizer_path;
static rcpp_tokenizer_t* g_htok = nullptr;
static time_t g_start_time = 0;

static void handle_sigint(int) { keep_running = false; }

// ── Mini GGUF scalar reader (duplicated from vision_qwen2vl_poc for
//     self-containedness — no cross-file dependency) ──
// Bounded reads (issue #1295): every fread is checked and file-controlled
// lengths/counts are capped, so a truncated/corrupt .gguf fails cleanly
// instead of bad_alloc'ing on a garbage length.
struct GgufReadError {};
struct GgufScanner {
    FILE* f = nullptr;
    uint64_t kc = 0;
    ~GgufScanner() { if (f) fclose(f); }
    bool open(const std::string& path) {
        f = fopen(path.c_str(), "rb");
        if (!f) return false;
        char magic[4];
        if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GGUF", 4) != 0) return false;
        uint32_t ver;
        if (fread(&ver, 4, 1, f) != 1) fail();
        if (fread(&kc, 8, 1, f) != 1) fail();
        uint64_t tc;
        if (fread(&tc, 8, 1, f) != 1) fail();
        if (kc > (1ull << 24)) fail();  // cap key count
        return true;
    }
    void fail() { throw GgufReadError{}; }
    void read_str(std::string& s) {
        uint64_t l;
        if (fread(&l, 8, 1, f) != 1) fail();
        if (l > (1ull << 28)) fail();  // 256 MB string cap
        s.resize(l);
        if (l && fread(&s[0], 1, l, f) != l) fail();
    }
    void skip_value(uint32_t vt) {
        switch (vt) {
            case 0: case 1: case 7: if (fseek(f, 1, SEEK_CUR)) fail(); break;
            case 2: case 3: if (fseek(f, 2, SEEK_CUR)) fail(); break;
            case 4: case 5: case 6: if (fseek(f, 4, SEEK_CUR)) fail(); break;
            case 8: { std::string tmp; read_str(tmp); break; }
            case 9: {
                uint32_t at; uint64_t an;
                if (fread(&at, 4, 1, f) != 1 || fread(&an, 8, 1, f) != 1) fail();
                if (at == 8) { for (uint64_t j = 0; j < an; j++) { std::string tmp; read_str(tmp); } }
                else if (an > (1ull << 30)) fail();
                else {
                    // element size by GGUF type (1/2/4/8 bytes); was hardcoded 4
                    size_t es = 4;
                    if (at == 0 || at == 1 || at == 7) es = 1;
                    else if (at == 2 || at == 3) es = 2;
                    else if (at == 10 || at == 11 || at == 12) es = 8;
                    if (fseek(f, (long)(an * es), SEEK_CUR)) fail();
                }
                break;
            }
            case 10: case 11: case 12: if (fseek(f, 8, SEEK_CUR)) fail(); break;
            default: fail();  // unknown type — refuse to guess
        }
    }
};

static bool read_gguf_uint32_kv(const std::string& path, const std::string& key, uint32_t& out) {
    try {
        GgufScanner gs;
        if (!gs.open(path)) return false;
        for (uint64_t i = 0; i < gs.kc; i++) {
            std::string k; gs.read_str(k);
            uint32_t vt;
            if (fread(&vt, 4, 1, gs.f) != 1) throw GgufReadError{};
            if ((vt == 4 || vt == 5) && k == key) {
                if (fread(&out, 4, 1, gs.f) != 1) throw GgufReadError{};
                return true;
            }
            gs.skip_value(vt);
        }
    } catch (const GgufReadError&) {
        fprintf(stderr, "[vision] WARNING: truncated/corrupt GGUF '%s' reading '%s'\n",
                path.c_str(), key.c_str());
    }
    return false;
}

// ── Load image from URL or data URL ──
// Returns a VlProcessor with loaded+processed pixels, or error string.
struct VlResult {
    VlProcessor proc;
    std::string error;
    bool ok() const { return error.empty() && proc.size() > 0; }
};

static VlResult load_image_from_content(const json& part) {
    VlResult result;

    // Extract URL from {"type":"image_url","image_url":{"url":"..."}}
    std::string url;
    if (part.contains("image_url")) {
        const auto& iu = part["image_url"];
        if (iu.is_string()) url = iu.get<std::string>();
        else if (iu.is_object() && iu.contains("url") && iu["url"].is_string()) url = iu["url"].get<std::string>();
    }

    if (url.empty()) {
        result.error = "no image_url found in content part";
        return result;
    }

    // Case 1: base64 data URL
    if (vl_is_data_url(url)) {
        auto raw = vl_decode_base64_image(url);
        if (raw.empty()) {
            result.error = "failed to decode base64 image";
            return result;
        }
        if (!result.proc.load_from_memory(raw.data(), raw.size(),
                                           VL_INPUT_SIZE, VL_INPUT_SIZE,
                                           VL_MEAN_QWEN2VL, VL_STD_QWEN2VL)) {
            result.error = "failed to process base64 image";
            return result;
        }
        return result;
    }

    // Case 2: remote URL — download via curl
    auto raw = vl_download_image(url);
    if (raw.empty()) {
        result.error = "failed to download image from " + url;
        return result;
    }
    if (!result.proc.load_from_memory(raw.data(), raw.size(),
                                       VL_INPUT_SIZE, VL_INPUT_SIZE,
                                       VL_MEAN_QWEN2VL, VL_STD_QWEN2VL)) {
        result.error = "failed to process downloaded image";
        return result;
    }

    return result;
}

// ── Simple GPT-2 BPE tokenizer (same as vision_qwen2vl_poc) ──
struct SimpleTokenizer {
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> vocab_ix;
    int eos_id = VL_EOS_ID;
    int bos_id = 151643; // <|im_start|>

    bool load(const std::string& path) {
        // Read GGUF string array metadata (bounded reads, issue #1295)
        try {
            GgufScanner gs;
            if (!gs.open(path)) return false;
            for (uint64_t i = 0; i < gs.kc; i++) {
                std::string k; gs.read_str(k);
                uint32_t vt;
                if (fread(&vt, 4, 1, gs.f) != 1) throw GgufReadError{};
                if (vt == 9 && k == "tokenizer.ggml.tokens") {
                    uint32_t at; uint64_t an;
                    if (fread(&at, 4, 1, gs.f) != 1 || fread(&an, 8, 1, gs.f) != 1) throw GgufReadError{};
                    if (at != 8 || an > (1ull << 24)) return false;  // not strings / absurd count
                    vocab.resize(an);
                    for (uint64_t j = 0; j < an; j++) gs.read_str(vocab[j]);
                    break;
                }
                gs.skip_value(vt);
            }
        } catch (const GgufReadError&) {
            fprintf(stderr, "[vision] WARNING: truncated/corrupt GGUF '%s' — tokenizer load failed\n",
                    path.c_str());
            return false;
        }

        for (size_t i = 0; i < vocab.size(); i++)
            vocab_ix[vocab[i]] = (int)i;

        read_gguf_uint32_kv(path, "tokenizer.ggml.eos_token_id", (uint32_t&)eos_id);
        return !vocab.empty();
    }

    std::vector<int> encode(const std::string& text) {
        std::vector<int> ids;
        std::string s;
        s.reserve(text.size() * 2);
        for (char c : text) {
            if (c == ' ') s += "\xC4\xA0"; // GPT2 space marker
            else s += c;
        }
        size_t pos = 0;
        while (pos < s.size()) {
            size_t best_len = 0; int best_id = -1;
            size_t max_try = std::min((size_t)24, s.size() - pos);
            for (size_t len = max_try; len >= 1; len--) {
                auto it = vocab_ix.find(s.substr(pos, len));
                if (it != vocab_ix.end()) { best_len = len; best_id = it->second; break; }
            }
            if (best_id < 0) { pos++; continue; }
            ids.push_back(best_id);
            pos += best_len;
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) {
        std::string out;
        out.reserve(ids.size() * 4);
        for (int id : ids) {
            if (id < 0 || (size_t)id >= vocab.size()) continue;
            std::string tok = vocab[id];
            size_t p;
            while ((p = tok.find("\xC4\xA0")) != std::string::npos)
                tok.replace(p, 2, " ");
            out += tok;
        }
        return out;
    }
};

// ── Tokenizer helpers: .htok (merge-BPE) when loaded, else GGUF greedy ──
static std::vector<int> encode_text(SimpleTokenizer& st, const std::string& text) {
    if (g_htok) {
        // Buffer is 2x text + 16: BPE output can never exceed 1 token/byte.
        std::vector<int> ids(text.size() * 2 + 16);
        size_t n = 0;
        rcpp_tokenizer_encode(g_htok, text.data(), text.size(), 1,
                              ids.data(), ids.size(), &n);
        // rcpp reports the TRUE count even when the buffer is too small —
        // clamp so resize() never materializes unwritten elements.
        ids.resize(std::min(n, ids.size()));
        return ids;
    }
    return st.encode(text);
}

static std::string decode_text(SimpleTokenizer& st, const std::vector<int>& ids) {
    if (g_htok) {
        char buf[65536];
        size_t l = 0;
        rcpp_tokenizer_decode(g_htok, ids.data(), ids.size(), buf, sizeof(buf), &l);
        // l is the TRUE byte count and may exceed the buffer — clamp before
        // constructing the string to avoid an out-of-bounds read.
        return std::string(buf, std::min(l, sizeof(buf)));
    }
    return st.decode(ids);
}

static int eos_id_of(SimpleTokenizer& st) {
    return g_htok ? rcpp_tokenizer_eos_id(g_htok) : st.eos_id;
}

// ── Main ──
#ifdef ONE_BIN_DISPATCH
int vision_server_main(int argc, char** argv) {
#else
int main(int argc, char** argv) {
#endif
    signal(SIGINT, handle_sigint);
    signal(SIGTERM, handle_sigint);

    static struct option long_opts[] = {
        {"port",      required_argument, nullptr, 'p'},
        {"mmproj",    required_argument, nullptr, 'm'},
        {"model",     required_argument, nullptr, 'M'},
        {"tokenizer", required_argument, nullptr, 't'},
        {"host",      required_argument, nullptr, 1001},
        {"api-token", required_argument, nullptr, 1002},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "p:m:M:t:", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'p': g_port = atoi(optarg); break;
            case 'm': g_mmproj_path = optarg; break;
            case 'M': g_model_path = optarg; break;
            case 't': g_tokenizer_path = optarg; break;
            case 1001: g_bind_host = optarg; break;
            case 1002: g_api_token = optarg; break;
        }
    }

    g_start_time = time(nullptr);

    if (g_mmproj_path.empty() || g_model_path.empty()) {
        fprintf(stderr, "Usage: %s --mmproj <mmproj.gguf> --model <text.gguf> [--port 8089] [--host 127.0.0.1] [--api-token TOKEN]\n", argv[0]);
        return 1;
    }

    // ── Load tokenizer ──
    // .htok (Qwen3/Mage-VL BPE via rcpp_tokenizer) takes precedence over the
    // GGUF-embedded vocab fallback.
    if (g_tokenizer_path.size() >= 5 &&
        g_tokenizer_path.substr(g_tokenizer_path.size() - 5) == ".htok") {
        if (rcpp_tokenizer_load(g_tokenizer_path.c_str(), &g_htok) == 0 && g_htok)
            fprintf(stderr, "Loaded htok tokenizer %s (BOS=%d EOS=%d)\n",
                    g_tokenizer_path.c_str(), rcpp_tokenizer_bos_id(g_htok),
                    rcpp_tokenizer_eos_id(g_htok));
        else
            fprintf(stderr, "WARNING: failed to load htok %s — falling back\n",
                    g_tokenizer_path.c_str());
    }
    SimpleTokenizer tokenizer;
    if (!g_htok && !tokenizer.load(g_model_path)) {
        fprintf(stderr, "WARNING: could not load tokenizer from %s\n", g_model_path.c_str());
    }

    // ── Load text model (GenericBackend CPU) ──
    fprintf(stderr, "Loading text model from %s ...\n", g_model_path.c_str());
    ModelConfig cfg;
    bool is_1bp = g_model_path.size() > 4 &&
                  g_model_path.substr(g_model_path.size() - 4) == ".1bp";
    if (is_1bp) {
        // 1BP text model: dims come from the OnebpHeader.
        OnebpModel mdl;
        if (!mdl.load(g_model_path.c_str())) {
            fprintf(stderr, "FAIL: could not read model header\n");
            return 1;
        }
        const auto& h = mdl.header;
        cfg.hidden = cfg.hidden_size = h.hidden_size;
        cfg.n_layers = cfg.num_layers = h.num_layers;
        cfg.n_heads = cfg.num_heads = cfg.num_attention_heads = h.num_attention_heads;
        cfg.n_kv_heads = cfg.num_kv_heads = h.num_kv_heads ? h.num_kv_heads : h.num_attention_heads;
        cfg.head_dim = h.head_dim ? h.head_dim : 128;
        cfg.n_ff = cfg.intermediate_size = h.intermediate_size;
        cfg.vocab = cfg.vocab_size = h.vocab_size;
        cfg.max_seq_len = h.max_seq_len;
        cfg.eos_token_id = h.eos_token_id;
        cfg.rope_theta = h.rope_theta();
        cfg.model_path = g_model_path;
        cfg.format = ModelFormat::ONEBP;
        cfg.model_name = g_model_path.substr(g_model_path.find_last_of('/') + 1);
    } else if (!read_gguf_header(g_model_path, cfg)) {
        fprintf(stderr, "FAIL: could not read model header\n");
        return 1;
    }
    cfg.max_seq_len = 1024;
    Backend* be = create_generic_backend();
    if (!be->init(cfg, g_model_path)) {
        fprintf(stderr, "FAIL: text model load failed\n");
        return 1;
    }
    fprintf(stderr, "Text model loaded: hidden=%d layers=%d vocab=%d\n",
            cfg.hidden, cfg.n_layers, cfg.vocab);

    // ── Load vision encoder weights (issue #1244) ──
    // .1bp -> Mage-ViT 1BP container (reserved[0..5] carry ViT dims);
    // otherwise a llama.cpp-style mmproj GGUF.
    VisionWeights vit;
    bool vit_ok = false;
    if (g_mmproj_path.size() > 4 &&
        g_mmproj_path.substr(g_mmproj_path.size() - 4) == ".1bp")
        vit_ok = mage_vit_load_weights_1bp(g_mmproj_path.c_str(), vit);
    else
        vit_ok = vit.load_from_gguf(g_mmproj_path);
    if (!vit_ok) {
        fprintf(stderr, "FAIL: mmproj load failed (%s)\n", g_mmproj_path.c_str());
        return 1;
    }
    fprintf(stderr, "Vision encoder loaded: H=%d L=%d NH=%d P=%d merger=%s\n",
            vit.config.hidden_size, vit.config.num_layers, vit.config.num_heads,
            vit.config.patch_size, vit.mm0_w.empty() ? "no" : "yes");

    // ── HTTP server ──
    httplib::Server svr;

    // Optional bearer-token auth (--api-token / VISION_API_TOKEN). When set,
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

    // Any unexpected exception returns a clean 500 — never the raw exception
    // text in an EXCEPTION_WHAT header (issue #1293).
    svr.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr) {
        res.status = 500;
        res.set_content(json({{"error", "Internal server error"}}).dump(), "application/json");
    });

    // ── GET /v1/health ──
    svr.Get("/v1/health", [&](const httplib::Request&, httplib::Response& res) {
        json j;
        j["status"] = "ok";
        j["service"] = "1bit-systems VL inference server";
        j["version"] = "vision-server-1.0";
        j["model"] = g_model_path;
        j["mmproj"] = g_mmproj_path;
        j["port"] = g_port;
        j["uptime"] = std::to_string(g_start_time) + "s";
        res.set_content(j.dump(2), "application/json");
    });

    // ── GET /v1/models ──
    svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        json j;
        j["object"] = "list";
        json models = json::array();
        json info;
        info["id"] = "zaya-vl";
        info["object"] = "model";
        info["owned_by"] = "1bit-systems";
        info["description"] = "Vision-language model (Qwen2-VL compatible)";
        models.push_back(info);
        j["data"] = models;
        res.set_content(j.dump(2), "application/json");
    });

    // ── POST /v1/chat/completions ──
    svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> infer_lock(g_inference_mutex);
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            json err = {{"error", "Invalid JSON body"}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }

        // ── Parse messages ──
        std::string text_prompt;
        std::vector<VlResult> images;

        // nlohmann value()/operator[] throw on non-object parts (type_error
        // 305/306/302) — catch and return 400 instead of escaping the handler
        // as a bare 500 (issue #1293).
        try {
            if (body.contains("messages") && body["messages"].is_array()) {
                for (auto& msg : body["messages"]) {
                    if (!msg.is_object()) {
                        res.status = 400;
                        res.set_content(json({{"error", "each message must be an object"}}).dump(), "application/json");
                        return;
                    }
                    std::string role = msg.value("role", "user");
                    const auto& content = msg["content"];

                    // ChatML (htok) mode: the generate phase feeds the
                    // <|im_start|>openers itself (VL_TMPL_SYS/USER_OPEN before
                    // the vision block, VL_TMPL_ASSIST after), so the raw text
                    // here must NOT carry a "role: " prefix — mixing both
                    // produced a broken prompt ("user: Say hi\n<|im_end|>...")
                    // and degenerate output. Without htok, keep the raw
                    // role-prefixed format the SimpleTokenizer path expects.
                    const bool chatml = (g_htok != nullptr);
                    const std::string prefix = chatml ? "" : (role + ": ");

                    if (content.is_string()) {
                        text_prompt += prefix + content.get<std::string>() + (chatml ? "" : "\n");
                    } else if (content.is_array()) {
                        for (const auto& part : content) {
                            if (part.is_string()) {  // OpenAI allows bare strings in content arrays
                                text_prompt += part.get<std::string>();
                                continue;
                            }
                            if (!part.is_object()) continue;
                            std::string type = part.value("type", "");
                            if (type == "text") {
                                const auto& t = part["text"];
                                if (t.is_string()) text_prompt += t.get<std::string>();
                            } else if (type == "image_url") {
                                auto result = load_image_from_content(part);
                                if (result.ok()) {
                                    images.push_back(std::move(result));
                                    fprintf(stderr, "[vision] loaded image: %dx%d\n",
                                            result.proc.width(), result.proc.height());
                                } else {
                                    fprintf(stderr, "[vision] WARNING: %s\n", result.error.c_str());
                                }
                            }
                        }
                    }
                }
            }
        } catch (const json::exception& e) {
            res.status = 400;
            res.set_content(json({{"error", std::string(e.what())}}).dump(), "application/json");
            return;
        }

        int max_tokens = body.value("max_tokens", 256);
        if (max_tokens < 1) max_tokens = 1;
        if (max_tokens > 32768) max_tokens = 32768;

        // ── Generate ──
        // Reset backend
        be->reset();

        // Track actual KV positions consumed for usage accounting (was a
        // hardcoded 66 tokens/image, issue #1294).
        size_t kv_used = 0;

        // 1. Feed vision embeddings through the text decoder (issue #1244).
        //    Real ViT forward: pixels -> mage_vit_forward (patch embed + 28
        //    transformer layers + 2x2 merger + projector) -> text-hidden
        //    embeddings, one forward_embed per token.
        //    With the Qwen3 template the vision block sits inside the user
        //    turn: <|im_start|>user\n <vision_start> embeds <vision_end> \n{text}
        std::vector<int> prompt_ids;
        int next = -1;
        if (g_htok) {
            // system + user-turn opener before the vision block (or the text
            // when there are no images)
            auto open = encode_text(tokenizer, std::string(VL_TMPL_SYS) + VL_TMPL_USER_OPEN);
            for (int t : open) be->generate(t);
            kv_used += open.size();
        }
        for (auto& vr : images) {
            be->generate(VL_VISION_START);
            kv_used++;
            std::vector<float> embs = mage_vit_forward(
                vit, vr.proc.pixels(), 3, 1,
                vr.proc.height(), vr.proc.width(), 1);
            if (embs.empty()) {
                fprintf(stderr, "[vision] FAIL: ViT forward produced no embeddings\n");
                continue;
            }
            int th = vit.config.hidden_size;
            // Projector output dim: merger mlp.2 rows (mm0 is [4H, 4H] ->
            // mlp.2 is [th, 4H]) when a merger exists, else the tower hidden.
            if (!vit.mm0_w.empty() && !vit.mm2_w.empty()) {
                int pm = (int)(vit.mm0_w.size() / (4 * vit.config.hidden_size));
                if (pm > 0) th = (int)(vit.mm2_w.size() / pm);
            }
            if (th <= 0) {  // malformed merger weights (issue #1297)
                fprintf(stderr, "[vision] FAIL: invalid projector output dim\n");
                continue;
            }
            if (th != cfg.hidden) {
                fprintf(stderr, "[vision] WARNING: projector dim %d != text hidden %d — "
                                "mismatched mmproj/model pair?\n", th, cfg.hidden);
            }
            int n_tiles = (int)(embs.size() / th);
            fprintf(stderr, "[vision] ViT: %d embeddings x %d dims through decoder\n",
                    n_tiles, th);
            std::vector<float> tok(th);
            for (int i = 0; i < n_tiles; i++) {
                const float* e = embs.data() + (size_t)i * th;
                // forward_embed wants cfg.hidden floats — pad/truncate on mismatch
                if (th == cfg.hidden) {
                    be->forward_embed(e);
                } else {
                    int n = std::min(th, cfg.hidden);
                    for (int j = 0; j < n; j++) tok[j] = e[j];
                    for (int j = n; j < cfg.hidden; j++) tok[j] = 0.0f;
                    be->forward_embed(tok.data());
                }
            }
            be->generate(VL_VISION_END);
            kv_used += (size_t)n_tiles + 1;
        }

        // 2. Tokenize and feed text prompt (template close for htok models)
        if (g_htok) text_prompt += VL_TMPL_ASSIST;
        prompt_ids = encode_text(tokenizer, text_prompt);
        if (prompt_ids.empty()) prompt_ids = {tokenizer.bos_id};
        kv_used += prompt_ids.size();

        fprintf(stderr, "Prompt: '%s' -> %zu tokens\n", text_prompt.c_str(), prompt_ids.size());
        for (size_t i = 0; i < prompt_ids.size(); i++) {
            int r = be->generate(prompt_ids[i]);
            if (r >= 0) next = r;   // keep the last prediction for step 3
        }

        // 3. Generate response. `next` already holds the prediction after the
        //    last prompt token (kept from step 2) — it is the first output token.
        std::vector<int> output_tokens;
        if (next >= 0) {
            output_tokens.push_back(next);
            for (int i = 0; i < max_tokens - 1; i++) {
                int nxt = be->generate(next);
                if (nxt < 0) break;
                next = nxt;
                output_tokens.push_back(nxt);
                if (nxt == eos_id_of(tokenizer)) break;
            }
        } else {
            // No prediction from the prompt feed — degenerate fallback.
            for (int i = 0; i < max_tokens; i++) {
                int nxt = be->generate(prompt_ids.back());
                if (nxt < 0) break;
                output_tokens.push_back(nxt);
                if (nxt == eos_id_of(tokenizer)) break;
            }
        }

        // ── Build response ──
        json response;
        response["id"] = "cmpl-vl-" + std::to_string(time(nullptr));
        response["object"] = "chat.completion";
        response["created"] = time(nullptr);
        response["model"] = "zaya-vl";

        json choice;
        choice["index"] = 0;
        json message;
        message["role"] = "assistant";
        message["content"] = decode_text(tokenizer, output_tokens);
        choice["message"] = message;
        choice["finish_reason"] = "stop";

        // Surface KV-cache overflow as an explicit error instead of a silent
        // empty 200 (issue #1294).
        if (kv_used >= (size_t)cfg.max_seq_len) {
            res.status = 400;
            json err = {{"error", "Prompt exceeds the model context window (max_seq_len=" +
                          std::to_string(cfg.max_seq_len) + ")"}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        json usage;
        usage["prompt_tokens"] = (int)kv_used;
        usage["completion_tokens"] = (int)output_tokens.size();
        usage["total_tokens"] = usage["prompt_tokens"].get<int>() + usage["completion_tokens"].get<int>();

        response["choices"] = json::array({choice});
        response["usage"] = usage;

        res.set_content(response.dump(2), "application/json");
    });

    // ── Start ──
    fprintf(stderr, "\n");
    fprintf(stderr, "╔════════════════════════════════════════╗\n");
    fprintf(stderr, "║  1bit.systems — VL Inference Server   ║\n");
    fprintf(stderr, "╚════════════════════════════════════════╝\n");
    fprintf(stderr, "  Port:    %d\n", g_port);
    fprintf(stderr, "  Host:    %s\n", g_bind_host.c_str());
    fprintf(stderr, "  Model:   %s\n", g_model_path.c_str());
    fprintf(stderr, "  MMProj:  %s\n", g_mmproj_path.c_str());
    if (!g_api_token.empty())
        fprintf(stderr, "  Auth:    bearer token required (Authorization: Bearer <token>)\n");
    else
        fprintf(stderr, "  Auth:    NONE — requests are unauthenticated\n");
    fprintf(stderr, "  Endpoints:\n");
    fprintf(stderr, "    POST /v1/chat/completions — VL inference\n");
    fprintf(stderr, "    GET  /v1/health           — Status\n");
    fprintf(stderr, "    GET  /v1/models           — Model list\n");
    fprintf(stderr, "\n  Try it:\n");
    fprintf(stderr, "    curl -X POST http://127.0.0.1:%d/v1/chat/completions \\\n", g_port);
    fprintf(stderr, "      -H \"Content-Type: application/json\" \\\n");
    fprintf(stderr, "      -d '{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What is this?\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"https://example.com/photo.jpg\"}}]}],\"max_tokens\":100}'\n");

    // SIGINT/SIGTERM set keep_running=false; a watchdog thread stops the
    // server so Ctrl-C / kill actually terminate it (issue #1292).
    if (g_bind_host != "127.0.0.1" && g_bind_host != "localhost" && g_bind_host != "::1") {
        fprintf(stderr,
            "\n  *** WARNING: binding to %s — server is network-reachable. ***\n",
            g_bind_host.c_str());
        if (g_api_token.empty())
            fprintf(stderr,
                "  *** No API token set (--api-token) — requests are UNAUTHENTICATED. ***\n"
                "  *** Use a reverse proxy / firewall, or set VISION_API_TOKEN. ***\n\n");
        else
            fprintf(stderr, "  *** Requests must send 'Authorization: Bearer <token>'. ***\n\n");
    }
    std::thread listener([&]() {
        if (!svr.listen(g_bind_host.c_str(), g_port)) {
            fprintf(stderr, "Failed to start server on port %d\n", g_port);
            _exit(1);
        }
    });
    while (keep_running)
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    svr.stop();
    listener.join();
    return 0;
}
