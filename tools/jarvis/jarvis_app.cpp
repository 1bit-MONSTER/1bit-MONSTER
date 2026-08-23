// jarvis_app.cpp — JARVIS v2: the clean-slate voice assistant.
//
// Pure C++23, in-process with the engine (BackendManager — no HTTP hop),
// no Python, no SaaS, no agent stack. One pipeline:
//
//   mic → VAD → STT (libwhisper) → LLM (engine) → TTS (piper) → speaker
//
// Voice cloning (codec training, voice packs), auth/billing/usage/beacon,
// RAG/planner/persona/tools and the old jarvis_server.cpp are GONE — see
// docs/jarvis.md for the rationale and the rebuild log.
//
// Built two ways: standalone binary (JARVIS_STANDALONE → main()) and as the
// `1bit jarvis` subcommand (jarvis_app_main, via onebin.cpp).
//
// usage:
//   jarvis --model <name> [--weights-dir DIR] [--text]
//          [--whisper whisper.gguf] [--piper PATH] [--piper-model voice.onnx]
//          [--mic DEVICE] [--system "prompt"] [--max-tokens N]

#include "jarvis/audio.h"
#include "jarvis/stt.h"
#include "jarvis/tts.h"
#include "jarvis/vad.h"

#include "backend_manager.h"
#include "model_discovery.h"
#include "simple_tokenizer.h"

#include "mesh/mesh.hpp"
#include "mesh/node_identity.hpp"
#include "mesh/peer_discovery.hpp"
#include "mesh/peer_api.hpp"
#include "mesh/mesh_agent.hpp"
#include "mesh/dispatch.hpp"

#include <httplib.h>

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using jarvis::Capture;
using jarvis::Playback;
using jarvis::STT;
using jarvis::TTS;
using jarvis::VAD;

namespace {

struct Options {
    std::string model_name;
    std::string weights_dir;
    std::string whisper_gguf;
    std::string piper_bin = "piper";
    std::string piper_model;
    std::string mic_device = "default";
    std::string system_prompt = "You are JARVIS, a helpful local voice assistant. "
                                "Answer concisely, in a few sentences.";
    bool text_mode = false;
    int max_tokens = 256;

    // ── Mesh / fleet dispatch (DSH awareness) ──
    // With --mesh-dispatch, JARVIS needs NO local model: the mic/STT/TTS
    // stay local, the LLM brain is dispatched to the best sibling install
    // on the mesh. JARVIS also announces itself as a node peers can route
    // to (POST /v1/jarvis/turn) — the DSH harness drives it from outside.
    bool mesh_dispatch = false;
    std::string mesh_name;      // friendly node name (default: <host>-jarvis)
    uint16_t node_port = 8081;  // this JARVIS's HTTP node port
    uint16_t mesh_port = 0;     // 0 = default 42424
};

void usage(const char* argv0) {
    fprintf(stderr,
        "JARVIS v2 — voice assistant on the 1bit engine (pure C++).\n\n"
        "usage: %s --model <name> [options]\n\n"
        "  --model NAME        model name (default: first Zyphra model found)\n"
        "  --weights-dir DIR   default: ./models or $HOME/.local/share/1bit-monster/weights\n"
        "  --text              chat over stdin instead of the mic\n"
        "  --whisper PATH      whisper GGUF model (voice mode)\n"
        "  --piper PATH        piper binary (default: piper on PATH)\n"
        "  --piper-model PATH  piper voice .onnx (voice replies)\n"
        "  --mic DEVICE        arecord device (default: default)\n"
        "  --system TEXT       system prompt\n"
        "  --max-tokens N      reply cap (default 256)\n"
        "  --mesh-dispatch     fleet mode: no local model — dispatch every\n"
        "                      turn to the best sibling install on the mesh\n"
        "                      (JARVIS becomes a mesh node on --port; DSH\n"
        "                      can drive it via POST /v1/jarvis/turn)\n"
        "  --mesh-name NAME    node name on the mesh (default: <host>-jarvis)\n"
        "  --port PORT         this JARVIS's HTTP node port (default 8081)\n"
        "  --mesh-port PORT    multicast port override (default 42424)\n",
        argv0);
}

std::string default_weights_dir() {
    const char* env = getenv("1BIT_WEIGHTS_DIR");
    if (env && *env) return env;
    if (std::filesystem::is_directory("models")) return "models";
    const char* home = getenv("HOME");
    if (home && *home) return std::string(home) + "/.local/share/1bit-monster/weights";
    return "/tmp";
}

// Case-insensitive + prefix model name match, mirroring unified_server.
bool match_name(const std::string& name, const std::string& cfg_name) {
    auto norm = [](std::string s) {
        for (auto& c : s) if (c == '-' || c == '_') c = ' ';
        return s;
    };
    std::string a = norm(name), b = norm(cfg_name);
    for (auto& c : a) c = (char)tolower((unsigned char)c);
    for (auto& c : b) c = (char)tolower((unsigned char)c);
    return b.find(a) != std::string::npos || a.find(b) != std::string::npos;
}

// Zyphra crown-jewel preference order — the default JARVIS stack
// (ZR1 router → ZAYA/BlackMamba/Zamba2 LLMs, all MIT open source).
// --model bypasses this entirely.
static const char* kZyphraPref[] = {
    "ZAYA1-8B", "ZAYA1-74B", "BlackMamba-2.8B", "BlackMamba-1.5B",
    "Zamba2-7B", "Zamba2-2.7B", "Zamba2-1.2B", "ZR1-1.5B",
};

// Pick the model: explicit --model wins; otherwise the first Zyphra model
// found in the weights dir; otherwise the first model of any kind.
// The Zyphra stack (ZAYA/ZR1/BlackMamba/Zamba2, MIT) is the crown jewel
// and JARVIS's default experience — see docs/jarvis.md.
static bool pick_model(const std::vector<ModelConfig>& models, const Options& opt, ModelConfig& out) {
    if (!opt.model_name.empty()) {
        for (const auto& m : models)
            if (match_name(opt.model_name, m.model_name)) { out = m; return true; }
        return false;
    }
    for (const char* pref : kZyphraPref) {
        for (const auto& m : models)
            if (match_name(pref, m.model_name)) { out = m; return true; }
    }
    if (!models.empty()) { out = models[0]; return true; }
    return false;
}

struct Session {
    Options opt;
    STT stt;
    TTS tts;
    VAD vad;
    std::vector<std::string> history;  // ["user: ...", "assistant: ...", ...]
    std::mutex mtx;
    std::atomic<bool> busy{false};

    // Fleet mode: when set, every turn is dispatched to a mesh peer
    // instead of the local engine. Owned by the caller (lives as long as
    // the session).
    mesh::MeshDispatcher* dispatcher = nullptr;

    std::string build_prompt(const std::string& user_text) {
        std::string p = opt.system_prompt + "\n\n";
        std::lock_guard<std::mutex> lock(mtx);
        size_t from = history.size() > 6 ? history.size() - 6 : 0;
        for (size_t i = from; i < history.size(); i++) p += history[i] + "\n";
        p += "user: " + user_text + "\nassistant:";
        return p;
    }

    void push_turn(const std::string& user_text, const std::string& reply) {
        std::lock_guard<std::mutex> lock(mtx);
        history.push_back("user: " + user_text);
        history.push_back("assistant: " + reply);
        if (history.size() > 40) {
            history.erase(history.begin(), history.begin() + (history.size() - 40));
        }
    }
};

// ── LLM turn: prompt → tokens, in-process BackendManager decode ──
std::string run_llm_turn(BackendManager& mgr, const std::string& prompt, int max_tokens) {
    std::vector<int> ids = g_tokenizer.encode(prompt);
    if (ids.empty()) return "";
    printf("[jarvis] tokens: %zu\n", ids.size());

    mgr.reset();
    // Prefill: feed all but the last prompt token; decode starts from it.
    for (size_t i = 0; i + 1 < ids.size(); i++) {
        if (mgr.generate(ids[i]) < 0) { fprintf(stderr, "[jarvis] prefill failed at %zu\n", i); return ""; }
    }

    std::string reply;
    int last = ids.back();
    for (int i = 0; i < max_tokens; i++) {
        int next = mgr.generate(last);
        if (next < 0 || next == g_tokenizer.eos_id) break;
        std::string piece = g_tokenizer.decode({next});
        printf("%s", piece.c_str());
        fflush(stdout);
        reply += piece;
        last = next;
    }
    printf("\n");
    return reply;
}

void handle_utterance(Session& s, BackendManager* mgr, const std::string& user_text) {
    printf("\n[jarvis] you: %s\n", user_text.c_str());
    if (s.opt.text_mode) {  // voice mode prints the transcript too
        printf("[jarvis] — speaking is disabled in --text mode; reply below\n");
    }

    std::string reply;
    if (s.dispatcher) {
        // ── Fleet mode: the LLM brain is on the mesh, not in this process.
        // The full prompt (system + history + user) is dispatched to the
        // best peer that serves the requested model.
        auto r = s.dispatcher->chat(s.opt.model_name, s.opt.system_prompt,
                                    s.build_prompt(user_text), s.opt.max_tokens);
        // Cold start: neighbors may not have been discovered yet (their
        // first beacon takes up to announce_interval_s). Retry briefly
        // before giving up.
        for (int attempt = 0; !r.ok && attempt < 5; ++attempt) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            r = s.dispatcher->chat(s.opt.model_name, s.opt.system_prompt,
                                   s.build_prompt(user_text), s.opt.max_tokens);
        }
        if (!r.ok) {
            fprintf(stderr, "[jarvis] dispatch failed (%s) — no LLM brain on the mesh?\n",
                    r.error.c_str());
            return;
        }
        printf("[jarvis] <%s/%s> %s\n", r.node_name.c_str(), r.model.c_str(),
               r.reply.c_str());
        reply = r.reply;
    } else {
        // ── Local mode: in-process engine decode.
        reply = run_llm_turn(*mgr, s.build_prompt(user_text), s.opt.max_tokens);
        if (reply.empty()) { fprintf(stderr, "[jarvis] empty reply — backend failed?\n"); return; }
    }
    s.push_turn(user_text, reply);

    if (!s.tts.loaded()) return;
    int rate = 0;
    std::vector<float> pcm = s.tts.synth(reply, rate);
    if (pcm.empty()) { fprintf(stderr, "[jarvis] TTS failed (piper missing?)\n"); return; }
    Playback pb;
    if (pb.start(rate)) {
        pb.write(pcm.data(), (int)pcm.size());
        pb.stop();
    }
}

}  // namespace

#if defined(JARVIS_STANDALONE)
int main(int argc, char** argv) {
#else
int jarvis_app_main(int argc, char** argv) {
#endif
    signal(SIGPIPE, SIG_IGN);  // aplay/piper children may die mid-write

    Options opt;
    for (int i = 1; i < argc; i++) {
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        std::string a = argv[i];
        if (a == "--model") opt.model_name = next();
        else if (a == "--weights-dir") opt.weights_dir = next();
        else if (a == "--whisper") opt.whisper_gguf = next();
        else if (a == "--piper") opt.piper_bin = next();
        else if (a == "--piper-model") opt.piper_model = next();
        else if (a == "--mic") opt.mic_device = next();
        else if (a == "--system") opt.system_prompt = next();
        else if (a == "--max-tokens") opt.max_tokens = atoi(next().c_str());
        else if (a == "--text") opt.text_mode = true;
        else if (a == "--mesh-dispatch") opt.mesh_dispatch = true;
        else if (a == "--mesh-name") opt.mesh_name = next();
        else if (a == "--port") opt.node_port = (uint16_t)atoi(next().c_str());
        else if (a == "--mesh-port") opt.mesh_port = (uint16_t)atoi(next().c_str());
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return 1; }
    }
    if (opt.model_name.empty() && !opt.mesh_dispatch) { usage(argv[0]); return 1; }
    if (!opt.text_mode && opt.whisper_gguf.empty()) {
        fprintf(stderr, "[jarvis] voice mode needs --whisper (or use --text)\n");
        return 1;
    }
    if (opt.weights_dir.empty()) opt.weights_dir = default_weights_dir();

    Session s;
    s.opt = opt;

    BackendManager* mgr = nullptr;  // null in fleet mode — the brain is remote

    // ── Mesh / fleet mode: JARVIS as a self-aware dispatch node ──────────
    // No local model required. JARVIS announces itself on the mesh, exposes
    // /v1/mesh/* + POST /v1/jarvis/turn (so the DSH harness or sibling
    // installs can drive it), and dispatches every turn to the best peer
    // that serves the requested model.
    std::unique_ptr<mesh::PeerDiscovery> disc;
    std::unique_ptr<mesh::MeshAgent> agent;
    std::unique_ptr<mesh::MeshDispatcher> dispatcher;
    httplib::Server node_svr;
    std::thread node_thread;

    if (opt.mesh_dispatch) {
        mesh::MeshConfig mcfg;
        mcfg.http_port = opt.node_port;
        mcfg.name      = opt.mesh_name;
        if (opt.mesh_port) mcfg.mesh_port = opt.mesh_port;
        mcfg.state_dir = mesh::default_state_dir();
        mcfg.agent_enabled = true;  // JARVIS says hi to new installs itself

        mesh::NodeIdentity me = mesh::load_or_create_identity(mcfg);
        me.host = mesh::detect_local_ip();
        me.api_base = mesh::make_api_base(me.host, me.port);
        me.caps.features = {"chat", "completions", "mesh", "jarvis"};
        mesh::MeshModelInfo mi;
        mi.name    = opt.model_name.empty() ? "auto" : opt.model_name;
        mi.backend = "mesh";
        me.caps.models.push_back(mi);

        disc = std::make_unique<mesh::PeerDiscovery>(mcfg, me);
        if (!disc->start()) {
            fprintf(stderr, "[jarvis] mesh discovery failed — is port %u free?\n",
                    mcfg.mesh_port);
            return 1;
        }
        agent = std::make_unique<mesh::MeshAgent>(*disc, mcfg);
        agent->start();
        dispatcher = std::make_unique<mesh::MeshDispatcher>(*disc, me);
        s.dispatcher = dispatcher.get();

        // ── JARVIS node server: /v1/mesh/* + /v1/jarvis/* ──
        mesh::register_mesh_handlers(node_svr, *disc, agent.get());
        node_svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
            res.set_content("JARVIS fleet node '" + me.name + "' (" + me.id +
                            ")\nmodel: " + mi.name +
                            "\npeers: " + std::to_string(disc->snapshot().size()) + "\n",
                            "text/plain");
        });
        node_svr.Get("/v1/jarvis/status", [&](const httplib::Request&, httplib::Response& res) {
            auto peers = disc->snapshot();
            nlohmann::json arr = nlohmann::json::array();
            for (const auto& p : peers) {
                arr.push_back({{"name", p.node.name}, {"id", p.node.id},
                               {"model", p.node.caps.models.empty()
                                             ? "" : p.node.caps.models[0].name},
                               {"integrated", p.integrated}});
            }
            nlohmann::json j{{"ok", true}, {"name", me.name}, {"id", me.id},
                             {"model", mi.name}, {"local_engine", false},
                             {"peers", arr}};
            res.set_content(j.dump(), "application/json");
        });
        node_svr.Post("/v1/jarvis/turn", [&](const httplib::Request& req, httplib::Response& res) {
            try {
                nlohmann::json body = nlohmann::json::parse(req.body);
                std::string text = body.value("text", std::string{});
                if (text.empty()) {
                    res.status = 400;
                    res.set_content(nlohmann::json{{"ok", false}, {"error", "text required"}}.dump(),
                                    "application/json");
                    return;
                }
                // Dispatch this turn to the fleet (a peer brain or our own
                // dispatcher); speak the reply if a voice is loaded.
                auto r = s.dispatcher->chat(opt.model_name, opt.system_prompt,
                                            s.build_prompt(text), opt.max_tokens);
                nlohmann::json j{{"ok", r.ok}, {"text", text}};
                if (r.ok) {
                    j["reply"] = r.reply;
                    j["node"]  = r.node_name;
                    j["model"] = r.model;
                    s.push_turn(text, r.reply);
                    if (s.tts.loaded()) {
                        int rate = 0;
                        std::vector<float> pcm = s.tts.synth(r.reply, rate);
                        if (!pcm.empty()) {
                            Playback pb;
                            if (pb.start(rate)) { pb.write(pcm.data(), (int)pcm.size()); pb.stop(); }
                        }
                    }
                } else {
                    j["error"] = r.error;
                }
                res.set_content(j.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(nlohmann::json{{"ok", false}, {"error", e.what()}}.dump(),
                                "application/json");
            }
        });

        printf("[jarvis] fleet node '%s' (%s) on %s:%u — model '%s' via the mesh\n",
               me.name.c_str(), me.id.substr(0, 8).c_str(),
               me.host.c_str(), opt.node_port, mi.name.c_str());
        node_thread = std::thread([&]() { node_svr.listen("0.0.0.0", opt.node_port); });
    } else {
        // ── Engine: in-process BackendManager (no HTTP hop) ──
        auto& mgr_ref = backend_manager();
        auto models = discover_models(opt.weights_dir);
        ModelConfig cfg;
        if (!pick_model(models, opt, cfg)) {
            if (opt.model_name.empty())
                fprintf(stderr, "[jarvis] no models found in %s — point --weights-dir at a Zyphra model dir\n", opt.weights_dir.c_str());
            else
                fprintf(stderr, "[jarvis] model '%s' not found in %s\n", opt.model_name.c_str(), opt.weights_dir.c_str());
            return 1;
        }
        if (opt.model_name.empty())
            printf("[jarvis] default Zyphra stack → %s\n", cfg.model_name.c_str());
        mgr_ref.discover();
        mgr_ref.set_strategy(SelectionStrategy::FASTEST);
        mgr_ref.set_fallback_policy(FallbackPolicy::SEQUENTIAL);
        if (!mgr_ref.init(cfg, opt.weights_dir, {})) {
            fprintf(stderr, "[jarvis] engine init failed for %s\n", cfg.model_name.c_str());
            return 1;
        }
        if (!g_tokenizer.load_from_gguf(cfg.model_path)) {
            fprintf(stderr, "[jarvis] warning: no tokenizer in model — replies may be empty\n");
        }
        printf("[jarvis] engine ready: %s (%s)\n", cfg.model_name.c_str(), cfg.model_path.c_str());
        mgr = &mgr_ref;
    }

    // ── TTS (piper) ──
    if (!opt.piper_model.empty()) {
        if (!s.tts.load(opt.piper_bin, opt.piper_model))
            fprintf(stderr, "[jarvis] TTS disabled (no piper model)\n");
    }

    // ── Voice mode: mic → VAD → STT → LLM ──
    if (!opt.text_mode) {
        if (!s.stt.load(opt.whisper_gguf)) {
            fprintf(stderr, "[jarvis] failed to load whisper model %s\n", opt.whisper_gguf.c_str());
            return 1;
        }
        printf("[jarvis] listening on '%s' — press Ctrl-C to quit.\n", opt.mic_device.c_str());

        Capture cap;
        bool ok = cap.start(16000, opt.mic_device, [&](const float* pcm, int n) {
            s.vad.process(pcm, n);
            if (!s.vad.is_speaking() && !s.vad.get_last_utterance().empty()) {
                if (s.busy.exchange(true)) { fprintf(stderr, "[jarvis] busy — dropped utterance\n"); }
                else {
                    auto utt = s.vad.get_last_utterance();
                    std::thread([&s, mgr, utt = std::move(utt)]() mutable {
                        std::string text = s.stt.transcribe(utt.data(), (int)utt.size());
                        if (!text.empty()) handle_utterance(s, mgr, text);
                        s.busy = false;
                    }).detach();
                }
            }
        });
        if (!ok) { fprintf(stderr, "[jarvis] arecord failed to start\n"); return 1; }
        for (;;) std::this_thread::sleep_for(std::chrono::seconds(1));  // until Ctrl-C
        cap.stop();
        return 0;
    }

    // ── Text mode: stdin lines ──
    printf("[jarvis] text chat — type and press Enter. Ctrl-D to quit.\n");
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        handle_utterance(s, mgr, line);
    }
    if (node_thread.joinable()) {
        node_svr.stop();
        node_thread.join();
    }
    return 0;
}
