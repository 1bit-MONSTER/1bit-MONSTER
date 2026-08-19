// mesh_peer.cpp — standalone 1bit-MONSTER Mesh node (demo / smoke test).
//
// A minimal node that runs the full mesh stack — identity, UDP-multicast
// discovery, /v1/mesh/* HTTP API, and the self-awareness agent loop — with
// NO model weights and NO inference backend. Two instances on a machine (or
// LAN) discover each other out of the box and start asking questions:
//
//   ./build/mesh_peer --name alice --port 18088 --state-dir /tmp/mesh-a
//   ./build/mesh_peer --name bob   --port 18089 --state-dir /tmp/mesh-b
//
// Then: curl http://127.0.0.1:18088/v1/mesh/peers
//
// Build: cmake --build . --target mesh_peer -j8
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

#include <httplib.h>

#include "mesh/mesh.hpp"
#include "mesh/mesh_agent.hpp"
#include "mesh/node_identity.hpp"
#include "mesh/peer_api.hpp"
#include "mesh/peer_discovery.hpp"

static std::atomic<bool> g_stop{false};
static void handle_sigint(int) { g_stop = true; }
static bool g_stub_chat = false;  // --stub-chat: serve a canned chat endpoint

static void usage(const char* argv0) {
    std::printf(
        "usage: %s [options]\n"
        "  --name NAME           friendly node name (default: hostname)\n"
        "  --port PORT           HTTP listen port (default: 8088)\n"
        "  --state-dir DIR       identity persistence dir (default: ~/.cache/1bit-mesh)\n"
        "  --mesh-group GROUP    multicast group (default: 239.255.42.42)\n"
        "  --mesh-port PORT      multicast port (default: 42424)\n"
        "  --announce N          announce interval seconds (default: 5)\n"
        "  --ttl N               peer expiry seconds (default: 15)\n"
        "  --no-agent            disable the self-awareness loop\n"
        "  --agent-interval N    agent scan interval seconds (default: 10)\n"
        "  --models 'A:bk,B:bk'  advertise models (name:backend, comma-separated)\n"
        "  --stub-chat           serve POST /v1/chat/completions with a canned\n"
        "                        reply (stand-in for a full server; for fleet\n"
        "                        dispatch demos and tests)\n"
        "  --verbose             print discovery chatter\n"
        "  -h, --help            show this help\n",
        argv0);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    mesh::MeshConfig cfg;
    cfg.verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "mesh_peer: %s requires a value\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--name") cfg.name = next("--name");
        else if (a == "--port") cfg.http_port = static_cast<uint16_t>(std::atoi(next("--port")));
        else if (a == "--state-dir") cfg.state_dir = next("--state-dir");
        else if (a == "--mesh-group") cfg.mesh_group = next("--mesh-group");
        else if (a == "--mesh-port") cfg.mesh_port = static_cast<uint16_t>(std::atoi(next("--mesh-port")));
        else if (a == "--announce") cfg.announce_interval_s = std::atoi(next("--announce"));
        else if (a == "--ttl") cfg.peer_ttl_s = std::atoi(next("--ttl"));
        else if (a == "--no-agent") cfg.agent_enabled = false;
        else if (a == "--agent-interval") cfg.agent_interval_s = std::atoi(next("--agent-interval"));
        else if (a == "--models") { setenv("ONEBIT_MESH_MODELS", next("--models"), 1); }
        else if (a == "--stub-chat") g_stub_chat = true;
        else if (a == "--verbose") cfg.verbose = true;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "mesh_peer: unknown option %s\n", a.c_str()); usage(argv[0]); return 2; }
    }
    if (cfg.state_dir.empty()) cfg.state_dir = mesh::default_state_dir();

    mesh::NodeIdentity me = mesh::load_or_create_identity(cfg);
    // Advertise the machine's reachable LAN address (not loopback) so peers
    // on other hosts can dial this node's api_base. The LAN IP is also
    // reachable from this same machine (single-box demos keep working).
    me.host = mesh::detect_local_ip();
    me.api_base = mesh::make_api_base(me.host, cfg.http_port);

    // --stub-chat: advertise chat capability so fleet dispatchers route to us.
    if (g_stub_chat) {
        bool has_chat = false;
        for (const auto& f : me.caps.features) has_chat |= (f == "chat");
        if (!has_chat) me.caps.features.push_back("chat");
        if (me.caps.models.empty()) {
            mesh::MeshModelInfo mi;
            mi.name = "stub-llm";
            mi.backend = "stub";
            me.caps.models.push_back(mi);
        }
    }

    mesh::PeerDiscovery disc(cfg, me);
    if (!disc.start()) {
        std::fprintf(stderr, "mesh_peer: discovery failed to start\n");
        return 1;
    }

    std::unique_ptr<mesh::MeshAgent> agent;
    if (cfg.agent_enabled) {
        agent = std::make_unique<mesh::MeshAgent>(disc, cfg);
        agent->start();
    }

    httplib::Server svr;
    mesh::register_mesh_handlers(svr, disc, agent.get());

    // --stub-chat: a canned OpenAI-compatible chat endpoint — lets fleet
    // dispatchers (JARVIS --mesh-dispatch, DSH brains) treat this peer as a
    // full model server for demos and tests.
    if (g_stub_chat) {
        svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
            try {
                nlohmann::json body = nlohmann::json::parse(req.body);
                std::string model = body.value("model", std::string("auto"));
                std::string user;
                for (const auto& m : body["messages"]) {
                    if (m.value("role", "") == "user") user = m.value("content", std::string{});
                }
                std::string reply = "[stub:" + me.name + ":" + model + "] re: \"" +
                                    (user.size() > 80 ? user.substr(0, 80) + "…" : user) + "\"";
                nlohmann::json out{
                    {"id", "stub-" + me.id.substr(0, 8)},
                    {"model", model},
                    {"choices", nlohmann::json::array({nlohmann::json{
                        {"index", 0},
                        {"message", {{"role", "assistant"}, {"content", reply}}},
                        {"finish_reason", "stop"},
                    }})},
                };
                res.set_content(out.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
            }
        });
    }

    svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        auto peers = disc.snapshot();
        std::string body = "1bit-MONSTER Mesh node '" + me.name + "' (" + me.id +
                           ")\nmesh proto " + me.proto +
                           "\npeers: " + std::to_string(peers.size()) + "\n";
        res.set_content(body, "text/plain");
    });

    signal(SIGINT, handle_sigint);
    signal(SIGTERM, handle_sigint);

    std::printf("╭──────────────────────────────────────────────────────╮\n");
    std::printf("│  1bit-MONSTER Mesh — self-aware network node         │\n");
    std::printf("╰──────────────────────────────────────────────────────╯\n");
    std::printf("  node      : %s (%s)\n", me.name.c_str(), me.id.c_str());
    std::printf("  api       : http://127.0.0.1:%u/v1 (mesh/1.0)\n", cfg.http_port);
    std::printf("  beacon    : %s:%u every %ds (ttl %ds)\n",
                cfg.mesh_group.c_str(), cfg.mesh_port,
                cfg.announce_interval_s, cfg.peer_ttl_s);
    std::printf("  agent     : %s\n", agent ? "self-awareness loop ON" : "off");
    std::printf("  state     : %s\n\n", cfg.state_dir.c_str());

    std::thread listener([&]() {
        if (!svr.listen("0.0.0.0", cfg.http_port)) {
            std::fprintf(stderr, "mesh_peer: listen(%u) failed\n", cfg.http_port);
        }
    });

    while (!g_stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    std::printf("\n[mesh] %s: shutting down...\n", me.name.c_str());
    svr.stop();
    listener.join();
    if (agent) agent->stop();
    disc.stop();
    return 0;
}
