// mesh.hpp — 1bit-MONSTER Mesh: self-aware installs on the network.
//
// Every 1bit-MONSTER install becomes a node that, out of the box:
//   1. announces itself on the LAN (UDP multicast beacon),
//   2. discovers sibling installs and keeps a live peer registry,
//   3. exposes /v1/mesh/* HTTP endpoints for handshake / ask / answer,
//   4. runs an optional self-awareness agent loop that proactively greets
//      new peers and starts integration conversations.
//
// The wire contract between installs is documented in docs/mesh-protocol.md.
// This module is pure C++ (POSIX sockets + nlohmann json). No Python, no
// Node, no model weights required for discovery to work.
//
// Protocol (mesh/1.0):
//   announce  — UDP multicast beacon: {"type":"announce","seq":N,"node":{...}}
//   handshake — POST /v1/mesh/handshake (capability exchange / "hook up")
//   ask       — POST /v1/mesh/ask      (deliver a question to a node)
//   answer    — POST /v1/mesh/answer   (reply to an ask)
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace mesh {

// ── Wire constants (docs/mesh-protocol.md) ───────────────────────────────
inline constexpr const char* kProtoName   = "mesh/1.0";
inline constexpr const char* kMeshGroup   = "239.255.42.42";  // reserved 1bit group
inline constexpr uint16_t    kMeshPort    = 42424;
inline constexpr const char* kEngineName  = "1bit-MONSTER";
inline constexpr int         kAnnounceIntervalSec = 5;   // beacon period
inline constexpr int         kPeerTtlSec          = 15;  // peer expiry (3 beacons)

// ── Configuration (flags / env, defaults work out of the box) ───────────
struct MeshConfig {
    std::string name;          // friendly node name; default = hostname
    uint16_t    http_port  = 8088;   // this node's HTTP listen port (server port)
    std::string mesh_group = kMeshGroup;
    uint16_t    mesh_port  = kMeshPort;
    int         announce_interval_s = kAnnounceIntervalSec;
    int         peer_ttl_s          = kPeerTtlSec;
    std::string state_dir;           // where the persistent node identity lives
    bool        agent_enabled = true; // self-awareness loop (greets new peers)
    int         agent_interval_s = 10; // how often the agent looks for new peers
    std::string agent_model;          // empty → templated questions (zero-model)
    bool        verbose = false;
};

// A model this node serves (capability advertisement).
struct MeshModelInfo {
    std::string name;     // "Qwen3-4B"
    std::string backend;  // "npu_flm", "ggml_vulkan", "hip", ...
    std::string quant;    // optional quant string
};

// The "capability card" a node publishes so peers know what it can do.
struct MeshCapabilities {
    std::vector<MeshModelInfo> models;
    std::vector<std::string>   backends;   // active backend names
    std::vector<std::string>   features;   // "chat", "completions", "mesh", ...
};

// Identity card — the JSON node object in every mesh message.
struct NodeIdentity {
    std::string id;         // persistent UUID, generated on first run
    std::string name;       // friendly name
    std::string host;       // IP/hostname peers should connect to
    uint16_t    port = 8088;
    std::string api_base;   // http://<host>:<port>/v1
    std::string version;    // engine version string
    std::string proto = kProtoName;
    MeshCapabilities caps;

    // nlohmann json serialization (declared here, defined in node_identity.cpp)
    friend void to_json(nlohmann::json& j, const NodeIdentity& n);
    friend void from_json(const nlohmann::json& j, NodeIdentity& n);
};

} // namespace mesh
