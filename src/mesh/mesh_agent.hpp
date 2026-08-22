// mesh_agent.hpp — the proactive self-awareness loop (add-on to PeerDiscovery).
//
// The node's mesh surface (PeerDiscovery + /v1/mesh/* API) records every ask
// and answer itself, so the network works with no agent at all. The agent is
// the extra layer that makes a node PROACTIVE: it notices new peers and
// reaches out first ("hi, want to hook up and integrate?") and auto-answers
// incoming intros with a templated accept (or a local-model reply when one
// is configured). The DSH brain (integrations/dsh) is the LLM-driven
// replacement for this loop, driven from outside the engine.
#pragma once

#include <atomic>
#include <string>
#include <thread>

#include "mesh/mesh.hpp"
#include "mesh/peer_discovery.hpp"

namespace mesh {

class MeshAgent {
public:
    MeshAgent(PeerDiscovery& disc, const MeshConfig& cfg);
    ~MeshAgent();

    MeshAgent(const MeshAgent&) = delete;
    MeshAgent& operator=(const MeshAgent&) = delete;

    void start();
    void stop();

    // Called by the /v1/mesh/ask handler after the node recorded the ask:
    // reply with a templated accept so the hookup completes immediately.
    void on_inbound_ask(const std::string& ask_id, const std::string& from_id,
                        const std::string& from_name, const std::string& type,
                        const std::string& question);

private:
    void loop();                              // periodic "new peer?" scan
    void greet_peer(PeerRecord& rec);         // build + send intro ask
    std::string build_question(const NodeIdentity& peer) const;
    void auto_answer(const AskRecord& a);     // templated accept reply
    bool ask_peer(const NodeIdentity& peer, const std::string& question);
    bool http_post(const std::string& api_base, const std::string& path,
                   const nlohmann::json& body, nlohmann::json& out) const;

    PeerDiscovery& disc_;
    MeshConfig     cfg_;
    std::atomic<bool> running_{false};

    std::thread loop_thread_;
};

} // namespace mesh
