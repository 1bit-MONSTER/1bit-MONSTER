// dispatch.hpp — fleet dispatch: route LLM turns to the best mesh node.
//
// JARVIS (and any engine client) can run with NO local model: the mic, STT
// and TTS stay local, but each LLM turn is dispatched over the mesh to the
// sibling install that best serves the requested model (or any node that
// speaks chat). This is what makes a 1bit fleet "self-aware": every install
// knows what its neighbors can do, and work flows to the machine that can
// do it.
//
// Dispatch is plain OpenAI-compatible HTTP (POST /v1/chat/completions) —
// every unified_server node and any peer running --stub-chat speaks it.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mesh/mesh.hpp"
#include "mesh/peer_discovery.hpp"

namespace mesh {

struct DispatchResult {
    bool   ok = false;
    std::string node_id;    // which node answered
    std::string node_name;
    std::string model;      // model actually served
    std::string reply;      // assistant text
    std::string error;
};

class MeshDispatcher {
public:
    MeshDispatcher(PeerDiscovery& disc, const NodeIdentity& me);

    // Run one chat turn. Candidates:
    //   1. peers whose caps advertise `model` (or any model when empty),
    //   2. peers advertising the "chat" feature,
    //   3. any alive peer.
    // Posts the OpenAI-compatible request and returns the assistant reply.
    DispatchResult chat(const std::string& model, const std::string& system,
                        const std::string& user, int max_tokens = 256);

    // Peers that can serve `model` (best → worst).
    std::vector<std::shared_ptr<PeerRecord>> candidates(const std::string& model) const;

private:
    bool post_chat(const NodeIdentity& node, const std::string& model,
                   const std::string& system, const std::string& user,
                   int max_tokens, DispatchResult& out) const;

    PeerDiscovery& disc_;
    NodeIdentity   me_;
};

} // namespace mesh
