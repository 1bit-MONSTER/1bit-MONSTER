// peer_discovery.hpp — LAN peer discovery + live peer registry.
//
// Announces this node's identity card on a UDP multicast group and listens
// for sibling installs announcing theirs. Peers land in a registry with a
// TTL (a peer that stops announcing expires and drops off the list).
// Works out of the box — no config, no central server. Loopback is enabled
// so two instances on one machine discover each other (demo / smoke test).
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "mesh/mesh.hpp"

namespace mesh {

// One discovered sibling install.
struct PeerRecord {
    NodeIdentity node;
    int64_t      first_seen_ms = 0;
    int64_t      last_seen_ms  = 0;
    bool         integrated = false;  // handshake completed (hooked up)
    bool         greeted    = false;  // agent already sent an intro ask
};

// A question this node received (or sent). The node records every ask it
// handles so the conversation is observable even with no agent attached.
struct AskRecord {
    std::string ask_id;
    std::string from_id;
    std::string from_name;
    std::string type;        // "intro" | "question" | "integration_offer"
    std::string question;
    std::string sent_at_ms;  // when the ask was recorded
    std::string answer;      // filled once answered
    bool        answered = false;
};

class PeerDiscovery {
public:
    PeerDiscovery(const MeshConfig& cfg, const NodeIdentity& me);
    ~PeerDiscovery();

    // Non-copyable.
    PeerDiscovery(const PeerDiscovery&) = delete;
    PeerDiscovery& operator=(const PeerDiscovery&) = delete;

    // Create the socket, join the multicast group, spawn the announce /
    // listen / sweeper threads. Returns false on socket failure.
    bool start();

    // Stop threads and close the socket (idempotent).
    void stop();

    const NodeIdentity& self() const { return me_; }

    // Upsert a peer from an announce beacon or a handshake request body.
    // Returns true if the peer is new (not seen before).
    bool upsert_peer(const NodeIdentity& node);

    // Live (non-expired) peers.
    std::vector<PeerRecord> snapshot() const;

    // Find a peer by node id (nullptr if unknown/expired).
    std::shared_ptr<PeerRecord> find(const std::string& id) const;

    // Mark a peer as integrated (handshake completed) / greeted (intro sent).
    void mark_integrated(const std::string& id);
    void mark_greeted(const std::string& id);

    // ── Conversation log (node-level, works with or without an agent) ──
    // Record an ask received from a peer.
    void record_inbound_ask(const std::string& ask_id, const std::string& from_id,
                            const std::string& from_name, const std::string& type,
                            const std::string& question);
    // Record an ask this node sent to a peer (outbound half of a chat).
    void record_outbound_ask(const std::string& ask_id, const std::string& question);
    // Record the answer to a local ask. When from_id/from_name are given and
    // the ask is unknown (sent by an external brain), a synthetic record is
    // appended so the conversation stays visible on this side.
    void record_answer(const std::string& ask_id, const std::string& answer,
                       const std::string& from_id = std::string{},
                       const std::string& from_name = std::string{});
    // All asks this node has handled, oldest first.
    std::vector<AskRecord> asks_snapshot() const;

    // A unique ask id for a message this node sends ("ask-<id8>-<n>").
    std::string next_ask_id();

private:
    void announce_loop();
    void listen_loop();
    void sweeper_loop();
    bool send_beacon(int seq);

    MeshConfig      cfg_;
    NodeIdentity    me_;
    int             sock_ = -1;
    std::atomic<bool> running_{false};
    std::atomic<int>  announce_seq_{0};

    mutable std::mutex reg_mu_;
    std::map<std::string, std::shared_ptr<PeerRecord>> registry_;

    mutable std::mutex inbox_mu_;
    std::deque<AskRecord> inbox_;
    std::atomic<int> ask_seq_{0};

    std::thread announce_thread_;
    std::thread listen_thread_;
    std::thread sweeper_thread_;
};

} // namespace mesh
