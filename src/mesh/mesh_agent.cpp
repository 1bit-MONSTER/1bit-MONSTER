// mesh_agent.cpp — the proactive self-awareness loop.
#include "mesh/mesh_agent.hpp"

#include <chrono>
#include <cstdio>

#include <httplib.h>

#include <nlohmann/json.hpp>

namespace mesh {

MeshAgent::MeshAgent(PeerDiscovery& disc, const MeshConfig& cfg)
    : disc_(disc), cfg_(cfg) {}

MeshAgent::~MeshAgent() { stop(); }

void MeshAgent::start() {
    if (running_.exchange(true)) return;
    loop_thread_ = std::thread([this] { loop(); });
}

void MeshAgent::stop() {
    if (!running_.exchange(false)) return;
    if (loop_thread_.joinable()) loop_thread_.join();
}

void MeshAgent::on_inbound_ask(const std::string& ask_id, const std::string& from_id,
                               const std::string& /*from_name*/,
                               const std::string& /*type*/,
                               const std::string& /*question*/) {
    auto peer = disc_.find(from_id);
    if (!peer) return;  // unknown peer — nothing to answer
    AskRecord a;
    a.ask_id = ask_id;
    a.from_id = from_id;
    auto_answer(a);
}

bool MeshAgent::ask_peer(const NodeIdentity& peer, const std::string& question) {
    std::string ask_id = disc_.next_ask_id();
    nlohmann::json body{
        {"from", disc_.self().id},
        {"from_name", disc_.self().name},
        {"ask_id", ask_id},
        {"type", "intro"},
        {"question", question},
        {"node", disc_.self()},
    };
    nlohmann::json resp;
    bool ok = http_post(peer.api_base, "/mesh/ask", body, resp);
    if (ok && resp.value("ok", false)) {
        disc_.mark_greeted(peer.id);
        disc_.record_outbound_ask(ask_id, question);
        std::printf("💬 [mesh] %s → %s: %s\n", disc_.self().name.c_str(),
                    peer.name.c_str(), question.c_str());
        return true;
    }
    return false;
}

void MeshAgent::loop() {
    while (running_.load()) {
        // Scan for peers we have not greeted yet.
        auto peers = disc_.snapshot();
        for (auto& rec : peers) {
            if (!running_.load()) break;
            if (rec.greeted) continue;
            if (!rec.node.api_base.empty()) {
                greet_peer(rec);
            }
        }
        for (int i = 0; i < cfg_.agent_interval_s && running_.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

void MeshAgent::greet_peer(PeerRecord& rec) {
    std::string q = build_question(rec.node);
    ask_peer(rec.node, q);
}

std::string MeshAgent::build_question(const NodeIdentity& peer) const {
    // Hook point for a DSH/LLM-driven brain: when cfg_.agent_model is set,
    // generate the question via the node's own /v1/chat/completions instead
    // of the template below. The template keeps the network functional with
    // zero model weights — the out-of-the-box guarantee.
    if (!cfg_.agent_model.empty()) {
        // TODO(mesh): model-driven question generation via local chat endpoint.
        // Left for the DSH plugin (integrations/dsh) which owns this prompt.
    }
    std::string mine = disc_.self().name;
    std::string mine_models;
    for (const auto& m : disc_.self().caps.models) {
        if (!mine_models.empty()) mine_models += ", ";
        mine_models += m.name + "@" + m.backend;
    }
    if (mine_models.empty()) mine_models = "(no models advertised)";

    std::string theirs = peer.name;
    std::string their_models;
    for (const auto& m : peer.caps.models) {
        if (!their_models.empty()) their_models += ", ";
        their_models += m.name + "@" + m.backend;
    }
    if (their_models.empty()) their_models = "(no models advertised)";

    return "Hi " + theirs + "! I'm " + mine + ", a 1bit-MONSTER node. "
           "I serve " + mine_models + ". You serve " + their_models + ". "
           "Want to hook up and integrate?";
}

void MeshAgent::auto_answer(const AskRecord& a) {
    std::string reply = "Yes! I'm " + disc_.self().name + ". "
                        "I serve ";
    bool first = true;
    for (const auto& m : disc_.self().caps.models) {
        if (!first) reply += ", ";
        reply += m.name + "@" + m.backend;
        first = false;
    }
    if (first) reply += "(no models advertised)";
    reply += ". Hook me up — route my requests through yours and vice versa.";

    // Send the answer back to the asker via their /v1/mesh/answer endpoint.
    auto peer = disc_.find(a.from_id);
    if (!peer) return;
    nlohmann::json body{
        {"ask_id", a.ask_id},
        {"from", disc_.self().id},
        {"from_name", disc_.self().name},
        {"answer", reply},
        {"accept", true},
    };
    nlohmann::json resp;
    if (http_post(peer->node.api_base, "/mesh/answer", body, resp) &&
        resp.value("ok", false)) {
        disc_.mark_integrated(peer->node.id);
        disc_.record_answer(a.ask_id, reply);  // local record too
    }
}

bool MeshAgent::http_post(const std::string& api_base, const std::string& path,
                          const nlohmann::json& body, nlohmann::json& out) const {
    // api_base may carry a path (e.g. "http://host:18089/v1"), but
    // httplib::Client(scheme_host_port) drops the path component. Extract it
    // and prepend it to the request path so the URL comes out right.
    std::string base = api_base;
    std::string base_path;
    size_t scheme_end = base.find("://");
    size_t host_start = scheme_end == std::string::npos ? 0 : scheme_end + 3;
    size_t slash = base.find('/', host_start);
    if (slash != std::string::npos) {
        base_path = base.substr(slash);
        base      = base.substr(0, slash);
    }
    httplib::Client cli(base);
    cli.set_connection_timeout(2, 0);
    cli.set_read_timeout(5, 0);
    auto res = cli.Post(base_path + path, body.dump(), "application/json");
    if (!res || res->status != 200) return false;
    try {
        out = nlohmann::json::parse(res->body);
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace mesh
