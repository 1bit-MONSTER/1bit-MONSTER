// dispatch.cpp — fleet dispatch: route LLM turns to the best mesh node.
#include "mesh/dispatch.hpp"

#include <cstdio>

#include <httplib.h>

#include <nlohmann/json.hpp>

namespace mesh {

namespace {

bool caps_offer(const MeshCapabilities& caps, const std::string& model) {
    if (model.empty()) return true;
    // Exact name or prefix match against advertised models.
    for (const auto& m : caps.models) {
        if (m.name == model) return true;
        if (model.size() >= 3 && m.name.rfind(model, 0) == 0) return true;
    }
    return false;
}

bool has_feature(const MeshCapabilities& caps, const std::string& feature) {
    for (const auto& f : caps.features)
        if (f == feature) return true;
    return false;
}

} // namespace

MeshDispatcher::MeshDispatcher(PeerDiscovery& disc, const NodeIdentity& me)
    : disc_(disc), me_(me) {}

std::vector<std::shared_ptr<PeerRecord>> MeshDispatcher::candidates(const std::string& model) const {
    auto peers = disc_.snapshot();

    // Pass 1: peers advertising the exact model.
    std::vector<std::shared_ptr<PeerRecord>> best;
    std::vector<std::shared_ptr<PeerRecord>> chat;
    std::vector<std::shared_ptr<PeerRecord>> any;
    for (auto& rec : peers) {
        if (caps_offer(rec.node.caps, model)) best.push_back(std::make_shared<PeerRecord>(rec));
        else if (has_feature(rec.node.caps, "chat")) chat.push_back(std::make_shared<PeerRecord>(rec));
        else any.push_back(std::make_shared<PeerRecord>(rec));
    }
    if (!best.empty()) return best;
    if (!chat.empty()) return chat;
    return any;
}

bool MeshDispatcher::post_chat(const NodeIdentity& node, const std::string& model,
                               const std::string& system, const std::string& user,
                               int max_tokens, DispatchResult& out) const {
    nlohmann::json body{
        {"model", model.empty() ? "auto" : model},
        {"messages", nlohmann::json::array()},
        {"max_tokens", max_tokens},
        {"temperature", 0.6},
        {"stream", false},
    };
    if (!system.empty()) {
        body["messages"].push_back({{"role", "system"}, {"content", system}});
    }
    body["messages"].push_back({{"role", "user"}, {"content", user}});

    // api_base may carry a path (".../v1"); httplib::Client drops it, so
    // split it off and prepend to the request path (same trick as the
    // mesh agent's http_post).
    std::string base = node.api_base;
    std::string base_path;
    size_t scheme_end = base.find("://");
    size_t host_start = scheme_end == std::string::npos ? 0 : scheme_end + 3;
    size_t slash = base.find('/', host_start);
    if (slash != std::string::npos) {
        base_path = base.substr(slash);
        base      = base.substr(0, slash);
    }

    httplib::Client cli(base);
    cli.set_connection_timeout(3, 0);
    cli.set_read_timeout(30, 0);
    auto res = cli.Post(base_path + "/chat/completions", body.dump(), "application/json");
    if (!res || res->status != 200) {
        out.error = "HTTP " + (res ? std::to_string(res->status) : std::string("no response"));
        return false;
    }
    try {
        nlohmann::json j = nlohmann::json::parse(res->body);
        auto& choices = j["choices"];
        if (choices.empty()) {
            out.error = "empty choices";
            return false;
        }
        out.reply = choices[0]["message"].value("content", std::string{});
        out.model = j.value("model", model);
        return !out.reply.empty();
    } catch (const std::exception& e) {
        out.error = e.what();
        return false;
    }
}

DispatchResult MeshDispatcher::chat(const std::string& model, const std::string& system,
                                    const std::string& user, int max_tokens) {
    DispatchResult result;
    auto cands = candidates(model);
    for (const auto& rec : cands) {
        DispatchResult r;
        if (post_chat(rec->node, model, system, user, max_tokens, r)) {
            r.node_id   = rec->node.id;
            r.node_name = rec->node.name;
            r.ok        = true;
            std::printf("📡 [dispatch] %s → %s (%s)\n",
                        me_.name.c_str(), rec->node.name.c_str(),
                        r.model.c_str());
            return r;
        }
    }
    result.error = cands.empty() ? "no peers on the mesh"
                                 : ("all " + std::to_string(cands.size()) +
                                    " candidate(s) failed: " + cands.front()->node.name);
    return result;
}

} // namespace mesh
