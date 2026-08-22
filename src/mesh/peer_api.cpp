// peer_api.cpp — /v1/mesh/* HTTP handlers (registered on an httplib server).
#include "mesh/peer_api.hpp"

#include <cstdio>

#include <nlohmann/json.hpp>

#include "mesh/mesh_agent.hpp"

namespace mesh {

namespace {

void send_json(httplib::Response& res, const nlohmann::json& j, int status = 200) {
    res.status = status;
    res.set_content(j.dump(), "application/json");
}

} // namespace

void register_mesh_handlers(httplib::Server& svr, PeerDiscovery& disc,
                            MeshAgent* agent) {
    // GET /v1/mesh/me — introspection: who am I?
    svr.Get("/v1/mesh/me", [&disc](const httplib::Request&, httplib::Response& res) {
        send_json(res, nlohmann::json{{"ok", true}, {"node", disc.self()}});
    });

    // GET /v1/mesh/peers — the neighborhood.
    svr.Get("/v1/mesh/peers", [&disc](const httplib::Request&, httplib::Response& res) {
        auto peers = disc.snapshot();
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& p : peers) {
            arr.push_back(nlohmann::json{
                {"node", p.node},
                {"state", "alive"},
                {"first_seen_ms", p.first_seen_ms},
                {"last_seen_ms", p.last_seen_ms},
                {"integrated", p.integrated},
                {"greeted", p.greeted},
            });
        }
        send_json(res, nlohmann::json{{"ok", true}, {"count", arr.size()}, {"peers", arr}});
    });

    // POST /v1/mesh/handshake — "hook up": exchange capability cards.
    svr.Post("/v1/mesh/handshake", [&disc](const httplib::Request& req, httplib::Response& res) {
        try {
            nlohmann::json body = nlohmann::json::parse(req.body);
            if (!body.contains("node")) {
                send_json(res, nlohmann::json{{"ok", false}, {"error", "missing 'node'"}}, 400);
                return;
            }
            NodeIdentity peer = body["node"].get<NodeIdentity>();
            if (peer.id.empty()) {
                send_json(res, nlohmann::json{{"ok", false}, {"error", "node.id required"}}, 400);
                return;
            }
            bool is_new = disc.upsert_peer(peer);
            disc.mark_integrated(peer.id);
            if (is_new) {
                std::printf("🤝 [mesh] %s: handshake with new peer %s (%s)\n",
                            disc.self().name.c_str(), peer.name.c_str(), peer.id.c_str());
            }
            send_json(res, nlohmann::json{
                               {"ok", true},
                               {"new_peer", is_new},
                               {"me", disc.self()},
                               {"greeting",
                                "hi from " + disc.self().name + " — hooked up"},
                           });
        } catch (const std::exception& e) {
            send_json(res, nlohmann::json{{"ok", false}, {"error", e.what()}}, 400);
        }
    });

    // POST /v1/mesh/ask — deliver a question to this node.
    svr.Post("/v1/mesh/ask", [&disc, agent](const httplib::Request& req, httplib::Response& res) {
        try {
            nlohmann::json body = nlohmann::json::parse(req.body);
            std::string ask_id   = body.value("ask_id", std::string{});
            std::string from_id  = body.value("from", std::string{});
            std::string from_nm  = body.value("from_name", std::string{});
            std::string type     = body.value("type", std::string("question"));
            std::string question = body.value("question", std::string{});
            if (ask_id.empty() || from_id.empty()) {
                send_json(res, nlohmann::json{{"ok", false},
                                              {"error", "ask_id and from required"}},
                          400);
                return;
            }
            // The sender's card rides along → register/refresh the peer.
            // (Never register ourselves — a node that sends itself an ask
            // would otherwise appear as its own peer.)
            if (body.contains("node")) {
                NodeIdentity sender = body["node"].get<NodeIdentity>();
                if (!sender.id.empty() && sender.id == from_id && sender.id != disc.self().id) {
                    disc.upsert_peer(sender);
                }
            }
            // The node records every ask (works without an agent); the agent
            // (if present) additionally auto-answers the intro.
            disc.record_inbound_ask(ask_id, from_id, from_nm, type, question);
            if (agent) {
                agent->on_inbound_ask(ask_id, from_id, from_nm, type, question);
            }
            send_json(res, nlohmann::json{{"ok", true}, {"ask_id", ask_id}, {"received", true}});
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[mesh] /v1/mesh/ask handler: %s (%s)\n",
                         typeid(e).name(), e.what());
            send_json(res, nlohmann::json{{"ok", false}, {"error", e.what()}}, 400);
        }
    });

    // POST /v1/mesh/answer — reply to a previous ask.
    svr.Post("/v1/mesh/answer", [&disc](const httplib::Request& req, httplib::Response& res) {
        try {
            nlohmann::json body = nlohmann::json::parse(req.body);
            std::string ask_id  = body.value("ask_id", std::string{});
            std::string from_id = body.value("from", std::string{});
            std::string answer  = body.value("answer", std::string{});
            if (ask_id.empty()) {
                send_json(res, nlohmann::json{{"ok", false}, {"error", "ask_id required"}}, 400);
                return;
            }
            disc.record_answer(ask_id, answer, from_id, body.value("from_name", std::string{}));
            if (body.value("accept", false) && !from_id.empty()) {
                disc.mark_integrated(from_id);
            }
            send_json(res, nlohmann::json{{"ok", true}, {"ask_id", ask_id}});
        } catch (const std::exception& e) {
            send_json(res, nlohmann::json{{"ok", false}, {"error", e.what()}}, 400);
        }
    });

    // GET /v1/mesh/asks — conversation log (inbound + outbound).
    svr.Get("/v1/mesh/asks", [&disc](const httplib::Request&, httplib::Response& res) {
        auto asks = disc.asks_snapshot();
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& a : asks) {
            arr.push_back(nlohmann::json{
                {"ask_id", a.ask_id},
                {"from", a.from_id},
                {"from_name", a.from_name},
                {"type", a.type},
                {"question", a.question},
                {"received_at_ms", a.sent_at_ms},
                {"answer", a.answer},
                {"answered", a.answered},
            });
        }
        send_json(res, nlohmann::json{{"ok", true}, {"asks", arr}});
    });
}

} // namespace mesh
