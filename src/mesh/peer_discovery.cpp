// peer_discovery.cpp — LAN peer discovery + live peer registry.
#include "mesh/peer_discovery.hpp"

#include <chrono>
#include <cstring>
#include <cstdio>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

namespace mesh {

namespace {

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

} // namespace

PeerDiscovery::PeerDiscovery(const MeshConfig& cfg, const NodeIdentity& me)
    : cfg_(cfg), me_(me) {}

PeerDiscovery::~PeerDiscovery() { stop(); }

bool PeerDiscovery::start() {
    if (running_.exchange(true)) return true;  // already running

    sock_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_ < 0) {
        std::fprintf(stderr, "[mesh] peer_discovery: socket() failed: %s\n",
                     std::strerror(errno));
        running_ = false;
        return false;
    }

    int reuse = 1;
    setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    struct sockaddr_in bind_addr{};
    bind_addr.sin_family      = AF_INET;
    bind_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    bind_addr.sin_port        = htons(cfg_.mesh_port);
    if (bind(sock_, reinterpret_cast<struct sockaddr*>(&bind_addr),
             sizeof(bind_addr)) < 0) {
        std::fprintf(stderr, "[mesh] peer_discovery: bind(%u) failed: %s\n",
                     cfg_.mesh_port, std::strerror(errno));
        close(sock_);
        sock_ = -1;
        running_ = false;
        return false;
    }

    // Join the multicast group (any interface).
    struct ip_mreq mreq{};
    inet_pton(AF_INET, cfg_.mesh_group.c_str(), &mreq.imr_multiaddr);
    mreq.imr_interface.s_addr = htonl(INADDR_ANY);
    if (setsockopt(sock_, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
        std::fprintf(stderr, "[mesh] peer_discovery: join %s failed: %s\n",
                     cfg_.mesh_group.c_str(), std::strerror(errno));
    }

    // Loopback ON so a single machine's instances find each other (demo,
    // smoke test, dev machines). TTL 1 = LAN only.
    unsigned char loop = 1;
    setsockopt(sock_, IPPROTO_IP, IP_MULTICAST_LOOP, &loop, sizeof(loop));
    unsigned char ttl = 1;
    setsockopt(sock_, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl));

    // Listen loop polls every 500 ms so it can observe the stop flag.
    struct timeval tv{};
    tv.tv_sec  = 0;
    tv.tv_usec = 500000;
    setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    announce_thread_ = std::thread([this] { announce_loop(); });
    listen_thread_   = std::thread([this] { listen_loop(); });
    sweeper_thread_  = std::thread([this] { sweeper_loop(); });

    std::printf("[mesh] %s: announcing on %s:%u (proto %s)\n",
                me_.name.c_str(), cfg_.mesh_group.c_str(), cfg_.mesh_port,
                me_.proto.c_str());
    return true;
}

void PeerDiscovery::stop() {
    if (!running_.exchange(false)) return;
    if (sock_ >= 0) {
        shutdown(sock_, SHUT_RDWR);
        close(sock_);
        sock_ = -1;
    }
    if (announce_thread_.joinable()) announce_thread_.join();
    if (listen_thread_.joinable())   listen_thread_.join();
    if (sweeper_thread_.joinable())  sweeper_thread_.join();
}

bool PeerDiscovery::send_beacon(int seq) {
    nlohmann::json beacon{
        {"type", "announce"},
        {"seq", seq},
        {"ts", now_ms()},
        {"node", me_},
    };
    std::string payload = beacon.dump();

    struct sockaddr_in dst{};
    dst.sin_family = AF_INET;
    dst.sin_port   = htons(cfg_.mesh_port);
    inet_pton(AF_INET, cfg_.mesh_group.c_str(), &dst.sin_addr);

    ssize_t n = sendto(sock_, payload.data(), payload.size(), 0,
                       reinterpret_cast<struct sockaddr*>(&dst), sizeof(dst));
    return n == static_cast<ssize_t>(payload.size());
}

void PeerDiscovery::announce_loop() {
    // Fire an immediate beacon so peers notice us right away, then every
    // announce_interval_s thereafter.
    send_beacon(announce_seq_.fetch_add(1));
    while (running_.load()) {
        for (int i = 0; i < cfg_.announce_interval_s * 2 && running_.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        if (!running_.load()) break;
        if (cfg_.verbose) {
            std::printf("[mesh] %s: announcing...\n", me_.name.c_str());
        }
        send_beacon(announce_seq_.fetch_add(1));
    }
}

void PeerDiscovery::listen_loop() {
    char buf[65536];
    struct sockaddr_in src{};
    socklen_t src_len = sizeof(src);
    while (running_.load()) {
        ssize_t n = recvfrom(sock_, buf, sizeof(buf) - 1, 0,
                             reinterpret_cast<struct sockaddr*>(&src), &src_len);
        if (n <= 0) continue;  // timeout or error → poll stop flag again
        buf[n] = '\0';
        try {
            nlohmann::json msg = nlohmann::json::parse(buf);
            if (msg.value("type", "") != "announce") continue;
            if (!msg.contains("node")) continue;
            NodeIdentity peer = msg["node"].get<NodeIdentity>();
            if (peer.id.empty() || peer.id == me_.id) continue;  // not ourselves
            bool is_new = upsert_peer(peer);
            if (is_new && cfg_.verbose) {
                std::printf("[mesh] %s: new peer %s (%s) via %s:%u\n",
                            me_.name.c_str(), peer.name.c_str(), peer.id.c_str(),
                            peer.host.c_str(), peer.port);
            }
        } catch (...) {
            /* ignore malformed beacons from non-mesh sources */
        }
    }
}

void PeerDiscovery::sweeper_loop() {
    const int64_t ttl_ms = static_cast<int64_t>(cfg_.peer_ttl_s) * 1000;
    while (running_.load()) {
        for (int i = 0; i < cfg_.peer_ttl_s && running_.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        if (!running_.load()) break;
        const int64_t cutoff = now_ms() - ttl_ms;
        std::lock_guard<std::mutex> lk(reg_mu_);
        for (auto it = registry_.begin(); it != registry_.end();) {
            if (it->second->last_seen_ms < cutoff) {
                if (cfg_.verbose) {
                    std::printf("[mesh] %s: peer %s expired (no beacon for %ds)\n",
                                me_.name.c_str(), it->second->node.name.c_str(),
                                cfg_.peer_ttl_s);
                }
                it = registry_.erase(it);
            } else {
                ++it;
            }
        }
    }
}

bool PeerDiscovery::upsert_peer(const NodeIdentity& node) {
    std::lock_guard<std::mutex> lk(reg_mu_);
    auto it = registry_.find(node.id);
    if (it != registry_.end()) {
        it->second->last_seen_ms = now_ms();
        it->second->node         = node;  // refresh card (capabilities may change)
        return false;
    }
    auto rec = std::make_shared<PeerRecord>();
    rec->node          = node;
    rec->first_seen_ms = now_ms();
    rec->last_seen_ms  = now_ms();
    registry_[node.id] = rec;
    return true;
}

std::vector<PeerRecord> PeerDiscovery::snapshot() const {
    std::lock_guard<std::mutex> lk(reg_mu_);
    std::vector<PeerRecord> out;
    out.reserve(registry_.size());
    for (const auto& [id, rec] : registry_) out.push_back(*rec);
    return out;
}

std::shared_ptr<PeerRecord> PeerDiscovery::find(const std::string& id) const {
    std::lock_guard<std::mutex> lk(reg_mu_);
    auto it = registry_.find(id);
    if (it == registry_.end()) return nullptr;
    return it->second;
}

void PeerDiscovery::mark_integrated(const std::string& id) {
    std::lock_guard<std::mutex> lk(reg_mu_);
    auto it = registry_.find(id);
    if (it != registry_.end()) it->second->integrated = true;
}

void PeerDiscovery::mark_greeted(const std::string& id) {
    std::lock_guard<std::mutex> lk(reg_mu_);
    auto it = registry_.find(id);
    if (it != registry_.end()) it->second->greeted = true;
}

// ── Conversation log ─────────────────────────────────────────────────────

namespace {

std::string now_str() {
    return std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::system_clock::now().time_since_epoch())
                              .count());
}

} // namespace

void PeerDiscovery::record_inbound_ask(const std::string& ask_id,
                                       const std::string& from_id,
                                       const std::string& from_name,
                                       const std::string& type,
                                       const std::string& question) {
    AskRecord a;
    a.ask_id     = ask_id;
    a.from_id    = from_id;
    a.from_name  = from_name;
    a.type       = type;
    a.question   = question;
    a.sent_at_ms = now_str();
    {
        std::lock_guard<std::mutex> lk(inbox_mu_);
        inbox_.push_back(a);
    }
    std::printf("📨 [mesh] %s: ask #%s from %s (%s): %s\n",
                me_.name.c_str(), ask_id.c_str(), from_name.c_str(),
                type.c_str(), question.c_str());
}

void PeerDiscovery::record_outbound_ask(const std::string& ask_id,
                                        const std::string& question) {
    AskRecord a;
    a.ask_id     = ask_id;
    a.from_id    = me_.id;
    a.from_name  = me_.name;
    a.type       = "intro";
    a.question   = question;
    a.sent_at_ms = now_str();
    {
        std::lock_guard<std::mutex> lk(inbox_mu_);
        inbox_.push_back(a);
    }
}

void PeerDiscovery::record_answer(const std::string& ask_id,
                                  const std::string& answer,
                                  const std::string& from_id,
                                  const std::string& from_name) {
    bool found = false;
    {
        std::lock_guard<std::mutex> lk(inbox_mu_);
        for (auto& a : inbox_) {
            if (a.ask_id == ask_id) {
                a.answer   = answer;
                a.answered = true;
                found = true;
                break;
            }
        }
    }
    if (!found) {
        // The ask was sent by an external brain (not recorded locally as
        // outbound) — synthesize the record so this side of the
        // conversation stays visible.
        AskRecord a;
        a.ask_id     = ask_id;
        a.from_id    = from_id;
        a.from_name  = from_name;
        a.type       = "reply";
        a.question   = "(outbound ask)";
        a.sent_at_ms = now_str();
        a.answer     = answer;
        a.answered   = true;
        std::lock_guard<std::mutex> lk(inbox_mu_);
        inbox_.push_back(a);
    }
    std::printf("✅ [mesh] %s: answer to ask #%s: %s\n",
                me_.name.c_str(), ask_id.c_str(), answer.c_str());
}

std::vector<AskRecord> PeerDiscovery::asks_snapshot() const {
    std::lock_guard<std::mutex> lk(inbox_mu_);
    return std::vector<AskRecord>(inbox_.begin(), inbox_.end());
}

std::string PeerDiscovery::next_ask_id() {
    return "ask-" + me_.id.substr(0, 8) + "-" + std::to_string(ask_seq_.fetch_add(1));
}

} // namespace mesh
