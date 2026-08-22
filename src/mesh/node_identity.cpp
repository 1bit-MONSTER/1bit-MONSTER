// node_identity.cpp — persistent node identity + capability card.
#include "mesh/node_identity.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

namespace mesh {

void to_json(nlohmann::json& j, const NodeIdentity& n) {
    j = nlohmann::json{
        {"id", n.id},
        {"name", n.name},
        {"host", n.host},
        {"port", n.port},
        {"api_base", n.api_base},
        {"version", n.version},
        {"proto", n.proto},
        {"caps", nlohmann::json{
                     {"models", nlohmann::json::array()},
                     {"backends", n.caps.backends},
                     {"features", n.caps.features},
                 }},
    };
    for (const auto& m : n.caps.models) {
        nlohmann::json mj{{"name", m.name}, {"backend", m.backend}};
        if (!m.quant.empty()) mj["quant"] = m.quant;
        j["caps"]["models"].push_back(mj);
    }
}

void from_json(const nlohmann::json& j, NodeIdentity& n) {
    n.id       = j.value("id", std::string{});
    n.name     = j.value("name", std::string{});
    n.host     = j.value("host", std::string{});
    n.port     = static_cast<uint16_t>(j.value("port", 8088));
    n.api_base = j.value("api_base", std::string{});
    n.version  = j.value("version", std::string{});
    n.proto    = j.value("proto", kProtoName);
    if (j.contains("caps")) {
        const auto& c = j["caps"];
        if (c.contains("backends")) n.caps.backends = c["backends"].get<std::vector<std::string>>();
        if (c.contains("features")) n.caps.features = c["features"].get<std::vector<std::string>>();
        if (c.contains("models")) {
            for (const auto& m : c["models"]) {
                MeshModelInfo mi;
                mi.name    = m.value("name", std::string{});
                mi.backend = m.value("backend", std::string{});
                mi.quant   = m.value("quant", std::string{});
                if (!mi.name.empty()) n.caps.models.push_back(mi);
            }
        }
    }
}

std::string generate_uuid() {
    unsigned char b[16];
    std::ifstream ur("/dev/urandom", std::ios::binary);
    if (ur) {
        ur.read(reinterpret_cast<char*>(b), sizeof(b));
    } else {
        std::random_device rd;
        for (auto& byte : b) byte = static_cast<unsigned char>(rd());
    }
    b[6] = (b[6] & 0x0f) | 0x40;  // version 4
    b[8] = (b[8] & 0x3f) | 0x80;  // variant 10
    char buf[40];
    std::snprintf(buf, sizeof(buf),
                  "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                  b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
    return std::string(buf);
}

std::string default_hostname() {
    char buf[256] = {0};
    if (gethostname(buf, sizeof(buf) - 1) == 0 && buf[0] != '\0') {
        return std::string(buf);
    }
    return "1bit-node";
}

std::string default_state_dir() {
    const char* xdg = getenv("XDG_CACHE_HOME");
    if (xdg && *xdg) return std::string(xdg) + "/1bit-mesh";
    const char* home = getenv("HOME");
    if (home && *home) return std::string(home) + "/.cache/1bit-mesh";
    return "/tmp/1bit-mesh";
}

std::string make_api_base(const std::string& host, uint16_t port) {
    // host may already be "http://..." — normalize.
    std::string h = host;
    if (h.rfind("http://", 0) != 0 && h.rfind("https://", 0) != 0) {
        h = "http://" + h;
    }
    // Strip a trailing slash so callers can append "/mesh/ask" cleanly.
    while (!h.empty() && h.back() == '/') h.pop_back();
    return h + ":" + std::to_string(port) + "/v1";
}

std::string detect_local_ip() {
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) return "127.0.0.1";
    // Classic trick: a UDP "connect" picks the outbound route without
    // sending any packets; getsockname then reports the local address.
    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(80);
    inet_pton(AF_INET, "8.8.8.8", &addr.sin_addr);
    if (connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) == 0) {
        struct sockaddr_in local{};
        socklen_t len = sizeof(local);
        if (getsockname(fd, reinterpret_cast<struct sockaddr*>(&local), &len) == 0) {
            char buf[INET_ADDRSTRLEN] = {0};
            if (inet_ntop(AF_INET, &local.sin_addr, buf, sizeof(buf))) {
                close(fd);
                return std::string(buf);
            }
        }
    }
    close(fd);
    return "127.0.0.1";
}

NodeIdentity load_or_create_identity(const MeshConfig& cfg) {
    NodeIdentity me;
    me.id      = generate_uuid();
    me.name    = cfg.name.empty() ? default_hostname() : cfg.name;
    me.host    = "127.0.0.1";  // refined by caller when it knows its address
    me.port    = cfg.http_port;
    me.version = "1bit-MONSTER (mesh/1.0)";
    me.proto   = kProtoName;
    me.api_base = make_api_base(me.host, me.port);

    // Optional capability seeding from env (no model registry dependency):
    //   ONEBIT_MESH_MODELS="Qwen3-4B:npu_flm,ZAYA1-74B:ggml_vulkan"
    const char* models_env = getenv("ONEBIT_MESH_MODELS");
    if (models_env && *models_env) {
        std::string s(models_env);
        size_t start = 0;
        while (start < s.size()) {
            size_t comma = s.find(',', start);
            std::string item = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            if (!item.empty()) {
                size_t colon = item.find(':');
                MeshModelInfo mi;
                mi.name = colon == std::string::npos ? item : item.substr(0, colon);
                mi.backend = colon == std::string::npos ? "auto" : item.substr(colon + 1);
                me.caps.models.push_back(mi);
            }
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
    }
    me.caps.features = {"chat", "completions", "mesh"};

    // Persist / restore.
    std::error_code ec;
    std::filesystem::create_directories(cfg.state_dir, ec);
    if (ec) return me;  // not writable → in-memory identity for this session

    std::string state_file = cfg.state_dir + "/node.json";
    std::ifstream in(state_file);
    if (in.good()) {
        try {
            nlohmann::json j;
            in >> j;
            NodeIdentity saved = j.get<NodeIdentity>();
            if (!saved.id.empty()) {
                saved.name  = cfg.name.empty() ? saved.name : cfg.name;
                saved.port  = cfg.http_port;
                saved.host  = me.host;
                saved.api_base = make_api_base(saved.host, saved.port);
                saved.proto = kProtoName;
                return saved;
            }
        } catch (...) { /* corrupt state → regenerate */ }
    }

    std::ofstream out(state_file);
    if (out.good()) {
        out << nlohmann::json(me).dump(2);
    }
    return me;
}

} // namespace mesh
