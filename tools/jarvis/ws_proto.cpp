#include "ws_proto.h"

namespace jarvis {

bool ws_parse_control(const std::string& text, std::string& type, nlohmann::json& payload) {
    try {
        payload = nlohmann::json::parse(text);
    } catch (...) { return false; }
    if (!payload.is_object() || !payload.contains("type") || !payload["type"].is_string()) return false;
    type = payload["type"].get<std::string>();
    return true;
}

std::string ws_meta_json(bool session) {
    nlohmann::json j = {{"type", "meta"}, {"session", session}};
    if (session) {
        j["sample_rate"] = 16000; j["channels"] = 1; j["format"] = "pcm16"; j["frame_ms"] = 20;
    } else {
        j["sample_rate"] = 24000; j["channels"] = 1; j["format"] = "float32";
    }
    return j.dump();
}

std::string ws_state_json(SessionState st) {
    const char* name = "idle";
    switch (st) {
        case SessionState::Idle: name = "idle"; break;
        case SessionState::Listening: name = "listening"; break;
        case SessionState::Processing: name = "processing"; break;
        case SessionState::Speaking: name = "speaking"; break;
    }
    return nlohmann::json{{"type", "state"}, {"state", name}}.dump();
}

} // namespace jarvis
