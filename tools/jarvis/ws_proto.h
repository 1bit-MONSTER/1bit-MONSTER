#pragma once
#include "voice_session.h"
#include <nlohmann/json.hpp>
#include <string>

namespace jarvis {

// Parse a WS control text frame. Returns false on malformed JSON.
bool ws_parse_control(const std::string& text, std::string& type, nlohmann::json& payload);

// Meta payload for a session handshake.
std::string ws_meta_json(bool session);

// State payload for the state machine.
std::string ws_state_json(SessionState st);

} // namespace jarvis
