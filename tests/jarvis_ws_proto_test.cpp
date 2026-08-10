#include "../tools/jarvis/ws_proto.h"
#include <cassert>
#include <cstdio>

using namespace jarvis;

int main() {
    std::string type;
    nlohmann::json payload;
    assert(ws_parse_control(R"({"type":"start"})", type, payload) && type == "start");
    assert(!ws_parse_control("not json", type, payload));
    auto meta = nlohmann::json::parse(ws_meta_json(true));
    assert(meta["session"] == true && meta["format"] == "pcm16");
    auto st = nlohmann::json::parse(ws_state_json(SessionState::Speaking));
    assert(st["type"] == "state" && st["state"] == "speaking");
    std::printf("PASS ws_proto_test\n");
    return 0;
}
