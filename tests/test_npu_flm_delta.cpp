// test_npu_flm_delta.cpp — multi-turn KV-reuse decision logic for the FLM
// backend. Pure host test: no NPU, no FLM binary required.
#include "../src/npu_flm_delta.h"
#include <cassert>
#include <cstdio>
#include <string>

int main() {
    std::string send;

    // New conversation (empty last): full prompt + reset
    assert(npu_flm_send_delta("", "user: hello\n", send) == true);
    assert(send == "user: hello\n");

    // Continuation: prompt extends last → delta only, no reset
    std::string last = "user: hello\n";
    std::string next = "user: hello\nassistant: hi\nuser: bye\n";
    assert(npu_flm_send_delta(last, next, send) == false);
    assert(send == "assistant: hi\nuser: bye\n");

    // Same-size prompt (edited history) → reset + full
    assert(npu_flm_send_delta(last, "user: hello!!\n", send) == true);
    assert(send == "user: hello!!\n");

    // Shorter prompt (history cleared) → reset + full
    assert(npu_flm_send_delta(next, "user: x\n", send) == true);
    assert(send == "user: x\n");

    // Empty delta (exact duplicate resend, e.g. retry after a failure) →
    // reset + full, self-healing rather than a no-op
    assert(npu_flm_send_delta(next, next, send) == true);
    assert(send == next);

    // Prefix lookalike ("user: hellooo" is NOT a prefix of "user: hello\n")
    // → reset + full, must not treat as continuation
    assert(npu_flm_send_delta(last, "user: hellooo\n", send) == true);
    assert(send == "user: hellooo\n");

    // ── Session-aware continuation ──
    std::string delta;
    std::string cur = "conv-1";
    // Same session, prompt extends last → continue with delta
    assert(npu_flm_session_continue("conv-1", cur, last, next, delta) == true);
    assert(delta == "assistant: hi\nuser: bye\n");
    // No session id → stateless full-send path
    assert(npu_flm_session_continue("", cur, last, next, delta) == false);
    // Different session → must reset (KV belongs to the other conversation)
    assert(npu_flm_session_continue("conv-2", cur, last, next, delta) == false);
    // Same session but diverged history → reset + full
    assert(npu_flm_session_continue("conv-1", cur, last, "user: hello!!\n", delta) == false);
    // No live session yet (empty current) → full
    assert(npu_flm_session_continue("conv-1", "", "", last, delta) == false);

    printf("test_npu_flm_delta: all assertions passed\n");
    return 0;
}
