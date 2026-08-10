// npu_flm_delta.h — multi-turn prompt decision for the FLM backend.
// Pure function so the KV-reuse logic is unit-testable without an NPU.
#pragma once
#include <string>

// Decide what to send to the FLM REPL for `prompt` given the previous
// prompt `last`. FLM keeps its KV cache resident across inserts, so a
// prompt that extends the previous one (client resends full history each
// turn) can continue the session with just the delta — no <<RESET>>, no
// full re-prefill of the whole conversation on every turn.
// Returns true when a session reset must precede the send.
// out_send receives the text to write (delta on continuation, full prompt
// on reset).
inline bool npu_flm_send_delta(const std::string& last, const std::string& prompt,
                               std::string& out_send) {
    if (!last.empty() && prompt.size() > last.size() &&
        prompt.compare(0, last.size(), last) == 0) {
        out_send = prompt.substr(last.size());
        if (!out_send.empty()) return false;  // continuation: no reset
    }
    out_send = prompt;  // new conversation / empty delta: full prompt + reset
    return true;
}

// Session-aware continuation decision for a server with multiple
// conversations sharing one FLM subprocess (the live session owns the
// device-resident KV cache). Returns true only when the request continues
// the live session: same session id AND the new prompt extends the previous
// one. out_delta receives the text to send without a reset. Any other case
// (no id, different session, diverged history) returns false — the caller
// must send the full prompt, which resets the session.
inline bool npu_flm_session_continue(const std::string& session_id,
                                     const std::string& current_session,
                                     const std::string& last_prompt,
                                     const std::string& prompt,
                                     std::string& out_delta) {
    if (session_id.empty() || session_id != current_session) return false;
    return !npu_flm_send_delta(last_prompt, prompt, out_delta);
}
