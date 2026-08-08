# JARVIS Mobile M1 — Server WebSocket Uplink (Voice-Active Session) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use supo-subagent-driven-development (recommended) or supo-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `1bit jarvis-server` so a phone can open a WebSocket voice session: stream mic audio in, get voice-active conversation (VAD → Whisper → LLM → TTS) with state/transcript control messages, and audio streamed back — without touching the existing HTTP API or the existing WS downlink.

**Architecture:** A new `jarvis::VoiceSession` class (pure logic, no sockets: PCM16 in → VAD → utterance → callbacks) is wired into the existing raw-socket `WebSocketServer` (tools/jarvis/audio_stream.*) on a new path `/v1/voice/session`. The existing `/v1/audio/stream` downlink path stays untouched. `jarvis_server.cpp` connects VoiceSession callbacks to the existing Whisper/LLM/TTS plumbing (`get_whisper_model`, `resolve_model` + `unified_chat`/`ollama_chat`, `g_persona_mgr`, `g_context_mem`, `g_codec_tts`).

**Tech Stack:** C++23, existing repo deps only (no new libraries). VAD from `tools/jarvis/vad.h`. Wire format: PCM16 @ 16 kHz mono, 20 ms frames (640 bytes) — **deliberate deviation from the spec's "Opus"**: the engine links no Opus codec, and VAD + Whisper are natively 16 kHz. Opus uplink is a documented follow-up (libopus). Downlink audio keeps the existing float32 @ 24 kHz framing (13 ms / 312 samples per frame).

## Global Constraints

- **Repo:** `~/1bit-systems`, branch `feat/jarvis-mobile` (created). Spec: `docs/superpowers/specs/2026-08-08-jarvis-mobile-design.md`.
- **Dirty tree:** The user has uncommitted WIP (e.g. `CMakeLists.txt`, `engine/npu/*`, `src/onnx_loader.cpp`). NEVER commit those. For CMakeLists.txt edits, commit only your hunks: `git add -p CMakeLists.txt`.
- **GitNexus (AGENTS.md):** Before editing any symbol in `tools/jarvis/audio_stream.cpp`/`.h` run impact analysis first: `node .gitnexus/run.cjs impact --target WebSocketServer` (if the CLI lacks that flag, use `node .gitnexus/run.cjs --help` and the query tool; if the index is stale run `node .gitnexus/run.cjs analyze`). Report HIGH/CRITICAL risk to the user before editing. Before committing run `node .gitnexus/run.cjs detect-changes`.
- **No new dependencies.** No libopus, no boost. Existing: httplib, nlohmann::json, ffmpeg (exec), `jarvis::VAD`, `jarvis::AuthManager` (tools/jarvis/auth.h).
- **Do not change** the existing `/v1/audio/stream` WS protocol (meta → float32 frames → end; client `cancel`) — the web UI depends on it. Add, don't modify.
- **Build:** `cmake -B build && cmake --build build --target onebin` — onebin sources are the `ONE_BIN_SOURCES` list in `CMakeLists.txt` (~line 1273). Tests: `enable_testing()` is on; test executables are `add_executable(test_X tests/test_X.cpp ...)` at root.
- **Commit style:** `feat(jarvis): ...` / `test(jarvis): ...`, one per task.

---

### Task 1: `VoiceSession` class — utterance detection + conversation state machine

Pure logic, zero sockets. This is the testable heart of the feature.

**Files:**
- Create: `tools/jarvis/voice_session.h`
- Create: `tools/jarvis/voice_session.cpp`
- Create: `tests/jarvis_voice_session_test.cpp`
- Modify: `CMakeLists.txt` — add `tools/jarvis/voice_session.cpp` to `ONE_BIN_SOURCES`; add test target near the other `test_*` targets (~line 854)

**Interfaces:**
- Consumes: `jarvis::VAD` (`tools/jarvis/vad.h` — `VADConfig{16000, 20, 0.01f, 200.0f, 500.0f, 300.0f, 200.0f}`, `process(const float*, int)`, `get_last_utterance()`, `reset()`), nothing else.
- Produces (used by Task 2/3):
  - `enum class SessionState { Idle, Listening, Processing, Speaking };`
  - `using StateCallback = std::function<void(SessionState)>;`
  - `using UtteranceCallback = std::function<void(const std::vector<int16_t>& pcm16)>;` — called once per detected speech segment (VAD-purified), 16 kHz mono.
  - `using ErrorCallback = std::function<void(const std::string& msg)>;`
  - `class VoiceSession { public: VoiceSession(); void start(); void stop(); void feed(const int16_t* pcm16, size_t n_samples); SessionState state() const; void set_callbacks(StateCallback, UtteranceCallback, ErrorCallback); void set_speaking(bool); };`
  - Semantics: `feed()` appends PCM16 → converts to float32 → `vad_.process()`. On utterance end (VAD `get_last_utterance()` non-empty), session sets `Processing` and fires `UtteranceCallback`. `set_speaking(true)` → `Speaking`; after client acknowledges or a 100 ms quiet timeout (test with `tick()`), session returns to `Listening` and VAD resets. `stop()` → `Idle`, VAD reset, buffered audio dropped. `start()` → `Listening`.

- [ ] **Step 1: Write the failing test**

`tests/jarvis_voice_session_test.cpp`:

```cpp
#include "../tools/jarvis/voice_session.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace jarvis;

static void assert_state(SessionState got, SessionState want, const char* what) {
    if (got != want) { std::printf("FAIL %s: got %d want %d\n", what, (int)got, (int)want); std::exit(1); }
}

int main() {
    // 1 second of 16k sine = "speech" (RMS >> 0.01 threshold)
    std::vector<int16_t> speech(16000);
    for (int i = 0; i < 16000; ++i) speech[i] = (int16_t)(12000 * std::sin(2 * 3.14159 * 440 * i / 16000.0));
    // 1 s of silence (well past min_silence_ms=500)
    std::vector<int16_t> silence(16000, 0);

    VoiceSession s;
    std::vector<SessionState> states;
    int utterances = 0;
    std::vector<std::vector<int16_t>> got_audio;
    s.set_callbacks(
        [&](SessionState st) { states.push_back(st); },
        [&](const std::vector<int16_t>& pcm) { utterances++; got_audio.push_back(pcm); },
        [&](const std::string&) {});
    s.start();
    assert_state(s.state(), SessionState::Listening, "start -> Listening");

    // speech then silence -> exactly one utterance, state Processing
    s.feed(speech.data(), speech.size());
    s.feed(silence.data(), silence.size());
    assert_state(s.state(), SessionState::Processing, "utterance -> Processing");
    assert(utterances == 1);
    assert(!got_audio[0].empty());

    // speaking -> tick -> back to listening (VAD re-arm)
    s.set_speaking(true);
    assert_state(s.state(), SessionState::Speaking, "set_speaking -> Speaking");
    s.tick(200);  // > 100 ms quiet timeout
    assert_state(s.state(), SessionState::Listening, "tick -> Listening");

    // stop drops audio, returns Idle
    s.feed(speech.data(), speech.size());
    s.stop();
    assert_state(s.state(), SessionState::Idle, "stop -> Idle");
    assert(utterances == 1);

    std::printf("PASS voice_session_test (%d utterances, %zu states)\n", utterances, states.size());
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/1bit-systems && g++ -std=c++23 -I. -o /tmp/vs_test tests/jarvis_voice_session_test.cpp tools/jarvis/voice_session.cpp tools/jarvis/vad.cpp 2>&1 | head -5
```
Expected: FAIL — `voice_session.h: No such file or directory`.

- [ ] **Step 3: Write minimal implementation**

`tools/jarvis/voice_session.h`:

```cpp
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace jarvis {

enum class SessionState { Idle, Listening, Processing, Speaking };

using StateCallback = std::function<void(SessionState)>;
using UtteranceCallback = std::function<void(const std::vector<int16_t>& pcm16)>;
using ErrorCallback = std::function<void(const std::string& msg)>;

class VoiceSession {
public:
    VoiceSession();
    ~VoiceSession();

    void start();                                  // Idle -> Listening
    void stop();                                   // any -> Idle, drop buffers
    void feed(const int16_t* pcm16, size_t n_samples);  // 16 kHz mono
    void tick(int ms_elapsed);                     // speaking -> listening timeout
    void set_speaking(bool speaking);              // enter/leave Speaking
    SessionState state() const;

    void set_callbacks(StateCallback on_state, UtteranceCallback on_utterance, ErrorCallback on_error);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace jarvis
```

`tools/jarvis/voice_session.cpp`:

```cpp
#include "voice_session.h"
#include "vad.h"
#include <algorithm>

namespace jarvis {

struct VoiceSession::Impl {
    VAD vad{VADConfig{}};
    SessionState st = SessionState::Idle;
    std::vector<float> pcm_f32;
    std::vector<int16_t> pending;   // pcm16 since utterance start
    int speaking_ms = 0;
    StateCallback on_state;
    UtteranceCallback on_utterance;
    ErrorCallback on_error;

    void set(SessionState next) {
        if (st == next) return;
        st = next;
        if (on_state) on_state(st);
    }
};

VoiceSession::VoiceSession() : impl_(new Impl) {}
VoiceSession::~VoiceSession() = default;

void VoiceSession::set_callbacks(StateCallback s, UtteranceCallback u, ErrorCallback e) {
    impl_->on_state = std::move(s);
    impl_->on_utterance = std::move(u);
    impl_->on_error = std::move(e);
}

SessionState VoiceSession::state() const { return impl_->st; }

void VoiceSession::start() {
    impl_->vad.reset();
    impl_->pending.clear();
    impl_->speaking_ms = 0;
    impl_->set(SessionState::Listening);
}

void VoiceSession::stop() {
    impl_->vad.reset();
    impl_->pending.clear();
    impl_->speaking_ms = 0;
    impl_->set(SessionState::Idle);
}

void VoiceSession::feed(const int16_t* pcm16, size_t n_samples) {
    if (impl_->st == SessionState::Idle || n_samples == 0) return;
    impl_->pcm_f32.resize(n_samples);
    for (size_t i = 0; i < n_samples; ++i) impl_->pcm_f32[i] = pcm16[i] / 32768.0f;
    impl_->vad.process(impl_->pcm_f32.data(), (int)n_samples);
    if (impl_->vad.is_speaking()) {
        impl_->pending.insert(impl_->pending.end(), pcm16, pcm16 + n_samples);
    }
    auto utt = impl_->vad.get_last_utterance();
    if (!utt.empty() && impl_->st == SessionState::Listening) {
        impl_->set(SessionState::Processing);
        if (impl_->on_utterance) impl_->on_utterance(impl_->pending);
        impl_->pending.clear();
        impl_->vad.reset();
    }
}

void VoiceSession::set_speaking(bool speaking) {
    if (speaking && impl_->st == SessionState::Processing) {
        impl_->speaking_ms = 0;
        impl_->set(SessionState::Speaking);
    } else if (!speaking && impl_->st == SessionState::Speaking) {
        impl_->set(SessionState::Listening);
        impl_->vad.reset();
    }
}

void VoiceSession::tick(int ms_elapsed) {
    if (impl_->st != SessionState::Speaking) return;
    impl_->speaking_ms += ms_elapsed;
    if (impl_->speaking_ms > 100) {  // quiet timeout after speech playback
        impl_->set(SessionState::Listening);
        impl_->vad.reset();
    }
}

} // namespace jarvis
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/1bit-systems && g++ -std=c++23 -I. -o /tmp/vs_test tests/jarvis_voice_session_test.cpp tools/jarvis/voice_session.cpp tools/jarvis/vad.cpp && /tmp/vs_test
```
Expected: `PASS voice_session_test (1 utterances, N states)`

- [ ] **Step 5: Commit** (test first, then sources)

```bash
cd ~/1bit-systems
git add tests/jarvis_voice_session_test.cpp tools/jarvis/voice_session.h tools/jarvis/voice_session.cpp
git commit -m "feat(jarvis): VoiceSession state machine (VAD utterance detection)"
```

---

### Task 2: WebSocket server — `/v1/voice/session` full-duplex path

Extend the existing raw-socket WS server with a second request path. Existing downlink behavior must not change.

**Files:**
- Modify: `tools/jarvis/audio_stream.h` — add `start_session` API + auth hook
- Modify: `tools/jarvis/audio_stream.cpp` — new session handler in the accept loop
- Create: `tests/jarvis_ws_proto_test.cpp` — pure protocol helpers test
- Modify: `CMakeLists.txt` — add the test target

**Interfaces:**
- Consumes: `VoiceSession` (Task 1), `jarvis::AuthManager::extract_bearer` / `validate` (tools/jarvis/auth.h).
- Produces:
  - `void WebSocketServer::set_session_handler(std::function<void(bool connected)> cb)` — not needed; instead:
  - `int WebSocketServer::start(int port, void* codec_tts_ptr, std::function<bool(const std::string& auth_header)> auth_check)` — overload; when `auth_check` is set, every WS upgrade (both paths) must pass it or be rejected with `403`.
  - Protocol for `/v1/voice/session`:
    - Client→server text: `{"type":"start"}`, `{"type":"stop"}`, `{"type":"cancel"}`
    - Client→server binary: PCM16 @ 16 kHz mono, 20 ms (640 bytes) frames
    - Server→client text: `{"type":"meta","session":true,"sample_rate":16000,"channels":1,"format":"pcm16","frame_ms":20}`, `{"type":"state","state":"listening|processing|speaking"}`, `{"type":"transcript","role":"user|assistant","text":"..."}`, `{"type":"end","reason":"done|stopped|error"}`, `{"type":"error","message":"..."}`
    - Server→client binary: float32 @ 24 kHz mono, 13 ms (312 samples) frames — identical framing to the existing downlink.
- Pure helpers to extract & test (static in audio_stream.cpp, exposed via a test header `tools/jarvis/ws_proto.h`):
  - `bool ws_parse_control(const std::string& text, std::string& type, nlohmann::json& payload)` — parses a text frame's JSON, returns false on malformed.
  - `std::string ws_meta_json(bool session)` — builds the meta text frame payload.
  - `std::string ws_state_json(SessionState st)` — builds `{"type":"state","state":"..."}`.

- [ ] **Step 1: Write the failing tests**

`tests/jarvis_ws_proto_test.cpp`:

```cpp
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/1bit-systems && g++ -std=c++23 -I. -o /tmp/ws_test tests/jarvis_ws_proto_test.cpp 2>&1 | head -3
```
Expected: FAIL — `ws_proto.h: No such file or directory`.

- [ ] **Step 3: Create `tools/jarvis/ws_proto.h` + `ws_proto.cpp`**

`tools/jarvis/ws_proto.h`:

```cpp
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
```

`tools/jarvis/ws_proto.cpp`:

```cpp
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/1bit-systems && g++ -std=c++23 -I. -o /tmp/ws_test tests/jarvis_ws_proto_test.cpp tools/jarvis/ws_proto.cpp && /tmp/ws_test
```
Expected: `PASS ws_proto_test`

- [ ] **Step 5: Extend `WebSocketServer`** — read `tools/jarvis/audio_stream.cpp` first (the accept loop + frame writer), then:

`tools/jarvis/audio_stream.h` — add:

```cpp
#include "ws_proto.h"   // top of header

// Auth hook: return true to accept the upgrade. Called with the raw
// Authorization header value (may be empty). Null = accept everything.
using WSAuthCheck = std::function<bool(const std::string& auth_header)>;

int start(int port, void* codec_tts_ptr, WSAuthCheck auth_check);
```

In `audio_stream.cpp`:
- Keep the existing `start(port, codec_tts_ptr)` delegate to the new overload with `WSAuthCheck{}`.
- In the upgrade handler: if `auth_check` is set and fails → reply `403 Forbidden` and close (do NOT send the 101).
- In the per-connection read loop, branch on the request path:
  - `/v1/audio/stream` → existing downlink behavior (unchanged).
  - `/v1/voice/session` → new session mode:
    - Send `ws_meta_json(true)` on connect.
    - Text frames → `ws_parse_control`; `start` → `session.start()`, `stop` → `session.stop()` + send `{"type":"end","reason":"stopped"}`, `cancel` → abort current TTS send.
    - Binary frames → treat as PCM16 (640-byte frames expected; tolerate partial frames by buffering), `session.feed(...)`.
    - Wire `session.set_callbacks`:
      - on_state → send `ws_state_json(st)`
      - on_utterance → invoked from jarvis_server.cpp wiring (Task 3), not here
    - The session-mode connection does NOT receive the downlink TTS feed that the `/v1/audio/stream` path uses; jarvis_server.cpp (Task 3) sends TTS audio to it via a new `send_audio(const float* samples, int n)` / `send_text(const std::string& json)` pair on the connection object. Add these as methods on the connection struct; downlink path keeps its existing send path.

- [ ] **Step 6: Rebuild + manual smoke**

```bash
cd ~/1bit-systems && cmake -B build >/dev/null 2>&1; cmake --build build --target onebin -j$(nproc) 2>&1 | tail -3
```
Expected: build succeeds. (Full protocol smoke happens in Task 3's integration script.)

- [ ] **Step 7: Commit** (use `git add -p CMakeLists.txt` for the test target hunk only)

```bash
cd ~/1bit-systems
git add tools/jarvis/ws_proto.h tools/jarvis/ws_proto.cpp tools/jarvis/audio_stream.h tools/jarvis/audio_stream.cpp tests/jarvis_ws_proto_test.cpp
git add -p CMakeLists.txt   # only the jarvis_ws_proto_test target hunk
git commit -m "feat(jarvis): WS /v1/voice/session full-duplex path with auth hook"
```

---

### Task 3: `jarvis_server.cpp` wiring — real Whisper/LLM/TTS behind the session

**Files:**
- Modify: `tools/jarvis_server.cpp` — WS start site (~line 1732), new session wiring block
- Create: `scripts/ws_session_smoke.js` — node WS client integration script
- Create: `scripts/ws_session_fixture.sh` — makes fixture PCM16 audio via ffmpeg

**Interfaces:**
- Consumes: `VoiceSession` + `WebSocketServer` session mode (Tasks 1–2); existing server globals: `get_whisper_model()`, `whisper_transcribe(model, pcm16, n)`, `g_persona_mgr.build_system_prompt()`, `g_context_mem` (has `.add_turn(role, text)` and history access — read `tools/jarvis/context.h` for the exact history getter), `resolve_model(model_id)` + `unified_chat(model, msgs, max_tokens, temp)` / `ollama_chat(...)`, `g_codec_tts.synthesize(text, voice)`, `synthesize_speech(text, voice)` (Piper fallback), `g_auth_manager` (`tools/jarvis/auth.h`).
- Produces: end-to-end `/v1/voice/session` — scripted client can hold a conversation.

- [ ] **Step 1: Read the existing `/v1/audio/chat` handler** (lines ~1026–1200) and `tools/jarvis/context.h` to copy the exact LLM message-building pattern (persona system prompt + history + user turn → `resolve_model` → `unified_chat`).

- [ ] **Step 2: Write the wiring block.** At the WS server start site, add a session handler. Sketch (fill exact signatures from Step 1's reading):

```cpp
// ── /v1/voice/session : full-duplex mobile voice loop ────────────────
struct MobileSession {
    jarvis::VoiceSession vs;
    bool active = false;
};
g_mobile_session = std::make_unique<MobileSession>();

// After g_ws_server->start(...): register session callbacks.
// on_utterance: whisper + LLM + TTS (mirrors /v1/audio/chat but from
// PCM16 directly — no ffmpeg round-trip needed):
//   1. WhisperModel* m = get_whisper_model(); if (!m) -> error frame + spoken error
//   2. text = whisper_transcribe(*m, utt.data(), (int)utt.size())
//   3. if (text.empty() || text == "[silence]") -> back to listening (vs.set_speaking(false))
//   4. g_context_mem.add_turn("user", text); send transcript user frame
//   5. build msgs: system (g_persona_mgr.build_system_prompt()) + history + user
//   6. Route route = resolve_model(""); llm_result = unified_chat(...) / ollama_chat(...)
//   7. reply text; g_context_mem.add_turn("assistant", reply); send transcript assistant
//   8. wav = g_codec_tts.synthesize(reply, default_voice); if empty -> synthesize_speech fallback
//   9. parse WAV (44-byte header, s16 @ 24 kHz) -> float32 -> send_audio() in
//      312-sample frames while vs.state()==Speaking; then vs.set_speaking(false)
// on_state: forward ws_state_json frames (Task 2 already sends them from
//           the connection loop — choose ONE owner: if the connection loop
//           sends on_state, the wiring only handles utterance->TTS).
```

To avoid double-sending state frames, the connection loop (Task 2) forwards on_state; Task 3 wires only `on_utterance` (STT→LLM→TTS→send_audio) and `on_error`. `set_speaking(true)` is called from the wiring before streaming TTS; `set_speaking(false)` after the last frame; the connection loop's `tick()` handles the 100 ms re-arm.

- [ ] **Step 3: Auth.** Pass `WSAuthCheck` to `start()`: when env `JARVIS_WS_TOKEN` is set, accept only if the header's bearer equals it (constant-time compare); otherwise (token unset) accept all — VPN-only deployment assumption (matches spec).

- [ ] **Step 4: Write the fixture + integration script**

`scripts/ws_session_fixture.sh`:

```bash
#!/usr/bin/env bash
# Generates 2 s of 440 Hz tone = "speech" fixture as PCM16 @ 16k.
set -euo pipefail
OUT="${1:-/tmp/jarvis_fixture.pcm16}"
ffmpeg -y -loglevel error -f lavfi -i "sine=frequency=440:duration=2" -ar 16000 -ac 1 -f s16le "$OUT"
echo "$OUT"
```

`scripts/ws_session_smoke.js` — node (uses global WebSocket in node ≥22):

```js
// Usage: node scripts/ws_session_smoke.js ws://127.0.0.1:8082/v1/voice/session /tmp/jarvis_fixture.pcm16 [token]
const fs = require('fs');
const wsUrl = process.argv[2];
const fixture = process.argv[3];
const token = process.argv[4] || '';
const pcm = fs.readFileSync(fixture);

const ws = new WebSocket(wsUrl, { headers: token ? { Authorization: 'Bearer ' + token } : {} });
let gotMeta = false, gotState = 0, gotTranscript = 0, gotAudio = 0, ended = false;

ws.onmessage = (ev) => {
  if (typeof ev.data === 'string') {
    const m = JSON.parse(ev.data);
    if (m.type === 'meta' && m.session) gotMeta = true;
    if (m.type === 'state') gotState++;
    if (m.type === 'transcript') { gotTranscript++; console.log(`  transcript[${m.role}]: ${m.text}`); }
    if (m.type === 'end') { ended = true; console.log(`  end reason=${m.reason}`); ws.close(); }
    if (m.type === 'error') console.log(`  ERROR: ${m.message}`);
  } else {
    gotAudio += ev.data.byteLength;
  }
};

ws.onopen = () => {
  ws.send(JSON.stringify({ type: 'start' }));
  // stream fixture in 640-byte frames (20 ms @ 16k), 10 ms apart
  let off = 0;
  const timer = setInterval(() => {
    if (off >= pcm.length) { clearInterval(timer); setTimeout(() => ws.send(JSON.stringify({ type: 'stop' })), 800); return; }
    const frame = pcm.subarray(off, off + 640);
    ws.send(frame.buffer.slice(frame.byteOffset, frame.byteOffset + frame.byteLength));
    off += 640;
  }, 10);
  setTimeout(() => { console.log(`SMOKE TIMEOUT: meta=${gotMeta} states=${gotState} transcripts=${gotTranscript} audioBytes=${gotAudio} ended=${ended}`); process.exit(2); }, 30000);
};

ws.onerror = (e) => { console.log('WS ERROR', e.message || e); process.exit(1); };
```

- [ ] **Step 5: Run the smoke test**

```bash
cd ~/1bit-systems && cmake --build build --target onebin -j$(nproc) && bash scripts/ws_session_fixture.sh /tmp/jarvis_fixture.pcm16
# start jarvis-server WITHOUT models (whisper unavailable path is still exercised):
# find the run command in the repo (onebin dispatch: build/1bit jarvis --help)
# with a stub WHISPER_MODEL_PATH unset, the session must send an error frame
# and the script must exit 0 with ended=true.
node scripts/ws_session_smoke.js ws://127.0.0.1:8082/v1/voice/session /tmp/jarvis_fixture.pcm16
```
Expected: meta + state frames arrive, an error/end frame arrives (transcription unavailable path — models absent on this dev box), script exits cleanly. **Full voice E2E (real Whisper/LLM/TTS) is a manual step on the Strix Halo box** — documented in the script header.

- [ ] **Step 6: Commit**

```bash
cd ~/1bit-systems
git add tools/jarvis_server.cpp scripts/ws_session_fixture.sh scripts/ws_session_smoke.js
git add -p CMakeLists.txt   # if the task touched it
git commit -m "feat(jarvis): wire VoiceSession to Whisper/LLM/TTS, WS token auth, smoke script"
```

---

## Self-Review Notes

- **Spec coverage:** M1 (server WS uplink) fully covered. M2 (Flutter app) and M3 (E2E) are intentionally separate plans, per the spec's milestone split. Spec items "voice-active loop", "control messages", "bearer token", "state/transcript frames", "error handling" all map to Tasks 1–3. Deviation flagged: uplink wire format is PCM16@16k (spec said Opus) — documented in the plan header and spec open-questions; Opus is a follow-up requiring libopus.
- **Type consistency:** `VoiceSession::feed(const int16_t*, size_t)` used identically in Task 1 and Task 3; `SessionState` enum used by `ws_state_json`; `set_speaking` semantics match the test.
- **Testing honesty:** the dev box has no whisper/LLM/TTS models; the smoke test verifies protocol + error paths; voice E2E is explicitly a Strix Halo manual step.
