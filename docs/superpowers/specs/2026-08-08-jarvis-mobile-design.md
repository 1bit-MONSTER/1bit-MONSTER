# JARVIS Mobile — Design

Date: 2026-08-08
Status: Approved (brainstorming complete)
Owner: 1bit.systems

## Goal

A mobile companion for the JARVIS voice pipeline: the phone is a **thin terminal** (mic + speaker + VPN client). The full pipeline — VAD → Whisper STT → router → LLM → codec TTS → cloned voice — runs on the user's Strix Halo box at home. **No data stays on the phone** (no audio persistence, transcripts are on-screen only and cleared on exit).

The reference product is "Lemonade mobile": a mobile app that connects to a local AI server you own.

## Non-Goals (v1)

- No on-device inference of any kind (no wake word, no on-device STT/TTS).
- No push notifications, no background wake, no widget.
- No iOS build (Android-first; Flutter keeps the door open).
- No WebRTC (WebSocket + Opus; ~200 ms added latency is acceptable for voice-active conversation).
- No multi-user/multi-device session management.

## Architecture

All code lives in the `1bit-systems/1bit-systems` repo (single repo, single CI):

```
phone (Flutter, Android)                  Strix Halo box
┌─────────────────────┐   WSS over VPN    ┌──────────────────────────────┐
│ mic → Opus → ───────┼─── WebSocket ────┼──▶ voice-gateway (C++23)      │
│ Opus → speaker ◀────┼───────────────────┼──▶ VAD → STT → LLM → TTS    │
│ tap-to-start,       │                   │         │                   │
│ no data stored      │                   │         ▼                   │
└─────────────────────┘                   │  engine: unified server     │
                                          │  + /v1/audio/transcriptions │
                                          │  + /v1/audio/speech (new)   │
                                          └──────────────────────────────┘
```

### Components

**1. `voice-gateway/` — C++23 daemon (new, standalone)**

- WebSocket server (reuse engine's existing HTTP/WS server code where practical).
- Opus decode (phone → PCM16 @ 16 kHz mono) / encode (PCM16 → phone).
- VAD: reuse the engine's existing `vad.*` stages (silence gating) to segment speech.
- Session state machine: `idle → listening → processing → speaking → listening …`; tap-stop from phone ends session.
- Never touches models directly. Talks to the engine over HTTP:
  - `POST /v1/audio/transcriptions` (Whisper STT) — new engine endpoint
  - `POST /v1/audio/speech` (codec TTS, voice-pack aware) — new engine endpoint
  - `POST /v1/chat/completions` (LLM/router/planner) — exists
- Config: `voice-gateway/config.toml` — listen addr/port, engine base URL, bearer token, persona name, voice-pack path.
- Binary name: `jarvis-gateway`, built by the repo's existing CMake build.

**2. `mobile/` — Flutter app (new, Android-first)**

- `connect` screen: server URL + token (persisted in secure storage).
- Main screen: big JARVIS button (tap to start/stop session), status lights (listening / thinking / speaking / offline), on-screen transcript log (cleared on exit).
- Audio: mic capture → Opus encode → WS send; Opus receive → decode → playback.
- Foreground service while a session is active (screen-off support).
- Dependencies: `web_socket_channel`, `record` (mic), `just_audio` or `audioplayers` (playback), `flutter_secure_storage`, an Opus codec binding (e.g. `opus_dart`/`flutter_opus` or a small FFI shim to libopus).

**3. Engine changes (small, additive)**

- `POST /v1/audio/transcriptions` on `unified` — accepts PCM16 or Opus, runs Whisper, returns text.
- `POST /v1/audio/speech` on `unified` — accepts text + persona/voice-pack, streams codec-TTS audio (PCM16 or Opus).
- The engine remains the single AI front door; the gateway is pure orchestration.

### WebSocket protocol (v1, JSON control + binary audio)

- Handshake: HTTP Upgrade with `Authorization: Bearer <token>`.
- Control messages (JSON, text frames):
  - `{"type":"hello","version":1}` — client → server on connect
  - `{"type":"start"}` / `{"type":"stop"}` — session control
  - `{"type":"state","state":"listening|processing|speaking"}` — server → client
  - `{"type":"transcript","role":"user|assistant","text":"…"}` — streaming transcript
  - `{"type":"error","message":"…"}` — spoken errors also flow as audio
  - `{"type":"bye"}` — server closes session
- Audio frames (binary): Opus packets, 20 ms, 16 kHz mono, tagged by frame order; server replies with Opus frames while in `speaking`.

### Data flow (voice-active session)

1. User taps button → app opens WS, sends `start`, begins streaming mic Opus frames.
2. Gateway decodes → feeds VAD. Speech end detected → STT via engine → transcript `user`.
3. Router/planner → LLM (chat completions) → assistant transcript.
4. TTS via engine → gateway encodes Opus → streams to phone while in `speaking`.
5. VAD re-arms → back to `listening`. Repeats until `stop` or WS drop.

### Auth & resilience

- Bearer token (shared secret in config on both ends). VPN (WireGuard/Tailscale) assumed for transport security; no TLS in v1 (documented limitation).
- WS drop → session aborts, gateway cleans up, phone returns to idle with "disconnected" state.
- STT/TTS/LLM failure → spoken error message + `error` control message; session continues.
- Engine down → gateway replies with `offline` state on connect attempt; phone shows offline.

### Error handling summary

| Failure | Behavior |
|---|---|
| WS drops mid-session | Abort session; phone → idle/disconnected |
| STT returns empty | Spoken "I didn't catch that"; back to listening |
| LLM/TTS error | Spoken error; back to listening |
| Engine unreachable | `offline` on connect; phone shows offline |
| Token rejected | 401 on handshake; phone shows auth error |

### Testing

- **Gateway unit tests (CTest):** VAD segmentation state machine, session state machine, protocol parsing, Opus round-trip (encode → decode → PCM equality within tolerance).
- **Integration test:** gateway + local `unified` + fixture audio file → expect transcript + spoken reply (scripted, CI-runnable where engine builds).
- **Flutter widget tests:** state UI (listening/thinking/speaking/offline), connect screen validation.
- **Manual E2E:** real phone over VPN against Strix Halo box.

### Milestones

1. **M1 — Engine audio endpoints:** `/v1/audio/transcriptions` + `/v1/audio/speech` on `unified`, curl-verified.
2. **M2 — Gateway:** WS server, Opus, VAD loop, engine calls, session state machine; CLI-driven test with fixture audio.
3. **M3 — App:** Flutter shell, connect screen, WS client, mic/playback, state UI, transcript log.
4. **M4 — E2E:** real VPN + phone session; polish (errors, reconnect, foreground service).

## Open Questions (tracked, not blockers)

- Opus binding choice for Flutter (pure-Dart vs FFI) — decide in M3.
- Exact VAD parameters reuse from `tools/jarvis/vad.*` — confirm in M2.
- Where the gateway daemon lives on the Strix Halo box (systemd unit) — packaging in M4.
