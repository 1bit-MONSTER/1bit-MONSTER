# JARVIS Mobile — Design

Date: 2026-08-08
Status: Approved (brainstorming complete)
Owner: 1bit.MONSTER

## Goal

A mobile companion for the JARVIS voice pipeline: the phone is a **thin terminal** (mic + speaker + VPN client). The full pipeline — VAD → Whisper STT → router → LLM → codec TTS → cloned voice — runs on the user's Strix Halo box at home. **No data stays on the phone** (no audio persistence, transcripts are on-screen only and cleared on exit).

The reference product is "Lemonade mobile": a mobile app that connects to a local AI server you own.

## Non-Goals (v1)

- No on-device inference of any kind (no wake word, no on-device STT/TTS).
- No push notifications, no background wake, no widget.
- ~~No iOS build~~ — **iOS IS a target**: cloud Mac mini (rentamac.io, macOS 26.6, Xcode 26.6, Flutter 3.44.9, CocoaPods 1.17.0) is provisioned; iOS 26.5 simulator runtime installed. Owner has an Apple Developer account — device signing + TestFlight available (Apple ID needed at signing time). Android remains a target too (Flutter cross-platform).
- No WebRTC (WebSocket + Opus; ~200 ms added latency is acceptable for voice-active conversation).
- No multi-user/multi-device session management.

## Architecture

All code lives in the `1bit-monster/1bit-monster` repo (single repo, single CI):

```
phone (Flutter, Android)                  Strix Halo box
┌─────────────────────┐   WS over VPN     ┌──────────────────────────────┐
│ mic → Opus → ───────┼─── WebSocket ────┼──▶ 1bit jarvis-server (C++23)│
│ Opus → speaker ◀────┼───────────────────┼──▶ VAD → STT → LLM → TTS    │
│ tap-to-start,       │                   │   (exists: /v1/audio/chat,  │
│ no data stored      │                   │    /v1/audio/stream WS)     │
└─────────────────────┘                   │         │                   │
                                          │         ▼                   │
                                          │  unified server (LLM)       │
                                          └──────────────────────────────┘
```

### Components

**1. `1bit jarvis-server` (existing, extended) — the gateway process**

Already in the repo (`tools/jarvis_server.cpp`, dispatched from the onebin; runs as its own process, separate from `unified`). Existing surface:
- `POST /v1/audio/transcriptions` — Whisper STT
- `POST /v1/audio/speech` — codec TTS (Piper fallback), WAV out
- `POST /v1/audio/chat` — voice-in/voice-out (VAD + Whisper + LLM + TTS)
- `GET /v1/audio/stream` — HTTP chunked TTS streaming
- `WebSocketServer` (tools/jarvis/audio_stream.*, raw POSIX RFC 6455, port 8082) — TTS **downlink** only today
- API-key auth (`/v1/api-key/*`, `tools/jarvis/auth.*`), personas, planner, RAG, usage/billing

**Gap-fill (M1) — WS uplink + control protocol:**
- Extend `WebSocketServer` to accept mic audio frames (Opus or PCM16) from the phone.
- Control messages (JSON): `hello`, `start`, `stop`, `state`, `transcript`, `error`, `bye` (per protocol below).
- Voice-active loop on the server: VAD segments incoming speech → STT → LLM (via existing planner/router → unified) → TTS → audio downlink → VAD re-arms.
- Bearer-token auth on WS handshake (config option; audio HTTP endpoints remain as-is for the web UI).

**2. `mobile/` — Flutter app (new, Android-first)**

**2. `mobile/` — Flutter app (new, Android-first)**

- `connect` screen: server URL + token (persisted in secure storage).
- Main screen: big JARVIS button (tap to start/stop session), status lights (listening / thinking / speaking / offline), on-screen transcript log (cleared on exit).
- Audio: mic capture → Opus encode → WS send; Opus receive → decode → playback.
- Foreground service while a session is active (screen-off support).
- Dependencies: `web_socket_channel`, `record` (mic), `just_audio` or `audioplayers` (playback), `flutter_secure_storage`, an Opus codec binding (e.g. `opus_dart`/`flutter_opus` or a small FFI shim to libopus).

### WebSocket protocol (v1, JSON control + binary audio)

- Handshake: HTTP Upgrade with `Authorization: Bearer <token>` (configurable; disabled = LAN/VPN only).
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

1. **M1 — Server WS uplink:** extend `WebSocketServer` + `jarvis_server.cpp` with control protocol, mic audio in, voice-active loop, token auth. Tested with a scripted WS client (node) + fixture audio.
2. **M2 — App:** Flutter shell, connect screen, WS client, mic/Opus, playback, state UI, transcript log.
3. **M3 — E2E:** real VPN + phone session; polish (errors, reconnect, foreground service).

## Open Questions (tracked, not blockers)

- VAD: reuse `tools/jarvis/vad.*` in the WS loop vs the existing `/v1/audio/chat` path — confirm in M1 (WS loop uses `jarvis::VAD` directly; the codec StreamingDecoder path from the existing downlink is reused for TTS).
- Uplink wire format: PCM16 @ 16 kHz (decided in M1 — VAD/Whisper native; Opus = follow-up with libopus).
- iOS signing team ID / Apple ID — needed at M2 device-build time (user has Apple Developer account).
- Opus binding choice for Flutter (pure-Dart vs FFI) — decide in M2 (downlink is float32 @ 24 kHz from the Zyphra codec; no Opus on the wire).
