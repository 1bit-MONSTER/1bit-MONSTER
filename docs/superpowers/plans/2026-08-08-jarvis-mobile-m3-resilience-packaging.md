# JARVIS Mobile M3 — Resilience, Packaging & E2E Runbook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use supo-subagent-driven-development (recommended) or supo-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the M3 milestone: the app survives connection loss with a sane offline→reconnect flow, the gateway shuts down cleanly, and the user has a runbook to deploy the gateway on the Strix Halo box and run the real-voice E2E from the phone.

**Architecture:** Three small workstreams, all on `feat/jarvis-mobile`:
1. **App resilience** (`mobile/`): on gateway disconnect the voice screen stops the mic, resets the session controller, and offers a Reconnect button that re-connects and starts a fresh session. No auto-reconnect loops in v1 (VPN blips are user-initiated events; a button is honest and debuggable).
2. **Gateway shutdown fix** (`tools/jarvis/audio_stream.cpp`): `WebSocketServer::stop()` joins the server thread, but the session read loop wakes every 50 ms and never checks `running_` — shutdown hangs while a session client is connected (M1 final-review minor M5).
3. **Packaging + runbook** (`scripts/` + `docs/mobile/`): a systemd unit for the gateway on the Strix Halo box and a step-by-step RUNBOOK.md covering box deploy (build onebin, env vars), phone setup (VPN, app connect), and the manual voice E2E checklist.

**Tech Stack:** Flutter/Dart (existing mobile/), C++23 (existing audio_stream.cpp), systemd unit file, markdown runbook.

## Global Constraints

- **Repo:** `~/1bit-systems`, branch `feat/jarvis-mobile`. Spec: `docs/superpowers/specs/2026-08-08-jarvis-mobile-design.md` (M3 milestone + error table). M1/M2 wire protocol is FROZEN — no protocol changes in this plan.
- **User WIP** files (engine/npu/*, tools/bitnet_decode.cpp, third_party/FastFlowLM, etc.) are uncommitted — NEVER commit them. Commits contain only their listed paths.
- **Tests:** `flutter test` via `/home/bcloud/flutter/bin/flutter` from `mobile/`; server compile check via the established g++ -c include set.
- **No new dependencies.** No new server features beyond the stop() fix.
- **Commit style:** `fix(jarvis-mobile): ...`, `fix(jarvis): ...`, `docs(jarvis-mobile): ...`.
- The real-voice E2E runs on the user's Strix Halo box — this plan delivers the runbook + checklist, not the run.

---

### Task 1: App — connection-loss UX + reconnect button

**Files:**
- Modify: `mobile/lib/screens/voice_screen.dart`
- Modify: `mobile/test/voice_screen_test.dart`

**Interfaces:**
- Consumes: `GatewayClient` (already emits `ErrorEvent('gateway connection lost')` on onDone — M2 final wave), `SessionController.setOffline(reason)`, existing mic/player/client lifecycle in `VoiceScreen`.
- Produces: voice screen behavior — on `ErrorEvent` with 'connection lost' (or any ErrorEvent while a session is active): stop mic, `controller.setOffline(...)`, button becomes **Reconnect** (label + icon change); tapping Reconnect calls `_connect()` then `startSession()` (re-establishes and starts a fresh session); while offline the button is enabled and labeled Reconnect; the transcript persists across the reconnect (it is per-screen memory — do NOT clear it; only `controller.reset()` on screen leave).

- [ ] **Step 1: Write the failing widget tests** (extend `voice_screen_test.dart`):

```dart
testWidgets('disconnect stops mic and shows Reconnect button', (tester) async {
  final mic = FakeMic();
  final client = FakeGatewayClient();
  await pumpVoiceScreen(tester, client: client, mic: mic);
  await tester.tap(find.byKey(const Key('jarvis-button')));
  await tester.pump();
  expect(mic.startCount, 1);
  client.emitError('gateway connection lost'); // fakes must expose emitError
  await tester.pump();
  expect(mic.stopCount, 1);                    // mic stopped
  expect(find.text('Reconnect'), findsOneWidget);
  expect(find.text('Offline'), findsOneWidget); // status label
});

testWidgets('Reconnect reconnects and starts a new session', (tester) async {
  final client = FakeGatewayClient();
  final mic = FakeMic();
  await pumpVoiceScreen(tester, client: client, mic: mic);
  await tester.tap(find.byKey(const Key('jarvis-button')));
  await tester.pump();
  client.emitError('gateway connection lost');
  await tester.pump();
  expect(client.connectCount, 1);
  await tester.tap(find.text('Reconnect'));
  await tester.pump();
  expect(client.connectCount, 2);              // re-connected
  expect(client.sent, contains('{"type":"start"}')); // fresh session started
  expect(mic.startCount, 2);
});
```
(Adjust to the existing fake APIs in the test file; the fakes must gain `emitError(String)` and `connectCount` if absent. Keep existing tests green.)

- [ ] **Step 2: Run to verify they fail** (`flutter test test/voice_screen_test.dart` — new tests fail: no Reconnect UI, mic not stopped).
- [ ] **Step 3: Implement** — in `VoiceScreen`'s events listener: on `ErrorEvent` whose message contains 'connection lost' (or when `controller.state` would go offline): stop mic (`_mic.stop()`), do NOT auto-clear the transcript, `controller.setOffline(...)`, flip a `_reconnecting`-style flag that renders the button as Reconnect; Reconnect tap = `_connect()` + `startSession()`. Keep the existing error banner behavior for other errors.
- [ ] **Step 4: Run to verify they pass** — full `flutter test` + `flutter analyze` (no new issues).
- [ ] **Step 5: Commit** — `fix(jarvis-mobile): connection-loss UX — stop mic, offline state, Reconnect button`.

---

### Task 2: Gateway — clean shutdown (stop() join hang)

**Files:**
- Modify: `tools/jarvis/audio_stream.cpp`

**Interfaces:**
- Consumes: existing `WebSocketServer` internals (server_thread_, running_, session read loop).
- Produces: `stop()` completes even with an active session client.

- [ ] **Step 1: Read the loop** — locate the session read loop that wakes every ~50 ms (tick cadence) and confirm it never checks `running_` (M1 final-review minor M5: `stop()` joins the server thread → hang while a client is connected).
- [ ] **Step 2: Fix** — add a `running_` check to the loop's wake path so `stop()` (which clears `running_` before joining) unblocks it: the loop must exit within one wake (~50 ms) of `stop()`. Keep the legacy downlink behavior identical (it has its own loop — check whether it shares the same hang; fix both loops if the same pattern exists, one change each).
- [ ] **Step 3: Verify** — compile check: `g++ -c -std=c++23 -DNDEBUG -I. -Isrc -Itools -Ithird_party/llama.cpp/vendor tools/jarvis/audio_stream.cpp` (established include set; report if a header is missing). If a standalone binary can be built per the M1 report instructions, run a stop-with-connected-client probe (connect a client, call stop via SIGTERM, assert exit within ~2 s); otherwise state exactly why and trace the fix in the report.
- [ ] **Step 4: Commit** — `fix(jarvis): WebSocketServer::stop() unblocks the session loop (no hang with active client)` — only `tools/jarvis/audio_stream.cpp` in the commit.

---

### Task 3: Deployment — systemd unit + RUNBOOK.md

**Files:**
- Create: `scripts/jarvis-gateway.service`
- Create: `docs/mobile/RUNBOOK.md`

**Interfaces:**
- Consumes: M1 gateway facts (binary `build/1bit` subcommand `jarvis`; env vars `WS_STREAM_PORT` (default 8082), `JARVIS_WS_TOKEN`, `WHISPER_MODEL_PATH`, `VOICE_PACKS_DIR`; unified server on port 8080 for LLM routing; personas via `PERSONA`/persona files — verify exact env names against `tools/jarvis_server.cpp` before writing, and use the real ones).
- Produces: deployable unit + human runbook.

- [ ] **Step 1: Verify env vars** — grep `tools/jarvis_server.cpp` for the exact environment variable names the gateway reads (WS_STREAM_PORT, JARVIS_WS_TOKEN, WHISPER_MODEL_PATH, VOICE_PACKS_DIR, any persona/voice defaults) and the unified-server port; record them in the report.
- [ ] **Step 2: Write `scripts/jarvis-gateway.service`** — systemd unit: `ExecStart=/home/<user>/1bit-systems/build/1bit jarvis` (user-adjustable), `Environment=` lines for the verified vars (commented defaults), `Restart=on-failure`, `After=network-online.target`, `WantedBy=multi-user.target`. Keep it a template with comments, not a hardcoded deploy.
- [ ] **Step 3: Write `docs/mobile/RUNBOOK.md`** — sections:
  1. **Box setup (Strix Halo):** build the engine (`cmake -B build && cmake --build build --target onebin`), start `unified` (port 8080) + `jarvis` (WS on 8082), set `JARVIS_WS_TOKEN`, whisper model path, voice packs; install the systemd unit (paths to adjust).
  2. **Phone setup:** VPN (WireGuard/Tailscale) to the home network, install the app (flutter build on the Mac per M2, or a dev build), connect screen values (ws://<box-lan-ip>:8082, token).
  3. **Manual voice E2E checklist** (the user runs this on the Strix Halo box): for each item, the expected observation:
     - [ ] gateway starts, WS port listening (`ss -tlnp | grep 8082`)
     - [ ] app connects (green light) with and without token (403 path)
     - [ ] tap JARVIS → light goes listening; speak → processing → assistant transcript appears
     - [ ] reply plays through the phone speaker (Zyphra codec voice/persona)
     - [ ] second turn works without re-tapping (voice-active)
     - [ ] `stop` tap ends the session; transcript stays until leaving the screen
     - [ ] kill the gateway mid-session → phone shows Offline + Reconnect; restart gateway, tap Reconnect → session works again
     - [ ] VPN off mid-session → same offline path; VPN back → Reconnect works
     - [ ] voice-cloned pack: switch persona/voice in the gateway config, verify the reply uses it
  4. **Troubleshooting:** common failures (STT unavailable → WHISPER_MODEL_PATH; TTS empty → voice pack missing; 403 → token mismatch; 400 on connect → stale gateway binary — rebuild after d3eb550d).
- [ ] **Step 4: Commit** — `docs(jarvis-mobile): gateway systemd unit + Strix Halo runbook and E2E checklist` — only `scripts/jarvis-gateway.service` + `docs/mobile/RUNBOOK.md`.

---

## Self-Review Notes

- **Spec coverage:** M3 spec items — errors/resilience (Task 1), reconnect (Task 1 button; no auto-loop — documented deviation, VPN-blip UX is user-initiated), foreground service (NOT in this plan: Android-only, screen-off sessions; park to M4 — the spec's M3 item stays tracked), packaging (Task 3), E2E (runbook + checklist delivered; the run itself is the user's manual step on Strix Halo). The M1 stop()-hang minor (M5) is Task 2.
- **Type consistency:** no interface changes; Task 1 consumes the existing `ErrorEvent('gateway connection lost')` emitted by GatewayClient (M2 final wave); fake APIs extended only in tests.
- **Honesty:** the E2E checklist's items are things I cannot run here (no models/voice packs on this box) — the plan says so; every item has an observable expectation so the user's run is pass/fail, not vibes.
