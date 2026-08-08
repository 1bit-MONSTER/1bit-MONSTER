# JARVIS Mobile — Strix Halo Deployment Runbook

Deploy and manually verify the JARVIS voice gateway on the Strix Halo box with
the mobile app (M1–M3). All paths assume the repo lives at
`/home/<user>/1bit-systems`; adjust for your box.

## Architecture (ports, verified in code)

| Process | Command | Port | Binds |
|---|---|---|---|
| Unified server (LLM routing) | `build/1bit unified` | 8088 (default) | 127.0.0.1 |
| Gateway HTTP API | `build/1bit jarvis` | 8080 (default) | 127.0.0.1 (`JARVIS_BIND_ADDR`) |
| Gateway WS voice session | `build/1bit jarvis` | 8082 (`WS_STREAM_PORT`) | 127.0.0.1 default (`WS_STREAM_BIND`) |

The gateway routes every LLM call to `UNIFIED_URL` (default
`http://127.0.0.1:8088`). The WS port binds loopback by default; set
`WS_STREAM_BIND=0.0.0.0` for the phone to reach it over LAN/VPN (see §1.5).

## 1. Box setup (Strix Halo)

1. Build the engine (single `build/1bit` binary, all subcommands):
   ```sh
   cd /home/<user>/1bit-systems
   cmake -B build
   cmake --build build --target onebin
   ```
2. Start the unified server first (the gateway depends on it for LLM turns):
   ```sh
   ./build/1bit unified &        # or: systemd-run --unit=jarvis-unified ./build/1bit unified
   curl -s http://127.0.0.1:8088/v1/health   # expect {"ok":true} (or similar 200)
   ```
3. Set gateway environment and start it once, by hand, to check startup logs:
   ```sh
   export JARVIS_WS_TOKEN=<token>                       # phone must send this; unset = no auth
   export WHISPER_MODEL_PATH=/home/<user>/models/whisper-small.gguf
   export VOICE_PACKS_DIR=/home/<user>/voice-packs      # zaya_default.voice and any cloned packs
   export PERSONAS_DIR=/home/<user>/personas            # optional; default ~/personas
   ./build/1bit jarvis
   ```
   Startup log should show: persona count, voice packs loaded, and
   `WS session: ws://127.0.0.1:8082/v1/voice/session`. Missing packs print
   `codec TTS: no voice packs found in <dir>`.
4. Install the systemd unit (template — edit `<user>` first):
   ```sh
   # edit scripts/jarvis-gateway.service: replace <user>, uncomment the
   # Environment= lines you need (token, model, voice packs)
   sudo cp scripts/jarvis-gateway.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now jarvis-gateway
   journalctl -u jarvis-gateway -f
   ```
5. LAN reachability (required for the phone): the WS port binds loopback by
   default, so make the gateway bind all interfaces — the phone then talks
   directly to `ws://<box-lan-ip>:8082/v1/voice/session`:
   ```sh
   # in the unit (scripts/jarvis-gateway.service):
   Environment=WS_STREAM_BIND=0.0.0.0
   # or, when running by hand:
   WS_STREAM_BIND=0.0.0.0 ./build/1bit jarvis
   ```
   `WS_STREAM_BIND` defaults to `127.0.0.1` (loopback only, safe). Binding
   `0.0.0.0` exposes the WS port on the network — keep `JARVIS_WS_TOKEN`
   set so the phone still needs the bearer token. Verify the listener is on
   all interfaces: `ss -tlnp | grep 8082` shows `*:8082` (or `0.0.0.0:8082`).

## 2. Phone setup

1. VPN to the home network: WireGuard or Tailscale on the phone, so the box's
   LAN/VPN IP is routable (the app talks plain `ws://`, no TLS).
2. Install the app: `flutter build apk` / `flutter build ios` from
   `mobile/` (per M2), or `flutter run` on a dev device.
3. Connect screen values:
   - Host: `<box-lan-ip>` (the LAN/VPN IP of the box)
   - Port: `8082` (the WS voice port)
   - Token: the same value as `JARVIS_WS_TOKEN` on the box (leave empty only
     if the gateway runs without a token)

## 3. Manual voice E2E checklist

Run on the Strix Halo box with the phone beside you. Each item has an
observable expectation — pass/fail is unambiguous.

- [ ] Gateway starts, WS port listening:
      `ss -tlnp | grep 8082` shows the listener; `journalctl -u jarvis-gateway`
      shows `WS session: ws://.../v1/voice/session`.
- [ ] App connects with correct token → connect screen shows a green light,
      no errors in `journalctl -u jarvis-gateway`.
- [ ] App connects with a wrong/empty token (while `JARVIS_WS_TOKEN` is set)
      → gateway returns 403 and the app shows the failure state, not green.
- [ ] Tap JARVIS → light goes "listening"; speak → light shows "processing" →
      assistant transcript line appears in the session list.
- [ ] Reply plays through the phone speaker with the Zyphra codec voice
      (persona's `voice_pack`), not silent.
- [ ] Second turn: speak again without re-tapping → same flow works
      (voice-active session stays armed).
- [ ] Tap stop → light returns to idle, transcript stays visible until you
      leave the screen.
- [ ] Kill the gateway mid-session (`sudo systemctl kill jarvis-gateway`) →
      phone shows Offline + Reconnect within seconds; `Restart=on-failure`
      brings the unit back within `RestartSec=2`, tap Reconnect → session
      works again (transcript may be reset).
- [ ] VPN off mid-session → same Offline + Reconnect path; VPN back on →
      tap Reconnect → session works again (no gateway restart needed).
- [ ] Voice-cloned pack: add a cloned pack to `VOICE_PACKS_DIR`, switch the
      active persona/voice (persona JSON `voice_pack` field; hot-reloaded
      every 5 s), verify the reply uses the new voice.

## 4. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Turn fails at STT; log says `STT unavailable (WHISPER_MODEL_PATH not set)` | `WHISPER_MODEL_PATH` unset or model failed to load | Set the var to a valid Whisper GGUF; check the path in the unit and reload |
| Reply comes back empty/silent | No voice pack found in `VOICE_PACKS_DIR` | Put at least `zaya_default.voice` (or the persona's pack) in the dir; startup log shows `codec TTS: no voice packs found` |
| App gets 403 on connect | `JARVIS_WS_TOKEN` on the box differs from the token in the app | Match them; unset the var on the box if you want auth off |
| App gets 400 right after connect / cannot establish session | Stale gateway binary predating the WS upgrade-parser fix (`d3eb550d`) | Rebuild: `cmake --build build --target onebin`, `sudo systemctl restart jarvis-gateway` |
| Phone cannot reach `ws://<box>:8082` at all | WS listener still on loopback (`WS_STREAM_BIND` unset) | Set `Environment=WS_STREAM_BIND=0.0.0.0` in the unit (see §1.5), `systemctl daemon-reload` + restart; verify with `curl http://127.0.0.1:8082/v1/voice/session` on the box (expect 400/403, not connection refused) |
| Voice turns fail at LLM step | Unified server not running | Start `build/1bit unified`; check `UNIFIED_URL` (default `http://127.0.0.1:8088`) |
