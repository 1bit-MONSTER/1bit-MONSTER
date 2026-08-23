# JARVIS Mobile M2 — Flutter App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use supo-subagent-driven-development (recommended) or supo-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Flutter app (`mobile/` in the 1bit-monster repo) that connects to the M1 gateway's `/v1/voice/session` WebSocket: tap-to-start voice-active conversation, status lights, transcript log. No data stored on the device. iOS simulator is the first build target (cloud Mac provisioned); Android follows.

**Architecture:** Three service layers + UI, all under `mobile/`:
- `lib/ws/protocol.dart` — pure Dart protocol constants + frame parsing (no plugins) — fully unit-testable.
- `lib/ws/gateway_client.dart` — WebSocket client (web_socket_channel): connect with bearer header, send control + PCM16 frames, dispatch received frames to callbacks.
- `lib/audio/audio_io.dart` — mic capture (record plugin, WAV-format output with the 44-byte header stripped → PCM16 @ 16 kHz) + reply playback (accumulate float32 @ 24 kHz → PCM16 → WAV → audioplayers BytesSource).
- `lib/state/session_controller.dart` — a ChangeNotifier mirroring the gateway state machine (idle/connecting/listening/processing/speaking/offline/error) + transcript list.
- UI: `lib/main.dart`, `lib/screens/connect_screen.dart`, `lib/screens/voice_screen.dart`.

**Tech Stack:** Flutter (stable, 3.44.x), Dart 3, `web_socket_channel`, `record` (mic), `audioplayers` (reply playback), `flutter_secure_storage` (URL + token). Builds: `flutter build ios --simulator` on the cloud Mac (rentamac.io; Flutter 3.44.9, Xcode 26.6, CocoaPods 1.17.0, iOS 26.5 runtime). Tests: `flutter test` on the EPYC dev box (Flutter Linux SDK at ~/flutter) — all unit/widget tests are pure-Dart and run headless.

## Global Constraints

- **Repo:** `~/1bit-monster`, branch `feat/jarvis-mobile`. Spec: `docs/superpowers/specs/2026-08-08-jarvis-mobile-design.md`. M1 (gateway) is DONE — the wire protocol below is implemented server-side; DO NOT change server behavior.
- **Wire protocol (from M1 — verbatim):**
  - URL: `ws://<host>:8082/v1/voice/session` (port overridable via `WS_STREAM_PORT`); handshake header `Authorization: Bearer <token>` (token optional when the server runs without `JARVIS_WS_TOKEN`).
  - Server → client text frames:
    - `{"type":"meta","session":true,"sample_rate":16000,"channels":1,"format":"pcm16","frame_ms":20}`
    - `{"type":"state","state":"idle|listening|processing|speaking"}`
    - `{"type":"transcript","role":"user|assistant","text":"..."}`
    - `{"type":"end","reason":"stopped|error"}` (session over — server keeps the socket open only until close)
    - `{"type":"error","message":"..."}`
  - Client → server text: `{"type":"start"}`, `{"type":"stop"}`, `{"type":"cancel"}`
  - Client → server binary: PCM16 @ 16 kHz mono, 20 ms frames (640 bytes)
  - Server → client binary: float32 @ 24 kHz mono, 13 ms frames (312 samples = 1248 bytes)
  - NOTE (M1 final review): `hello`/`bye`/`version` frames from the spec's protocol section are NOT on the wire — the meta frame is the handshake. Track this: the client greets nothing, expects meta immediately after connect.
- **No data on device:** nothing persisted except the connect settings (URL + token in secure storage). Transcript log is in-memory only and cleared when leaving the voice screen.
- **User WIP:** uncommitted user files exist in the repo (engine/npu/*, tools/bitnet_decode.cpp, third_party/FastFlowLM, etc.) — NEVER commit them. `mobile/` is a NEW directory; commits must contain only `mobile/` paths (plus `.gitignore` root hunks if needed — prefer a `mobile/.gitignore` to avoid touching the root one).
- **No new server work.** Any server-side need discovered here goes into the report as an M1-follow-up note, not a server edit.
- **Commit style:** `feat(jarvis-mobile): ...` / `test(jarvis-mobile): ...` / `build(jarvis-mobile): ...`.
- **Builds:** run `flutter build ios --simulator --debug` on the Mac by rsyncing `mobile/` over (credentials in `~/Documents/Mac mini SSH credentials.txt`; PATH on the Mac needs `$HOME/flutter/bin:$HOME/ruby/bin`).

---

### Task 1: App scaffold + pure protocol layer + session controller (all unit-testable)

**Files:**
- Create: `mobile/pubspec.yaml` (name `jarvis_mobile`, deps: web_socket_channel ^3, record ^5, audioplayers ^6, flutter_secure_storage ^9; dev: flutter_test, flutter_lints)
- Create: `mobile/analysis_options.yaml` (flutter_lints defaults)
- Create: `mobile/.gitignore` (standard Flutter ignores: build/, .dart_tool/, ios/Pods, etc.)
- Create: `mobile/lib/ws/protocol.dart`
- Create: `mobile/lib/state/session_controller.dart`
- Create: `mobile/test/protocol_test.dart`
- Create: `mobile/test/session_controller_test.dart`

**Interfaces:**
- Consumes: nothing (pure Dart).
- Produces (Tasks 2–4 consume):
  - `class ControlFrame { static const start = '{"type":"start"}'; static const stop = '{"type":"stop"}'; static const cancel = '{"type":"cancel"}'; }`
  - `class MetaFrame { final int sampleRate; final int channels; final String format; final int frameMs; final bool session; }`
  - `enum GatewayState { idle, listening, processing, speaking }` — parseable from state frames.
  - `class TranscriptLine { final String role; final String text; }`
  - `class GatewayEvent { const GatewayEvent.meta(MetaFrame m); const GatewayEvent.state(GatewayState s); const GatewayEvent.transcript(TranscriptLine t); const GatewayEvent.end(String reason); const GatewayEvent.error(String message); const GatewayEvent.audio(Uint8List float32); }`
  - `GatewayEvent? parseGatewayFrame(String text)` — returns null for unknown/malformed text frames.
  - `const kUplinkFrameBytes = 640;` `const kDownlinkFrameBytes = 1248;` `const kWavHeaderSize = 44;`
  - `Uint8List pcm16ToWav(Uint8List pcm16, int sampleRate, int channels)` — 44-byte RIFF/WAVE header + PCM16 payload (24 kHz replies, 16 kHz mic).
  - `Uint8List float32ToPcm16(Uint8List float32)` — little-endian float32 → int16 clamp [-1,1]*32767.
  - `class SessionController extends ChangeNotifier { GatewayState state; List<TranscriptLine> transcript; String? errorMessage; bool connected; void reset(); void onEvent(GatewayEvent e); void setConnecting(); void setOffline(String reason); }` — transcript capped at 100 lines; `reset()` clears transcript + error.

- [ ] **Step 1: Scaffold + write the failing tests**

```bash
cd ~/1bit-monster && flutter create --platforms=ios,android --org systems.onebit --project-name jarvis_mobile mobile
```

Then replace `mobile/test/widget_test.dart` with two real test files:

`mobile/test/protocol_test.dart`:
```dart
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:jarvis_mobile/ws/protocol.dart';

void main() {
  test('parses meta frame', () {
    final e = parseGatewayFrame(
        '{"type":"meta","session":true,"sample_rate":16000,"channels":1,"format":"pcm16","frame_ms":20}');
    expect(e, isA<GatewayEvent>());
    final m = (e as GatewayEvent).maybeMeta;
    expect(m!.sampleRate, 16000);
    expect(m.format, 'pcm16');
    expect(m.frameMs, 20);
  });

  test('parses state, transcript, end, error', () {
    expect((parseGatewayFrame('{"type":"state","state":"speaking"}') as GatewayEvent).maybeState, GatewayState.speaking);
    final t = (parseGatewayFrame('{"type":"transcript","role":"assistant","text":"hi"}') as GatewayEvent).maybeTranscript!;
    expect(t.role, 'assistant');
    expect(t.text, 'hi');
    expect((parseGatewayFrame('{"type":"end","reason":"stopped"}') as GatewayEvent).maybeEnd, 'stopped');
    expect((parseGatewayFrame('{"type":"error","message":"boom"}') as GatewayEvent).maybeError, 'boom');
  });

  test('rejects malformed frames', () {
    expect(parseGatewayFrame('not json'), isNull);
    expect(parseGatewayFrame('{"type":"bogus"}'), isNull);
    expect(parseGatewayFrame('{"state":"listening"}'), isNull);
  });

  test('float32ToPcm16 converts and clamps', () {
    final f32 = Float32List.fromList([0.0, 0.5, -0.5, 2.0, -2.0]);
    final pcm = float32ToPcm16(Uint8List.view(f32.buffer));
    expect(pcm.length, 10);
    final i16 = ByteData.sublistView(pcm);
    expect(i16.getInt16(0, Endian.little), 0);
    expect(i16.getInt16(2, Endian.little), 16384);
    expect(i16.getInt16(4, Endian.little), -16384);
    expect(i16.getInt16(6, Endian.little), 32767); // clamped
    expect(i16.getInt16(8, Endian.little), -32768); // clamped
  });

  test('pcm16ToWav builds a playable header', () {
    final pcm = Uint8List(6400);
    final wav = pcm16ToWav(pcm, 24000, 1);
    expect(wav.length, 6400 + 44);
    final h = ByteData.sublistView(wav);
    expect(String.fromCharCodes(wav.sublist(0, 4)), 'RIFF');
    expect(String.fromCharCodes(wav.sublist(8, 12)), 'WAVE');
    expect(h.getUint32(24, Endian.little), 24000); // sample rate
    expect(h.getUint16(22, Endian.little), 1);     // channels
    expect(h.getUint16(34, Endian.little), 16);    // bits per sample
  });
}
```

`mobile/test/session_controller_test.dart`:
```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:jarvis_mobile/state/session_controller.dart';
import 'package:jarvis_mobile/ws/protocol.dart';

void main() {
  test('state transitions + transcript from events', () {
    final c = SessionController();
    c.setConnecting();
    expect(c.connected, false);
    c.onEvent(parseGatewayFrame(
        '{"type":"meta","session":true,"sample_rate":16000,"channels":1,"format":"pcm16","frame_ms":20}')!);
    c.onEvent(parseGatewayFrame('{"type":"state","state":"listening"}')!);
    expect(c.state, GatewayState.listening);
    expect(c.connected, true);
    c.onEvent(parseGatewayFrame('{"type":"transcript","role":"user","text":"hello"}')!);
    c.onEvent(parseGatewayFrame('{"type":"transcript","role":"assistant","text":"hi there"}')!);
    expect(c.transcript.length, 2);
    expect(c.transcript.last.text, 'hi there');
  });

  test('error sets message, reset clears everything', () {
    final c = SessionController();
    c.onEvent(parseGatewayFrame('{"type":"error","message":"STT unavailable"}')!);
    expect(c.errorMessage, 'STT unavailable');
    c.onEvent(parseGatewayFrame('{"type":"state","state":"listening"}')!);
    c.reset();
    expect(c.transcript, isEmpty);
    expect(c.errorMessage, isNull);
  });

  test('transcript capped at 100 lines', () {
    final c = SessionController();
    for (var i = 0; i < 120; i++) {
      c.onEvent(parseGatewayFrame('{"type":"transcript","role":"user","text":"t$i"}')!);
    }
    expect(c.transcript.length, 100);
    expect(c.transcript.first.text, 't20');
  });
}
```

`mobile/lib/ws/protocol.dart`:
```dart
import 'dart:convert';
import 'dart:typed_data';

const int kUplinkFrameBytes = 640;    // PCM16 @ 16 kHz, 20 ms
const int kDownlinkFrameBytes = 1248; // float32 @ 24 kHz, 13 ms
const int kWavHeaderSize = 44;

class ControlFrame {
  static const String start = '{"type":"start"}';
  static const String stop = '{"type":"stop"}';
  static const String cancel = '{"type":"cancel"}';
}

enum GatewayState { idle, listening, processing, speaking }

GatewayState? gatewayStateFromString(String s) => switch (s) {
      'idle' => GatewayState.idle,
      'listening' => GatewayState.listening,
      'processing' => GatewayState.processing,
      'speaking' => GatewayState.speaking,
      _ => null,
    };

class MetaFrame {
  final int sampleRate;
  final int channels;
  final String format;
  final int frameMs;
  final bool session;
  MetaFrame(this.sampleRate, this.channels, this.format, this.frameMs, this.session);
}

class TranscriptLine {
  final String role;
  final String text;
  TranscriptLine(this.role, this.text);
}

sealed class GatewayEvent {
  const GatewayEvent();
  MetaFrame? get maybeMeta => null;
  GatewayState? get maybeState => null;
  TranscriptLine? get maybeTranscript => null;
  String? get maybeEnd => null;
  String? get maybeError => null;
}

class MetaEvent extends GatewayEvent {
  final MetaFrame meta;
  const MetaEvent(this.meta);
  @override MetaFrame? get maybeMeta => meta;
}
class StateEvent extends GatewayEvent {
  final GatewayState state;
  const StateEvent(this.state);
  @override GatewayState? get maybeState => state;
}
class TranscriptEvent extends GatewayEvent {
  final TranscriptLine line;
  const TranscriptEvent(this.line);
  @override TranscriptLine? get maybeTranscript => line;
}
class EndEvent extends GatewayEvent {
  final String reason;
  const EndEvent(this.reason);
  @override String? get maybeEnd => reason;
}
class ErrorEvent extends GatewayEvent {
  final String message;
  const ErrorEvent(this.message);
  @override String? get maybeError => message;
}

/// Parses a server text frame. Returns null for malformed or unknown frames.
GatewayEvent? parseGatewayFrame(String text) {
  final Object? decoded;
  try {
    decoded = jsonDecode(text);
  } catch (_) {
    return null;
  }
  if (decoded is! Map<String, dynamic>) return null;
  final type = decoded['type'];
  if (type is! String) return null;
  switch (type) {
    case 'meta':
      return MetaEvent(MetaFrame(
        decoded['sample_rate'] as int? ?? 0,
        decoded['channels'] as int? ?? 1,
        decoded['format'] as String? ?? '',
        decoded['frame_ms'] as int? ?? 0,
        decoded['session'] as bool? ?? false,
      ));
    case 'state':
      final s = gatewayStateFromString(decoded['state'] as String? ?? '');
      return s == null ? null : StateEvent(s);
    case 'transcript':
      final role = decoded['role'] as String? ?? '';
      final text = decoded['text'] as String? ?? '';
      if (role.isEmpty || text.isEmpty) return null;
      return TranscriptEvent(TranscriptLine(role, text));
    case 'end':
      return EndEvent(decoded['reason'] as String? ?? '');
    case 'error':
      return ErrorEvent(decoded['message'] as String? ?? '');
    default:
      return null;
  }
}

/// Little-endian float32 -> clamped int16 PCM.
Uint8List float32ToPcm16(Uint8List float32) {
  final inF = Float32List.view(float32.buffer, float32.offsetInBytes, float32.lengthInBytes ~/ 4);
  final out = ByteData(float32.lengthInBytes ~/ 2);
  for (var i = 0; i < inF.length; i++) {
    final v = (inF[i] * 32767.0).clamp(-32768.0, 32767.0);
    out.setInt16(i * 2, v.round(), Endian.little);
  }
  return out.buffer.asUint8List();
}

/// Prepends a RIFF/WAVE header to PCM16 audio.
Uint8List pcm16ToWav(Uint8List pcm16, int sampleRate, int channels) {
  final dataLen = pcm16.length;
  final wav = ByteData(kWavHeaderSize + dataLen);
  void put(int offset, String s) {
    for (var i = 0; i < s.length; i++) {
      wav.setUint8(offset + i, s.codeUnitAt(i));
    }
  }
  put(0, 'RIFF');
  wav.setUint32(4, 36 + dataLen, Endian.little);
  put(8, 'WAVE');
  put(12, 'fmt ');
  wav.setUint32(16, 16, Endian.little);
  wav.setUint16(20, 1, Endian.little);            // PCM
  wav.setUint16(22, channels, Endian.little);
  wav.setUint32(24, sampleRate, Endian.little);
  wav.setUint32(28, sampleRate * channels * 2, Endian.little);
  wav.setUint16(32, channels * 2, Endian.little);
  wav.setUint16(34, 16, Endian.little);
  put(36, 'data');
  wav.setUint32(40, dataLen, Endian.little);
  final out = wav.buffer.asUint8List();
  out.setRange(kWavHeaderSize, out.length, pcm16);
  return out;
}
```

`mobile/lib/state/session_controller.dart`:
```dart
import 'package:flutter/foundation.dart';
import '../ws/protocol.dart';

class SessionController extends ChangeNotifier {
  GatewayState _state = GatewayState.idle;
  final List<TranscriptLine> _transcript = [];
  String? _errorMessage;
  bool _connected = false;

  GatewayState get state => _state;
  List<TranscriptLine> get transcript => List.unmodifiable(_transcript);
  String? get errorMessage => _errorMessage;
  bool get connected => _connected;

  static const int _maxTranscript = 100;

  void setConnecting() {
    _connected = false;
    _state = GatewayState.idle;
    _errorMessage = null;
    notifyListeners();
  }

  void setOffline(String reason) {
    _connected = false;
    _state = GatewayState.idle;
    _errorMessage = reason;
    notifyListeners();
  }

  void onEvent(GatewayEvent e) {
    final s = e.maybeState;
    if (s != null) _state = s;
    final t = e.maybeTranscript;
    if (t != null) {
      _transcript.add(t);
      if (_transcript.length > _maxTranscript) {
        _transcript.removeRange(0, _transcript.length - _maxTranscript);
      }
    }
    if (e.maybeMeta != null) _connected = true;
    final err = e.maybeError;
    if (err != null) _errorMessage = err;
    notifyListeners();
  }

  void reset() {
    _transcript.clear();
    _errorMessage = null;
    _state = GatewayState.idle;
    _connected = false;
    notifyListeners();
  }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/1bit-monster/mobile && ~/flutter/bin/flutter pub get >/dev/null && ~/flutter/bin/flutter test
```
Expected: FAIL — `protocol.dart` / `session_controller.dart` missing (compile errors).

- [ ] **Step 3: Create the two lib files** (exact code above), delete `mobile/test/widget_test.dart` (scaffold template test — it references the counter app and would fail).

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/1bit-monster/mobile && ~/flutter/bin/flutter test
```
Expected: all protocol + controller tests PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/1bit-monster && git add mobile && git commit -m "feat(jarvis-mobile): app scaffold + protocol layer + session controller"
```
Verify `git show --stat HEAD`: only `mobile/` paths.

---

### Task 2: Gateway client + audio I/O services (with test doubles)

**Files:**
- Create: `mobile/lib/ws/gateway_client.dart`
- Create: `mobile/lib/audio/audio_io.dart`
- Create: `mobile/test/gateway_client_test.dart`
- Create: `mobile/test/audio_io_test.dart`

**Interfaces:**
- Consumes: `protocol.dart` (Task 1).
- Produces (Tasks 3–4 consume):
  - `class GatewayClient { Future<void> connect(String host, int port, String? token); void start(); void stop(); void cancel(); Future<void> sendPcm16(Uint8List frame); Future<void> close(); Stream<GatewayEvent> get events; bool get connected; }` — `connect` throws `GatewayException` on failure (socket error, non-101, 403). Audio frames: raw bytes → `WebSocketChannel.sink.add`.
  - `abstract class MicSource { Stream<Uint8List> get pcm16Frames; Future<void> start(); Future<void> stop(); }` + `class RecordMicSource implements MicSource` (record plugin: `AudioEncoder.wav`, sampleRate 16000, numChannels 1; strip the leading 44-byte WAV header from the stream, chunk into 640-byte PCM16 frames; if the plugin cannot do wav/pcm16 on a platform, throw a clear `MicUnavailableException`).
  - `abstract class ReplyPlayer { Future<void> playPcm16(Uint8List pcm16, int sampleRate); Future<void> stop(); }` + `class WavReplyPlayer implements ReplyPlayer` (audioplayers `AudioPlayer().play(BytesSource(wavBytes))`, volume 1.0; buffer is the full reply).
  - `class ReplyBuffer { void addFloat32(Uint8List f32); Uint8List? takePcm16(); void reset(); }` — accumulates float32 frames; `takePcm16()` returns `float32ToPcm16` of the buffer and clears it (used by the controller between `state==speaking` and the next state).
- Test doubles: `FakeGatewaySocket` (a `StreamController<String/Uint8List>`-backed pair exposing `sink`/`stream` and capturing sent frames) and `FakeMic`/`FakePlayer` implementations of the abstract classes.

**Tests (verify real behavior):**
- `gateway_client_test.dart`:
  - client sends `{"type":"start"}` after `start()` and `{"type":"stop"}` after `stop()`;
  - `sendPcm16` forwards exact bytes;
  - incoming text frames are parsed and emitted as `GatewayEvent`s (feed meta + state frames, collect events);
  - incoming binary frames emitted as-is (assert byte-for-byte);
  - `connect` failure: with a URL pointing at a closed port, `connect` throws `GatewayException` (this test needs a real socket — bind a `HttpServer` on `127.0.0.1:0` in the test and use its port, or use `WebSocketChannel.connect` against a dead port with a short timeout; implement with a tiny local `HttpServer` that responds 403 to prove the 403 path, and a closed port for the socket-error path).
  - NOTE: `web_socket_channel` on the Dart VM connects via `dart:io` — this works under `flutter test` on Linux.
- `audio_io_test.dart`:
  - `ReplyBuffer`: add two 1248-byte float32 frames (write known floats, e.g. 0.25 → expect int16 8192), `takePcm16` returns 2496 bytes and clears;
  - `WavReplyPlayer` with a `FakePlayer`? (audioplayers is a plugin — abstract `ReplyPlayer` and test only `ReplyBuffer` + WAV path via `pcm16ToWav`; the player impl is thin glue verified on the simulator in Task 4);
  - `RecordMicSource` header-strip logic: factor the header-strip/chunk logic into a pure function `Uint8List stripWavHeader(Uint8List bytes)` + `List<Uint8List> chunkPcm16(Uint8List pcm, int frameBytes)` and unit-test those (feed 44 header + 1280 bytes → two 640-byte frames, header gone).

- [ ] **Step 1: Write the failing tests** (as specified above; exact assertions for the pure parts, behavior-level for the client with the local test HttpServer).
- [ ] **Step 2: Run to verify they fail** (`flutter test`).
- [ ] **Step 3: Implement** `gateway_client.dart`, `audio_io.dart` per the interfaces. Client details: `WebSocketChannel.connect(Uri.parse('ws://$host:$port/v1/voice/session'), headers: token == null ? null : {'Authorization': 'Bearer $token'})`; events stream maps text → `parseGatewayFrame` (skip nulls), binary → raw bytes; `sendPcm16` uses `sink.add(frame)`; errors → `GatewayException`.
- [ ] **Step 4: Run to verify they pass.**
- [ ] **Step 5: Commit** — `feat(jarvis-mobile): gateway WS client + audio IO with test doubles`.

---

### Task 3: UI — connect screen + voice screen + state lights + transcript

**Files:**
- Create: `mobile/lib/screens/connect_screen.dart`
- Create: `mobile/lib/screens/voice_screen.dart`
- Create: `mobile/lib/main.dart` (replace scaffold)
- Create: `mobile/lib/state/app_state.dart` — tiny `AppState extends ChangeNotifier { String? host; int port; String? token; }` with secure-storage load/save (`flutter_secure_storage`; keys `jarvis_host`, `jarvis_port`, `jarvis_token`).
- Create: `mobile/test/voice_screen_test.dart` (widget test with a fake `GatewayClient` injected — the voice screen takes a `GatewayClient Function()` factory + `MicSource`/`ReplyPlayer` factories so the test can inject fakes)
- Modify: `mobile/ios/Runner/Info.plist` — add `NSMicrophoneUsageDescription` ("JARVIS listens only while you hold the session open.") and `NSAppTransportSecurity` → `NSAllowsArbitraryLoads` true (plain `ws://` to a LAN/VPN host; document why in a comment).

**UI spec (from the design spec):**
- Connect screen: server host field, port field (default 8082), token field (obscured, optional), "Connect" button → saves to secure storage → navigates to voice screen. Errors shown inline.
- Voice screen: JARVIS button (large circular, tap = start session, tap again = stop; disabled while connecting), status lights row (idle=gray, connecting=amber pulse, listening=green, processing=blue, speaking=cyan, offline=red) with a label, transcript list (auto-scroll to bottom), "Disconnect" back button, error banner when `errorMessage != null`.
- Widget test: fake client emits meta + state listening → button enabled, green light; fake client emits transcript → line appears; tap button → fake client received `start`; second tap → `stop`.

- [ ] **Step 1: Write the failing widget tests** (fake client per the interfaces; the screen must accept injected factories — constructor params with defaults).
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement screens + main + app_state.**
- [ ] **Step 4: Run to verify they pass** (`flutter test` — widget tests run headless on Linux).
- [ ] **Step 5: Commit** — `feat(jarvis-mobile): connect + voice screens, state lights, transcript UI`.

---

### Task 4: Integration — live gateway protocol test + iOS simulator proof

**Files:**
- Create: `mobile/test/live_gateway_test.dart` (skipped by default — `@Tags(['live'])`, run explicitly)
- Create: `mobile/tool/stub_server.dart` — a small Dart WS server (using `dart:io` `HttpServer` + manual RFC 6455 upgrade is NOT needed — use `package:web_socket_channel/web_socket_channel.dart`'s `WebSocketChannel` server side? The package has no server; use `dart:io` `WebSocketTransformer.upgrade`) that replays a canned session: meta → state listening → transcript user ("hello from the simulator") → state processing → state speaking → 1 s of 440 Hz sine as float32@24k frames → transcript assistant ("Hi! JARVIS mobile is alive.") → state listening → (idle until stop) → end stopped on stop.
- Modify: `mobile/test/live_gateway_test.dart` — connects `GatewayClient` to a local stub server on `127.0.0.1:0`, asserts: meta parsed, state sequence received, audio bytes received (byte count > 0), start/stop frames sent. Then a second variant with `WS_HOST`/`WS_PORT` env vars pointing at the real EPYC gateway (`ws://<host>:8082`) — skipped unless `JARVIS_LIVE=1`.

**Steps:**
- [ ] **Step 1: Write `stub_server.dart`** (WebSocketTransformer upgrade on an HttpServer; canned session per above; sine generator in pure Dart).
- [ ] **Step 2: Write `live_gateway_test.dart`** (stub-server variant always runs; env-var variant tagged live/skipped).
- [ ] **Step 3: Run `flutter test`** — stub variant passes headless on the EPYC box.
- [ ] **Step 4: Live run against the real gateway** on the EPYC box: start `build/1bit jarvis` (per M1 report instructions) and run `JARVIS_LIVE=1 flutter test --tags live test/live_gateway_test.dart` — expect protocol-level pass (meta/state/end frames; STT-unavailable error frame is acceptable — the point is the client handles the real server's frames). If the onebin can't run on this box, report exactly why and mark this step as a Mac/Strix-Halo manual step.
- [ ] **Step 5: iOS simulator proof on the Mac:**
  ```bash
  # on EPYC box:
  rsync -a --exclude build --exclude .dart_tool ~/1bit-monster/mobile/ rentamac@<mac>:/tmp/  # via the Mac SSH creds
  # on Mac (PATH as per Global Constraints):
  cd mobile && flutter pub get && flutter build ios --simulator --debug
  xcrun simctl boot <udid> && xcrun simctl install <udid> build/ios/iphonesimulator/Runner.app
  # run the stub server ON the Mac: dart run tool/stub_server.dart &
  xcrun simctl launch <udid> com.example.jarvisMobile  # (bundle id per org systems.onebit: systems.onebit.jarvisMobile — verify in project)
  # simulator can reach 127.0.0.1:8082 on the Mac; connect in-app to 127.0.0.1:8082, tap JARVIS, screenshot
  xcrun simctl io <udid> screenshot /tmp/jarvis_mobile.png
  ```
  Deliverable: screenshot + the app's transcript showing the stub reply, saved back to the EPYC box.
- [ ] **Step 6: Commit** — `feat(jarvis-mobile): live-gateway integration test + stub server` (+ any fixes the integration surfaced).

---

## Self-Review Notes

- **Spec coverage:** M2 spec items — connect screen (settings), main screen with JARVIS button + status lights, on-screen transcript cleared on exit, mic → PCM16 → WS, float32 → playback, foreground-service note (Android-only; v1 keeps screen-on use — spec's foreground service is deferred to M3 with the reconnect/polish milestone, noted), no data persistence beyond secure-storage settings — all mapped to Tasks 1–4.
- **Type consistency:** `GatewayEvent` sealed hierarchy + `maybeX` accessors used identically across Tasks 1–4; `float32ToPcm16`/`pcm16ToWav` names consistent in tests and lib; `GatewayClient.connect(host, port, token)` signature fixed in Task 2 and consumed unchanged in Tasks 3–4.
- **Honest gaps flagged:** audioplayers latency (reply is buffered then played — streaming playback is a follow-up); mic plugin encoder support varies per platform (wav + header-strip fallback chosen); simulator-proof relies on a stub server when the EPYC gateway is unreachable from the cloud Mac; real-voice E2E stays a Strix Halo manual step (M1 report).
- **Deferred (parked) from M1 final review:** stop() join hang (M3 note), hello/bye frames (client greets nothing — protocol section of the spec already tracks this).
