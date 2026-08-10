// JARVIS voice-gateway stub: pure dart:io WebSocket server that replays a
// canned /v1/voice/session session for integration tests and the iOS
// simulator demo:
//
//   connect → meta → listening → transcript(user "hello from the simulator")
//   → processing → speaking → ~1 s of 440 Hz sine (float32 @ 24 kHz,
//   1248-byte frames) → transcript(assistant "Hi! JARVIS mobile is alive.")
//   → listening → idle.
//
//   {"type":"stop"} → {"type":"end","reason":"stopped"} + close.
//   {"type":"start"} → begins the canned sequence.
//
// Run:  dart run tool/stub_server.dart [port]     (default 8082)
//
// Also importable from tests (start() in-process on port 0) — `main` only
// runs when the file is executed directly. No package imports, so it runs
// with a bare `dart run` even without pub get.

import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

const String _kStart = '{"type":"start"}';
const String _kStop = '{"type":"stop"}';

const int kSamplesPerFrame = 312; // 13 ms @ 24 kHz = 1248 bytes float32
const double _kSineAmp = 0.2;
const double _kSineHz = 440.0;
const int _kSampleRate = 24000;
const int _kFrames = 77; // 77 * 13 ms ≈ 1.0 s

const String _kMeta = '{"type":"meta","sample_rate":24000,"channels":1,'
    '"format":"float32","frame_ms":13,"session":true}';

class StubGateway {
  HttpServer? _http;

  /// Text frames received from the connected client, in order.
  final List<String> received = <String>[];

  double _phase = 0; // sine phase across frames (one connection at a time)

  int get port => _http!.port;

  Future<void> start({int port = 8082}) async {
    _http = await HttpServer.bind(InternetAddress.anyIPv4, port);
    _http!.listen(_onRequest);
  }

  Future<void> close() async {
    await _http?.close(force: true);
    _http = null;
  }

  void _onRequest(HttpRequest request) async {
    if (request.uri.path != '/v1/voice/session' ||
        !WebSocketTransformer.isUpgradeRequest(request)) {
      request.response.statusCode = HttpStatus.notFound;
      request.response.write('stub gateway: connect to '
          'ws://host:port/v1/voice/session');
      await request.response.close();
      return;
    }
    final ws = await WebSocketTransformer.upgrade(request);
    stdout.writeln('stub: client connected');
    ws.add(_kMeta); // meta right after the 101, like the real gateway
    ws.listen((data) {
      if (data is! String) return; // uplink PCM16: ignored by the stub
      received.add(data);
      stdout.writeln('stub: received $data');
      if (data == _kStop) {
        stdout.writeln('stub: stop -> end/stopped, close');
        ws.add('{"type":"end","reason":"stopped"}');
        ws.close();
      } else if (data == _kStart) {
        stdout.writeln('stub: start -> replaying canned session');
        unawaited(_replay(ws));
      }
    });
  }

  Future<void> _replay(WebSocket ws) async {
    Future<void> text(String s) async {
      if (ws.readyState != WebSocket.open) return;
      ws.add(s);
    }

    await text('{"type":"state","state":"listening"}');
    await text('{"type":"transcript","role":"user","text":"hello from the simulator"}');
    await text('{"type":"state","state":"processing"}');
    await text('{"type":"state","state":"speaking"}');

    // ~1 s of 440 Hz sine as float32 @ 24 kHz, paced at real time.
    final samples = Float32List(kSamplesPerFrame);
    for (var f = 0; f < _kFrames; f++) {
      if (ws.readyState != WebSocket.open) return;
      for (var i = 0; i < kSamplesPerFrame; i++) {
        samples[i] = _kSineAmp * sin(2 * pi * _kSineHz * _phase / _kSampleRate);
        _phase += 1;
      }
      ws.add(samples.buffer.asUint8List());
      await Future<void>.delayed(const Duration(milliseconds: 13));
    }

    await text('{"type":"transcript","role":"assistant","text":"Hi! JARVIS mobile is alive."}');
    await text('{"type":"state","state":"listening"}');
  }
}

Future<void> main(List<String> args) async {
  final port = args.isNotEmpty ? int.parse(args[0]) : 8082;
  final stub = StubGateway();
  await stub.start(port: port);
  stdout.writeln('JARVIS stub gateway on '
      'ws://127.0.0.1:${stub.port}/v1/voice/session (Ctrl-C to stop)');
}
