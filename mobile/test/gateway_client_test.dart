import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:jarvis_mobile/ws/gateway_client.dart';
import 'package:jarvis_mobile/ws/protocol.dart';

/// Polls [cond] until true or [timeout] elapses.
Future<void> waitUntil(bool Function() cond,
    {Duration timeout = const Duration(seconds: 5)}) async {
  final end = DateTime.now().add(timeout);
  while (!cond()) {
    if (DateTime.now().isAfter(end)) fail('condition not met within $timeout');
    await Future<void>.delayed(const Duration(milliseconds: 20));
  }
}

/// Test WebSocket server: upgrades every request, records received frames,
/// answers `{"type":"start"}` with meta + state text frames and one
/// 1248-byte binary frame.
class TestWsServer {
  final HttpServer http;
  final List<Object> received = [];
  String? authHeader;

  TestWsServer._(this.http);

  int get port => http.port;

  static Future<TestWsServer> start() async {
    final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    final s = TestWsServer._(server);
    server.listen((request) async {
      s.authHeader = request.headers.value('authorization');
      final ws = await WebSocketTransformer.upgrade(request);
      ws.listen((data) {
        s.received.add(data);
        if (data == ControlFrame.start) {
          ws.add(
              '{"type":"meta","sample_rate":24000,"channels":1,"format":"float32","frame_ms":13,"session":true}');
          ws.add('{"type":"state","state":"speaking"}');
          ws.add(Uint8List.fromList(List.generate(1248, (i) => i % 251)));
        }
      });
    });
    return s;
  }

  Future<void> close() => http.close(force: true);
}

void main() {
  test('connect sends start/stop, forwards PCM, emits events and audio', () async {
    final server = await TestWsServer.start();
    final client = GatewayClient();
    try {
      await client.connect('127.0.0.1', server.port, 'secret-token');
      expect(client.connected, isTrue);
      expect(server.authHeader, 'Bearer secret-token');

      final events = <GatewayEvent>[];
      final audio = <Uint8List>[];
      final evSub = client.events.listen(events.add);
      final auSub = client.audio.listen(audio.add);
      addTearDown(evSub.cancel);
      addTearDown(auSub.cancel);

      client.start();
      client.stop();

      final frame = Uint8List.fromList(List.generate(640, (i) => i * 3 % 256));
      await client.sendPcm16(frame);

      await waitUntil(() => events.length >= 2 && audio.isNotEmpty);
      // Server-received binary frame: compare bytes (Uint8List has identity ==).
      await waitUntil(() => server.received
          .where((d) => d is List<int> && d.length == frame.length)
          .isNotEmpty);

      expect(events, hasLength(2));
      expect(events[0].maybeMeta?.sampleRate, 24000);
      expect(events[0].maybeMeta?.format, 'float32');
      expect(events[1].maybeState, GatewayState.speaking);

      // Binary downlink arrives on the audio stream, byte-for-byte.
      expect(audio, hasLength(1));
      expect(audio.single, orderedEquals(List.generate(1248, (i) => i % 251)));

      // Server received start, stop and the exact PCM bytes.
      expect(server.received.whereType<String>(),
          containsAllInOrder([ControlFrame.start, ControlFrame.stop]));
      expect(server.received
          .whereType<Uint8List>()
          .single, orderedEquals(frame));

      await client.close();
      expect(client.connected, isFalse);
    } finally {
      await server.close();
    }
  });

  test('skips malformed and unknown text frames', () async {
    final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    server.listen((request) async {
      final ws = await WebSocketTransformer.upgrade(request);
      ws.add('not json');
      ws.add('{"type":"unknown"}');
      ws.add('{"type":"state","state":"idle"}');
    });
    final client = GatewayClient();
    try {
      await client.connect('127.0.0.1', server.port, null);
      final events = <GatewayEvent>[];
      addTearDown(client.events.listen(events.add).cancel);
      await waitUntil(() => events.isNotEmpty);
      expect(events, hasLength(1));
      expect(events.single.maybeState, GatewayState.idle);
    } finally {
      await server.close(force: true);
    }
  });

  test('connect throws GatewayException on 403', () async {
    final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    server.listen((request) {
      request.response.statusCode = 403;
      request.response.close();
    });
    final client = GatewayClient();
    try {
      await expectLater(client.connect('127.0.0.1', server.port, null),
          throwsA(isA<GatewayException>()));
      expect(client.connected, isFalse);
    } finally {
      await server.close(force: true);
    }
  });

  test('connect throws GatewayException on socket error', () async {
    // Bind a port, then close it so nothing listens there.
    final probe = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    final deadPort = probe.port;
    await probe.close(force: true);

    final client = GatewayClient();
    await expectLater(client.connect('127.0.0.1', deadPort, null),
        throwsA(isA<GatewayException>()));
    expect(client.connected, isFalse);
  });
}
