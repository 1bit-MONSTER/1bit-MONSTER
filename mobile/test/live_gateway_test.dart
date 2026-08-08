import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:jarvis_mobile/ws/gateway_client.dart';
import 'package:jarvis_mobile/ws/protocol.dart';

import '../tool/stub_server.dart';

/// Polls [cond] until true or [timeout] elapses.
Future<void> waitUntil(bool Function() cond,
    {Duration timeout = const Duration(seconds: 15)}) async {
  final end = DateTime.now().add(timeout);
  while (!cond()) {
    if (DateTime.now().isAfter(end)) fail('condition not met within $timeout');
    await Future<void>.delayed(const Duration(milliseconds: 20));
  }
}

void main() {
  test('GatewayClient drives a stub gateway through a full canned session',
      () async {
    final stub = StubGateway();
    await stub.start(port: 0);
    addTearDown(stub.close);

    final client = GatewayClient();
    final events = <GatewayEvent>[];
    var audioBytes = 0;
    final evSub = client.events.listen(events.add);
    final auSub = client.audio.listen((f) => audioBytes += f.length);
    addTearDown(() async {
      await evSub.cancel();
      await auSub.cancel();
      await client.close();
    });

    await client.connect('127.0.0.1', stub.port, null);
    client.start();

    // Full canned session: 4 states, 2 transcripts, ~1 s of audio.
    await waitUntil(
        () =>
            events.whereType<StateEvent>().length >= 4 &&
            events.whereType<TranscriptEvent>().length >= 2 &&
            audioBytes > 0,
        timeout: const Duration(seconds: 20));

    // Meta frame parsed.
    final meta = events.map((e) => e.maybeMeta).firstWhere((m) => m != null)!;
    expect(meta.sampleRate, 24000);
    expect(meta.channels, 1);
    expect(meta.format, 'float32');
    expect(meta.frameMs, 13);
    expect(meta.session, isTrue);

    // State sequence listening → processing → speaking → listening.
    final states =
        events.whereType<StateEvent>().map((e) => e.state).toList();
    expect(states, [
      GatewayState.listening,
      GatewayState.processing,
      GatewayState.speaking,
      GatewayState.listening,
    ]);

    // Canned transcripts.
    final lines =
        events.whereType<TranscriptEvent>().map((e) => e.line).toList();
    expect(lines[0].role, 'user');
    expect(lines[0].text, 'hello from the simulator');
    expect(lines[1].role, 'assistant');
    expect(lines[1].text, 'Hi! JARVIS mobile is alive.');

    // Audio downlink carried 1248-byte float32 frames.
    expect(audioBytes, greaterThan(0));
    expect(audioBytes % 1248, 0);

    // Start and stop frames actually reached the server.
    expect(stub.received, contains('{"type":"start"}'));
    client.stop();
    await waitUntil(() => stub.received.contains('{"type":"stop"}'));

    // End frame after stop.
    await waitUntil(() => events.whereType<EndEvent>().isNotEmpty);
    expect(events.whereType<EndEvent>().first.reason, 'stopped');
  });

  test('GatewayClient talks to a real gateway (protocol-level)',
      () async {
    final env = Platform.environment;
    if (env['JARVIS_LIVE'] != '1') {
      markTestSkipped(
          'set JARVIS_LIVE=1 plus WS_HOST/WS_PORT to run against the real gateway');
      return;
    }
    final host = env['WS_HOST'];
    final port = int.tryParse(env['WS_PORT'] ?? '');
    if (host == null || port == null) {
      markTestSkipped('WS_HOST/WS_PORT not set');
      return;
    }
    // Definite non-null copies: markTestSkipped does not halt execution.
    final h = host;
    final p = port;

    final client = GatewayClient();
    final events = <GatewayEvent>[];
    var audioBytes = 0;
    final evSub = client.events.listen(events.add);
    final auSub = client.audio.listen((f) => audioBytes += f.length);
    addTearDown(() async {
      await evSub.cancel();
      await auSub.cancel();
      await client.close();
    });

    await client.connect(h, p, null);
    client.start();

    // The point is the client handles real-server frames: some parseable
    // event or audio frame must arrive. Meta/state/transcript expected;
    // error/end frames are acceptable (models may be absent on CI boxes).
    await waitUntil(() => events.isNotEmpty || audioBytes > 0,
        timeout: const Duration(seconds: 10));

    client.stop();
    await waitUntil(
        () =>
            events.whereType<EndEvent>().isNotEmpty ||
            events.whereType<ErrorEvent>().isNotEmpty ||
            !client.connected,
        timeout: const Duration(seconds: 10));

    expect(events.isNotEmpty || audioBytes > 0, isTrue,
        reason: 'expected at least one parseable frame from the real gateway');
  }, tags: ['live']);
}
