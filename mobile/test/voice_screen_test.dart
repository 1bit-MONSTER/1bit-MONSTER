import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:jarvis_mobile/audio/audio_io.dart';
import 'package:jarvis_mobile/screens/voice_screen.dart';
import 'package:jarvis_mobile/ws/gateway_client.dart';
import 'package:jarvis_mobile/ws/protocol.dart';

/// Test double for [GatewayClient] — Dart's implicit interface only requires
/// the public members, so `implements` works without touching the real class.
class FakeGatewayClient implements GatewayClient {
  final _events = StreamController<GatewayEvent>.broadcast();
  final _audio = StreamController<Uint8List>.broadcast();

  final List<String> sent = [];
  String? lastHost;
  int? lastPort;
  String? lastToken;
  bool closed = false;
  int connectCount = 0;

  /// When set, connect() waits on it — lets tests observe the connecting
  /// state before the connection completes.
  Completer<void>? connectGate;

  @override
  Stream<GatewayEvent> get events => _events.stream;

  @override
  Stream<Uint8List> get audio => _audio.stream;

  @override
  bool get connected => true;

  @override
  Future<void> connect(String host, int port, String? token) async {
    connectCount++;
    lastHost = host;
    lastPort = port;
    lastToken = token;
    final gate = connectGate;
    if (gate != null) await gate.future;
  }

  @override
  void start() => sent.add('start');

  @override
  void stop() => sent.add('stop');

  @override
  void cancel() => sent.add('cancel');

  @override
  Future<void> sendPcm16(Uint8List frame) async {}

  @override
  Future<void> close() async {
    closed = true;
  }

  void emit(GatewayEvent e) => _events.add(e);

  void emitMeta() => emit(parseGatewayFrame(
      '{"type":"meta","session":true,"sample_rate":16000,"channels":1,'
      '"format":"pcm16","frame_ms":20}')!);

  void emitState(GatewayState s) => emit(StateEvent(s));

  void emitTranscript(String role, String text) =>
      emit(TranscriptEvent(TranscriptLine(role, text)));

  void emitError(String message) => emit(ErrorEvent(message));

  void emitAudio(Uint8List bytes) => _audio.add(bytes);
}

class FakeMic implements MicSource {
  final _frames = StreamController<Uint8List>.broadcast();
  int startCalls = 0;
  int stopCalls = 0;
  bool failStart = false;

  @override
  Stream<Uint8List> get pcm16Frames => _frames.stream;

  @override
  Future<void> start() async {
    startCalls++;
    if (failStart) throw StateError('mic unavailable');
  }

  @override
  Future<void> stop() async {
    stopCalls++;
  }
}

class FakePlayer implements ReplyPlayer {
  final List<(Uint8List, int)> played = [];

  @override
  Future<void> playPcm16(Uint8List pcm16, int sampleRate) async {
    played.add((pcm16, sampleRate));
  }

  @override
  Future<void> stop() async {}

  @override
  Future<void> dispose() async {}
}

Future<void> pumpVoice(WidgetTester tester, FakeGatewayClient client,
    {FakeMic? mic, FakePlayer? player}) async {
  await tester.pumpWidget(MaterialApp(
    home: VoiceScreen(
      host: '192.168.1.10',
      port: 8082,
      token: 'sekret',
      clientFactory: () => client,
      micFactory: () => mic ?? FakeMic(),
      playerFactory: () => player ?? FakePlayer(),
    ),
  ));
  await tester.pump(); // flush the initState connect() future
}

Color? lightColor(WidgetTester tester, String state) {
  final container = tester.widget<Container>(find.descendant(
      of: find.byKey(Key('light-$state')),
      matching: find.byType(Container)));
  // Ignore alpha: the connecting light pulses its opacity.
  return (container.decoration as BoxDecoration?)?.color?.withValues(alpha: 1.0);
}

void main() {
  testWidgets('meta + listening -> green light, button enabled',
      (tester) async {
    final gate = Completer<void>();
    final client = FakeGatewayClient()..connectGate = gate;
    await tester.pumpWidget(MaterialApp(
      home: VoiceScreen(
        host: '192.168.1.10',
        port: 8082,
        token: 'sekret',
        clientFactory: () => client,
      ),
    ));

    // While connecting: button disabled, amber light on.
    expect(
        tester.widget<InkWell>(find.byKey(const Key('jarvis-button'))).onTap,
        isNull);
    expect(lightColor(tester, 'connecting')!.value, Colors.amber.value);

    gate.complete();
    await tester.pump();
    expect(client.lastHost, '192.168.1.10');
    expect(client.lastPort, 8082);
    expect(client.lastToken, 'sekret');

    client.emitMeta();
    client.emitState(GatewayState.listening);
    await tester.pump();

    expect(lightColor(tester, 'listening')!.value, Colors.green.value);
    expect(
        tester.widget<InkWell>(find.byKey(const Key('jarvis-button'))).onTap,
        isNotNull);
    expect(find.text('LISTENING'), findsOneWidget);
  });

  testWidgets('transcript events appear as lines', (tester) async {
    final client = FakeGatewayClient();
    await pumpVoice(tester, client);
    client.emitMeta();
    await tester.pump();

    client.emitTranscript('user', 'hello there');
    client.emitTranscript('assistant', 'hi!');
    await tester.pump();

    expect(find.text('hello there'), findsOneWidget);
    expect(find.text('hi!'), findsOneWidget);
    expect(find.text('YOU'), findsOneWidget);
    expect(find.text('JARVIS'), findsWidgets);
  });

  testWidgets('tap sends start, second tap sends stop', (tester) async {
    final client = FakeGatewayClient();
    final mic = FakeMic();
    await pumpVoice(tester, client, mic: mic);
    client.emitMeta();
    await tester.pump();

    await tester.tap(find.byKey(const Key('jarvis-button')));
    await tester.pump();
    expect(client.sent, ['start']);
    expect(mic.startCalls, 1);
    expect(find.text('STOP'), findsOneWidget);

    await tester.tap(find.byKey(const Key('jarvis-button')));
    await tester.pump();
    expect(client.sent, ['start', 'stop']);
    expect(mic.stopCalls, 1);
  });

  testWidgets('audio during speaking is buffered and flushed to the player',
      (tester) async {
    final client = FakeGatewayClient();
    final player = FakePlayer();
    await pumpVoice(tester, client, player: player);
    client.emitMeta();
    await tester.pump();

    await tester.tap(find.byKey(const Key('jarvis-button')));
    await tester.pump();

    client.emitState(GatewayState.speaking);
    await tester.pump();
    final f32 = Float32List.fromList([0.25, -0.5, 0.0, 1.0]);
    client.emitAudio(Uint8List.view(f32.buffer));
    await tester.pump();

    // Leaving speaking flushes the buffer as PCM16 @ 24 kHz.
    client.emitState(GatewayState.listening);
    await tester.pump();
    await tester.pump();

    expect(player.played, hasLength(1));
    expect(player.played.single.$2, 24000);
    expect(player.played.single.$1.lengthInBytes, f32.lengthInBytes ~/ 2);
  });

  testWidgets('disconnect stops mic and shows Reconnect button',
      (tester) async {
    final client = FakeGatewayClient();
    final mic = FakeMic();
    await pumpVoice(tester, client, mic: mic);
    client.emitMeta();
    await tester.pump();

    await tester.tap(find.byKey(const Key('jarvis-button')));
    await tester.pump();
    expect(mic.startCalls, 1);
    expect(find.text('STOP'), findsOneWidget);

    client.emitError('gateway connection lost');
    await tester.pump();

    expect(mic.stopCalls, 1); // mic stopped
    expect(find.text('Reconnect'), findsOneWidget); // button relabeled
    expect(find.text('OFFLINE'), findsOneWidget); // status label
  });

  testWidgets('Reconnect reconnects and starts a new session', (tester) async {
    final client = FakeGatewayClient();
    final mic = FakeMic();
    await pumpVoice(tester, client, mic: mic);
    client.emitMeta();
    await tester.pump();

    await tester.tap(find.byKey(const Key('jarvis-button')));
    await tester.pump();
    client.emitTranscript('user', 'hello there');
    await tester.pump();
    client.emitError('gateway connection lost');
    await tester.pump();
    expect(client.connectCount, 1);

    await tester.tap(find.text('Reconnect'));
    await tester.pump();

    expect(client.connectCount, 2); // re-connected
    expect(client.sent.last, 'start'); // fresh session started
    expect(mic.startCalls, 2);
    expect(find.text('hello there'), findsOneWidget); // transcript survives
  });

  testWidgets('reconnect mic failure stays on recoverable Reconnect state',
      (tester) async {
    final client = FakeGatewayClient();
    final mic = FakeMic()..failStart = true;
    await pumpVoice(tester, client, mic: mic);
    client.emitMeta();
    await tester.pump();

    await tester.tap(find.byKey(const Key('jarvis-button')));
    await tester.pump();
    client.emitError('gateway connection lost');
    await tester.pump();

    await tester.tap(find.text('Reconnect'));
    await tester.pump();

    // Mic failed after a successful socket reconnect: the button must stay
    // an enabled Reconnect (offline), not fall into the dead busy-spinner.
    expect(mic.startCalls, 2);
    expect(find.text('Reconnect'), findsOneWidget);
    expect(find.text('OFFLINE'), findsOneWidget);
    expect(
        tester.widget<InkWell>(find.byKey(const Key('jarvis-button'))).onTap,
        isNotNull);
  });

  testWidgets('error event shows the banner', (tester) async {
    final client = FakeGatewayClient();
    await pumpVoice(tester, client);
    client.emitMeta();
    await tester.pump();
    expect(find.byKey(const Key('error-banner')), findsNothing);

    client.emitError('STT unavailable');
    await tester.pump();

    expect(find.byKey(const Key('error-banner')), findsOneWidget);
    expect(find.text('STT unavailable'), findsOneWidget);
  });
}
