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
