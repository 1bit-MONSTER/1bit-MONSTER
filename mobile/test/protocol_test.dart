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
