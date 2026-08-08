import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:jarvis_mobile/audio/audio_io.dart';
import 'package:jarvis_mobile/ws/protocol.dart';

/// 1248-byte float32 frame of constant value [v].
Uint8List f32Frame(double v) {
  final b = ByteData(1248);
  for (var i = 0; i < 1248 ~/ 4; i++) {
    b.setFloat32(i * 4, v, Endian.little);
  }
  return b.buffer.asUint8List();
}

void main() {
  group('stripWavHeader', () {
    test('removes the 44-byte header', () {
      final wav = Uint8List(kWavHeaderSize + 1280);
      final pcm = stripWavHeader(wav);
      expect(pcm.length, 1280);
    });

    test('partial header: removes up to min(44, len)', () {
      expect(stripWavHeader(Uint8List(10)), isEmpty);
      expect(stripWavHeader(Uint8List(0)), isEmpty);
      expect(stripWavHeader(Uint8List(100)).length, 56);
    });
  });

  group('WavPcm16Framer', () {
    test('header split across chunks: zero PCM loss, byte-identical frames',
        () {
      final header = Uint8List.fromList(List.generate(44, (i) => 0xAA));
      final pcm = Uint8List.fromList(List.generate(640 * 3, (i) => i % 256));
      final frames = <Uint8List>[];
      final framer = WavPcm16Framer(kUplinkFrameBytes);

      // 20 header bytes, then 24 header + 76 PCM bytes, then the rest.
      framer.add(Uint8List.sublistView(header, 0, 20), frames.add);
      framer.add(
          Uint8List.fromList([
            ...header.sublist(20),
            ...pcm.sublist(0, 76),
          ]),
          frames.add);
      framer.add(Uint8List.sublistView(pcm, 76), frames.add);

      expect(frames, hasLength(3));
      expect(frames[0], orderedEquals(pcm.sublist(0, 640)));
      expect(frames[1], orderedEquals(pcm.sublist(640, 1280)));
      expect(frames[2], orderedEquals(pcm.sublist(1280, 1920)));
    });

    test('header entirely inside one chunk, then a partial frame remainder',
        () {
      final frames = <Uint8List>[];
      final framer = WavPcm16Framer(kUplinkFrameBytes);
      framer.add(Uint8List(kWavHeaderSize + 640 + 10), frames.add);
      framer.add(Uint8List(630), frames.add); // 630 + 10 carried = 640
      expect(frames, hasLength(2));
      expect(frames[0].length, 640);
      expect(frames[1].length, 640);
    });
  });

  group('chunkPcm16', () {
    test('44-byte header + 1280 bytes -> two 640-byte frames', () {
      final wav = Uint8List(kWavHeaderSize + 1280);
      final chunks = chunkPcm16(stripWavHeader(wav), kUplinkFrameBytes);
      expect(chunks, hasLength(2));
      expect(chunks[0].length, 640);
      expect(chunks[1].length, 640);
    });

    test('drops trailing partial frame', () {
      final chunks = chunkPcm16(Uint8List(1300), kUplinkFrameBytes);
      expect(chunks, hasLength(2));
    });

    test('frame bytes are preserved in order', () {
      final pcm = Uint8List.fromList(List.generate(1280, (i) => i % 256));
      final chunks = chunkPcm16(pcm, kUplinkFrameBytes);
      expect(chunks[0], orderedEquals(pcm.sublist(0, 640)));
      expect(chunks[1], orderedEquals(pcm.sublist(640, 1280)));
    });
  });

  group('ReplyBuffer', () {
    test('accumulates float32 and converts to PCM16', () {
      final buf = ReplyBuffer();
      buf.addFloat32(f32Frame(0.25));
      buf.addFloat32(f32Frame(0.25));

      final pcm = buf.takePcm16();
      expect(pcm, isNotNull);
      // float32ToPcm16 halves bytes: 2 * 1248 float32 -> 1248 int16.
      expect(pcm!.length, 1248);
      // 0.25 * 32767 = 8191.75 -> rounds to 8192.
      final view = ByteData.sublistView(pcm);
      expect(view.getInt16(0, Endian.little), 8192);
      expect(view.getInt16(1246, Endian.little), 8192);

      // Buffer cleared after take.
      expect(buf.takePcm16(), isNull);
    });

    test('takePcm16 on empty buffer returns null', () {
      expect(ReplyBuffer().takePcm16(), isNull);
    });

    test('reset clears accumulated audio', () {
      final buf = ReplyBuffer();
      buf.addFloat32(f32Frame(0.25));
      buf.reset();
      expect(buf.takePcm16(), isNull);
    });
  });
}
