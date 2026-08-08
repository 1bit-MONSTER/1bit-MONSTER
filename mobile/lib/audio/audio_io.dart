import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:audioplayers/audioplayers.dart';
import 'package:record/record.dart';

import '../ws/protocol.dart';

/// Thrown when the platform cannot record wav/PCM16 audio.
class MicUnavailableException implements Exception {
  final String message;
  MicUnavailableException(this.message);

  @override
  String toString() => 'MicUnavailableException: $message';
}

/// Removes a leading WAV header from [bytes]. Handles partial headers:
/// removes up to min([kWavHeaderSize], bytes.length) leading bytes.
Uint8List stripWavHeader(Uint8List bytes) =>
    bytes.sublist(math.min(kWavHeaderSize, bytes.length));

/// Splits PCM16 [pcm] into [frameBytes]-byte frames, dropping a trailing
/// partial frame.
List<Uint8List> chunkPcm16(Uint8List pcm, int frameBytes) {
  final chunks = <Uint8List>[];
  for (var i = 0; i + frameBytes <= pcm.length; i += frameBytes) {
    chunks.add(Uint8List.sublistView(pcm, i, i + frameBytes));
  }
  return chunks;
}

/// Strips a leading WAV header (handles the header split across chunks) and
/// emits complete [frameBytes]-byte PCM16 frames, carrying the partial-frame
/// remainder across chunks. Testable without the record plugin.
class WavPcm16Framer {
  WavPcm16Framer(this.frameBytes);

  final int frameBytes;
  final BytesBuilder _buf = BytesBuilder();
  int _headerLeft = kWavHeaderSize;

  /// Feeds [bytes] (header and/or PCM) and emits complete frames via [emit].
  void add(Uint8List bytes, void Function(Uint8List) emit) {
    var data = bytes;
    if (_headerLeft > 0) {
      final strip = math.min(_headerLeft, bytes.length);
      data = bytes.sublist(strip);
      _headerLeft -= strip;
    }
    if (data.isEmpty) return;
    _buf.add(data);
    final pcm = _buf.takeBytes();
    for (final frame in chunkPcm16(pcm, frameBytes)) {
      emit(frame);
    }
    final rest = pcm.length % frameBytes;
    if (rest > 0) _buf.add(Uint8List.sublistView(pcm, pcm.length - rest));
  }
}

/// Source of microphone PCM16 frames.
abstract class MicSource {
  Stream<Uint8List> get pcm16Frames;
  Future<void> start();
  Future<void> stop();
}

/// Records via the `record` plugin (wav, 16 kHz mono), strips the WAV header
/// and emits complete 640-byte PCM16 frames.
class RecordMicSource implements MicSource {
  static const _config = RecordConfig(
    encoder: AudioEncoder.wav,
    sampleRate: 16000,
    numChannels: 1,
  );

  final AudioRecorder _recorder = AudioRecorder();
  final _frames = StreamController<Uint8List>();
  final _framer = WavPcm16Framer(kUplinkFrameBytes);
  StreamSubscription<Uint8List>? _sub;
  bool _started = false;

  @override
  Stream<Uint8List> get pcm16Frames => _frames.stream;

  @override
  Future<void> start() async {
    if (_started) return;
    try {
      if (!await _recorder.isEncoderSupported(AudioEncoder.wav)) {
        throw MicUnavailableException(
            'wav/pcm16 recording is not supported on this platform.');
      }
      if (!await _recorder.hasPermission()) {
        throw MicUnavailableException('microphone permission denied.');
      }
      final stream = await _recorder.startStream(_config);
      _started = true;
      _sub = stream.listen(_onChunk);
    } on MicUnavailableException {
      rethrow;
    } catch (e) {
      throw MicUnavailableException('recording unavailable: $e');
    }
  }

  @override
  Future<void> stop() async {
    if (!_started) return;
    _started = false;
    await _sub?.cancel();
    _sub = null;
    await _recorder.stop();
  }

  void _onChunk(Uint8List bytes) => _framer.add(bytes, _frames.add);
}

/// Plays back PCM16 reply audio.
abstract class ReplyPlayer {
  Future<void> playPcm16(Uint8List pcm16, int sampleRate);
  Future<void> stop();
}

/// Thin audioplayers glue: converts PCM16 to WAV and plays it from bytes.
class WavReplyPlayer implements ReplyPlayer {
  final AudioPlayer _player = AudioPlayer();

  @override
  Future<void> playPcm16(Uint8List pcm16, int sampleRate) async {
    final wav = pcm16ToWav(pcm16, sampleRate, 1);
    await _player.stop();
    await _player.play(BytesSource(wav), volume: 1.0);
  }

  @override
  Future<void> stop() => _player.stop();

  void dispose() => _player.dispose();
}

/// Accumulates float32 downlink frames; converts to PCM16 on demand.
class ReplyBuffer {
  final BytesBuilder _buf = BytesBuilder();

  void addFloat32(Uint8List f32) => _buf.add(f32);

  /// Returns float32ToPcm16 of everything added so far and clears the buffer.
  /// Returns null if nothing was added.
  Uint8List? takePcm16() {
    final bytes = _buf.takeBytes();
    if (bytes.isEmpty) return null;
    return float32ToPcm16(bytes);
  }

  void reset() => _buf.clear();
}
