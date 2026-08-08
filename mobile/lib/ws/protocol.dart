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
