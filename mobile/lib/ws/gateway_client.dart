import 'dart:async';
import 'dart:typed_data';

import 'package:web_socket_channel/io.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

import 'protocol.dart';

/// Thrown when the gateway connection cannot be established
/// (socket error, non-101 response, or 403).
class GatewayException implements Exception {
  final String message;
  GatewayException(this.message);

  @override
  String toString() => 'GatewayException: $message';
}

/// WebSocket client for the JARVIS voice gateway.
///
/// - `events` carries parsed server text frames (meta/state/transcript/end/error).
/// - `audio` carries raw binary downlink frames (float32 @ 24 kHz, 1248 bytes)
///   as a SEPARATE stream — binary frames never appear on `events`.
///
/// The server sends the meta frame immediately after the 101; this client
/// does not send a hello — the session is started with `start()`.
class GatewayClient {
  static const _path = '/v1/voice/session';
  static const _connectTimeout = Duration(seconds: 5);

  WebSocketChannel? _channel;
  StreamSubscription<dynamic>? _sub;
  final _events = StreamController<GatewayEvent>.broadcast();
  final _audio = StreamController<Uint8List>.broadcast();

  /// Parsed text frames from the server (malformed/unknown frames skipped).
  Stream<GatewayEvent> get events => _events.stream;

  /// Raw binary downlink frames (float32 @ 24 kHz, 1248 bytes each).
  Stream<Uint8List> get audio => _audio.stream;

  bool get connected => _channel != null;

  /// Connects to `ws://host:port/v1/voice/session`, optionally with a bearer
  /// token. Throws [GatewayException] on socket error / non-101 / 403.
  ///
  /// Uses [IOWebSocketChannel] because web_socket_channel 3.0.x only exposes
  /// `headers` on the dart:io implementation.
  Future<void> connect(String host, int port, String? token) async {
    final uri = Uri.parse('ws://$host:$port$_path');
    try {
      final channel = IOWebSocketChannel.connect(
        uri,
        headers: token == null ? null : {'Authorization': 'Bearer $token'},
        connectTimeout: _connectTimeout,
      );
      await channel.ready;
      _channel = channel;
      _sub = channel.stream.listen(
        _onData,
        onError: (Object e) {
          _channel = null;
          _events.add(ErrorEvent('gateway connection lost: $e'));
        },
        onDone: () => _channel = null,
      );
    } on GatewayException {
      rethrow;
    } catch (e) {
      throw GatewayException('connect to $uri failed: $e');
    }
  }

  void start() => _send(ControlFrame.start);

  void stop() => _send(ControlFrame.stop);

  void cancel() => _send(ControlFrame.cancel);

  /// Sends one uplink PCM16 frame (640 bytes @ 16 kHz).
  Future<void> sendPcm16(Uint8List frame) async {
    _channel?.sink.add(frame);
  }

  /// Closes the connection.
  Future<void> close() async {
    await _sub?.cancel();
    _sub = null;
    final ch = _channel;
    _channel = null;
    await ch?.sink.close();
  }

  void _send(String frame) => _channel?.sink.add(frame);

  void _onData(dynamic data) {
    if (data is String) {
      final event = parseGatewayFrame(data);
      if (event != null) _events.add(event);
    } else if (data is List<int>) {
      _audio.add(Uint8List.fromList(data));
    }
  }
}
