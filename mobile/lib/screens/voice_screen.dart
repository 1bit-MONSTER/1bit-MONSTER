import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../audio/audio_io.dart';
import '../state/session_controller.dart';
import '../ws/gateway_client.dart';
import '../ws/protocol.dart';

GatewayClient _defaultClientFactory() => GatewayClient();
MicSource _defaultMicFactory() => RecordMicSource();
ReplyPlayer _defaultPlayerFactory() => WavReplyPlayer();

/// Status shown in the lights row. `connecting` and `offline` are screen-level
/// (connect() in flight / failed), the rest mirror the gateway state machine.
enum _Status { idle, connecting, listening, processing, speaking, offline }

/// Voice session screen: big JARVIS toggle button, status lights, live
/// transcript.
///
/// Factories are injectable for tests; defaults build the real services.
/// Stream subscriptions are attached BEFORE `connect()` so the meta frame
/// (sent by the server right after the 101) is not missed.
class VoiceScreen extends StatefulWidget {
  const VoiceScreen({
    super.key,
    required this.host,
    required this.port,
    this.token,
    this.clientFactory = _defaultClientFactory,
    this.micFactory = _defaultMicFactory,
    this.playerFactory = _defaultPlayerFactory,
  });

  final String host;
  final int port;
  final String? token;
  final GatewayClient Function() clientFactory;
  final MicSource Function() micFactory;
  final ReplyPlayer Function() playerFactory;

  @override
  State<VoiceScreen> createState() => _VoiceScreenState();
}

class _VoiceScreenState extends State<VoiceScreen>
    with SingleTickerProviderStateMixin {
  late final GatewayClient _client = widget.clientFactory();
  late final MicSource _mic = widget.micFactory();
  late final ReplyPlayer _player = widget.playerFactory();
  final SessionController _controller = SessionController();
  final ReplyBuffer _replyBuffer = ReplyBuffer();
  final _scrollCtrl = ScrollController();
  late final AnimationController _pulse = AnimationController(
      vsync: this, duration: const Duration(milliseconds: 800))
    ..repeat(reverse: true);

  StreamSubscription<GatewayEvent>? _eventsSub;
  StreamSubscription<Uint8List>? _audioSub;
  StreamSubscription<Uint8List>? _micSub;

  bool _connecting = true;
  bool _offline = false;
  bool _sessionActive = false;

  static const Map<_Status, Color> _lightColors = {
    _Status.idle: Colors.grey,
    _Status.connecting: Colors.amber,
    _Status.listening: Colors.green,
    _Status.processing: Colors.blue,
    _Status.speaking: Colors.cyan,
    _Status.offline: Colors.red,
  };

  @override
  void initState() {
    super.initState();
    // Subscribe BEFORE connect(): the server sends the meta frame immediately
    // after the 101, and the client only buffers it until the first listener.
    _controller.addListener(_onControllerChanged);
    _eventsSub = _client.events.listen(_onEvent);
    _audioSub = _client.audio.listen(_replyBuffer.addFloat32);
    _micSub = _mic.pcm16Frames.listen((f) => _client.sendPcm16(f));
    _connect();
  }

  Future<void> _connect() async {
    _controller.setConnecting();
    try {
      await _client.connect(widget.host, widget.port, widget.token);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _connecting = false;
        _offline = true;
      });
      _controller.setOffline(
          'Cannot reach ${widget.host}:${widget.port} — ${e.toString()}');
      return;
    }
    if (!mounted) return;
    setState(() => _connecting = false);
  }

  void _onEvent(GatewayEvent e) {
    final prevState = _controller.state;
    final end = e.maybeEnd;
    if (end != null) {
      // Session over: stop capturing. Server keeps the socket open.
      _sessionActive = false;
      _mic.stop();
      _flushReplyAudio();
      if (end == 'error') {
        _controller.setOffline('Session ended with an error.');
      } else {
        _controller.onEvent(e);
      }
      if (mounted) setState(() {});
      return;
    }
    _controller.onEvent(e);
    if (e.maybeState != null &&
        prevState == GatewayState.speaking &&
        _controller.state != GatewayState.speaking) {
      _flushReplyAudio();
    }
  }

  /// Converts buffered float32 downlink frames to PCM16 and plays them.
  Future<void> _flushReplyAudio() async {
    final pcm = _replyBuffer.takePcm16();
    if (pcm != null) await _player.playPcm16(pcm, 24000);
  }

  Future<void> _onButtonTap() async {
    if (_sessionActive) {
      _client.stop();
      await _mic.stop();
      if (mounted) setState(() => _sessionActive = false);
    } else {
      try {
        await _mic.start();
      } catch (e) {
        _controller.setOffline('Microphone unavailable: $e');
        return;
      }
      _client.start();
      if (mounted) setState(() => _sessionActive = true);
    }
  }

  void _onControllerChanged() {
    if (mounted) setState(() {});
    if (!_scrollCtrl.hasClients) return;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollCtrl.hasClients) {
        _scrollCtrl.animateTo(_scrollCtrl.position.maxScrollExtent,
            duration: const Duration(milliseconds: 200),
            curve: Curves.easeOut);
      }
    });
  }

  _Status get _status {
    if (_offline) return _Status.offline;
    if (_connecting) return _Status.connecting;
    return switch (_controller.state) {
      GatewayState.idle => _Status.idle,
      GatewayState.listening => _Status.listening,
      GatewayState.processing => _Status.processing,
      GatewayState.speaking => _Status.speaking,
    };
  }

  @override
  void dispose() {
    _pulse.dispose();
    _scrollCtrl.dispose();
    _eventsSub?.cancel();
    _audioSub?.cancel();
    _micSub?.cancel();
    _controller.removeListener(_onControllerChanged);
    _mic.stop();
    _player.stop();
    _client.close();
    _controller.reset();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final status = _status;
    return Scaffold(
      appBar: AppBar(title: const Text('JARVIS')),
      body: SafeArea(
        child: Column(
          children: [
            _buildLights(theme, status),
            const SizedBox(height: 8),
            _buildTranscript(theme),
            const SizedBox(height: 16),
            _buildButton(theme, status),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }

  Widget _buildLights(ThemeData theme, _Status status) {
    return Column(
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            for (final s in _Status.values) ...[
              _StatusLight(
                key: Key('light-${s.name}'),
                status: s,
                active: s == status,
                pulse: s == status && s == _Status.connecting,
                animation: _pulse,
              ),
              if (s != _Status.values.last) const SizedBox(width: 10),
            ],
          ],
        ),
        const SizedBox(height: 6),
        Text(status.name.toUpperCase(),
            key: const Key('status-label'),
            style: theme.textTheme.labelLarge
                ?.copyWith(color: _lightColors[status], letterSpacing: 2)),
      ],
    );
  }

  Widget _buildTranscript(ThemeData theme) {
    return Expanded(
      child: Container(
        margin: const EdgeInsets.symmetric(horizontal: 16),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.35),
          borderRadius: BorderRadius.circular(12),
        ),
        child: _controller.transcript.isEmpty
            ? Center(
                child: Text('Tap JARVIS to start talking.',
                    style: theme.textTheme.bodyMedium
                        ?.copyWith(color: Colors.white54)))
            : ListView.builder(
                controller: _scrollCtrl,
                itemCount: _controller.transcript.length,
                itemBuilder: (context, i) {
                  final line = _controller.transcript[i];
                  final isUser = line.role == 'user';
                  return Padding(
                    padding: const EdgeInsets.symmetric(vertical: 4),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(isUser ? 'YOU' : 'JARVIS',
                            style: theme.textTheme.labelSmall?.copyWith(
                                color: isUser
                                    ? const Color(0xFF00E5C3)
                                    : const Color(0xFF4FC3F7),
                                fontWeight: FontWeight.bold)),
                        const SizedBox(width: 8),
                        Expanded(child: Text(line.text)),
                      ],
                    ),
                  );
                },
              ),
      ),
    );
  }

  Widget _buildButton(ThemeData theme, _Status status) {
    final busy = _connecting || _offline || !_controller.connected;
    final isActive = _sessionActive;
    return Column(
      children: [
        if (_controller.errorMessage != null) ...[
          Container(
            key: const Key('error-banner'),
            margin: const EdgeInsets.only(bottom: 12),
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            decoration: BoxDecoration(
              color: theme.colorScheme.errorContainer,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Row(
              children: [
                const Icon(Icons.error_outline, size: 18),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(_controller.errorMessage!,
                      style: TextStyle(color: theme.colorScheme.onErrorContainer)),
                ),
              ],
            ),
          ),
        ],
        InkWell(
          key: const Key('jarvis-button'),
          onTap: busy && !isActive ? null : _onButtonTap,
          customBorder: const CircleBorder(),
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 300),
            width: 150,
            height: 150,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              gradient: RadialGradient(colors: isActive
                  ? [const Color(0xFF0E3B34), const Color(0xFF00E5C3)]
                  : [const Color(0xFF123C63), const Color(0xFF4FC3F7)]),
              boxShadow: [
                BoxShadow(
                  color: (isActive
                          ? const Color(0xFF00E5C3)
                          : const Color(0xFF4FC3F7))
                      .withValues(alpha: 0.35),
                  blurRadius: 28,
                  spreadRadius: 2,
                ),
              ],
            ),
            child: Center(
              child: busy && !isActive
                  ? const SizedBox(
                      width: 36,
                      height: 36,
                      child: CircularProgressIndicator(
                          color: Colors.white, strokeWidth: 3))
                  : Text(isActive ? 'STOP' : 'JARVIS',
                      style: theme.textTheme.titleLarge?.copyWith(
                          color: Colors.white, fontWeight: FontWeight.bold,
                          letterSpacing: 2)),
            ),
          ),
        ),
      ],
    );
  }
}

class _StatusLight extends StatelessWidget {
  const _StatusLight({
    super.key,
    required this.status,
    required this.active,
    required this.pulse,
    required this.animation,
  });

  final _Status status;
  final bool active;
  final bool pulse;
  final Animation<double> animation;

  @override
  Widget build(BuildContext context) {
    final color = _VoiceScreenState._lightColors[status]!;
    return AnimatedBuilder(
      animation: animation,
      builder: (context, _) {
        final effectiveColor = pulse
            ? color.withValues(alpha: 0.35 + 0.65 * animation.value)
            : (active ? color : Colors.grey.withValues(alpha: 0.25));
        return Container(
          width: 16,
          height: 16,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: effectiveColor,
            boxShadow: active && !pulse
                ? [BoxShadow(color: color.withValues(alpha: 0.6), blurRadius: 8)]
                : null,
          ),
        );
      },
    );
  }
}
