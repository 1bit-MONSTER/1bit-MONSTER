import 'package:flutter/material.dart';

import '../state/app_state.dart';
import 'voice_screen.dart';

/// Server host/port/token entry. Saves settings to secure storage, then
/// pushes the voice screen.
class ConnectScreen extends StatefulWidget {
  const ConnectScreen({super.key, this.settings});

  /// Injected for tests; a real [AppState] is created when omitted.
  final AppState? settings;

  @override
  State<ConnectScreen> createState() => _ConnectScreenState();
}

class _ConnectScreenState extends State<ConnectScreen> {
  late final AppState _settings = widget.settings ?? AppState();
  final _hostCtrl = TextEditingController();
  final _portCtrl = TextEditingController();
  final _tokenCtrl = TextEditingController();
  String? _error;
  bool _saving = false;

  @override
  void initState() {
    super.initState();
    _settings.load().then((_) {
      if (!mounted) return;
      setState(() {
        _hostCtrl.text = _settings.host ?? '';
        _portCtrl.text = '${_settings.port}';
        _tokenCtrl.text = _settings.token ?? '';
      });
    });
  }

  @override
  void dispose() {
    _hostCtrl.dispose();
    _portCtrl.dispose();
    _tokenCtrl.dispose();
    super.dispose();
  }

  Future<void> _connect() async {
    final host = _hostCtrl.text.trim();
    final port = int.tryParse(_portCtrl.text.trim());
    if (host.isEmpty) {
      setState(() => _error = 'Enter the gateway host (e.g. 192.168.1.10).');
      return;
    }
    if (port == null || port < 1 || port > 65535) {
      setState(() => _error = 'Port must be a number between 1 and 65535.');
      return;
    }
    final token = _tokenCtrl.text.trim();
    setState(() {
      _error = null;
      _saving = true;
    });
    _settings
      ..host = host
      ..port = port
      ..token = token.isEmpty ? null : token;
    await _settings.save();
    if (!mounted) return;
    setState(() => _saving = false);
    Navigator.of(context).push(MaterialPageRoute<void>(
      builder: (_) => VoiceScreen(host: host, port: port, token: _settings.token),
    ));
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('JARVIS')),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 420),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                const Icon(Icons.smart_toy, size: 72, color: Color(0xFF00E5C3)),
                const SizedBox(height: 8),
                Text('Voice gateway', textAlign: TextAlign.center,
                    style: theme.textTheme.titleMedium),
                const SizedBox(height: 24),
                TextField(
                  key: const Key('host-field'),
                  controller: _hostCtrl,
                  decoration: const InputDecoration(
                    labelText: 'Host',
                    hintText: '192.168.1.10',
                    prefixIcon: Icon(Icons.dns),
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 16),
                TextField(
                  key: const Key('port-field'),
                  controller: _portCtrl,
                  keyboardType: TextInputType.number,
                  decoration: const InputDecoration(
                    labelText: 'Port',
                    prefixIcon: Icon(Icons.router),
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 16),
                TextField(
                  key: const Key('token-field'),
                  controller: _tokenCtrl,
                  obscureText: true,
                  decoration: const InputDecoration(
                    labelText: 'Token (optional)',
                    prefixIcon: Icon(Icons.key),
                    border: OutlineInputBorder(),
                  ),
                ),
                if (_error != null) ...[
                  const SizedBox(height: 16),
                  Text(_error!, key: const Key('connect-error'),
                      style: TextStyle(color: theme.colorScheme.error)),
                ],
                const SizedBox(height: 24),
                FilledButton.icon(
                  key: const Key('connect-button'),
                  onPressed: _saving ? null : _connect,
                  icon: _saving
                      ? const SizedBox(
                          width: 18, height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2))
                      : const Icon(Icons.wifi_tethering),
                  label: const Text('Connect'),
                  style: FilledButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 16)),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
