import 'package:flutter/foundation.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

/// Connect settings (gateway host/port/token) persisted in secure storage.
///
/// `load()` never throws: missing/empty values fall back to defaults
/// (port 8082), and storage failures are swallowed so a fresh install
/// or an unavailable platform simply starts with defaults.
class AppState extends ChangeNotifier {
  AppState({FlutterSecureStorage? storage})
      : _storage = storage ?? FlutterSecureStorage();

  static const String hostKey = 'jarvis_host';
  static const String portKey = 'jarvis_port';
  static const String tokenKey = 'jarvis_token';
  static const int defaultPort = 8082;

  final FlutterSecureStorage _storage;

  String? host;
  int port = defaultPort;
  String? token;

  Future<void> load() async {
    try {
      final h = await _storage.read(key: hostKey);
      final p = await _storage.read(key: portKey);
      final t = await _storage.read(key: tokenKey);
      host = (h != null && h.isNotEmpty) ? h : null;
      port = int.tryParse(p ?? '') ?? defaultPort;
      token = (t != null && t.isNotEmpty) ? t : null;
      notifyListeners();
    } catch (_) {
      // Nothing stored yet, or secure storage unavailable: keep defaults.
    }
  }

  Future<void> save() async {
    await _storage.write(key: hostKey, value: host);
    await _storage.write(key: portKey, value: '$port');
    await _storage.write(key: tokenKey, value: token);
  }
}
