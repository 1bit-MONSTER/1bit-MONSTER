import 'package:flutter/foundation.dart';
import '../ws/protocol.dart';

class SessionController extends ChangeNotifier {
  GatewayState _state = GatewayState.idle;
  final List<TranscriptLine> _transcript = [];
  String? _errorMessage;
  bool _connected = false;

  GatewayState get state => _state;
  List<TranscriptLine> get transcript => List.unmodifiable(_transcript);
  String? get errorMessage => _errorMessage;
  bool get connected => _connected;

  static const int _maxTranscript = 100;

  void setConnecting() {
    _connected = false;
    _state = GatewayState.idle;
    _errorMessage = null;
    notifyListeners();
  }

  void setOffline(String reason) {
    _connected = false;
    _state = GatewayState.idle;
    _errorMessage = reason;
    notifyListeners();
  }

  void onEvent(GatewayEvent e) {
    final s = e.maybeState;
    if (s != null) _state = s;
    final t = e.maybeTranscript;
    if (t != null) {
      _transcript.add(t);
      if (_transcript.length > _maxTranscript) {
        _transcript.removeRange(0, _transcript.length - _maxTranscript);
      }
    }
    if (e.maybeMeta != null) _connected = true;
    final err = e.maybeError;
    if (err != null) _errorMessage = err;
    notifyListeners();
  }

  void reset() {
    _transcript.clear();
    _errorMessage = null;
    _state = GatewayState.idle;
    _connected = false;
    notifyListeners();
  }
}
