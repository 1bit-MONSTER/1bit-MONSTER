// audio_stream.h — Real-time streaming audio codec decoder + WebSocket server.
//
// StreamingDecoder generates audio frame-by-frame (13ms frames at 24kHz =
// 312 samples/frame) from codec tokens.  Uses ONNX Runtime when available
// (USE_ONNXRUNTIME), falls back to frame-chunking the full decode.
//
// WebSocketServer runs a minimal WebSocket server on a separate port (so it
// doesn't conflict with httplib which lacks built-in WebSocket support in
// v0.18.1).  jarvis_server registers /v1/audio/stream as a redirect / info
// endpoint pointing to this server.

#pragma once
#include "ws_proto.h"
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace jarvis {

/// Callback type for streaming audio chunks
using AudioChunkCallback = std::function<void(const float* samples, int num_samples, int sample_rate)>;

/// Streaming audio codec decoder.
/// Generates audio frame-by-frame (13ms frames at 24kHz = 312 samples).
struct StreamingDecoderImpl;

class StreamingDecoder {
public:
    StreamingDecoder();
    ~StreamingDecoder();

    /// Load decoder model
    bool load(const std::string& onnx_model_path);

    /// Start decoding codec tokens, calling callback for each audio frame.
    /// @param tokens: codec token indices [n_codebooks, seq_len]
    /// @param n_tokens: number of codec frames
    /// @param speaker_emb: [512] float speaker embedding
    /// @param callback: called with each audio chunk as it's decoded
    void decode_streaming(
        const int32_t* tokens, int n_tokens, const float* speaker_emb,
        AudioChunkCallback callback
    );

    bool is_loaded() const;
    int sample_rate() const { return 24000; }

private:
    std::unique_ptr<StreamingDecoderImpl> impl_;
};

// ── Session connection handle ─────────────────────────────────────────
//
// Handle to an active /v1/voice/session WebSocket connection.  The server
// thread owns the read loop (control frames + PCM16 uplink) and the
// VoiceSession state machine; jarvis_server (Task 3 wiring) pushes TTS
// audio/text downlink through send_audio/send_text and receives user
// utterances via set_utterance_callback.
//
// Lifetime: WebSocketServer::session_conn() returns non-null from the
// moment a session client connects until the *next* session connects or
// the server stops.  Check active() before sending — a dead handle's
// send_* return false and are safe to call.
class WsSessionConn {
public:
    WsSessionConn() = default;
    ~WsSessionConn();
    WsSessionConn(const WsSessionConn&) = delete;
    WsSessionConn& operator=(const WsSessionConn&) = delete;

    /// Send one float32 PCM chunk (24 kHz mono) to the session client.
    /// Returns false if the connection is gone or the client sent cancel.
    bool send_audio(const float* samples, int n_samples);

    /// Send a text (JSON) frame to the session client.
    bool send_text(const std::string& json);

    /// The VoiceSession driving this connection.  feed/tick/start/stop run
    /// on the server thread; set_speaking may be called from jarvis_server.
    VoiceSession& session() { return *session_; }

    /// True while a session client is connected.
    bool active() const { return fd_ >= 0; }

    /// True after the client sent {"type":"cancel"} until the next start.
    bool cancelled() const { return cancelled_; }

    /// Hook for jarvis_server wiring: receives VAD-detected user
    /// utterances (16 kHz mono PCM16).  Set once, from jarvis_server.
    void set_utterance_callback(UtteranceCallback cb) { utterance_cb_ = std::move(cb); }

    // Implementation detail; manipulated by the server thread.
    int fd_ = -1;
    std::mutex send_mu_;                       // serializes downlink writes
    std::vector<uint8_t> pcm_buf_;             // partial 20ms uplink frames
    std::atomic<bool> cancelled_{false};
    std::unique_ptr<VoiceSession> session_;
    UtteranceCallback utterance_cb_;
};

// ── WebSocket audio stream server ────────────────────────────────────
//
// Implemented using raw POSIX sockets because httplib v0.18.1 does not
// include built-in WebSocket support.  Runs on a configurable port
// (default 8082).  jarvis_server starts it on a background thread.
//
// Protocol (RFC 6455):
//   1. Client sends HTTP GET upgrade with Sec-WebSocket-Key
//   2. Server replies 101 Switching Protocols
//   3. Server sends text frame: {"type":"meta","sample_rate":24000,...}
//   4. Server sends binary frames: raw float32 PCM (~312 samples = 13ms)
//   5. Server sends text frame: {"type":"end","reason":"done"}
//   6. Client can send text frame: {"type":"cancel"} to abort

// Auth hook: return true to accept the upgrade. Called with the raw
// Authorization header value (may be empty). Null = accept everything.
using WSAuthCheck = std::function<bool(const std::string& auth_header)>;

struct WebSocketServerImpl;

class WebSocketServer {
public:
    WebSocketServer();
    ~WebSocketServer();

    /// Start the WebSocket server on a background thread.
    /// @param port  listening port (0 = auto-assign)
    /// @param codec_tts_ptr  pointer to a jarvis::CodecTts instance (void* to avoid include)
    /// @return the actual port number, or -1 on failure
    int start(int port, void* codec_tts_ptr);

    /// Start with an auth gate: every upgrade (both request paths) must
    /// pass auth_check or is rejected with 403 (no 101).  Null = open.
    int start(int port, void* codec_tts_ptr, WSAuthCheck auth_check);

    /// Handle for the active /v1/voice/session connection, or nullptr if
    /// no session client has connected yet (see WsSessionConn lifetime).
    WsSessionConn* session_conn() const { return session_conn_.get(); }

    /// Stop the server and join the background thread.
    void stop();

    /// Check if the server is running.
    bool is_running() const { return running_; }

    /// Get the port the server is listening on.
    int port() const { return port_; }

private:
    std::atomic<bool> running_{false};
    std::atomic<int> port_{-1};
    int listen_fd_{-1};
    std::unique_ptr<std::thread> server_thread_;
    WSAuthCheck auth_check_;
    std::unique_ptr<WsSessionConn> session_conn_;  // latest session conn (kept after close)
};

} // namespace jarvis
