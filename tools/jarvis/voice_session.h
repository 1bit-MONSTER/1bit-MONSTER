#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace jarvis {

enum class SessionState { Idle, Listening, Processing, Speaking };

using StateCallback = std::function<void(SessionState)>;
using UtteranceCallback = std::function<void(const std::vector<int16_t>& pcm16)>;
using ErrorCallback = std::function<void(const std::string& msg)>;

class VoiceSession {
public:
    VoiceSession();
    ~VoiceSession();

    void start();                                  // Idle -> Listening
    void stop();                                   // any -> Idle, drop buffers
    void feed(const int16_t* pcm16, size_t n_samples);  // 16 kHz mono
    void tick(int ms_elapsed);                     // speaking -> listening timeout
    void set_speaking(bool speaking);              // enter/leave Speaking
    SessionState state() const;

    void set_callbacks(StateCallback on_state, UtteranceCallback on_utterance, ErrorCallback on_error);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace jarvis
