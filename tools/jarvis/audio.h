// audio.h — Mic capture + speaker playback via arecord/aplay child processes.
// Pure C++ (fork/exec + pipes, the same idiom as tts.cpp and backend_npu.cpp).
// ponytail: arecord/aplay subprocesses instead of ALSA lib — ALSA dev headers
// aren't installed here; swap to snd_pcm when latency tuning matters.
#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace jarvis {

/// Captures mono PCM from the mic via `arecord`, delivered as float32.
class Capture {
public:
    using SamplesCb = std::function<void(const float* samples, int n)>;

    /// Spawn arecord (s16le mono @ sample_rate). cb is called from a
    /// dedicated reader thread with f32 chunks. Returns false if spawn fails.
    bool start(int sample_rate_hz, const std::string& device, SamplesCb cb);
    void stop();  // idempotent; kills child
    ~Capture() { stop(); }

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

/// Plays float32 PCM through `aplay` (s16le mono @ sample_rate).
class Playback {
public:
    bool start(int sample_rate_hz);  // spawn aplay; false if spawn fails
    void write(const float* samples, int n);  // f32 -> s16 -> pipe
    void stop();  // flush + close stdin, reap child
    ~Playback() { stop(); }

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace jarvis
