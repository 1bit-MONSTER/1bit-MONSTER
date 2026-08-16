// stt.h — Speech-to-text via the engine's own libwhisper.
// GPU-accelerated when a HIP device is present (src/whisper_hip.hip),
// scalar CPU otherwise. WHISPER_GPU=0 forces the scalar path.
#pragma once

#include <mutex>
#include <string>
#include <vector>

namespace jarvis {

class STT {
public:
    /// Load a whisper GGUF model (llama.cpp whisper format).
    bool load(const std::string& gguf_path);

    bool loaded() const { return loaded_; }

    /// Transcribe 16 kHz mono f32 PCM. Returns trimmed text ("" on failure).
    /// Serialized internally — whisper is not thread-safe.
    std::string transcribe(const float* pcm, int n_samples);

private:
    struct WhisperModel;  // fwd (whisper.h is heavy)
    WhisperModel* model_ = nullptr;
    bool loaded_ = false;
    std::mutex mtx_;
};

}  // namespace jarvis
