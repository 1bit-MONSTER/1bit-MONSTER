// stt.cpp — libwhisper-backed STT. See stt.h.

#include "jarvis/stt.h"

#include <algorithm>
#include <cstring>

#include "whisper.h"

namespace jarvis {

struct STT::WhisperModel {
    ::WhisperModel m;
};

bool STT::load(const std::string& gguf_path) {
    std::lock_guard<std::mutex> lock(mtx_);
    delete model_;
    model_ = new WhisperModel;
    if (!model_->m.load_from_gguf(gguf_path)) {
        delete model_;
        model_ = nullptr;
        loaded_ = false;
        return false;
    }
    loaded_ = true;
    return true;
}

std::string STT::transcribe(const float* pcm, int n_samples) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!loaded_ || !model_ || n_samples <= 0) return "";
    std::string text = whisper_transcribe(model_->m, pcm, n_samples);

    // Trim whitespace; collapse nothing else — keep the model's casing.
    size_t b = text.find_first_not_of(" \t\r\n");
    size_t e = text.find_last_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    return text.substr(b, e - b + 1);
}

}  // namespace jarvis
