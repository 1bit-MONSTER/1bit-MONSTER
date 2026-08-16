// tts.h — Text-to-speech via the `piper` binary (fork/exec, pure C++).
// Out-of-the-box voice: Piper runs anywhere, one ONNX voice file, no
// training. The trained RVQ codec voice (voice cloning) was deliberately
// gutted — see docs/jarvis.md.
#pragma once

#include <string>
#include <vector>

namespace jarvis {

class TTS {
public:
    /// piper_bin: path to piper binary (default "piper" on PATH).
    /// model_path: path to a piper .onnx voice.
    bool load(const std::string& piper_bin, const std::string& model_path);

    bool loaded() const { return loaded_; }

    /// Synthesize text; returns f32 mono PCM at out_sample_rate (22050).
    /// Empty vector on failure (missing piper, model, or synth error).
    std::vector<float> synth(const std::string& text, int& out_sample_rate);

private:
    std::string piper_bin_;
    std::string model_path_;
    bool loaded_ = false;
};

}  // namespace jarvis
