// whisper_demo.cpp — Whisper speech-to-text demo
// Build: cmake --build . --target whisper_demo -j8
// Run:   ./build/whisper_demo <whisper.gguf> <audio.wav>
//
// If no audio file provided, generates a synthetic test tone.

#include "whisper.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>

// Generate a synthetic test tone (440Hz sine wave, 3 seconds)
static std::vector<float> generate_test_tone(int sample_rate = 16000, int duration_sec = 3, float freq = 440.0f) {
    int n = sample_rate * duration_sec;
    std::vector<float> pcm(n);
    for (int i = 0; i < n; i++) {
        float t = (float)i / sample_rate;
        pcm[i] = 0.3f * sinf(2.0f * M_PI * freq * t);  // 440Hz A4 tone
        // Add a bit of harmonic
        pcm[i] += 0.15f * sinf(2.0f * M_PI * freq * 2 * t);
        pcm[i] += 0.1f * sinf(2.0f * M_PI * freq * 3 * t);
    }
    return pcm;
}

int main(int argc, char** argv) {
    fprintf(stderr, "╔══════════════════════════════════════════╗\n");
    fprintf(stderr, "║        Whisper Speech-to-Text Demo       ║\n");
    fprintf(stderr, "╚══════════════════════════════════════════╝\n\n");
    
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <whisper.gguf> [audio.wav]\n", argv[0]);
        fprintf(stderr, "\n  If no audio file given, generates a synthetic 440Hz test tone.\n");
        return 1;
    }
    
    std::string model_path = argv[1];
    
    // 1. Load model
    fprintf(stderr, "1. Loading Whisper model: %s\n", model_path.c_str());
    WhisperModel model;
    if (!model.load_from_gguf(model_path)) {
        fprintf(stderr, "❌ Failed to load model\n");
        return 1;
    }
    fprintf(stderr, "   ✅ Model loaded: %d enc layers, %d dec layers, %d hidden\n",
            model.cfg.n_audio_layer, model.cfg.n_text_layer, model.cfg.n_audio_state);
    
    // 2. Load or generate audio
    std::vector<float> audio_pcm;
    int sample_rate = 16000;
    
    if (argc > 2) {
        fprintf(stderr, "\n2. Loading audio: %s\n", argv[2]);
        audio_pcm = whisper_load_wav(argv[2], &sample_rate);
        if (audio_pcm.empty()) {
            fprintf(stderr, "❌ Failed to load WAV file\n");
            return 1;
        }
    } else {
        fprintf(stderr, "\n2. Generating test tone (440Hz, 3 seconds)\n");
        audio_pcm = generate_test_tone();
        sample_rate = 16000;
    }
    
    fprintf(stderr, "   Audio: %d samples @ %d Hz (%.1f seconds)\n",
            (int)audio_pcm.size(), sample_rate, (float)audio_pcm.size() / sample_rate);
    
    // 3. Compute mel spectrogram
    fprintf(stderr, "\n3. Computing mel spectrogram (%d bands)...\n", model.cfg.n_mels);
    auto mel = whisper_log_mel_spectrogram(audio_pcm.data(), (int)audio_pcm.size(), 
                                            sample_rate, model.cfg.n_mels);
    int n_frames = (int)(mel.size() / model.cfg.n_mels);
    fprintf(stderr, "   Mel spectrogram: %d frames × %d bands\n", n_frames, model.cfg.n_mels);

    // 3b. GPU correctness gate: scalar vs HIP, same weights, same mel.
    // Usage: whisper_demo <model.gguf> <audio.wav> --check-gpu
    if (argc > 3 && strcmp(argv[3], "--check-gpu") == 0) {
        fprintf(stderr, "\n[gpu-check] scalar vs HIP comparison\n");
        setenv("WHISPER_GPU", "0", 1);  // reference transcription must be scalar

        std::string ref_text = whisper_transcribe(model, audio_pcm.data(), (int)audio_pcm.size());

        if (!whisper_gpu_available()) {
            fprintf(stderr, "[gpu-check] FAIL — no HIP device, nothing to compare\n");
            return 2;
        }

        // Encoder comparison
        std::vector<float> enc_gpu((size_t)model.cfg.n_audio_ctx * model.cfg.n_audio_state);
        int ctx = 0;
        if (whisper_gpu_encode(&model, mel.data(), n_frames, enc_gpu.data(), &ctx) != 0) {
            fprintf(stderr, "[gpu-check] FAIL — GPU encode returned error\n");
            return 2;
        }
        enc_gpu.resize((size_t)ctx * model.cfg.n_audio_state);
        auto enc_cpu = whisper_encode(model, mel.data(), n_frames);
        float enc_maxdiff = 0.0f;
        size_t nmin = std::min(enc_cpu.size(), enc_gpu.size());
        for (size_t i = 0; i < nmin; i++)
            enc_maxdiff = std::max(enc_maxdiff, fabsf(enc_cpu[i] - enc_gpu[i]));
        fprintf(stderr, "[gpu-check] encoder max abs diff: %g (%zu/%zu elems)\n",
                enc_maxdiff, nmin, enc_cpu.size());

        // Decoder loop on GPU, comparing logits + final text
        std::vector<int> tokens = {WHISPER_SOT, WHISPER_TRANSCRIBE, WHISPER_ENGLISH};
        std::vector<int> out;
        std::vector<float> kv_cache;  // unused by both paths, API compatibility
        float logits_maxdiff = 0.0f;
        int steps = 0;
        for (int step = 0; step < model.cfg.n_text_ctx; step++) {
            std::vector<float> logits(model.cfg.n_vocab);
            if (whisper_gpu_decode_step(&model, tokens.data(), (int)tokens.size(), logits.data()) != 0) {
                fprintf(stderr, "[gpu-check] FAIL — GPU decode error at step %d\n", step);
                return 2;
            }
            auto ref_logits = whisper_decode_step(model, tokens, enc_cpu.data(), ctx, kv_cache);
            for (int i = 0; i < model.cfg.n_vocab; i++)
                logits_maxdiff = std::max(logits_maxdiff, fabsf(ref_logits[i] - logits[i]));
            int next = 0;
            float m = logits[0];
            for (int i = 1; i < model.cfg.n_vocab; i++) if (logits[i] > m) { m = logits[i]; next = i; }
            if (next == WHISPER_EOT) break;
            out.push_back(next);
            tokens.push_back(next);
            steps++;
        }
        fprintf(stderr, "[gpu-check] decoder max abs diff over %d steps: %g\n", steps, logits_maxdiff);

        std::string gpu_text;
        for (int id : out) {
            if (id >= 0 && (size_t)id < model.vocab.size() && !model.vocab[id].empty())
                gpu_text += whisper_decode_bpe_token(model.vocab[id]);
            else if (id < 256) gpu_text += (char)id;
        }
        fprintf(stderr, "[gpu-check] scalar: \"%s\"\n", ref_text.c_str());
        fprintf(stderr, "[gpu-check] gpu:    \"%s\"\n", gpu_text.c_str());
        bool pass = enc_maxdiff < 1e-2f && logits_maxdiff < 1e-2f && ref_text == gpu_text;
        fprintf(stderr, "[gpu-check] %s\n", pass ? "PASS — GPU matches scalar" : "FAIL — mismatch");
        return pass ? 0 : 2;
    }

    // 4. Run encoder
    fprintf(stderr, "\n4. Running encoder (%d layers)...\n", model.cfg.n_audio_layer);
    auto enc_out = whisper_encode(model, mel.data(), n_frames);
    int n_enc_ctx = (int)(enc_out.size() / model.cfg.n_audio_state);
    fprintf(stderr, "   Encoder output: %d tokens × %d dims\n", n_enc_ctx, model.cfg.n_audio_state);
    
    // 5. Run decoder loop
    fprintf(stderr, "\n5. Running decoder loop...\n");
    auto text = whisper_transcribe(model, audio_pcm.data(), (int)audio_pcm.size());
    
    // 6. Output
    fprintf(stderr, "\n══════════════════════════════════════════\n");
    fprintf(stderr, "   Transcription: \"%s\"\n", text.c_str());
    fprintf(stderr, "══════════════════════════════════════════\n");
    
    return 0;
}
