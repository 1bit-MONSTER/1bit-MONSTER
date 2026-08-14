// e2e_whisper.cpp — Whisper STT e2e gate: load a GGUF + WAV, transcribe.
// usage: e2e_whisper <model.gguf> <audio.wav> [tokens.txt]
#include <cstdio>
#include <string>
#include <fstream>
#include "whisper.h"
int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: %s <model.gguf> <audio.wav> [tokens.txt]\n", argv[0]); return 2; }
    WhisperModel model;
    if (!model.load_from_gguf(argv[1])) { printf("FAIL load\n"); return 1; }
    if (argc > 3) {  // load the detokenization vocab (HF tokenizer -> one token per line)
        std::ifstream f(argv[3]);
        std::string line;
        while (std::getline(f, line)) model.vocab.push_back(line);
        fprintf(stderr, "[whisper] vocab loaded: %zu\n", model.vocab.size());
    }
    int sr = 0;
    std::vector<float> pcm = whisper_load_wav(argv[2], &sr);
    if (pcm.empty()) { printf("FAIL wav\n"); return 1; }
    if (sr != 16000) {
        fprintf(stderr, "[whisper] resampling %d -> 16000\n", sr);
        // crude linear resample (the harness only needs 16k inputs)
        std::vector<float> out;
        out.reserve((size_t)(pcm.size() * 16000.0 / sr));
        for (size_t i = 0; i < pcm.size() * 16000 / sr; i++) {
            double src = (double)i * sr / 16000.0;
            size_t i0 = (size_t)src; size_t i1 = i0 + 1 < pcm.size() ? i0 + 1 : i0;
            double f = src - i0;
            out.push_back((float)(pcm[i0] * (1 - f) + pcm[i1] * f));
        }
        pcm = std::move(out);
    }
    std::string text = whisper_transcribe(model, pcm.data(), (int)pcm.size());
    printf("TRANSCRIPT: %s\n", text.c_str());
    return text.empty() ? 1 : 0;
}
