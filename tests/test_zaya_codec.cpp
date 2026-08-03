// test_zaya_codec.cpp — self-check for the C++ RVQ-VAE codec decoder
// (issue #1368).  Plain checks, no test framework, works with or without
// NDEBUG (asserts would be compiled out of release builds).
//
// Usage:
//   python3 tools/export_codec_gguf.py --random --out /tmp/codec_test.gguf
//   ./build/test_zaya_codec /tmp/codec_test.gguf
//
// The exporter also writes <base>.tokens.bin / .speaker_emb.bin /
// .ref_output.bin next to the GGUF; when present, this test additionally
// compares the C++ decode numerically against the PyTorch reference.
#include "rocm_cpp/zaya_codec.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

static int failures = 0;

#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (cond) {                                                          \
            printf("  OK   %s\n", msg);                                      \
        } else {                                                             \
            printf("  FAIL %s\n", msg);                                      \
            failures++;                                                      \
        }                                                                    \
    } while (0)

static bool read_file(const std::string& path, std::vector<uint8_t>& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return false; }
    out.resize((size_t)n);
    bool ok = n == 0 || fread(out.data(), 1, (size_t)n, f) == (size_t)n;
    fclose(f);
    return ok;
}

template <typename T>
static bool load_bin(const std::string& path, std::vector<T>& out) {
    std::vector<uint8_t> raw;
    if (!read_file(path, raw) || raw.size() % sizeof(T) != 0) return false;
    out.resize(raw.size() / sizeof(T));
    memcpy(out.data(), raw.data(), raw.size());
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s codec.gguf\n"
                "  Generate one with:\n"
                "    python3 tools/export_codec_gguf.py --random --out /tmp/codec_test.gguf\n",
                argv[0]);
        return 2;
    }
    const std::string gguf = argv[1];
    const std::string base = gguf.substr(0, gguf.size() - 5);  // strip ".gguf"

    ZayaCodecDecoder dec;
    std::string err;
    CHECK(dec.load(gguf, &err), "load GGUF");
    if (!dec.is_loaded()) {
        fprintf(stderr, "load error: %s\n", err.c_str());
        return 1;
    }
    printf("  info: %d codebooks x %d entries, code_dim=%d latent=%d "
           "speaker_dim=%d, n_res=%d, sr=%d Hz, strides=[",
           dec.n_codebooks(), dec.codebook_size(), dec.code_dim(), dec.latent_dim(),
           dec.speaker_dim(), dec.n_res_blocks(), dec.sample_rate());
    for (int s : dec.decoder_strides()) printf("%d,", s);
    printf("] outpad=[");
    for (int p : dec.decoder_output_paddings()) printf("%d,", p);
    printf("]\n");

    // Length law: 56 latent frames = 72000 samples (3 s) for the default
    // config; 1 frame = 1600 samples.  Only checked for default strides.
    if (dec.decoder_strides() == std::vector<int>({4, 4, 4, 5, 4}) &&
        dec.decoder_output_paddings() == std::vector<int>({0, 3, 3, 4, 0})) {
        CHECK(dec.expected_output_samples(56) == 72000,
              "length law: 56 frames -> 72000 samples");
        CHECK(dec.expected_output_samples(1) == 1600,
              "length law: 1 frame -> 1600 samples");
    }

    // ── Tokens: use the exporter's sidecar when present, else synthetic ──
    const int n_cb = dec.n_codebooks();
    const int T = 56;
    std::vector<int32_t> tokens;
    std::vector<float> emb;
    if (!load_bin(base + ".tokens.bin", tokens) || (int)tokens.size() != n_cb * T) {
        printf("  note: no tokens sidecar — using synthetic tokens\n");
        tokens.resize((size_t)n_cb * T);
        for (size_t i = 0; i < tokens.size(); i++)
            tokens[i] = (int32_t)((i * 2654435761u) % (uint32_t)dec.codebook_size());
    }
    if (!load_bin(base + ".speaker_emb.bin", emb) ||
        (int)emb.size() != dec.speaker_dim()) {
        printf("  note: no speaker_emb sidecar — using zero embedding\n");
        emb.clear();
    }
    std::vector<float> ref;
    const bool have_ref = load_bin(base + ".ref_output.bin", ref);

    std::vector<float> out;
    CHECK(dec.decode(tokens.data(), T, emb.empty() ? nullptr : emb.data(), out, &err),
          "decode tokens -> PCM");
    const int expected = dec.expected_output_samples(T);
    CHECK(expected > 0 && (int)out.size() == expected,
          "output length == expected_output_samples");
    printf("  info: %d frames -> %zu samples (%.3f s @ %d Hz), expected %d\n",
           T, out.size(), (double)out.size() / dec.sample_rate(), dec.sample_rate(),
           expected);
    if (have_ref) {
        CHECK((int)ref.size() == expected,
              "output length == PyTorch reference length");
    }

    // All samples finite.
    bool finite = true;
    for (float v : out) finite = finite && std::isfinite(v);
    CHECK(finite, "all output samples finite");

    // Deterministic: decoding twice gives bit-identical audio.
    std::vector<float> out2;
    dec.decode(tokens.data(), T, emb.empty() ? nullptr : emb.data(), out2, &err);
    CHECK(out == out2, "decode is deterministic (two runs identical)");

    if (have_ref) {
        double max_abs = 0.0;
        for (size_t i = 0; i < out.size() && i < ref.size(); i++)
            max_abs = fmax(max_abs, fabs((double)out[i] - ref[i]));
        printf("  info: max |C++ - PyTorch| = %.3e (tolerance 1e-3)\n", max_abs);
        CHECK(out.size() == ref.size() && max_abs < 1e-3,
              "numeric match vs PyTorch reference (max abs err < 1e-3)");
    } else {
        printf("  note: SKIP numeric comparison — %s.ref_output.bin not found\n",
               base.c_str());
    }

    if (failures == 0) {
        printf("PASS\n");
        return 0;
    }
    printf("FAIL (%d check(s) failed)\n", failures);
    return 1;
}
