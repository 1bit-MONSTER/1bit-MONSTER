#pragma once
// zaya_codec.h — C++ decoder for the zaya_audio RVQ-VAE audio codec (5.87M
// params), issue #1368.  Embedded / llama.cpp-style deployment: no HIP, no
// ONNX Runtime, no Python — stdlib + the repo's GgufReader only.
//
// Pipeline (mirrors zaya_audio/codec.py RVQVAE.decode + the ONNX decoder
// export in zaya_audio/export_onnx.py):
//
//   tokens (n_codebooks x T) --codebook lookup+sum--> z_q [code_dim][T]
//     --post_vq 1x1 conv--> [latent_dim][T]
//     --dec_proj 3x1 conv (k=3,pad=1)--> [dec_in][T]
//     --5x DecoderBlock: FiLM(speaker_emb) -> ConvTranspose1d(k=2s,p=s//2,
//        output_padding) -> ReLU -> n_res_blocks x ResidualBlock--> upsample
//     --post_conv 7x1 conv (k=7,pad=3)--> mono PCM @ 24 kHz
//
// Weights come from a GGUF file written by tools/export_codec_gguf.py
// (architecture "zaya_codec"; F32 tensors; structural hyperparameters in
// "zaya_codec.*" KV metadata).  All convs preserve time length except the
// transposed upsampling convs; total upsampling for the default config is
// 1280x + output padding, e.g. 56 latent frames -> 72000 samples (3 s).

#include <cstdint>
#include <string>
#include <vector>

class ZayaCodecDecoder {
public:
    ZayaCodecDecoder() = default;
    ~ZayaCodecDecoder() = default;
    ZayaCodecDecoder(const ZayaCodecDecoder&) = delete;
    ZayaCodecDecoder& operator=(const ZayaCodecDecoder&) = delete;

    /// Load codec weights from a GGUF file produced by export_codec_gguf.py.
    /// On failure returns false and (if err != nullptr) fills err.
    bool load(const std::string& gguf_path, std::string* err = nullptr);

    bool is_loaded() const { return loaded_; }

    // ── Hyperparameters (as read from the GGUF KV metadata) ───────────
    int n_codebooks() const { return n_codebooks_; }
    int codebook_size() const { return codebook_size_; }
    int code_dim() const { return code_dim_; }
    int latent_dim() const { return latent_dim_; }
    int speaker_dim() const { return speaker_dim_; }
    int sample_rate() const { return sample_rate_; }
    int n_res_blocks() const { return n_res_blocks_; }
    const std::vector<int>& decoder_strides() const { return strides_; }
    const std::vector<int>& decoder_output_paddings() const { return outpads_; }

    /// Exact output sample count for `n_frames` latent frames, using the
    /// ConvTranspose1d length law L_out = (L_in-1)*s - 2*p + k + op per block.
    int expected_output_samples(int n_frames) const;

    /// Decode RVQ token frames to PCM audio.
    ///
    /// tokens:     n_codebooks_ x n_frames int32 codes, row-major
    ///             (codebook c, frame t at tokens[c*n_frames + t]).
    /// speaker_emb: speaker_dim_ floats; may be nullptr for a zero embedding
    ///             (matches codec.py's zero-embedding fallback).
    /// out:        mono float32 PCM at sample_rate() Hz.
    /// Returns false (with err) on failure, e.g. not loaded.
    bool decode(const int32_t* tokens, int n_frames, const float* speaker_emb,
                std::vector<float>& out, std::string* err = nullptr) const;

    // Weight containers (implementation detail; public so src/zaya_codec.cpp
    // can build them from GGUF tensors without a friend dance).
    struct Conv1d {
        int cin = 0, cout = 0, k = 0, pad = 0;
        std::vector<float> w;  // [cout][cin][k], PyTorch Conv1d layout
        std::vector<float> b;  // [cout]
    };
    struct ConvT {
        int cin = 0, cout = 0, k = 0, stride = 0, pad = 0, outpad = 0;
        std::vector<float> w;  // [cin][cout][k], PyTorch ConvTranspose1d layout
        std::vector<float> b;  // [cout]
    };
    struct Film {
        int channels = 0, speaker_dim = 0;
        std::vector<float> w;  // [2*channels][speaker_dim]
        std::vector<float> b;  // [2*channels]
    };
    struct ResBlock {
        Conv1d c1, c2;  // k=3, pad=1, cin==cout==block channels
    };
    struct DecBlock {
        Film film;
        ConvT up;
        std::vector<ResBlock> res;
    };

private:
    bool loaded_ = false;
    int n_codebooks_ = 0, codebook_size_ = 0, code_dim_ = 0;
    int latent_dim_ = 0, speaker_dim_ = 0, n_res_blocks_ = 0;
    int sample_rate_ = 24000;
    std::vector<int> strides_, outpads_;
    std::vector<float> embed_;  // [n_codebooks][codebook_size][code_dim]
    Conv1d post_vq_, dec_proj_, post_conv_;
    std::vector<DecBlock> blocks_;
};
