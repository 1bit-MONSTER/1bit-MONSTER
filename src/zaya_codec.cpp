// zaya_codec.cpp — implementation of ZayaCodecDecoder (issue #1368).
// CPU-only, float32, stdlib + GgufReader.  Ops needed: Conv1d, ConvTranspose1d,
// ReLU, per-channel affine (FiLM), residual add — all plain loops.
#include "rocm_cpp/zaya_codec.h"
#include "gguf_reader.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using Conv1d = ZayaCodecDecoder::Conv1d;
using ConvT = ZayaCodecDecoder::ConvT;
using Film = ZayaCodecDecoder::Film;
using ResBlock = ZayaCodecDecoder::ResBlock;

namespace {

void set_err(std::string* err, const std::string& msg) {
    if (err) *err = msg;
}

// Parse a GGUF string-array KV ("4","4","4","5","4") into ints.
// Values may be 0 (output paddings) — only reject nonsense.
bool parse_int_array(const std::vector<std::string>& arr, std::vector<int>& out) {
    out.clear();
    for (const auto& s : arr) {
        char* end = nullptr;
        long v = std::strtol(s.c_str(), &end, 10);
        if (end == s.c_str() || *end != '\0' || v < 0 || v > 1 << 20) return false;
        out.push_back((int)v);
    }
    return !out.empty();
}

// Load a GGUF F32 tensor, validating each dim against `want` (0 = wildcard,
// checked by the caller).  GGUF shape order is shape[0]-fastest, which is
// exactly PyTorch's contiguous flat order, so the flat data needs no
// reordering.  Returns false with a message on stderr if anything is off.
bool load_flat(GgufReader& r, const std::string& name,
               const std::vector<int>& want, std::vector<float>& w) {
    const GgufTensorInfo* ti = r.tensor_info(name);
    if (!ti) { std::fprintf(stderr, "zaya_codec: missing tensor %s\n", name.c_str()); return false; }
    if (ti->shape.size() != want.size() || ti->dtype != GGUF_DTYPE_F32) {
        std::fprintf(stderr, "zaya_codec: %s: bad ndim or non-F32 dtype\n", name.c_str());
        return false;
    }
    for (size_t i = 0; i < want.size(); i++)
        if (want[i] > 0 && ti->shape[i] != (uint64_t)want[i]) {
            std::fprintf(stderr, "zaya_codec: %s dim %zu is %llu, expected %d\n",
                         name.c_str(), i, (unsigned long long)ti->shape[i], want[i]);
            return false;
        }
    size_t n = 0;
    return r.get_tensor_f32(name, w, &n) && n == ti->numel;
}

// Load a Conv1d (w: [cout][cin][k], b: [cout]); k odd, pad = (k-1)/2.
// cout == 0 means "derive from the weight size" (dec_proj's out channels are
// only known from the GGUF itself).
bool load_conv1d(GgufReader& r, const std::string& prefix, int cin, int cout, int k,
                 Conv1d& c) {
    if (!load_flat(r, prefix + ".weight", {k, cin, cout}, c.w) ||
        !load_flat(r, prefix + ".bias", {0}, c.b)) {
        std::fprintf(stderr, "zaya_codec: failed to load %s\n", prefix.c_str());
        return false;
    }
    if (cout == 0) {
        if (c.w.size() % ((size_t)cin * k) != 0) {
            std::fprintf(stderr, "zaya_codec: %s: bad weight size\n", prefix.c_str());
            return false;
        }
        cout = (int)(c.w.size() / ((size_t)cin * k));
    }
    if ((size_t)cout != c.b.size()) {
        std::fprintf(stderr, "zaya_codec: %s: bias size != cout\n", prefix.c_str());
        return false;
    }
    c.cin = cin; c.cout = cout; c.k = k; c.pad = (k - 1) / 2;
    return true;
}

// ── Ops (PyTorch semantics, float32) ───────────────────────────────────────

// Cross-correlation Conv1d: out[t] = sum_k x[t-pad+k]*w[k], zero-padded.
// pad == (k-1)/2 for every conv here, so T_out == T_in.
void conv1d(const std::vector<float>& x, int t_in, const Conv1d& c, std::vector<float>& y) {
    const int cin = c.cin;
    y.assign((size_t)c.cout * t_in, 0.f);
    for (int co = 0; co < c.cout; co++) {
        const float* wco = &c.w[(size_t)co * cin * c.k];
        float* yc = &y[(size_t)co * t_in];
        for (int t = 0; t < t_in; t++) {
            float acc = c.b[co];
            for (int ci = 0; ci < cin; ci++) {
                const float* xc = &x[(size_t)ci * t_in];
                const float* wci = &wco[(size_t)ci * c.k];
                for (int kk = 0; kk < c.k; kk++) {
                    int idx = t - c.pad + kk;
                    if (idx >= 0 && idx < t_in) acc += xc[idx] * wci[kk];
                }
            }
            yc[t] = acc;
        }
    }
}

// Fractionally-strided transposed conv: L_out = (L_in-1)*s - 2*p + k + op.
void conv_transpose1d(const std::vector<float>& x, int t_in, const ConvT& c,
                      std::vector<float>& y) {
    const int t_out = (t_in - 1) * c.stride - 2 * c.pad + c.k + c.outpad;
    y.assign((size_t)c.cout * t_out, 0.f);
    for (int ci = 0; ci < c.cin; ci++) {
        const float* xc = &x[(size_t)ci * t_in];
        for (int co = 0; co < c.cout; co++) {
            float* yc = &y[(size_t)co * t_out];
            const float* wco = &c.w[((size_t)ci * c.cout + co) * c.k];
            for (int t = 0; t < t_in; t++) {
                const float xv = xc[t];
                for (int kk = 0; kk < c.k; kk++) {
                    int l = t * c.stride - c.pad + kk;
                    if (l >= 0 && l < t_out) yc[l] += xv * wco[kk];
                }
            }
        }
    }
    for (int co = 0; co < c.cout; co++) {
        float* yc = &y[(size_t)co * t_out];
        const float bv = c.b[co];
        for (int l = 0; l < t_out; l++) yc[l] += bv;
    }
}

// FiLM: gamma_beta = W*emb + b (rows 0..C-1 = gamma, C..2C-1 = beta), then
// x[c][t] = gamma[c]*x[c][t] + beta[c].  In-place on x ([channels][T]).
void film_apply(const Film& f, const float* emb, std::vector<float>& x) {
    const int T = (int)(x.size() / f.channels);
    std::vector<float> gb(2 * f.channels, 0.f);
    if (emb) {
        for (int r = 0; r < 2 * f.channels; r++) {
            float acc = f.b[r];
            const float* wr = &f.w[(size_t)r * f.speaker_dim];
            for (int c = 0; c < f.speaker_dim; c++) acc += wr[c] * emb[c];
            gb[r] = acc;
        }
    } else {
        for (int r = 0; r < 2 * f.channels; r++) gb[r] = f.b[r];
    }
    for (int c = 0; c < f.channels; c++) {
        const float g = gb[c], bet = gb[f.channels + c];
        float* xc = &x[(size_t)c * T];
        for (int t = 0; t < T; t++) xc[t] = g * xc[t] + bet;
    }
}

void relu(std::vector<float>& x) {
    for (float& v : x)
        if (v < 0.f) v = 0.f;
}

// ResidualBlock: x + conv2(relu(conv1(x))).
void res_block(const ResBlock& rb, int channels, int T, std::vector<float>& x) {
    std::vector<float> t1, t2;
    conv1d(x, T, rb.c1, t1);
    relu(t1);
    conv1d(t1, T, rb.c2, t2);
    for (size_t i = 0; i < x.size(); i++) x[i] += t2[i];
}

} // namespace

// ── Public API ─────────────────────────────────────────────────────────────

bool ZayaCodecDecoder::load(const std::string& gguf_path, std::string* err) {
    loaded_ = false;
    blocks_.clear();
    embed_.clear();

    GgufReader r;
    if (!r.open(gguf_path)) {
        set_err(err, "zaya_codec: cannot open GGUF file: " + gguf_path);
        return false;
    }
    std::string arch;
    r.get_string("general.architecture", arch);
    if (arch != "zaya_codec") {
        set_err(err, "zaya_codec: not a zaya_codec GGUF (architecture=\"" + arch + "\")");
        return false;
    }

    auto get_u32 = [&](const char* key, int& out) -> bool {
        uint32_t v;
        if (!r.get_u32(key, v)) return false;
        out = (int)v;
        return true;
    };
    if (!get_u32("sample_rate", sample_rate_) ||
        !get_u32("n_codebooks", n_codebooks_) ||
        !get_u32("codebook_size", codebook_size_) ||
        !get_u32("code_dim", code_dim_) ||
        !get_u32("latent_dim", latent_dim_) ||
        !get_u32("speaker_dim", speaker_dim_) ||
        !get_u32("n_res_blocks", n_res_blocks_)) {
        set_err(err, "zaya_codec: missing/invalid zaya_codec.* KV metadata");
        return false;
    }
    std::vector<std::string> sarr;
    if (!r.get_string_array("decoder_strides", sarr) || !parse_int_array(sarr, strides_)) {
        set_err(err, "zaya_codec: missing decoder_strides KV array");
        return false;
    }
    if (!r.get_string_array("decoder_output_paddings", sarr) ||
        !parse_int_array(sarr, outpads_) || outpads_.size() != strides_.size()) {
        set_err(err, "zaya_codec: missing/bad decoder_output_paddings KV array");
        return false;
    }

    // ── Tensors ──────────────────────────────────────────────────────────
    // Codebook embeddings [n_codebooks][codebook_size][code_dim].
    if (!load_flat(r, "res_vq.embed", {code_dim_, codebook_size_, n_codebooks_}, embed_)) {
        set_err(err, "zaya_codec: failed to load res_vq.embed");
        return false;
    }

    if (!load_conv1d(r, "post_vq", code_dim_, latent_dim_, 1, post_vq_) ||
        !load_conv1d(r, "dec_proj", latent_dim_, 0, 3, dec_proj_)) {
        set_err(err, "zaya_codec: failed to load post_vq/dec_proj");
        return false;
    }

    const int n_blocks = (int)strides_.size();
    for (int i = 0; i < n_blocks; i++) {
        DecBlock blk;
        const int s = strides_[i];
        const int k = 2 * s, pad = s / 2, op = outpads_[i];
        const std::string pfx = "decoder." + std::to_string(i);

        // FiLM proj: w [2*cin][speaker_dim], b [2*cin].  cin is wildcarded in
        // load_flat and derived here, then cross-checked against the ConvT.
        if (!load_flat(r, pfx + ".film.proj.weight", {speaker_dim_, 0}, blk.film.w) ||
            !load_flat(r, pfx + ".film.proj.bias", {0}, blk.film.b)) {
            set_err(err, "zaya_codec: failed to load " + pfx + ".film.proj");
            return false;
        }
        if (blk.film.w.size() % (size_t)speaker_dim_ != 0 ||
            blk.film.b.size() != blk.film.w.size() / (size_t)speaker_dim_ ||
            blk.film.b.size() % 2 != 0) {
            set_err(err, "zaya_codec: bad FiLM shape in " + pfx);
            return false;
        }
        blk.film.channels = (int)(blk.film.b.size() / 2);
        blk.film.speaker_dim = speaker_dim_;

        // ConvTranspose1d: w [cin][cout][k], b [cout].  Both cin/cout
        // wildcarded, derived from the flat size.
        if (!load_flat(r, pfx + ".conv_transpose.weight", {k, 0, 0}, blk.up.w) ||
            !load_flat(r, pfx + ".conv_transpose.bias", {0}, blk.up.b)) {
            set_err(err, "zaya_codec: failed to load " + pfx + ".conv_transpose");
            return false;
        }
        if (blk.up.w.size() % (size_t)k != 0 || blk.up.b.empty() ||
            blk.up.w.size() / k % blk.up.b.size() != 0) {
            set_err(err, "zaya_codec: bad ConvTranspose shape in " + pfx);
            return false;
        }
        blk.up.cout = (int)blk.up.b.size();
        blk.up.cin = (int)(blk.up.w.size() / k / blk.up.b.size());
        blk.up.k = k; blk.up.stride = s; blk.up.pad = pad; blk.up.outpad = op;
        if (blk.up.cin != blk.film.channels) {
            set_err(err, "zaya_codec: FiLM/ConvT channel mismatch in " + pfx);
            return false;
        }

        // Residual blocks at the output channel count.
        const int C = blk.up.cout;
        blk.res.resize(n_res_blocks_);
        for (int j = 0; j < n_res_blocks_; j++) {
            const std::string rp = pfx + ".res_blocks." + std::to_string(j) + ".net.";
            if (!load_conv1d(r, rp + "0", C, C, 3, blk.res[j].c1) ||
                !load_conv1d(r, rp + "2", C, C, 3, blk.res[j].c2)) {
                set_err(err, "zaya_codec: failed to load " + rp);
                return false;
            }
        }
        blocks_.push_back(std::move(blk));
    }

    if (blocks_.empty() || dec_proj_.cout == 0) {
        set_err(err, "zaya_codec: no decoder blocks");
        return false;
    }
    if (blocks_[0].film.channels != dec_proj_.cout) {
        set_err(err, "zaya_codec: dec_proj channels != first decoder block channels");
        return false;
    }
    // post_conv: w [1][C][7], b [1].
    if (!load_conv1d(r, "post_conv", blocks_.back().up.cout, 1, 7, post_conv_)) {
        set_err(err, "zaya_codec: failed to load post_conv");
        return false;
    }

    loaded_ = true;
    return true;
}

int ZayaCodecDecoder::expected_output_samples(int n_frames) const {
    if (!loaded_ || n_frames <= 0) return 0;
    long L = n_frames;
    for (size_t i = 0; i < blocks_.size(); i++) {
        const ConvT& up = blocks_[i].up;
        L = (L - 1) * (long)up.stride - 2L * up.pad + up.k + up.outpad;
    }
    return (int)L;
}

bool ZayaCodecDecoder::decode(const int32_t* tokens, int n_frames, const float* speaker_emb,
                              std::vector<float>& out, std::string* err) const {
    if (!loaded_) { set_err(err, "zaya_codec: decoder not loaded"); return false; }
    if (!tokens || n_frames <= 0) { set_err(err, "zaya_codec: bad tokens/n_frames"); return false; }
    const int T = n_frames;

    // 1) Codebook lookup + sum across codebooks -> z_q [code_dim][T].
    std::vector<float> z_q((size_t)code_dim_ * T, 0.f);
    for (int cb = 0; cb < n_codebooks_; cb++) {
        const float* emb = &embed_[(size_t)cb * codebook_size_ * code_dim_];
        const int32_t* tok = tokens + (size_t)cb * T;
        for (int t = 0; t < T; t++) {
            int v = tok[t];
            if (v < 0) v = 0; else if (v >= codebook_size_) v = codebook_size_ - 1;
            const float* vec = emb + (size_t)v * code_dim_;
            for (int d = 0; d < code_dim_; d++) z_q[(size_t)d * T + t] += vec[d];
        }
    }

    // 2) post_vq (1x1), 3) dec_proj (k=3, pad=1).
    std::vector<float> z, x;
    conv1d(z_q, T, post_vq_, z);
    conv1d(z, T, dec_proj_, x);

    // 4) Decoder blocks: FiLM -> ConvTranspose1d -> ReLU -> res blocks.
    for (size_t i = 0; i < blocks_.size(); i++) {
        const DecBlock& blk = blocks_[i];
        const int t_in = (int)(x.size() / blk.film.channels);
        film_apply(blk.film, speaker_emb, x);
        std::vector<float> y;
        conv_transpose1d(x, t_in, blk.up, y);
        relu(y);
        const int t_out = (int)(y.size() / blk.up.cout);
        for (const ResBlock& rb : blk.res) res_block(rb, blk.up.cout, t_out, y);
        x = std::move(y);
    }

    // 5) post_conv -> mono audio.
    const int t_audio = (int)(x.size() / post_conv_.cin);
    conv1d(x, t_audio, post_conv_, out);
    return true;
}
