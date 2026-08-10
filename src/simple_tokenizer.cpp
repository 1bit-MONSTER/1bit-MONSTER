// simple_tokenizer.cpp — single definition point for the shared g_tokenizer
// instance declared in include/simple_tokenizer.h, plus all method bodies
// moved out of the header to avoid recompilation cascades (issue #375).
// Lives in the backend_manager static lib so every target that links it
// (unified_server, backend_demo, ...) gets the symbol, regardless of whether
// that target actually loads/uses a real tokenizer.
#include "simple_tokenizer.h"

#ifndef ZINC_DISABLED
extern "C" {
    void* zinc_tokenizer_init(const char* gguf_path);
    void zinc_tokenizer_destroy(void* state);
    uint32_t zinc_tokenizer_bos_id(void* state);
    uint32_t zinc_tokenizer_eos_id(void* state);
    long long zinc_tokenizer_encode(void* state, const char* text, uint32_t* out_ids, size_t out_cap);
    long long zinc_tokenizer_decode(void* state, const uint32_t* ids, size_t n_ids, char* out_buf, size_t out_cap);
}
#endif

// ── SimpleTokenizer method bodies ──────────────────────────────────────────

bool SimpleTokenizer::load_from_gguf(const std::string& gguf_path) {
#ifndef ZINC_DISABLED
    if (gguf_path.size() < 5 || gguf_path.substr(gguf_path.size() - 5) != ".gguf")
        return false;
    void* st = zinc_tokenizer_init(gguf_path.c_str());
    if (!st) {
        fprintf(stderr, "SimpleTokenizer: no usable embedded vocab in %s\n", gguf_path.c_str());
        return false;
    }
    unload_gguf_native();
    gguf_tok_state = st;
    use_gguf_native = true;
    bos_id = (int)zinc_tokenizer_bos_id(st);
    eos_id = (int)zinc_tokenizer_eos_id(st);
    fprintf(stderr, "Loaded GGUF-native tokenizer from %s (BOS=%d, EOS=%d)\n",
            gguf_path.c_str(), bos_id, eos_id);
    return true;
#else
    (void)gguf_path;
    return false;
#endif
}

bool SimpleTokenizer::load(const std::string& vocab_path) {
    if (vocab_path.size() >= 5 && vocab_path.substr(vocab_path.size() - 5) == ".htok") {
        rcpp_tokenizer_t* tok = nullptr;
        rcpp_status_t st = rcpp_tokenizer_load(vocab_path.c_str(), &tok);
        if (st == RCPP_OK && tok) {
            bpe_tok = tok;
            use_bpe = true;
            bos_id = rcpp_tokenizer_bos_id(bpe_tok);
            eos_id = rcpp_tokenizer_eos_id(bpe_tok);
            fprintf(stderr, "Loaded BPE tokenizer from %s (BOS=%d, EOS=%d)\n",
                    vocab_path.c_str(), bos_id, eos_id);
            return true;
        }
        fprintf(stderr, "Failed to load BPE tokenizer from %s\n", vocab_path.c_str());
    }
    fprintf(stderr, "No .htok at %s — using fallback ASCII tokenizer\n", vocab_path.c_str());
    return false;
}

void SimpleTokenizer::unload_gguf_native() {
#ifndef ZINC_DISABLED
    if (gguf_tok_state) { zinc_tokenizer_destroy(gguf_tok_state); gguf_tok_state = nullptr; }
#endif
    use_gguf_native = false;
}

SimpleTokenizer::~SimpleTokenizer() {
    if (bpe_tok) rcpp_tokenizer_free(bpe_tok);
    unload_gguf_native();
}

std::vector<int> SimpleTokenizer::encode(const std::string& text) {
#ifndef ZINC_DISABLED
    if (use_gguf_native && gguf_tok_state) {
        std::vector<uint32_t> r(4096);
        long long n = zinc_tokenizer_encode(gguf_tok_state, text.c_str(), r.data(), r.size());
        if (n > 0) {
            if ((size_t)n > r.size()) n = (long long)r.size();  // clamp OOB (issue #961)
            return std::vector<int>(r.begin(), r.begin() + n);
        }
        return {bos_id};
    }
#endif
    if (use_bpe && bpe_tok) {
        std::vector<int> r(4096);
        size_t out_n = 0;
        rcpp_status_t st = rcpp_tokenizer_encode(bpe_tok, text.c_str(), text.size(),
                                                  1, r.data(), r.size(), &out_n);
        if (st == RCPP_OK && out_n > 0) {
            if (out_n > r.size()) out_n = r.size();  // clamp OOB (issue #961)
            r.resize(out_n);
            return r;
        }
        return {bos_id};
    }
    // Fallback: ASCII + UTF-8 byte passthrough
    std::vector<int> r = {bos_id};
    for (unsigned char c : text) {
        if (c >= 32 && c <= 126)
            r.push_back((int)c + 100);
        else if (c != 0)
            r.push_back((int)c + 200);
    }
    return r;
}

std::vector<int> SimpleTokenizer::encode_with_logprobs(const std::string& text,
                                                       std::vector<double>& logprobs_out) {
    logprobs_out.clear();
    if (use_bpe && bpe_tok) {
        std::vector<int> r(4096);
        std::vector<double> lp(4096);
        size_t out_n = 0;
        rcpp_status_t st = rcpp_tokenizer_encode_with_logprobs(
            bpe_tok, text.c_str(), text.size(),
            1, r.data(), lp.data(), r.size(), &out_n);
        if (st == RCPP_OK && out_n > 0) {
            if (out_n > r.size()) out_n = r.size();  // clamp OOB (issue #961)
            r.resize(out_n);
            logprobs_out.assign(lp.begin(), lp.begin() + out_n);
            return r;
        }
        return {bos_id};
    }
    // Fallback (also used for GGUF-native, which has no logprob API):
    // encode without logprobs
    auto tokens = encode(text);
    logprobs_out.resize(tokens.size(), -1.0);
    return tokens;
}

std::string SimpleTokenizer::decode(const std::vector<int>& tokens) {
#ifndef ZINC_DISABLED
    if (use_gguf_native && gguf_tok_state) {
        std::vector<uint32_t> ids(tokens.begin(), tokens.end());
        std::string r(8192, '\0');
        long long n = zinc_tokenizer_decode(gguf_tok_state, ids.data(), ids.size(), r.data(), r.size());
        if (n >= 0) {
            if ((size_t)n > r.size()) n = (long long)r.size();  // clamp OOB (issue #961)
            r.resize((size_t)n); return r;
        }
        return "";
    }
#endif
    if (use_bpe && bpe_tok) {
        // NPU FLM backend convention: shifted char-tokens, not vocab IDs
        // (ASCII -> +100, raw bytes -> +300, EOS=106). Real BPE streams
        // from a 128000-vocab model are never confined to this narrow set,
        // so all-in-range is a safe detector (same heuristic as the
        // fallback below). Without this, FLM-generated text decodes as
        // garbage vocab ids once a real .htok is loaded.
        bool npu_shifted = true;
        for (int v : tokens) {
            if (v == 106) continue;
            if (!((v >= 132 && v <= 226) || (v >= 300 && v <= 555))) {
                npu_shifted = false;
                break;
            }
        }
        if (npu_shifted && !tokens.empty()) {
            std::string r;
            for (int v : tokens) {
                if (v == 106) continue;
                if (v >= 132 && v <= 226) r += (char)(v - 100);
                else if (v >= 300 && v <= 555) r += (char)(v - 300);
            }
            return r;
        }
        std::string r(4096, '\0');
        size_t out_len = 0;
        rcpp_status_t st = rcpp_tokenizer_decode(bpe_tok, tokens.data(), tokens.size(),
                                                  r.data(), r.size(), &out_len);
        if (st == RCPP_OK && out_len > 0) {
            if (out_len > r.size()) out_len = r.size();  // clamp OOB (issue #961)
            r.resize(out_len);
            return r;
        }
        return "";
    }
    // Fallback: ASCII + UTF-8 byte passthrough.
    // Bands must match the backend encoders (backend_npu.cpp forward):
    // printable ASCII 32-126 -> id 132-226 (+100); control/raw bytes ->
    // id 300-555 (+300). (The old +200 scheme's band 201-226 decoded
    // letters as control chars — fixed 2026-08-08.)
    std::string r;
    for (int v : tokens) {
        if (v == bos_id || v == eos_id) continue;
        if (v >= 132 && v <= 226)
            r += (char)(v - 100);
        else if (v >= 300 && v <= 555)
            r += (char)(v - 300);
        else {
            r += '['; r += std::to_string(v); r += ']';
        }
    }
    return r;
}

SimpleTokenizer g_tokenizer;
