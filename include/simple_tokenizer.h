#pragma once
// simple_tokenizer.h — shared tokenizer wrapper, extracted from
// tools/unified_server.cpp so backends that need to bridge token IDs to text
// (e.g. FlmBackend, which talks to an external text-only subprocess) can use
// the exact same vocabulary as the rest of the server. Uses the RCPP BPE
// tokenizer (.htok) when available, falls back to ASCII/UTF-8 byte
// passthrough when not — behavior unchanged from the original inline struct.

#include "rocm_cpp/tokenizer.h"
#include <string>
#include <vector>
#include <cstdio>

struct SimpleTokenizer {
    int bos_id = 2;
    int eos_id = 1;
    bool use_bpe = false;
    rcpp_tokenizer_t* bpe_tok = nullptr;

    bool load(const std::string& vocab_path) {
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

    ~SimpleTokenizer() {
        if (bpe_tok) rcpp_tokenizer_free(bpe_tok);
    }

    std::vector<int> encode(const std::string& text) {
        if (use_bpe && bpe_tok) {
            std::vector<int> r(4096);
            size_t out_n = 0;
            rcpp_status_t st = rcpp_tokenizer_encode(bpe_tok, text.c_str(), text.size(),
                                                      1, r.data(), r.size(), &out_n);
            if (st == RCPP_OK && out_n > 0) {
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

    /// Encode with per-token log-probabilities (for cascade strategy).
    /// Returns token IDs and fills logprobs_out with corresponding logprobs.
    std::vector<int> encode_with_logprobs(const std::string& text,
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
                r.resize(out_n);
                logprobs_out.assign(lp.begin(), lp.begin() + out_n);
                return r;
            }
            return {bos_id};
        }
        // Fallback: encode without logprobs
        auto tokens = encode(text);
        logprobs_out.resize(tokens.size(), -1.0);
        return tokens;
    }

    std::string decode(const std::vector<int>& tokens) {
        if (use_bpe && bpe_tok) {
            std::string r(4096, '\0');
            size_t out_len = 0;
            rcpp_status_t st = rcpp_tokenizer_decode(bpe_tok, tokens.data(), tokens.size(),
                                                      r.data(), r.size(), &out_len);
            if (st == RCPP_OK && out_len > 0) {
                r.resize(out_len);
                return r;
            }
            return "";
        }
        // Fallback: ASCII + UTF-8 byte passthrough
        std::string r;
        for (int v : tokens) {
            if (v == bos_id || v == eos_id) continue;
            if (v > 100 && v < 200)
                r += (char)(v - 100);
            else if (v > 200 && v < 456)
                r += (char)(v - 200);
            else {
                r += '['; r += std::to_string(v); r += ']';
            }
        }
        return r;
    }
};

// Shared global instance — defined once in tools/unified_server.cpp, used
// there and by any backend (e.g. FlmBackend) that needs to bridge token IDs
// to text using the exact same vocabulary the rest of the server uses.
extern SimpleTokenizer g_tokenizer;
