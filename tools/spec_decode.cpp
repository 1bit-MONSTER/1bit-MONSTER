// spec_decode.cpp — speculative decoding demo: draft (Qwen3-0.6B) proposes
// N tokens, target (Llama-3.2-1B-Instruct) verifies them all in ONE
// llama_decode batch, the longest greedy-consistent prefix is accepted.
// Lossless vs greedy baseline (acceptance is exact argmax comparison).
//
// Build: cmake --build build --target spec_decode -j8
// Run:   ./build/spec_decode models/Qwen3-0.6B.Q8_0.gguf \
//            models/Llama-3.2-1B-Instruct.Q8_0.gguf "What is 2+2?" [n_draft=4] [max_tokens=64]
//
// Uses the same vendored llama.cpp (ggml-vulkan) as the ggml_vulkan backend.

#include "llama.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

struct Lm {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;
    llama_memory_t mem = nullptr;
    std::string name;

    bool load(const std::string& path, const std::string& nm, int n_ctx = 4096) {
        name = nm;
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 99;
        model = llama_model_load_from_file(path.c_str(), mp);
        if (!model) { fprintf(stderr, "[%s] model load failed\n", nm.c_str()); return false; }
        auto cp = llama_context_default_params();
        cp.n_ctx = n_ctx;
        cp.n_batch = 512;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "[%s] context failed\n", nm.c_str()); return false; }
        vocab = llama_model_get_vocab(model);
        mem = llama_get_memory(ctx);
        fprintf(stderr, "[%s] loaded %s (ctx %d)\n", nm.c_str(), path.c_str(), n_ctx);
        return true;
    }

    // Decode a batch; logits_ith(i) gives the logits AFTER position i.
    int decode(const std::vector<llama_token>& toks, const std::vector<llama_pos>& poss,
               const std::vector<int8_t>& want_logits) {
        int rc;
        if (toks.size() == 1) {
            // Single token: the proven path (same as the ggml_vulkan backend).
            llama_token t = toks[0];
            auto b = llama_batch_get_one(&t, 1);
            rc = llama_decode(ctx, b);
        } else {
            llama_batch b = llama_batch_init((int)toks.size(), 0, 1);
            for (size_t i = 0; i < toks.size(); i++) {
                b.token[i] = toks[i];
                b.pos[i] = poss[i];
                b.n_seq_id[i] = 1;
                b.seq_id[i][0] = 0;
                b.logits[i] = want_logits.empty() ? 0 : want_logits[i];
            }
            b.n_tokens = (int)toks.size();
            rc = llama_decode(ctx, b);
            llama_batch_free(b);
        }
        if (rc != 0) fprintf(stderr, "[%s] decode rc=%d at batch start pos %lld\n", name.c_str(), rc, (long long)(poss.empty() ? -1 : poss[0]));
        return rc;
    }

    int greedy(const std::vector<llama_token>& toks, const std::vector<llama_pos>& poss) {
        if (decode(toks, poss, {1}) != 0) return -1;
        float* lg = llama_get_logits_ith(ctx, -1);
        int nv = llama_vocab_n_tokens(vocab);
        int best = 0;
        for (int v = 1; v < nv; v++) if (lg[v] > lg[best]) best = v;
        return best;
    }

    // Current sequence length in the KV cache.
    llama_pos pos() const { return llama_memory_seq_pos_max(mem, 0) + 1; }

    // Copy of the logits from the most recent decode (vocab-sized).
    std::vector<float> last_logits() const {
        int nv = llama_vocab_n_tokens(vocab);
        float* lg = llama_get_logits_ith(ctx, -1);
        return std::vector<float>(lg, lg + nv);
    }

    // Roll the KV cache back to `keep` positions (drop positions >= keep).
    void rollback(llama_pos keep) {
        llama_memory_seq_rm(mem, 0, keep, -1);
    }

    void clear() {
        if (ctx) { llama_free(ctx); ctx = nullptr; }
        if (model) { llama_model_free(model); model = nullptr; }
    }
};

static int greedy_argmax(const std::vector<float>& lg) {
    int best = 0;
    for (int v = 1; v < (int)lg.size(); v++) if (lg[v] > lg[best]) best = v;
    return best;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s draft.gguf target.gguf prompt [n_draft=4] [max_tokens=64]\n", argv[0]);
        return 1;
    }
    const std::string draft_path = argv[1];
    const std::string target_path = argv[2];
    const std::string prompt = argv[3];
    int N = argc > 4 ? atoi(argv[4]) : 4;
    int max_tokens = argc > 5 ? atoi(argv[5]) : 64;

    Lm draft, target;
    if (!draft.load(draft_path, "draft") || !target.load(target_path, "target")) return 1;

    // Tokenize the prompt with the target's vocab (same family; ids align for
    // the shared BOS/word tokens in these two models).
    std::vector<llama_token> prompt_toks;
    int n = (int)prompt.size();
    prompt_toks.resize(n + 4);
    int nt = llama_tokenize(target.vocab, prompt.c_str(), n, prompt_toks.data(), (int)prompt_toks.size(), true, false);
    prompt_toks.resize(nt);
    fprintf(stderr, "prompt: %d tokens\n", nt);

    int accepted = 0, rejected = 0;
    auto run = [&](bool speculative) -> std::pair<std::vector<llama_token>, double> {
        // Fresh KV for both contexts.
        llama_memory_clear(target.mem, true);
        llama_memory_clear(draft.mem, true);
        auto t0 = std::chrono::steady_clock::now();
        std::vector<llama_token> out;
        std::vector<llama_token> dtoks, ttoks;
        std::vector<llama_pos> dposs, tposs;
        int dpos = 0, tpos = 0;

        // Prefill: one single-token decode per prompt token (this fork's KV
        // rejects multi-token batches entirely — the backend does the same).
        for (int i = 0; i < nt - 1; i++) {
            if (draft.decode({prompt_toks[i]}, {}, {}) != 0 ||
                target.decode({prompt_toks[i]}, {}, {}) != 0) {
                fprintf(stderr, "prefill failed at %d\n", i); return {{}, 0};
            }
        }
        dpos = tpos = nt - 1;

        // First target token: the prefill above already decoded the whole
        // prompt; re-derive the first output by decoding the last prompt
        // token once more is impossible (no KV replacement), so instead we
        // decode the last prompt token as part of the prefill with logits.
        // Simplest robust approach: re-run the prefill with the last token
        // carrying logits via a fresh single-token decode is not possible,
        // so decode the full prompt for the target and take the argmax of
        // the final position by decoding the last token AGAIN at its own
        // position is not allowed either — so we keep a separate "first
        // token" pass: decode the last prompt token one more time via the
        // get_one path (auto position = mem_max+1 — the prompt token is
        // duplicated at a new position; harmless for the draft, and the
        // target re-decodes it too, keeping both KVs symmetric).
        int tok = target.greedy({prompt_toks.back()}, {});
        if (tok < 0) { fprintf(stderr, "first token failed\n"); return {{}, 0}; }
        // Decode the first output token into BOTH KVs; its logits predict
        // the next position (the acceptance anchor for proposal[0]).
        if (target.decode({tok}, {}, {}) != 0) return {{}, 0};
        std::vector<float> prev_logits = target.last_logits();
        if (draft.greedy({prompt_toks.back()}, {}) < 0) return {{}, 0};
        if (draft.decode({tok}, {}, {}) != 0) return {{}, 0};
        tpos = target.pos();
        dpos = draft.pos();
        out.push_back(tok);

        while ((int)out.size() < max_tokens) {
            if (speculative) {
                // 1. Draft proposes N tokens (greedy, its own context). The
                // first proposal comes from the draft's last decode's logits
                // (the tok is already in its KV — no re-decode).
                std::vector<llama_token> proposals;
                std::vector<float> dlogits = draft.last_logits();
                for (int i = 0; i < N; i++) {
                    int nxt = greedy_argmax(dlogits);
                    proposals.push_back(nxt);
                    if (draft.decode({nxt}, {}, {}) != 0) break;
                    dlogits = draft.last_logits();
                }
                if (proposals.empty()) { fprintf(stderr, "draft stalled\n"); break; }

                // 2. Target verifies ALL proposals in one batch decode,
                // positioned at the target's current KV length.
                std::vector<llama_token> vtoks = proposals;
                std::vector<llama_pos> vposs;
                std::vector<int8_t> vlogits(proposals.size(), 1);  // logits at every draft position
                llama_pos vbase = target.pos();
                for (int i = 0; i < (int)proposals.size(); i++) vposs.push_back(vbase + i);
                if (target.decode(vtoks, vposs, vlogits) != 0) { fprintf(stderr, "verify decode failed\n"); break; }

                // 3. Accept the longest greedy-consistent prefix. proposal[0]
                // must match the prediction from the LAST REAL decode
                // (prev_logits); proposal[i>0] matches the verify batch's
                // logits at position i-1.
                std::vector<std::vector<float>> verify_logits;
                for (int i = 0; i < (int)proposals.size(); i++) {
                    int nv = llama_vocab_n_tokens(target.vocab);
                    float* lg = llama_get_logits_ith(target.ctx, i);
                    verify_logits.emplace_back(lg, lg + nv);
                }
                auto argmax = [&](const std::vector<float>& lg) {
                    int best = 0;
                    for (int v = 1; v < (int)lg.size(); v++) if (lg[v] > lg[best]) best = v;
                    return best;
                };
                int n_accept = 0;
                for (int i = 0; i < (int)proposals.size(); i++) {
                    const std::vector<float>& lg = (i == 0) ? prev_logits : verify_logits[i - 1];
                    if (argmax(lg) == proposals[i]) n_accept++;
                    else break;
                }
                accepted += n_accept;
                rejected += (int)proposals.size() - n_accept;
                // 4. Emit the accepted tokens (they're already in both KVs),
                // capped at max_tokens like the baseline.
                for (int i = 0; i < n_accept && (int)out.size() < max_tokens; i++) out.push_back(proposals[i]);
                if (n_accept == (int)proposals.size()) {
                    // All accepted: the last proposal's logits predict the next
                    // token — decode it into the target KV (like the reject
                    // path's fix decode) so the next verify starts fresh.
                    tok = argmax(verify_logits.back());
                    if (target.decode({tok}, {}, {}) != 0) { fprintf(stderr, "allacc decode failed\n"); break; }
                    prev_logits = target.last_logits();
                } else {
                    // Rejected at n_accept: the prediction there IS the correct
                    // next token. Roll both KVs back to the accepted prefix and
                    // decode the fix token into the target KV.
                    const std::vector<float>& lg = (n_accept == 0) ? prev_logits : verify_logits[n_accept - 1];
                    tok = argmax(lg);
                    // Keep only the accepted prefix — the rejected proposal at
                    // vbase+n_accept must go, then the fix token lands exactly
                    // there (auto position = memory max + 1).
                    target.rollback(vbase + n_accept);
                    if (target.decode({tok}, {}, {}) != 0) { fprintf(stderr, "fix decode failed\n"); break; }
                    prev_logits = target.last_logits();
                    draft.rollback(draft.pos() - ((int)proposals.size() - n_accept));
                }
                if ((int)out.size() >= max_tokens) break;
                out.push_back(tok);
                // Sync the draft KV with the new token (accepted proposals are
                // already there — the draft proposed them).
                if (draft.decode({tok}, {}, {}) != 0) { fprintf(stderr, "draft sync failed\n"); break; }
                tpos = target.pos();  // resync from the actual KV length
                dpos = draft.pos();
            } else {
                // Greedy baseline: the next token is already predicted by the
                // last decode's logits; decode it to advance the KV.
                int nxt = greedy_argmax(prev_logits);
                if (target.decode({nxt}, {}, {}) != 0) break;
                prev_logits = target.last_logits();
                tok = nxt;
                out.push_back(tok);
                tpos++;
            }
            if (tok == llama_vocab_eos(target.vocab)) break;
        }
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return {out, secs};
    };

    // Baseline (greedy, no speculation).
    auto [base_out, base_s] = run(false);
    // Speculative.
    auto [spec_out, spec_s] = run(true);

    // Lossless check: identical token streams.
    bool lossless = (base_out == spec_out);
    auto piece = [&](llama_token t) {
        char buf[64]; int l = llama_token_to_piece(target.vocab, t, buf, sizeof(buf), 0, true);
        return std::string(buf, l);
    };

    fprintf(stderr, "\n══════════════════════════════════════════════\n");
    fprintf(stderr, "  draft=%s  target=%s  N=%d\n", draft_path.c_str(), target_path.c_str(), N);
    fprintf(stderr, "  baseline: %d tokens in %.2fs = %.1f tok/s\n",
            (int)base_out.size(), base_s, base_out.size() / base_s);
    fprintf(stderr, "  spec:     %d tokens in %.2fs = %.1f tok/s  (accept %d / reject %d)\n",
            (int)spec_out.size(), spec_s, spec_out.size() / spec_s, accepted, rejected);
    fprintf(stderr, "  lossless: %s\n", lossless ? "YES (identical greedy output)" : "NO — BUG");
    fprintf(stderr, "  baseline ids: ");
    for (auto t : base_out) fprintf(stderr, "%d ", t);
    fprintf(stderr, "\n  spec ids:     ");
    for (auto t : spec_out) fprintf(stderr, "%d ", t);
    fprintf(stderr, "\n  output:   ");
    for (auto t : spec_out) fprintf(stderr, "%s", piece(t).c_str());
    fprintf(stderr, "\n══════════════════════════════════════════════\n");

    return (lossless && !spec_out.empty()) ? 0 : 1;
}
