# Speculative Decoding Demo (roadmap Phase 2 core loop)

Lossless speculative decoding: a draft model proposes N tokens, the target
verifies them all in ONE `llama_decode` batch, and the longest
greedy-consistent prefix is accepted. Output is bit-identical to greedy
(verified: `lossless: YES`).

## Build & run

```
cmake --build build --target spec_decode -j8
./build/spec_decode models/Qwen3-0.6B.Q8_0.gguf \
    models/Llama-3.2-1B-Instruct.Q8_0.gguf "What is 2+2?" [n_draft=4] [max_tokens=64]
```

## How it works

1. Prefill both contexts with the prompt (single-token decodes — this
   llama.cpp fork's KV rejects multi-token batches).
2. Target generates the first token; its logits are the acceptance anchor.
3. Draft proposes N tokens from its own logits (no re-decode).
4. Target decodes all N proposals in ONE batch with per-position logits.
5. Accept while `argmax(target_logits_at_i) == draft_token_{i+1}`.
6. On rejection: roll the target KV back to the accepted prefix
   (`llama_memory_seq_rm`), decode the fix token at the rejection position,
   resync the draft KV, repeat.

## Caveats

- **Lossless vs greedy** — the acceptance is exact-argmax comparison.
- **Speedup requires a well-matched draft** (same family/precision). The
  demo's cross-family pair (Qwen3-0.6B → Llama-1B) has ~15-30% acceptance,
  so it runs slower than the baseline. With a matched pair the acceptance is
  ~90%+; the batched verify then amortizes the target's cost. The roadmap's
  NPU path (batched INT8 verification) is where the real speedup lands.
- This fork's multi-output batch logits have a quirk at the EOS boundary
  (same-model runs can diverge on the final token); the cross-model case is
  clean across all tested prompts.

The loop is intentionally standalone — it's also wired into the unified
server: the ggml-vulkan backend implements the three primitives
(`decode_one` / `verify_batch` / `rollback`) and the server runs the loop
when launched with a draft model:

```
./build/1bit unified -p 8088 -w ./models -m Llama-3.2-1B-Instruct \
    --draft-model ./models/Qwen3-0.6B.Q8_0.gguf --spec-decode
```

The draft runs as a second side-by-side ggml-vulkan backend (any smaller
GGUF); the served model is the target. Per-request behavior:

- Both contexts are reset and prefilled with the same prompt tokens
  (single-token decodes — the fork's KV rejects multi-token batches
  outside the verify step).
- The first output token is the TARGET's greedy argmax; the draft's logits
  are kept separate (draft and target have different vocabs).
- Each round: draft proposes N greedy tokens → target verifies them in
  ONE batch → longest greedy-consistent prefix accepted → rejected
  positions rolled back (`llama_memory_seq_rm`) with the fix token
  re-decoded in place, both KVs re-synced.
- If the target backend can't do `verify_batch` (e.g. NPU FLM), the
  request falls back to the normal loop with a stderr notice.

Caveats
-------

- **Lossless vs greedy** — the acceptance is exact-argmax comparison.
- **Speedup requires a well-matched draft** (same family/precision). The
  demo's cross-family pair (Qwen3-0.6B → Llama-1B) has ~15-30% acceptance,
  so it runs slower than the baseline. With a matched pair the acceptance is
  ~90%+; the batched verify then amortizes the target's cost. The roadmap's
  NPU path (batched INT8 verification) is where the real speedup lands.
- This fork's multi-output batch logits have a quirk at the EOS boundary
  (same-model runs can diverge on the final token); the cross-model case is
  clean across all tested prompts.
- The server's normal path samples (temp 0.8); the spec path is greedy, so
  outputs can legitimately differ per request — the spec output is what
  greedy would produce.
