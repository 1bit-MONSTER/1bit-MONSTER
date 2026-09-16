# Family attention via the AttnCtx adapter (Nanbeige), 2026-09-16

## The ABI gap

The generated family attention kernel (`n1_core_attn.py`, nh20/nkv4/hd128, N=2048)
has the **AttnCtx** signature — `attn(k(3, instrBO, instrSize, Q, KT, C2, V, SCR))`,
the same ABI `zaya_decode.cpp` drives. `Bf16Mm::run_attn` drives a *different*
kernel: FLM's captured `(act, out, kv)` ABI. So pointing Nanbeige's bf16 prefill at
the generated ELF cannot work by construction, and the prefill fell through to the
captured short-context ELF / the CPU fallback (the doc-level "not through that ELF").

## The adapter

`engine/npu/src/npu_engine_universal.cpp` now carries an `NPU_ATTN_CTX=1` path in
the bf16 prefill. When the shape is nh20/nkv4/hd128 it constructs the same `AttnCtx`
`zaya_decode.cpp` uses (`NPU_ATTN_XCLBIN` / `NPU_ATTN_INSTS` / `NPU_ATTN_MAX_SEQ`)
and, because the generated kernel is decode-shaped (M=8), issues one query row per
call over the prefill block, writing the float result back to `bA` as bf16. It is a
**correctness** path (much slower than the captured short-context ELF), gated off by
default.

The float Q/K/V are already present (`bqo` after host q_norm/k_norm + RoPE, and
`kv_caches[l][0].k/v`), so no new packing was needed.

## Measurements (device lock held; NPU_ATTN_KV_REGION=4194304)

```
NPU_ATTN_CTX=1 NPU_PREFILL_BF16=1 NPU_BF16=1 NPU_ATTN_KV_REGION=4194304 \
NPU_PREFILL_MAX=2048 NPU_ATTN_COLS=4 \
NPU_ATTN_XCLBIN=.../attn_gen_2048_nh20_hd128.xclbin \
NPU_ATTN_INSTS=.../attn_gen_2048_nh20_hd128_insts.txt NPU_ATTN_MAX_SEQ=2048 \
  ./engine/npu/build/npu_engine_nanbeige4_1_3b <model.q4nx> 4 <ids.txt>
```

| prompt | keys | prefill | boot | notes |
|---|---:|---:|---:|---|
| repeated passage (raw ids) | 1172 | 340.4 s | **13** | 32/32 layers `[NPU_ATTN_CTX]`, attn 338 s, 0 crash |
| `/tmp/nb_ids1500.txt` | 1500 | 457.8 s | 152343 | 32/32 layers, attn 456 s, 0 crash |

No hang and no ERT at either length — the AttnCtx ABI drives the generated kernel
end to end, which is the ABI-adapter result the family lane needed.

## Identity vs `flm run nanbeige4.1:3b`

Same passage, same model dir, `flm run nanbeige4.1:3b -c 4096`:

- FLM reports `Prefill chunk 1/1 with 1202 tokens` and its first generated token is
  `<n` (its markup for a newline) = **token 13**. Our path's boot is **13**. First
  token matches.
- **Caveat (why this is not yet a strict identity):** FLM applies the model chat
  template (1202 tokens) while the engine consumed the raw 1172 ids. A strict
  token identity needs the template-tokenised ids on both sides. Also the earlier
  1500-key run was not compared against FLM (only the 1172-key one was).

## Boundary

- With the CPU / captured-ELF fallback the same prompts give context-free-looking
  boots (13 and 51752 across runs) — the signature the family doc recorded. The
  AttnCtx path is the first family path that runs the **generated** attention.
- The `NPU_ATTN_CTX` default xclbin path points at the goal tree's
  `engine/npu/xclbins/`, which does not carry the nh20 build yet — pass
  `NPU_ATTN_XCLBIN`/`NPU_ATTN_INSTS` (the family worktree's artifacts) until it is
  vendored.
