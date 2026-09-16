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

## Addendum: the strict (template-matched) identity FAILS

The 1172-token raw-id run's "boot 13 == FLM's first token 13" was a **coincidence**
(newline is a common opener); the strict test resolves it.

Building the prompt with the model's own chat template (`tokenizer_config.json`'s
`chat_template`, rendered with jinja2) yields **1202 tokens** — exactly the count
`flm run nanbeige4.1:3b` reported for the same passage. Running the engine on those
ids:

```
NPU_ATTN_CTX=1 ... /tmp/nb_ids_tmpl.txt   (1202 ids, template-matched)
-> Prefill 1202 [bf16], 32/32 layers [NPU_ATTN_CTX], attn 334 s
-> boot = 764
```

FLM's first generated token on the same passage is **13**. So with the tokenisation
matched, the engine's first token is **764 vs FLM 13 — NOT identical.** The earlier
"first token matches" line must be read as raw-id-only and is superseded by this.

Conclusion for the family status: the AttnCtx adapter is a working **vehicle** (the
generated kernel drives a real prefill, 32/32 layers, no hang/ERT, and the kernel
itself gates NPU==EMU on the bench), but Nanbeige is **not at token parity**: at
>1024 keys the end-to-end prefill diverges from FLM (764 vs 13). The divergence is
downstream of the attention kernel — candidates are the bf16 prefill's Q/K/V
normalisation/RoPE ordering relative to the kernel's pre-RoPE contract, the GDN
layers in the 32-layer stack, and the chat-template system prefix (the template is
Nanbeige's tool-calling template, whose rendered 1202 tokens include a system
section that may not be what FLM actually feeds).

So: cited **partial** for Nanbeige — attention ABI route established and driven;
token parity at >1024 keys NOT established, with the mismatch pinned to the prefill
stack rather than the kernel (the kernel is separately gated).

## Addendum 2: the AttnCtx path is NON-DETERMINISTIC in the engine (blocker)

Chasing the 764-vs-166103-vs-13 discrepancy exposed something more basic: **the
`NPU_ATTN_CTX` path does not return the same answer twice.**

Same 8-token prompt, byte-identical command, repeated:

| path | runs |
|---|---|
| `NPU_ATTN_CTX=1` (generated kernel via AttnCtx) | 152369, 152367, 8900, 6037, 152360, 152552, **2092** |
| captured-ELF bf16 prefill (no AttnCtx) | 152343, 152343 — **stable** |
| plain fallback (no bf16 prefill) | 152367, 152367 — **stable** |

Ruled out:

- **Device contention** — no other process on `/dev/accel/accel0`, no new
  `Firmware timeout state capture` in dmesg during these runs, `device.lock` held.
- **Host-thread race on the KV cache** — `NPU_HOST_THREADS=1` still varies
  (152360 / 152552 / 2092).
- **A stale-KV cache in `AttnCtx::run`** — `repack` is true on every call here
  (`seq` strictly increases within a layer, and `kv_caches[l]` moves per layer).
- The kernel itself syncs (`r.wait(); bC2->sync(FROM_DEVICE)`), and on the standalone
  bench the same AttnCtx reproduces NPU==EMU to 8.575258e-02 across runs.

So the variance is introduced by running AttnCtx **interleaved with the engine's
other NPU contexts** (the bf16 GEMM contexts on the same device) — a condition the
standalone bench never exercised. It is not a property of the generated kernel.

**Consequence: no token identity is claimable for Nanbeige.** The adapter is a
working *vehicle* (32/32 layers, no hang, no ERT, and the kernel gates NPU==EMU on
the bench), but this path cannot support a parity claim until the variance is
root-caused. The earlier "boot 13 == FLM 13" line is fully retracted; the 1202-token
template-matched run (764) and the CPU-attention run (166103) are both unreliable
**for the AttnCtx side**, though the CPU-attention run is itself deterministic.

Next diagnostic (if resumed): dump the AttnCtx's layer-0 output inside the engine
and compare against the host packing for the same row, and check the engine's
hw_context count at the time AttnCtx launches (`hwctx_limit=16`) — if the engine
already holds its full complement of contexts, the AttnCtx launch may be silently
degraded. Also worth testing: create the AttnCtx **before** the bf16 contexts, and
force an explicit device-level sync between the last GEMM and the AttnCtx launch.
