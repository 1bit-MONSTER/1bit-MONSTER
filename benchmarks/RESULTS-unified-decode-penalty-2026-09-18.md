# The unified decode penalty: 3.3 ms/token, device-side, in the bf16→runlist KV handoff

Goal `mu35shsg-i3hlyi`, criterion (c). Follows
`RESULTS-decode-window-warmup-2026-09-18.md`, which showed the native decode is ~1–7%
behind FLM at a warm-up-free window and that this is now the clause (c) fails on.

This localises that shortfall to one place, and shows it is recoverable: **the runlist
decode executes ~3.3 ms/token faster when the KV it reads was written by the runlist prefill
than when it was written by the bf16 prefill** — same engine binary, same prompt, same decode
length, only the prefill path different.

## Measurement

Qwen3-0.6B, `/tmp/p_1k.txt` (1024 ids), `NPU_RUNLIST_STATS=1`, decode 16 tokens:

| prefill path | device exec / token | host build / token | decode | prefill |
|---|---:|---:|---:|---:|
| bf16 (unified, default) | **15.46–15.94 ms** | 1.34–1.97 ms | 17.1 ms/tok (**59** tok/s) | 0.55 ms/token |
| runlist (NPU_RUNLIST=1) | **12.48–12.50 ms** | 1.43–1.83 ms | 12.2 ms/tok (**82** tok/s) | 14 ms/token |

And at 32 decode tokens for the comparable rows:

| configuration (1k) | decode | FLM on-box |
|---|---:|---:|
| pure runlist (runlist prefill + runlist decode) | **80 tok/s** | 75.25 |
| unified (bf16 prefill + runlist decode) | 69.9 tok/s | 75.25 |

**The host side is identical** (build ~1.5 ms/token in both), so the difference is entirely
device execution: the runlist's per-token kernels do ~3.3 ms more work, or wait ~3.3 ms
longer, when the KV BO was produced by the bf16 prefill.

## Why this matters for criterion (c)

The pure runlist decode (80 tok/s at 1k) **beats FLM (75.25)**. The default path does not
(69.9), and the default path is the one the criterion measures because its prefill/TTFT are
the reason it is the default (0.535 ms/prompt-token against the runlist's 14 ms).

So the decode clause is not blocked by the decode kernels; it is blocked by the **handoff**.
Recovering those 3.3 ms/token would make the default path's decode >= FLM *while keeping*
prefill at 1.33x and TTFT faster at 1k — i.e. it would close criterion (c) for Qwen3-0.6B
outright, and likely for the H=2560 pair (which shows the same 0.99x at 1k).

## Candidate mechanisms (named, not tested)

The bf16 prefill drives the captured FLM attention ELF (`Bf16Mm::run_attn`) and writes
`kv_caches[l][0].k/v`; the runlist path writes its KV through the per-ctx whole-layer ELF.
Both then hand over to the same `RuntimeLayerEngine` decode, so the difference is in what the
decode reads:

1. **KV layout/quantisation**: if the bf16 prefill's KV is stored in a wider or differently
   packed form than the runlist ELF expects, every decode step's KV read spans more memory
   (or a per-step repack happens on device). The `KV_REGION` knob (`NPU_ATTN_KV_REGION`) and
   the `AttnCtx` N-split work in `RESULTS-family-attn-ctx-adapter-2026-09-16.md` are the
   existing body of knowledge on this layout.
2. **KV length/alignment**: the bf16 prefill caps at 8185/8161 of the requested tokens
   (`NPU_PROMPT_MAX`), so the runlist session may be advancing a position counter that does
   not match the KV actually present, costing a padded row read per step.
3. **BO residency**: the unified path holds the bf16 context BOs alive alongside the runlist
   session (the engine's own note: "the bf16 prefill block … hands its final hidden and its KV
   to the runlist session and `_exit()`s from there"), so the decode may contend for device
   memory/banks it does not need.

The cheapest discriminator is (1): dump the first 4 KB of the KV BO after each prefill and
compare the two byte streams for the same prompt and position — a `NPU_ATTN_DUMP`-style
comparison the `AttnCtx` work already has hooks for.


## Code reading refutes the "extra keys" hypothesis (2026-09-18, same session)

The first guess from the 12.5 -> 15.8 ms arithmetic was "the unified decode attends over
more keys": 15.8 ms is about the pure-runlist rate at ~2048 keys (15.1 ms), so a doubled
key count would fit. Reading both paths shows that is **not** what happens — the key count
is identical:

```cpp
// unified (npu_engine_universal.cpp, bf16 prefill -> runlist decode)
sp += npt;                       // sp = prefill length
int ctx = sp;                    // = 1024
... npu_runlist_forward(++ctx, lg, NV);      // first forward at ctx = 1025

// pure runlist (npu_runlist_bridge.cpp, npu_runlist_decode)
for (int t : ids) { rt.embed(t); rt.forward(++ctx); }   // ctx = 1024 after prefill
int c1 = ctx + 1;                // = 1025
```

Both arms issue their first decode forward with `ctx == 1025`, and the per-ctx ELF attends
the same 1025 keys. So the 3.3 ms/token is **not** more attention work. It is also not a
second prefill: the bf16 arm's reported prefill is 548 ms (0.535 ms/token), and the decode
block calls `npu_runlist_lmhead` then `npu_runlist_forward` directly — it never re-enters
`npu_runlist_decode`.

That leaves the two remaining candidates, now ordered by cost to test:

1. **KV BO contents/layout in the handoff.** The bf16 arm writes KV from the host through
   `npu_runlist_write_kv(l, sp, npt, bKv.data(), kv_region)`, which re-packs when the source
   region stride differs from the session's (`src_region_stride_u16 != g_sess_kv_region_u16`,
   e.g. 4 MB nh16 source into the session's 8 MB region). If the host write lands at an offset
   or interleave the device prefill does not use, the decode's KV reads span a different
   footprint for the same key count. Cheap test: the `RT_KV_DUMP_DIR` / `RT_DUMP_POST` hooks in
   `runtime_layer.cpp` already dump `kv_bos_[0]` (32 MB) — dump it after each prefill and diff
   the two byte streams for the same prompt and position.
2. **Device-memory / BO residency.** The unified process keeps the whole bf16 GEMM context
   alive (per-layer weight BOs, `bA`/`bC`, the attention ELFs) alongside the runlist session's
   per-layer 32 MB KV BOs. A different BO address map can change bank/channel distribution and
   hence kernel exec time on this part. Cheap test: free the bf16 BOs before entering the
   unified decode and re-measure the same 16-token window.

Neither is run here; both are named with their instrument. This correction matters because
candidate (1) as originally written in this document ("KV layout/quantisation") and the
length-based reading are different diagnoses with different fixes.


### Instrument caveats for candidate (1) — read before running the dump

`RT_KV_DUMP_DIR` fires at the **start** of `RuntimeLayerEngine::forward(ctx_len)` and writes the
full 32 MB of `kv_bos_[0]` to `<dir>/kv_ctx<ctx_len>.bin`. Two consequences, both of which
must be handled or the diff will be read wrong:

- **Volume.** It fires on *every* forward call, so a 1024-token prompt writes 1024 files of
  32 MB (32 GB) per run. `/tmp` is tmpfs on this box and a previous lane already wedged every
  agent tool by filling it; use a real-disk directory and a "keep only the newest file"
  watcher (`while :; do ls -t | tail -n +2 | xargs -r rm -f; sleep 0.2; done`), then read the
  surviving file.
- **The two paths dump different points.** The pure-runlist decode calls
  `build_runlist`/`execute_runlist`, **not** `forward()`, so its last dump is
  `kv_ctx1024.bin` — the BO *before* the 1024th token executes, i.e. 1023 tokens of KV. The
  unified path's `npu_runlist_forward(++ctx, …)` *does* call `forward()`, so its first dump is
  `kv_ctx1025.bin` — the full 1024-token KV written by the bf16 handoff. Comparing "the same
  ctx file" across the two runs therefore compares 1023 vs 1024 tokens of KV, not the same
  state. To make them comparable, add a one-line dump call right after the pure path's prefill
  loop (or use `RT_DUMP_POST` with the same alignment), rather than diffing `kv_ctx<N>.bin`.

## Status

Criterion (c) remains unmet. What changed here is that its remaining decode shortfall is now
a **named, ~3.3 ms/token device-side handoff cost with a measured beat-FLM target behind it**,
rather than "native decode is a few percent slower". That is a bounded engineering target,
not a mystery.
