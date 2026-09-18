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


## MEASURED (2026-09-18): the KV handoff is topologically identical — the penalty is device state, plus a lost decode overlap

Two env-gated dumps were added and used (inert by default):

- `RuntimeLayerEngine::dump_kv_bo(path, n)` — syncs the session's `kv_bos_[0]` back and writes it.
- `NPU_KV_DUMP_PURE=<path>` in `npu_runlist_decode` (dumps the **local** engine's BO — note
  `g_sess_rt` is null on this path, so the bridge helper cannot be used here), and
  `NPU_KV_DUMP_UNIFIED=<path>` in the unified path right after `npu_runlist_write_act`.

Both arms ran the same 1024-id prompt, 0.6B, and dumped exactly the same logical point: the
KV BO with all 1024 prompt tokens, before any decode exec. Result over the layer-0 32 MB BO
(session region stride 8 MB, 1024 B per token; valid span 1 MB per region):

| comparison | differing bytes |
|---|---:|
| same position, region 0 valid span | **47.8%** |
| ±1-token shift (pure[0:] vs uni[1MB:]) | 81.6% (worse) |
| identical 1024-byte token blocks | **0 / 1024** |
| tail beyond the valid span | 0 |

So the two streams are **aligned by token** (a shift makes it worse, not better), occupy the
same offsets and the same span, and are equally populated (~1.046M non-zero bytes each); the
differences are confined to low-order bits of the bf16 values, e.g. token 0 starts
`pure 234,62 …` / `uni 230,62 …` (0x3EEA vs 0x3EE6) and `pure 165,189` / `uni 75,189`
(0xBDA5 vs 0xBD4B). **The KV topology and layout are identical; only the values differ,
slightly.**

**Therefore the KV handoff content is NOT the cause of the exec-time penalty** — identical
shape, offsets, span and key count cannot make the same kernels run 27% longer. What the
numbers do decompose into is two effects:

| | pure runlist | unified | delta |
|---|---:|---:|---:|
| device exec / token | 12.48 ms | 15.84 ms | **+3.36 ms** |
| host build / token | ~1.5 ms | ~1.5 ms | 0 |
| reported decode / token | **12.2 ms** | **17.1 ms** | +4.9 ms |

1. **+3.36 ms of device exec** is a property of the *process's device state*, not the data:
   the unified process keeps the entire bf16 GEMM context alive (per-layer `layerB` weight
   BOs, `bA`/`bC`, the bf16mm device weight BOs from `bf16mm_dequant_dev` — hundreds of MB)
   alongside the runlist session's 28 × 32 MB KV BOs. A different BO address map on this part
   can change bank/channel utilisation for the same kernels. Next test: free the bf16 BOs
   before entering the unified decode and re-measure the same 16-token window.
2. **~+1.3–1.5 ms of lost overlap** comes from the code path, not the device. The pure
   decode uses `build_runlist(slot)` / `execute_runlist(slot)` on alternating slots, so the
   next token's host build overlaps the current token's device exec (12.2 reported against
   12.48 exec — the build is hidden). The unified decode calls
   `npu_runlist_forward(++ctx, …)` → `RuntimeLayerEngine::forward()`, which is:

   ```cpp
   if (!build_runlist(0, ctx_len)) return false;
   if (!execute_runlist(0)) return false;
   if (!wait_runlist(0)) return false;      // single slot, strictly serial
   ```

   so it pays build + exec. Making the unified decode use the double-buffered slots recovers
   this ~1.3–1.5 ms/token; it does **not** close the gap on its own (≈15.9 ms/token → ≈63
   tok/s, still under FLM's 75.25), which is why effect 1 is the priority.

**Net:** the 3.3 ms/token is real and reproducible, the KV content is ruled out by
measurement, the key count was already ruled out by reading the code, and the remaining
mechanism is device-memory/BO residency — with a second, independently fixable ~1.4 ms/token
of lost decode overlap on top. Two targets, both named, one of them a small code change.


## WITHDRAWN (2026-09-18, later the same session): the 3.3 ms "handoff penalty" was cross-window variance

After rebuilding, the same unified configuration measures **12.52–12.61 ms/token of device
exec**, not the 15.46–15.94 ms that the earlier window reported. Four consecutive 1k/16-token
runs and three 1k/32-token runs on the current binary:

| configuration (Qwen3-0.6B, 1k, current binary) | decode ms/tok | tok/s |
|---|---:|---:|
| unified (bf16 prefill + runlist decode), 16 tok ×4 | 13.5, 13.5, 13.5, 14.0 | 74, 74, 74, 71 |
| unified, 32 tok ×3 | 13.9, 13.9, 14.0 | 72, 72, 71 |
| pure runlist, 32 tok ×2 | 12.8, 13.0 | 78, 77 |
| FLM on-box, 32 tok (this window) | — | **72.33** |
| FLM on-box, 32 tok (earlier window) | — | 75.25 |

So:

- **The unified path's exec is not 3.3 ms slower than the pure path's.** Both are ~12.5 ms
  in the current binary (pure: reported 12.8–13.0 ms/tok; unified: 13.9–14.0, the residual
  being the lost build overlap). The earlier 15.46–15.94 ms reading was stable *within* that
  process but did not reproduce *across* builds/sessions — the numbers are stable within a
  process and vary by up to ~27% between them, which is the signature of device/BO
  allocation state (other engines resident, allocation order), not of the KV handoff.
- **The A/B intended to remove it found nothing to remove**: `NPU_UNIFIED_FREE_BF16=1`
  (release the bf16 weight/scratch BOs and hw contexts before the decode) gives
  12.51–12.55 ms exec against 12.52–12.61 baseline — no effect, as expected if there was no
  penalty. The hook is kept, inert and opt-in, as a tested negative.
- **The KV-content comparison is unaffected**: it independently ruled out the handoff *values*
  (48% low-order-bit differences, identical offsets/span/topology), which remains valid.
- **The "decode clause fails for 0.6B at 0.93x" reading is superseded.** In the current
  window native unified is 71–72 tok/s against FLM's 72.33 (≈0.98–1.00x) and pure runlist is
  77–78 (≈1.07x). FLM itself moved 75.25 → 72.33 between windows. The honest statement at 1k
  is therefore **at parity within measurement variance (~0.98–1.00x), not a decisive deficit**
  — and *any* decode comparison here needs repeated runs and a quiet device before it is
  cited, which is the same lesson as the 8-token warm-up correction.

The two candidate mechanisms this document listed (KV layout, BO residency) are both now
closed: layout/values by the byte diff, residency by the A/B. What remains true is the
smaller, code-path finding: the unified decode uses `RuntimeLayerEngine::forward()`
(single-slot `build → execute → wait`), so it does not hide the ~1.0–1.5 ms/token host build
the way `npu_runlist_decode`'s alternating slots do. That is worth ~1.0–1.5 ms/token, is
reproducible in the numbers above (12.5 ms exec vs 13.9–14.0 ms/tok), and is the only
remaining decode optimisation named here — **now fixed**: the unified decode was routed
through the alternating slots and measures 78–80 tok/s at 1k against 72 serial
(`RESULTS-unified-decode-overlap-2026-09-18.md`).

## Status

**Correction: the headline of this document no longer stands.** There is no reproducible
3.3 ms/token device-side handoff penalty; the reading that produced it did not survive a
rebuild, and the A/B that was meant to remove it measured nothing (12.51 vs 12.52 ms exec).
What survives is (a) the KV-content exclusion, (b) the ~1.0–1.5 ms/token lost-overlap cost of
using the single-slot `forward()` in the unified decode, and (c) the process-level finding
that runlist exec time varies up to ~27% between builds/sessions, which means every decode
comparison in this lane must be repeated before it is cited.

At 1k/32 tok in the current window the default unified decode is 71–72 tok/s against FLM's
72.33 (~0.98–1.00x) and the pure runlist is 77–78 (≈1.07x), so the criterion-(c) decode
clause is at parity within variance rather than decisively failed for Qwen3-0.6B.

Instrumentation landed: `RuntimeLayerEngine::dump_kv_bo`, `NPU_KV_DUMP_PURE`,
`NPU_KV_DUMP_UNIFIED`, and `NPU_UNIFIED_FREE_BF16` (all inert unless set).
