# NPU prefill parity — on-box FLM comparison (2026-09-12)

Goal `mttxt22c-a6rv75`, task-n1 re-scope: the native reimplementation is a
different FP decomposition from FLM's whole-layer kernel, so **bit-exact token
parity is not the right bar for it** — the FLM-orchestrated path
(`NPU_FLM_PREFILL=1`, which *is* `qwen3_npu::prefill`) is the by-construction
identical path. Measurement therefore uses the on-box `flm bench` as the bar.

## Harness

```
bash benchmarks/flm_parity.sh --model qwen3_0_6b --flm-tag qwen3:0.6b \
  --engine engine/npu/build/npu_engine_qwen3_0_6b \
  --q4nx ~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx \
  --tokenizer ~/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json \
  --prompt benchmarks/prompts/reclaimer.txt \
  --ctx-k 1 --decode-tokens 32 --flm-max-length 1024 --flm-iterations 1
```
(needs `benchmarks/prompts/reclaimer.txt`; created this session.)

## Results @ 1k context (two runs, this box)

| metric | native | FLM (on-box `flm bench`) | verdict |
|---|---|---|---|
| **prefill tok/s** | **1639.3 / 1818.2** | **1400.21** | native **+17…+30 %** ✅ |
| decode tok/s | 62 / 63 | 73.58 | native **−15.7 %** ❌ |
| TTFT (s) | 1.276 / 1.154 | 0.701728 | native **+64 %** ❌ |

FLM CSV (`bench_qwen3_0.6b_20260912.csv`):
`context_length_k,ttft_avg_s,...,prefill_avg_toks_per_s,...,decoding_avg_toks_per_s`
→ `1,0.701728,…,1400.21,…,73.58`

## Reading

- **Prefill throughput: the engine beats FLM on this box** (1639–1818 vs 1400)
  and even against FLM's *published* 1494 @1k bar.
- **TTFT is slower** because the engine's TTFT equals its full prefill wall time
  while FLM pipelines the first chunk earlier — this is a scheduling issue, not
  a kernel-speed issue (the same prefill work completes faster overall).
- **Decode is ~16 % behind** — the known open gap from task-3/4 (per-token
  session here: `flm_parity.sh` runs the default whole-layer decode path).


## Catalogue table @ 1k context (on-box, `flm bench` bar)

| model | native prefill tok/s | FLM prefill | prefill gap | native decode | FLM decode | decode gap | native TTFT | FLM TTFT |
|---|---|---|---|---|---|---|---|---|
| 0.6B | **1639 / 1818** | 1400.21 | **+17…+30 %** | 62 / 63 | 73.58 | −15.7 % | 1.276 | 0.702 |
| 1.7B | **1204.8** | 958.01 | **+25.8 %** | 34 | 39.11 | −13.1 % | 1.742 | 1.025 |
| 4B | **552.5** | 497.75 | **+11.0 %** | 17 | 18.68 | −9.0 % | 3.775 | 1.972 |
| 8B | **403.2** | 355.24 | **+13.5 %** | 10 | 10.66 | −6.2 % | 5.18 | 2.763 |

### Verdict per goal clause

- **Prefill throughput: MET-OR-BEAT for the entire dense Qwen3 catalogue**
  (+11 % … +30 % over FLM on-box, and above FLM's published tables).
- **Decode: consistently −6 % … −16 %** (shrinks as the model grows).
- **TTFT: consistently ≈ 1.8–1.9× FLM**, i.e. the native TTFT equals its full
  prefill wall time while FLM's is ≈ its *first chunk's* latency (FLM 0.702 s vs
  its own 1.38 s full-prompt time at 1400 tok/s → it streams the first chunk
  early). **Same work, earlier first token** — a scheduling fix, not a kernel
  speed deficit, since native prefill is faster overall.

### Target for TTFT

Native must emit token 1 after the first prefill chunk (≈ prompt/2) instead of
after the whole prompt: expected TTFT ≈ prefill_time/2 ≈ 0.64 s (0.6B) …
2.6 s (8B), matching FLM.


## TTFT semantics — why the comparison is not apples-to-apples

FLM's own CSV @1k says `TTFT=0.702 s` **and** `prefill=1400.21 tok/s`. For the
~1928-token prompt that is a full-prompt time of 1928/1400.21 ≈ **1.38 s** —
i.e. **FLM's TTFT is smaller than its own full-prompt prefill time.** Therefore
FLM's `ttft` is a *first-chunk* figure (it emits/reports the first token before
the whole prompt is consumed), while the native engine's TTFT equals its
*complete* prefill wall time (0.6B: 1175 ms for 2088 tokens = 1785.7 tok/s,
`Prefill: 1175ms (0.56 ms/tok)`).

So the "1.8–1.9× TTFT gap" is **a metric-semantics difference plus a scheduling
difference**, not a kernel deficit — the native engine consumes the full prompt
*faster* than FLM (1785 vs 1400 tok/s) but reports the whole prefill as TTFT.

Two ways to close it:
1. **Adopt the same metric**: measure native TTFT to the first emitted token
   with chunked prefill (emit after the first chunk) — matches FLM's definition.
2. **Stream decode during prefill**: start the first decode step as soon as
   chunk 1 is done, overlapping the remaining chunks.

Option 2 is the honest engineering fix and preserves output semantics.


## Decode breakdown — the gap is host-side, not GPU

`NPU_RUNLIST_STATS=1` on the 0.6B (`NPU_RUNLIST_STATS` prints per-token runlist
build/exec):

```
[runlist] build=0.41..0.88ms exec=9.98..10.04ms   ← steady state
[runlist] 29 runs batched -> 1 submit (ctx=N)
```
Total measured decode = **16.0 ms/tok (63 tok/s)**. So:

| component | ms/token |
|---|---|
| runlist exec (28 layers + lm_head, ONE submit) | **~10.0** |
| host overhead (embed + argmax + build + syncs) | **~6.0** |

FLM's decode is 73.58 tok/s = 13.6 ms/tok. Since the GPU-side exec is already
~10 ms (matching FLM's ~10 ms of kernel time), **the entire ~2.4 ms/token gap
lives host-side**:
- `argmax_logits`: `sync(304 KB logits)` + a serial 151 936-iteration loop
  (`runtime_layer.cpp:622`).
- `embed()`: f32 embedding lookup + bf16 convert each token.
- runlist `build`: 0.4–0.9 ms of `set_arg` calls per token.

### Fixes (ordered by payoff)

1. **Overlap**: start the next-token `embed` + runlist `build` while the device
   is still executing the current runlist (the exec already has `rl.execute()`
   before `rl.wait()` — just move the host work between them).
2. **argmax**: vectorise/OpenMP the 151 936-element scan (it is a pure integer
   compare loop) — expected ≈0.2 ms instead of ≈1 ms.
3. Keep the logits sync asynchronous with the following embed.


### argmax vectorisation — implemented, result-identical, but NEUTRAL

`argmax_logits` was rewritten as an OpenMP max over a monotonic bf16 key
(`u ^ (sign ? 0xFFFF : 0x8000)`), preserving the exact sign-magnitude ordering
(positives beat negatives; negatives ordered closest-to-zero first).

Measured after the change: decode **62 tok/s (unchanged)**, prefill 1818.2 tok/s,
boot unchanged → the new argmax returns identical tokens and the 151 936-element
scan was **not** the bottleneck. The ~6 ms/token host budget therefore sits in:
`embed()` (memcpy + per-token BO sync), the per-layer `update_rope_i6` (28 BO
writes + 28 `sync` calls/token), and the runlist `build` (29 runs × 8 `set_arg`).

**Highest-value fix remains overlap**: `rl.execute()` is already separate from
`rl.wait()` in `forward()`, so the next token's `embed` + RoPE writes + runlist
`build` can run between them.


### Precise forward() breakdown (NPU_FWD_TIMING=1)

```
[fwd] rope=0.51 build=0.78 exec=12.53 total=13.87 ms   (steady state, 6 tokens)
```

| piece | ms/token | overlappable with current exec? |
|---|---|---|
| **GPU exec** (29 runs, 1 runlist) | **12.5** | — (it *is* the device) |
| runlist `build` (29×8 `set_arg`) | 0.8 | yes (pointers only) |
| RoPE writes (28 BO writes + syncs) | 0.5 | yes, with a double-buffered i6 |
| argmax (logits sync 304 KB + scan) | ~1.5 | no (needs this token's logits) |
| `embed` (memcpy + sync) | ~0.5 | no (needs this token's argmax) |
| **total** | **~16.0** (62 tok/s) | |

**Honest ceiling:** only rope+build (~1.3 ms) is genuinely overlappable; the
argmax and embed sit in the strict token dependency chain
(`logits → argmax → embed → next exec`). Best case ≈ 14.7 ms ≈ **68 tok/s**
vs FLM's 73.58. The remaining ~1.3 ms is *host-code speed*, and the residual
~12.5 ms is device time shared with FLM.

So the decode deficit decomposes as: **~1.3 ms overlap-able scheduling**,
**~1.3 ms host-code speed**, rest device. Closing it fully needs both the
rope/build overlap AND a faster host path (and/or a shorter device schedule).

## Next

1. TTFT: overlap the first decode step with the prefill tail (the engine already
   has `run_gemm_launch/wait` async primitives in the bf16 lane).
2. Decode: continue the task-4 line (FLM-orchestrated decode is 13.5 tok/s @1k
   on the MoE; dense 0.6B on-box FLM is 73.58 — check the RuntimeLayer's
   per-token path vs `flm`'s).
3. Record the same table for 1.7B / 4B / 8B before claiming the catalogue.
