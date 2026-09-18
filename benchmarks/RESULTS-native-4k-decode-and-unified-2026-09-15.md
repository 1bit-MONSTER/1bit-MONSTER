# 4k context: decode, and the unified path that makes the fast prefill reachable (2026-09-15)

Follow-on to `RESULTS-native-4096-context-2026-09-15.md`, which put the *prefill*
of the dense-Qwen3 family on the NPU at 4k. This one measures **decode** at 4k
(never measured before) and closes the larger gap that measurement exposed: the
two fast halves of the engine could not be used together.

## 1. Decode at 4095 — the runlist path, 8 tokens

`NPU_RUNLIST=1` with the per-context ELF sets regenerated to ctx 4200
(`~/npu-build/elfs-4k-*`). Reference is FLM's own forward in the same harness
(`NPU_FLM_PREFILL=1 NPU_FLM_DECODE=1`), same prompt, same token count.

| model | native dec | native prefill | FLM dec | FLM prefill | native / FLM dec |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 17.7 ms/tok (57) | 70417 ms | 20.6 (49) | 1927 ms | **1.16×** |
| Qwen3-1.7B | 29.2 ms/tok (34) | 126607 ms | 34.0 (29) | 2939 ms | **1.16×** |
| Qwen3-4B | 55.2 ms/tok (18) | 248843 ms | 64.1 (16) | 6471 ms | **1.16×** |
| Qwen3-8B | 90.4 ms/tok (11) | 414851 ms | 104.2 (10) | 9056 ms | **1.15×** |
| Qwen3-VL-4B | 55.2 ms/tok (18) | 247866 ms | 64.1 (16) | 6348 ms | **1.16×** |

**Decode holds its ~1.16× lead at 4k, and the tokens are right**: 1.7B, 4B, 8B
and VL-4B each produced **8 of 8 tokens identical to FLM**, token for token, from
the boot onward. 0.6B differs at the boot token only, at the documented unstable
position (see §5).

**But the prefill is the story.** 70 s to 415 s against FLM's 1.9 s to 9.1 s —
**36× to 46× worse TTFT** on the only path that delivers that decode. At 1k the
same effect is already 13.8 s against FLM's 0.79 s (17×), and this is the
path a plain `npu_engine_<model> model.q4nx N ids` invocation takes with no
environment set.

Why: the runlist prefill is **one whole-layer forward per prompt token**, so it
costs decode speed per token (17 ms/tok at 0.6B, 101 ms/tok at 8B). The engine
has a fast prefill too — the bf16 path — but on its own it stops before decode
(`bf16-only: int8 ctxs unavailable — prefill+boot done, skipping int8 decode`).

That is the real state at 4k: **fast prefill OR fast decode, and the default is
the slow-prefill one.**

## 2. The unified path, and why it was unreachable

The combination already existed in the tree: the bf16 block writes its final
hidden and its KV into a `RuntimeLayerEngine` session and `_exit()`s from a
decode loop that uses the runlist's own `forward` (`npu_engine_universal.cpp`
§4676-4710). It is selected by `NPU_UNIFIED=1`.

It did nothing. `npu_runlist_decode` runs earlier (`:733`) and `return 0`s on
success, so execution never reached the bf16 block; with `NPU_RUNLIST` unset the
guard `(!rl || atoi(rl) != 0)` is true, so the flag had no effect at all.
Reaching it required knowing to also pass `NPU_RUNLIST=0`, which the flag's name
does not suggest.

Fixed (commit `53005444b`) with a deliberately narrow exception: the flag is
honoured only together with `NPU_PREFILL_BF16=1`, because the unified session is
initialised *inside* the bf16 block — and skipping the runlist block for
`NPU_UNIFIED` alone would silently demote the fast DECODE to the 112-launch split
path, i.e. a flag typo would cost both halves. Set alone it now says so:

```
[unified] NPU_UNIFIED=1 needs NPU_PREFILL_BF16=1 (the unified session is
initialised inside the bf16 prefill block) — ignoring it and using the runlist path
```

## 3. A second bug: the KV handoff used the wrong region stride

With the flag reachable, the unified decode was still wrong — and wrong in a way
that named its own cause. **nh16 models were correct and nh32 models were not:**

| model | shape | unified tokens (before fix) | FLM |
|---|---|---|---|
| Qwen3-0.6B | nh16 | correct | — |
| Qwen3-1.7B | nh16 | correct, 8/8 | 8/8 |
| Qwen3-4B | nh32 | `59277 27737 65 82 …` | `59277 16 4489 58907 …` |
| Qwen3-8B | nh32 | `44353 75 437 10692 …` | `44353 16 4489 58907 …` |
| Qwen3-VL-4B | nh32 | wrong | — |

The boot token was right in every case; everything after it was not. That is the
signature of a handoff fault rather than a compute fault, because the prefill
consumes the buffer directly and only the *decode* reads it back through the
session.

`npu_runlist_write_kv` copied four regions at the **session's** stride (8 MB,
`g_sess_kv_region_u16`, hardcoded). The buffer it was handed was laid out by the
bf16 attention at the **attention ELF's** stride, and that one is per-shape:

| shape | model | `kv_region` (u16) | bytes |
|---|---|---:|---:|
| nh16 | 0.6B (H=1024), 1.7B (H=2048) | 4194304 | 8 MB |
| nh32 | 4B/VL-4B (H=2560), 8B (H=4096) | 2097152 | 4 MB |

So for nh32 region 0 was copied correctly and regions 1..3 were read from double
the offsets they were written at — the K heads in the second region and all of V.
**4194304 / 2097152 is exactly the pair that separates the working models from
the broken ones**, which is what makes this the cause and not a correlate.

Fixed (commit `c300aa2f5`): `npu_runlist_write_kv` now takes the caller's region
stride and **re-packs region by region** when it differs from the session's; when
they match the copy is unchanged. The first call logs the translation.

This is not a one-off mismatch: every runlist layer ELF is baked at
`MAX_L=8192` → 8 MB, while the attention captures are baked at their own
context, so the two strides are a permanent property of the two files.

## 4. Result — native beats FLM at 4k on both metrics, every size

Unified path, npt=4095, 8 decode tokens, against FLM's own runtime:

| model | native prefill | FLM prefill | native dec | FLM dec | tokens |
|---|---:|---:|---:|---:|---|
| Qwen3-0.6B | **1725 ms** | 1927 ms | **19.8** (51) | 20.6 (49) | 7/7 after the unstable boot |
| Qwen3-1.7B | **2503 ms** | 2886 ms | **30.9** (32) | 33.9 (30) | **8/8** |
| Qwen3-4B | **5556 ms** | 6464 ms | **58.6** (17) | 64.2 (16) | 7/8 (step 8 is a tie) |
| Qwen3-8B | **8070 ms** | 9029 ms | **94.2** (11) | 104.3 (10) | **8/8** |
| Qwen3-VL-4B | **5566 ms** | 6528 ms | **58.5** (17) | 64.2 (16) | **8/8** |

Prefill gains 1.10–1.18×, decode 1.05–1.11×, and **TTFT at 4k goes from 70–415 s
to 1.7–8.1 s** — the prefill is **25× faster at npt=1024 and 40× at npt=4095**
than the runlist path it replaces (0.6B: 13783 → 546 ms, 70417 → 1755 ms).

## 5. Caveats, stated rather than buried

- **It is opt-in.** The default is still the runlist path, i.e. the 36–46× TTFT.
  Making the combination the default for dense Qwen3 is a behaviour change that
  has not been measured broadly enough to make here.
- **Unified decode is 10–20% slower than runlist decode** (0.6B: 19.8 vs 17.7
  ms/tok at 4k; 12.5 vs 11.4 at 1k). It buys a 25–40× prefill for a ~15% decode,
  which is the right trade for interactive use and the wrong one for long
  generation with a short prompt.
- **4B step 8 is a tie** (native 16, FLM 57). The runlist path returns 57 there,
  so both values are reachable by legitimate numerics; 7 of 8 is the measured
  result, not 8 of 8.
- **0.6B at npt=4095 is an unstable position** — the same documented class as
  npt=1280. FLM and native-with-CPU-attention say 59277; the byte-exact int8
  runlist path and native-with-NPU-attention both say 44353. Four
  implementations, two values, two each. The gate there carries no evidence,
  and the 7 tokens after it are the real result.
- **No nh20/nh24 4096 capture**, so Nanbeige and Phi4 keep the CPU fallback.

## Reproduce

```sh
export NPU_XCLBIN_DIR=$PWD/engine/npu/xclbins

# unified: fast prefill + runlist decode  (needs the per-ctx ELFs to ctx >= prompt+decode)
NPU_LAYER_ELF_DIR=$HOME/npu-build/elfs-4k-0p6b NPU_PREFILL_BF16=1 NPU_UNIFIED=1 \
  NPU_PREFILL_MAX=4096 engine/npu/build/npu_engine_qwen3_0_6b \
  ~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx 8 ~/npu-build/parity/ids4095.txt

# the same prompt on FLM's own runtime, in this harness
NPU_FLM_PREFILL=1 NPU_FLM_DECODE=1 engine/npu/build/npu_engine_qwen3_0_6b \
  ~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx 8 ~/npu-build/parity/ids4095.txt

# the comparison this supersedes for TTFT: the default runlist path
NPU_LAYER_ELF_DIR=$HOME/npu-build/elfs-4k-0p6b NPU_RUNLIST=1 engine/npu/build/npu_engine_qwen3_0_6b \
  ~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx 8 ~/npu-build/parity/ids4095.txt
```

`ids4095.txt` is the first 4095 tokens of `ids4096.txt`, which is `ids2048.txt`
twice; `ids1024`/`ids2048`/`ids4096` are prefixes of one sequence (checked).

## Provenance

One session, 2026-09-15. The 4096-context attention captures and the per-context
ELF sets this measurement depends on are from the preceding two commits
(`d9d287288`, `847192002`) and the script fix before them (`38914ae28`).
