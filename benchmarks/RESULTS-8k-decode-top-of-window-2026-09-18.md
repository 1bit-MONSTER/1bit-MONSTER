# Criterion (c) decode clause at the top of the window: measurable, and at parity — 2026-09-18

Goal `mu35shsg-i3hlyi`. The 8k decode clause was recorded as "cannot be measured by any current
path" because a prompt of 8192 with `ng>=2` needs `ctx=8194`, past the baked `MAX_L=8192`
per-ctx ELF window. That is true of a *8192-token* prompt, and it is the whole range only if you
insist on decoding at a 8192-token prompt. It is not true of the window itself:

> **A prompt of `8192 - ng` tokens puts every decode forward at `ctx <= 8192`.** With
> `ng = 32` and a 8160-token prompt the forwards run at `ctx = 8161..8192`, i.e. the top of the
> supported window, and the ELF for each of those contexts exists.

So the decode clause is measurable at the top of the range after all, and this is what it says.

## Result: decode at ~8.2k context, all six models

`benchmarks/d8k.sh <Model> <flm_tag> <engine> <abmodel>`: native = `ng=32` at
`/tmp/p_8160.txt` (`NPU_PREFILL_BF16=1 NPU_BF16=1 NPU_GREEDY=1 NPU_PREFILL_MAX=8192
NPU_PROMPT_MAX=8192`); FLM = `npu_ab.sh --ctx-k 8 --decode-tokens 32 --skip-native`. Two pairs
per model, native and FLM interleaved.

| model | native ms/tok | native tok/s | FLM decode t/s | ratio (per pair) |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 28.2, 28.2 | 35.46 | 33.86, 33.86 | **1.047 / 1.047** |
| Qwen3-1.7B | 42.2, 41.1 | 23.70, 24.33 | 24.32, 24.03 | 0.975 / 1.013 |
| Qwen3-4B | 73.1, 73.2 | 13.68, 13.66 | 13.53, 13.57 | 1.011 / 1.007 |
| Qwen3-VL-4B | 73.9, 73.0 | 13.53, 13.70 | 10.09†, 13.44 | 1.341† / **1.019** |
| Qwen3-8B | 113.6, 112.1 | 8.80, 8.92 | 8.76, 8.76 | 1.005 / 1.018 |
| Llama-3.1-8B | 123.7, 106.4 | 8.08, 9.40 | 8.74, 9.27 | 0.925† / **1.014** |

† contaminated pair: Llama pair 1 ran with a foreign `accel0` holder (Prism `/tmp/pfhip`, the
Bonsai model); VL-4B pair 1's **FLM** leg read 10.09 t/s against 13.44 in pair 2 (33% spread)
while native's two readings agree to 1.2%. On the clean pairs the ratio is **1.019 (VL-4B) and
1.014 (Llama)**.

**Verdict: at the top of the supported window, native decode is at parity or ahead for all six
models** — 0.6B clearly ahead (+4.7%, both pairs identical), and the others inside ±2.5%
(1.7B −2.5%/+1.3%, 4B +1.1%/+0.7%, VL-4B +1.9%, 8B +0.5%/+1.8%, Llama +1.4%). Native's own
spread is 0.1–2.2% in seven of twelve pairs, so most of the residual is FLM's leg.

## The warm-up trap this avoids

The first attempt used `ng=8`, which is warm-up-dominated: native read **31.5 tok/s vs FLM's
33.83 (0.93x)**, i.e. it would have been recorded as a 7% decode loss. The same model with
`ng=32` reads **35.46 vs 33.86 (1.047x)**. This is exactly the transient documented in
`RESULTS-decode-window-warmup` for the 1k rows — and it is why `d8k.sh` uses `ng=32` and says so
in its header. Any future decode row in this lane at any context should use `ng>=32`.

## What this changes for criterion (c)

- **Before:** 8k context = prefill/TTFT measured (native ahead 1.04–1.10x), decode "impossible".
- **Now:** decode at ~8.2k context is also measured and at parity-or-ahead, so **all three
  clauses hold across 1k..8192 context**, which is the range the criterion names.
- **Remaining bound, stated exactly:** `ctx <= 8192` is hard. A prompt of 8191–8192 tokens can
  only decode `8192 - prompt_len` tokens, so a *sustained* decode run at a 8192-token prompt
  still needs contexts >8192 and remains phase-2 work. This is a window bound, not a
  performance regression, and it is the same bound the `runlist` `MAX_L` ELF baking imposes on
  every path in the engine.

## Commands / logs

```
# prompt: 8160 = 8192 - 32 ids from /tmp/p_8k.txt
python3 -c 'ids=[int(x) for x in open("/tmp/p_8k.txt").read().split()];open("/tmp/p_8160.txt","w").write(" ".join(map(str,ids[:8160])))'
bash benchmarks/d8k.sh <Model-NPU2> <flm_tag> <engine> <abmodel> 2          # Llama adds NPU_UNIFIED=1 NPU_LAYER_ELF_DIR=<warm dir>
# logs: /tmp/d8k-<engine>-<n>.log (native), /tmp/d8k-flm-<engine>-<n>.log (FLM)
```
