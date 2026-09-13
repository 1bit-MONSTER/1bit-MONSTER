# Coverage pass — dense Qwen3 family (pi agent, 2026-09-13)

Goal `mttxt22c-a6rv75`: "meet-or-beat FLM's measured performance … for **every model**
the native NPU engine supports." This is the first per-model pass beyond Qwen3-0.6B.

All runs: on-box, `NPU_XCLBIN_DIR=engine/npu/xclbins`, prompt = `benchmarks/prompts/reclaimer.txt`
tokenized to `/tmp/ids_<n>.txt`, correctness gate = native greedy boot token vs FLM's
`run_qwen3_prefill` `GREEDY_NEXT` on the same ids file.

## 1. The attention-shape matrix (why some models need no new capture)

| model | hidden | nh | nkv | head_dim | layers | inter | attention shape |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 1024 | 16 | 8 | 128 | 28 | 3072 | **nh16/nkv8/hd128** |
| Qwen3-1.7B | 2048 | 16 | 8 | 128 | 28 | 6144 | **nh16/nkv8/hd128 — identical to 0.6B** |
| Qwen3-4B   | 2560 | 32 | 8 | 128 | 36 | 9728 | nh32/nkv8/hd128 |
| Qwen3-8B   | 4096 | 32 | 8 | 128 | 36 | 12288 | nh32/nkv8/hd128 |
| Nanbeige4.1-3B | 2560 | 20 | 4 | 128 | 32 | 10752 | nh20/nkv4/hd128 |
| Phi4-mini  | 3072 | 24 | 8 | 128 | 32 | 8192 | nh24/nkv8/hd128 |
| Llama-3.2-1B | 2048 | 32 | 8 | **64** | 16 | 8192 | nh32/nkv8/**hd64** |
| Gemma4-E2B | 1536 | 8 | 1 | **256** | 35 | 6144 | nh8/nkv1/**hd256** |

**Key consequence:** the attention kernel is parameterised by `(nh, nkv, head_dim)`, not by
hidden size. **Qwen3-1.7B is shape-identical to 0.6B**, so the already-captured
`attn_mha_256/1024/2048_nh16.elf` apply unchanged — **no new capture required**.
4B/8B (nh32) and every other family need their own capture.

## 2. Results

| model | npt | native boot | FLM boot | gate | native prefill | FLM on-box prefill | verdict |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B @1k | 1024 | 25 | 25 | ✅ | **1440.5 tok/s** | 1123.1 tok/s | **+28.3% (beats)** *(prior work)* |
| **Qwen3-1.7B @1k** | 1024 | **220** | **220** | ✅ | **838.9 tok/s** | **942.57 tok/s** | **-11.0% (loses)** |
| Qwen3-4B @256 | 256 | **1614** | **1614** | ✅ | 328.2 tok/s | (not benched) | correct at ≤256 |
| Qwen3-4B @1k | 1024 | 87672 | 220 | ❌ | (meaningless) | (not benched) | **broken >256** |
| Qwen3-8B | — | — | — | not run | — | — | predict = 4B (same nh32 shape) |

Raw native lines:

```
Qwen3-1.7B @1024: Prefill: 1220ms (1.192 ms/tok) [GEMM 153ms, attn 188ms, conv+other 1195ms]
                  [0] boot=220 (24ms)
Qwen3-4B   @256 : Prefill:  780ms (3.047 ms/tok) [GEMM  76ms, attn 165ms, conv+other  758ms]
                  [0] boot=1614 (22ms)
Qwen3-4B   @1024: Prefill: 2517ms (2.458 ms/tok) [GEMM 319ms, attn 286ms, conv+other 2480ms]
                  [0] boot=87672 (21ms)      <-- WRONG
FLM qwen3:1.7b   : TTFT 1.04186s, Prefill 942.57 tok/s, Decode 39.53 tok/s (ctx 981)
```

## 3. Findings

1. **Qwen3-1.7B is now covered and CORRECT** (boot 220 == FLM 220) with **zero new capture** —
   the nh16 attention ELFs transfer directly. This is one model off the "unmeasured" list.
2. **But it does not beat FLM on prefill**: native 838.9 vs FLM on-box 942.57 tok/s = **-11.0%**
   (per-token: 1.192 vs 1.062 ms/tok). The 0.6B prefill win (+28.3%) **does not generalise**.
   Native degrades faster with model size (0.6B→1.7B: 1440→839 = -42%) than FLM (1123→943 = -16%),
   and the dominant term is host-side `conv+other` (1195 ms), consistent with the earlier
   "prefill bottleneck is host math, not NPU GEMM" finding.
3. **nh32 models (4B/8B) are gated at ≤256** (4B @256: 1614 == 1614) **and break above it**
   (4B @1024: 87672 ≠ 220) — exactly the `attn_mha_1024_nh16.elf` being nh16 while the model is nh32.
   Fix = one nh32 >256 attention capture, not a kernel project.
4. **Decode for 1.7B was not measured in this pass** (only the boot token). FLM on-box decode is
   39.53 tok/s; native decode needs a separate `NPU_RUNLIST=1` run.

## 4. Recovered capture recipe (for the models that need it)

`benchmarks/RESULTS-npu-prefill-1k-SOLVED-2026-09-12.md` + `~/npu-build/cap1024/run.log`:

```
cd npu-infer/tools/capture
RT_TOKENS=<ids file> \
LD_PRELOAD=$PWD/cap_interposer.so CAP_DIR=~/npu-build/cap<model> \
  ./run_qwen3_prefill ~/.config/flm/models/<Model>-NPU2
# -> elf_0001..elf_NNNN.bin ; the attention ELF is elf_0012 (98848 B @1024 for nh16)
```

`run_qwen3_prefill` is model-agnostic at the driver level (`argv[1]` = model dir, `RT_TOKENS` = ids,
prints `GREEDY_NEXT`) but hardcodes the `qwen3_npu` class — dense-Qwen3 only. Other families need
the family driver (see `gen_attn_chunk_llama/_nb/_phi4` for the generator side).

## 5. Repro

```
cd /home/bcloud/1bit-MONSTER-goal
export NPU_XCLBIN_DIR=$PWD/engine/npu/xclbins
# native prefill + boot token
NPU_RUNLIST=0 NPU_PREFILL_BF16=1 NPU_PREFILL_MAX=<npt> \
  ./engine/npu/build/npu_engine_qwen3_1_7b ~/.config/flm/models/Qwen3-1.7B-NPU2/model.q4nx 1 /tmp/ids_<npt>.txt
# FLM reference boot token
cd npu-infer/tools/capture && RT_TOKENS=/tmp/ids_<npt>.txt ./run_qwen3_prefill ~/.config/flm/models/Qwen3-1.7B-NPU2
# FLM on-box bench
mkdir -p ~/npu-build/flmbench && cd ~/npu-build/flmbench
python3 -c "import json;json.dump({'max_length':1024,'iterations':1,'input_text':open('/home/bcloud/1bit-MONSTER-goal/benchmarks/prompts/reclaimer.txt').read()},open('cfg.json','w'))"
/opt/fastflowlm/bin/flm bench qwen3:1.7b -i cfg.json   # -> bench_qwen3_1.7b_<date>.csv
```

## 6. Remaining coverage work (precise)

| group | models | needs |
|---|---|---|
| nh16/hd128 | Qwen3-1.7B | **done** (correct; prefill gap -11% is a perf task) |
| nh32/hd128 | Qwen3-4B, Qwen3-8B | capture nh32 >256 attention ELF |
| nh20/nkv4/hd128 | Nanbeige4.1-3B | capture (unique shape) |
| nh24/nkv8/hd128 | Phi4-mini | capture (unique shape) |
| nh32/nkv8/hd64 | Llama-3.2-1B/3B | capture (hd64) |
| nh8/nkv1/hd256 | Gemma4-E2B/E4B | capture (hd256) |
| MoE | Qwen3.6-35B-A3B | capture (MoE layer ELFs) |
| no engine | LFM2-1.2B/2.6B | build engine first |
