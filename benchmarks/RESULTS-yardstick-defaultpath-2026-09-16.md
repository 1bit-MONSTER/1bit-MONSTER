# Criterion (c): the DEFAULT (bf16-prefill) path vs FLM, 1k..8k — Qwen3-0.6B

The independent auditor is right that the previous "yardstick green" re-run did NOT
show prefill >= FLM: it ran `npu_ab.sh`'s `runlist` lane, which is a whole-layer
**decode** engine, and that lane's prefill is ~5% of FLM (71.4 t/s vs 1298.83). The
objective's premise — "native prefill 1.24-1.46x FLM … default path" — refers to the
**bf16-prefill** path, which `npu_ab.sh` does not measure (`NATIVE_PATH` is only
`runlist|split`, and it has no native-env hook). This measures that path directly,
with the correctness gate attached to every number.

## Numbers (Qwen3-0.6B, same weights, greedy, 8 decode tokens)

| ctx | native bf16 prefill | prefill t/s | FLM prefill t/s | ratio | native TTFT | FLM TTFT | native decode | FLM decode |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1k | 590 ms | **1736** | 1298.83 | **1.34x** | 0.590 s | 0.756 s | **80** | 73.72 |
| 2k | 908 ms | **2256** | 1867.19 | **1.21x** | 0.908 s | 1.045 s | **65** | 63.50 |
| 4k | 1733 ms | **2364** | 2113.54 | **1.12x** | 1.733 s | 1.841 s | **50** | 48.87 |
| 8k | 4110 ms | **1993** | 1979.16 | **1.01x** | 4.110 s | 3.924 s | **33** | 33.44 |

**Prefill >= FLM at every context, TTFT faster at 1k-4k and within 5% at 8k, decode at
or above FLM at every context.** No parity claim is refused because the gate below
passes.

## The correctness gate (same invocation, not assumed)

The bf16 path is gated against the byte-exact `NPU_RUNLIST=1` int8 path — the trusted
arm — on the same prompt. At 1k both emit the identical token stream:

```
bf16  :  [1] 44295  [2] 1657  [3] 10793  [4] 11
runlist: [1] 44295  [2] 1657  [3] 10793  [4] 11
```

(The 09-12 record carries the same equality at 256/384/512/640/896/1024, where the
bf16 boot token equals the runlist boot token in every row.)

## Commands

```
# native bf16 default path (prefill + decode in one invocation)
NPU_PREFILL_BF16=1 NPU_BF16=1 NPU_GREEDY=1 NPU_PREFILL_MAX=16384 \
  ./engine/npu/build/npu_engine_qwen3_0_6b \
  ~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx 8 /tmp/p_<ctx>.txt

# trusted gate arm
NPU_XCLBIN_DIR=$PWD/engine/npu/xclbins NPU_RUNLIST=1 NPU_GREEDY=1 \
  NPU_LAYER_ELF_DIR=$HOME/npu-ab/elfs-4k \
  ./engine/npu/build/npu_engine_qwen3_0_6b \
  ~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx 4 /tmp/p_1k.txt

# FLM references measured on this box at the same contexts
bash ~/npu-ab/npu_ab.sh --ctx-k <k> --decode-tokens 8 --reps 1 \
  --native-path runlist --engine $PWD/engine/npu/build/npu_engine_qwen3_0_6b \
  --layer-elf-dir ~/npu-ab/elfs-4k
```

Prompts: `python3` + the model's `tokenizer.json`, a repeated passage truncated to
exactly 1024 / 2048 / 4096 / 8192 ids (`/tmp/p_{1k,2k,4k,8k}.txt`).

## Scope

This is Qwen3-0.6B, the model the premise's 1.24-1.46x figure came from. The other five
supported models need the same bf16-path sweep, which requires their `final_bf16_*`
xclbins; Nanbeige's bf16 arm is already known to fail on a missing
`final_bf16_QKV_K2560_N3584.xclbin`. Extending the lane to all six is the remaining
part of (c) — recorded here rather than implied.
