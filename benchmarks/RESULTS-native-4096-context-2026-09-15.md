# Context 2049–4095 on the NPU — a 4096-context attention capture (2026-09-15)

Follow-on to `RESULTS-native-vs-flm-dense-qwen3-2k-2026-09-14.md`, whose "Still
open" list led with this: *"Context > 2048: no attention capture exists, so the
published table's 4k/8k/16k/32k columns are unreachable today."* The 4k part of
that is now closed for the dense-Qwen3 shapes.

## What was wrong

`Bf16Mm::run_attn` returned `nullptr` for every `attn_tokens > 2048`
(`npu_engine_bf16_mm.h`). `nullptr` there is not a failure — it routes the call
to the CPU attention reference, which is correct and slow. So a prompt longer
than 2048 tokens worked, at **17 tok/s prefill**, and the 2048 slot's comment
said why in three words: *"no capture this long"*.

## The capture

Same technique as the 1024 and 2048 kernels: drive FLM's own runtime through
`npu-infer/tools/capture/run_qwen3_prefill` under `cap_interposer.so` with a
4096-token prompt and `CAP_NO_SYNC=1`, and take the attention ELF out of the
`xrt::elf` dump.

| file | bytes | sha256 (first 16) | captured from |
|---|---:|---|---|
| `attn_mha_4096_nh16.elf` | 386512 | `21087fada04e362d` | Qwen3-0.6B |
| `attn_mha_4096_nh32.elf` | 701904 | `a1f663b7a9cc0084` | Qwen3-4B |

### Identification is measured, not assumed

Two independent checks, both cheap and both decisive:

1. **The interposer's ELF index is stable across captures.** The attention
   kernel is `elf_0012` in the 256/1024/2048 runs, in the nh16 4096 capture and
   in the nh32 4096 capture. Same object, same position.
2. **The size is affine in the captured length, and the fit is exact.** Fitting
   the two *committed* captures of each shape and evaluating at 4096:

   | shape | fit from the committed pair | predicts @4096 | captured |
   |---|---|---:|---:|
   | nh16 | `2960.0 + 93.641·L` | 386512.0 | **386512** |
   | nh32 | `2960.0 + 170.641·L` | 701904.0 | **701904** |

   Both shapes share the same constant term (2960.0) and both predict their
   capture to the byte. The nh16 fit also reproduces the committed 256-context
   kernel (predicts 26932, committed 26928).

Each capture also self-validates: the interposed run prints FLM's own
`GREEDY_NEXT`, and both printed exactly what the uninterposed run printed.

## Correctness gates

Gate = the boot token equals FLM's own runtime on the same tokens
(`run_qwen3_prefill` `GREEDY_NEXT`). Note this is a *different* implementation
from the native engine, so agreement is evidence and not tautology.

### Length sweep, Qwen3-0.6B (the new nh16 kernel in every row)

| npt | native | FLM | |
|---:|---:|---:|---|
| 1024 | 25 | 25 | match |
| 2048 | 220 | 220 | match |
| 2560 | 220 | 220 | match |
| 3072 | 220 | 220 | match |
| 3584 | 220 | 220 | match |
| 3840 | 11066 | 11066 | match |
| 4000 | 39690 | 39690 | match |
| 4094 | 220 | 220 | match |
| 4095 | **44353** | 59277 | **see below** |

Forced onto the 1024 and 2048 slots via `NPU_ATTN_ELF_1024` / `_2048`, the same
file returns 25 and 220 — a capture is valid *below* its length as well as at it.

### The family at 4095 (the longest prompt the CLI accepts)

| model | shape | native boot | FLM boot | |
|---|---|---|---:|---|
| Qwen3-0.6B | nh16 | 44353 | 59277 | unstable position, below |
| Qwen3-1.7B | nh16 | 59277 | 59277 | match |
| Qwen3-4B | nh32 | 59277 | 59277 | match |
| Qwen3-8B | nh32 | 44353 | 44353 | match |
| Qwen3-VL-4B | nh32 | 59277 | 59277 | match |

`npt` is 4095, not 4096: the CLI caps the prompt at 4095 by design
(`npu_engine_universal.cpp:4202`, announced — max_seq_len is 4096).

## Performance at 4095 — native vs FLM's own prefill, same box, same tokens

| model | native NPU attn | ms/tok | FLM own prefill | ms/tok | native / FLM |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 1696–1702 ms | 0.415 | 1981 ms | 0.480 | **1.17×** |
| Qwen3-1.7B | 2494 ms | 0.609 | 2960 ms | 0.720 | **1.19×** |
| Qwen3-4B | 5431 ms | 1.326 | 8189 ms | 2.000 | **1.51×** |
| Qwen3-8B | 9311 ms | 2.274 | 9043 ms | 2.210 | 0.97× (FLM 3% ahead) |
| Qwen3-VL-4B | 5419 ms | 1.323 | 6455 ms | 1.580 | **1.19×** |

Two things to say honestly about this table:

- **The win here is range more than speed.** Every row above was previously the
  CPU-attention fallback: at 0.6B that is **240614 ms (58.8 ms/tok, 17 tok/s)**,
  i.e. this is a **142×** improvement over what the engine did at this context
  before the capture, and the 4k row became reachable at all.
- **Qwen3-8B is now the one size where FLM edges ahead** (3%, single pass), at
  the longest context. It is also the size that trails in the published decode
  table. The 1k/2k lead of 1.6–1.7× on prefill does not extend to 4k on the
  largest model; that is the honest reading of one measurement.

## The 4095 disagreement, and why it is not the kernel

Qwen3-0.6B at npt=4095 is a **numerically unstable position** — the class
`SESSION-FINDINGS-2026-09-14.md` already documents at npt=1280, where *four*
implementations gave four different boot tokens. There are four independent
measurements here and they split two/two:

| implementation | boot | uses the new ELF? |
|---|---:|---|
| FLM's own runtime | 59277 | no |
| native bf16 + CPU attention | 59277 | no |
| native int8 runlist (byte-exact) | **44353** | **no** |
| native bf16 + NPU attention | **44353** | yes |

The decisive ones are the third and fourth. The byte-exact int8 runlist path
never loads this kernel and lands on the *same* answer as the NPU path, so the
disagreement is between two legitimate numerics, not between "the kernel" and
"the reference". Two further controls:

- the per-layer `ATTN-DIFF` at 4095 is **identical** to 4094's (same max, same
  row, same values), so the attention step did not move at the failing length;
- a **different** 4095-token prompt (the same sequence shifted by one token)
  gives **16** on all three of FLM, CPU attention *and* the new kernel.

So: the kernel is correct on 8 of 9 gated lengths, on a second prompt at the
ninth, and on a second shape (nh32 matches FLM at 4095 on 4B/8B/VL-4B). At one
position on one model the greedy argmax falls the other way, as it did at 1280.

## What is still open

- **Context > 4096.** A capture at length L serves contexts ≤ L, so 4097+ needs
  another capture (and the CLI's own 4095 cap lifted). The published table's
  8k/16k/32k columns remain unreachable.
- **nh20 (Nanbeige) and nh24 (Phi4) at 4096** have no capture; they keep the
  CPU fallback, which is the correct-answer path for them — the shape guard
  deliberately does not hand them the nh16/nh32 kernel.
- **Qwen3-8B at 4095**, the one size where FLM's prefill is ahead (3%).
- **The 4095 CLI cap** is intentional (max_seq_len 4096) but means a 4096-token
  prompt is silently shortened to 4095 *by announcement*, so a 4096-token
  comparison is not expressible through this entry point.

## Reproduce

```sh
cd npu-infer/tools/capture
RT_TOKENS=~/npu-build/parity/ids4096.txt CAP_NO_SYNC=1 \
  CAP_DIR=~/npu-build/cap4096_06b LD_PRELOAD=$PWD/cap_interposer.so \
  ./run_qwen3_prefill ~/.config/flm/models/Qwen3-0.6B-NPU2     # -> elf_0012_386512.bin
RT_TOKENS=~/npu-build/parity/ids4096.txt CAP_NO_SYNC=1 \
  CAP_DIR=~/npu-build/cap4096_4b  LD_PRELOAD=$PWD/cap_interposer.so \
  ./run_qwen3_prefill ~/.config/flm/models/Qwen3-4B-NPU2       # -> elf_0012_701904.bin

export NPU_XCLBIN_DIR=$PWD/engine/npu/xclbins
engine/npu/build/npu_engine_qwen3_0_6b \
  ~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx 1 ~/npu-build/parity/ids4095.txt
# run under: NPU_RUNLIST=0 NPU_PREFILL_BF16=1 NPU_PREFILL_MAX=4096
```

`ids4096.txt` is the 2048-token fixture repeated twice, so
`ids1024`/`ids2048`/`ids4096` are prefixes of one sequence (checked).

## Provenance

One session, 2026-09-15. The ELF-regeneration script fix that made the 4096
per-context ELF set buildable (`benchmarks/gen-layer-elfs.sh`, unusable since
`4db62ecd5`) is the preceding commit; the ELF sets for all four dense sizes now
exist to ctx 4200 in `~/npu-build/elfs-4k-*` and reproduce the committed sets
byte-for-byte.
