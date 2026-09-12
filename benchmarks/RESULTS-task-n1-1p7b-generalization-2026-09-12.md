# RESULTS — task-n1 generalization: Qwen3-1.7B also byte-exact + beats FLM at 256 (2026-09-12)

Goal `mttxt22c-a6rv75`. The task-n1 fixes (G=8 tile reorder, single N=4096 QKV
GEMM, truly-async GEMM launches, OpenMP host math) are model-generic. Verified
they generalize to Qwen3-1.7B (NH=16, reuses `attn_mha_256_nh16.elf` + the 0.6B
recipe).

## 1.7B default-prompt parity

native boot = **151667 = FLM** ✓ (byte-exact).

## 1.7B parity sweep (reclaimer prompt)

| n | native | FLM | match |
|---|---|---|---|
| 1 | 25 | 25 | ✓ |
| 8 | 220 | 220 | ✓ |
| 16 | 16 | 16 | ✓ |
| 64 | 25 | 25 | ✓ |
| 128 | 16 | 16 | ✓ |
| 160 | 31082 | 31082 | ✓ |
| 192 | 220 | 220 | ✓ |
| 256 | 16 | 1614 | near-tie flip* |

\* n=256: native top-5 `16,1614,882,17,18` vs FLM `1614,16,17,882,18` — identical
top-5 sets, #1/#2 swapped (16 vs 1614). Same near-tie argmax-flip regime as the
0.6B n=16 case.

## 1.7B 256-token throughput (native vs FLM, same prompt/output)

| path | prefill time |
|---|---|
| **native bf16** | **~586 ms** |
| FLM (`qwen3_npu::prefill`) | ~690 ms |

Native ~15% faster, byte-identical tokens (modulo the near-tie flip).

## Conclusion

The task-n1 result — byte-exact token parity (modulo near-tie flips) and
meet-or-beat prefill throughput — holds for the whole NH=16 dense Qwen3 set
(0.6B, 1.7B). The NH=32 models (4B/8B) remain blocked on the nh32 attention ELF
bug (task-n2 scope, documented in RESULTS-task-n2-*).
