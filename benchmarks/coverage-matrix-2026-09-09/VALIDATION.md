# Independent validation — re-run 2026-09-10

A clean re-run of every lane's key result (full output, no pipes, clean exit
codes) to confirm the matrix. Raw output in VALIDATION.log.

| Lane | Check | Result |
|---|---|---|
| HRX2 (Q4NX) | all 9 q4nx-converted models, `--device HRX20` | ✅ all " Paris…", EXIT 0 |
| Vulkan | Qwen3-0.6B + Qwen3-Coder-30B (standard GGUF), `--device Vulkan0` | ✅ " Paris…", EXIT 0 |
| native NPU | zaya-8B `.q4nx` | ✅ `EMB dbg corr=1.0000000 argmax 30334 vs 30334 (SAME)`, EXIT 0 |
| FLM NPU | qwen3:0.6b `/v1/chat/completions` | ✅ "The capital of France is **Paris**." 78.4 tps |

Device enumeration confirms the two accelerators: `HRX0: AMD Radeon 8060S
(gfx1151)` and `Vulkan0: AMD Radeon 8060S (RADV STRIX_HALO)`.

Fork fix commit: `df58715ca` (in /home/bcloud/hrx-ws/hrx-v2-src, branch hrx-v2).

Notes on earlier caveats:
- minicpm4-8b `EXIT: 13` in the first sweep was SIGPIPE from `| head -1`, not a
  real failure — clean re-run exits 0 and prints " Paris.\n\n- Paris is the most
  populous city in France…".
- Matrix row count corrected: 17 model rows (15 unique checkpoints + 2
  Q4NX-format variants of already-listed checkpoints), all 17 covered.
