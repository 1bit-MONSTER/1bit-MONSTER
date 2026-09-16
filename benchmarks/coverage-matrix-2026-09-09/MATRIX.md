# Coverage Matrix — "2 NPU engines + (HRX + Vulkan) = any model"

Machine: Ryzen AI MAX+ 395 (Strix Halo), 32t / 125 GB · NPU XDNA 2 · GPU Radeon 8060S (gfx1151)
Legend: ✅ verified output · ⚠️ runs but wrong/degenerate/unverified · ❌ fails · — not applicable (no local file for that lane)

## Model × lane matrix (every cell = an actual run, logged in the .log files)

| Model (family) | native-npu | npu-flm | hrx | vulkan |
|---|---|---|---|---|
| Qwen3-0.6B (dense) | ⚠️ runlist → token 0 | ✅ "Paris" 73.9 t/s | ❌ GET_ROWS q6_K | ✅ "Paris" |
| Qwen3-1.7B (dense) | ⚠️ runlist → token 0 | ✅ thinking 43.3 t/s | ❌ GET_ROWS q6_K | ✅ "Paris" |
| Qwen3-4B (dense) | ⚠️ runlist → token 0 | ✅ thinking 19.6 t/s | ❌ GET_ROWS q6_K | ✅ "Paris" 76.4 t/s |
| Qwen2.5-7B-Instruct (dense) | — | — | ❌ GET_ROWS (q6_K) | ✅ "Paris" |
| Qwen3-Coder-30B-A3B (MoE) | — | — | ✅ "Paris" | ✅ "Paris" |
| Qwen3.6-35B-A3B (MoE) | ⚠️ runs ~0.3 t/s | ⚠️ "////" degenerate | ❌ abort Q8_0 | ✅ "Paris" |
| Llama-3.2-1B (dense) | ❌ missing xclbin | ✅ "Paris" 59.9 t/s | — | — |
| zaya-8B (Zyphra MoE) | ✅ corr 1.0 vs CPU | — | ❌ dim mismatch | ❌ arch 'zaya' |
| qwen3-06b-q4nx | — | — | ✅ "Paris" | N/A |
| qwen25-05b-q4nx | — | — | ✅ "Paris" | N/A |
| qwen25-3b-q4nx | — | — | ✅ "Paris" | N/A |
| qwen25-7b-q4nx | — | — | ✅ "Paris" | N/A |
| minicpm4-8b-q4nx | — | — | ✅ "Paris" | N/A |
| minicpm5-1b-q4nx | — | — | ✅ "Paris, …" | N/A |
| glm47-flash-q4nx | — | — | ✅ "Paris" | N/A |
| qwen3coder-30b-q4nx | — | — | ✅ "Paris" | N/A |
| qwen3next-80b-q4nx | — | — | ✅ "Paris" | N/A |

Out of scope: ZAYA1-74B-A4B + .1bp (vek385-eval — VEK385 platform, not Strix Halo); ZAYA1-8B safetensors (conversion source).

## Verdict

**The thesis holds: every model row in the matrix (17/17) now runs
end-to-end with verified output on at least one of the four lanes.**

- **Dense Qwen3 / Qwen2.5** → FLM + Vulkan ✅ (native-NPU runlist path is still
  mid-development and emits token 0 — noted, not gating).
- **MoE (30B/35B)** → Vulkan ✅ (30B also HRX ✅).
- **Llama-3.2-1B** → FLM ✅.
- **zaya-8B** → native-NPU ✅ (corr 1.0 vs CPU reference).
- **Q4NX-converted set (9)** → HRX2 fork ✅, after three fork fixes committed as
  `df58715ca` (MoE `nselected` + pointwise-stride JIT range widening, FA0 fusion
  scoped to H≤16). The 3 previously-failing models — MiniCPM4-8B, GLM-4.7-Flash,
  Qwen3-Next-80B — all now return "Paris."

Coverage is the UNION of lanes, not a single universal lane. Vulkan is the most
reliable standard-GGUF lane; FLM is the production NPU lane; HRX2 covers the
custom Q4NX tile format; the reverse-engineered native NPU engine verifiably
covers zaya-8B (dense-Qwen3 native path remains in development).

## Reproducible commands (see per-lane .md + .log)
```bash
cd /home/bcloud/1bit-MONSTER
export NPU_XCLBIN_DIR=/home/bcloud/1bit-MONSTER/engine/npu/xclbins   # native NPU
B=/home/bcloud/hrx-slice/hrx-llamacpp/out/llama-hrx-b59              # HRX0 + Vulkan0
B2=/home/bcloud/hrx-ws/hrx-v2-src/build/bin                         # HRX20 (Q4NX, fixed fork)

# native NPU:  engine/npu/build/npu_engine <model.q4nx> <n> [ids_file]
# FLM:         /opt/fastflowlm/bin/flm serve <tag> -p <port>  → POST /v1/chat/completions
# HRX/Vulkan:  LD_LIBRARY_PATH=$B/lib:$B/bin $B/bin/llama-completion -m <gguf> \
#                --device HRX0|Vulkan0 -p "The capital of France is" -n 16 --temp 0 -no-cnv --no-display-prompt
# HRX2 (Q4NX): LD_LIBRARY_PATH=$B2:/home/bcloud/hrx-ws/install-new/lib $B2/llama-completion \
#                -m <q4nx.gguf> --device HRX20 -p "The capital of France is" -n 16 --temp 0 -no-cnv --no-display-prompt
```

## Evidence
- PREFLIGHT.md · ROSTER.md · native-npu.md + native-npu.log · flm.md + flm.log
- hrx.md + hrx.log · vulkan.md + vulkan.log · hrx2-fork-fixes.patch (fork commit df58715ca)
