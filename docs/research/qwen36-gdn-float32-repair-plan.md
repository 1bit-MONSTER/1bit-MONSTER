# Qwen3.6-35B-A3B GateDeltaNet float32 repair — plan & progress

Goal `mttxt22c-a6rv75` task-4 unblock. Status: **in progress** (user directive
2026-09-11: "start working on that — requires binary-level work").

## ✅ SOLVED (2026-09-11): v0.9.46 drop-in, not a GDN rewrite
The bf16-GDN regression is v1.0.x-only. The v0.9.46 stack works and produces
non-NaN prefill + decode. **The earlier "runlist timeout" and "config JSON
error" were a HEADER ABI mismatch** (v0.9.46 lib loaded against v1.0.4
`lm_config.hpp`; the `LM_Config` struct layout changed between v0.9.46 and
v1.0.x). Compiling against the v0.9.46 headers fixes it.

Working combo (verified end-to-end):
- headers = third_party/FastFlowLM/src/include (v0.9.46)
- lib = v0.9.46 .deb (libqwen3_6_moe_npu.so md5 39a6c36a)
- xclbins = v0.9.46 set (share/flm/xclbins in the .deb)
- XRT = 2.21.75 OR 2.26.0 (both work once headers match)
- model dir must be NAMED `Qwen3.6-35B-A3B-NPU2` (xclbin manager keys off it)

Result: prefill nan=0 (boot=760="The"), forward() nan=0 at ~72 ms/tok ≈
11–14 tok/s — matches the 07-30 run (11.66 tok/s).

## ⚠️ CONTRACT BAR DISCREPANCY (needs a user decision)

The task-4 contract references `qwen3.6_results.md` (decode **17.48** / prefill
**102.45** @1k). But the 07-30 on-box capture (`benchmarks/
RESULTS-qwen3.6-35b-a3b-npu-flm-2026-07-30.md`) records the ACTUAL published
v0.9.46 table as decode **13.65** / prefill **78.98** @1k (v0.9.46 release notes:
"New 13.65 vs Old 12.41 @1k"), and on-box v0.9.46 measured **11.66 / 98.05**.
The `qwen3.6_results.md` 17.48/102.45 matches neither — it is stale (likely a
different build/HW). The real v0.9.46 bar is LOWER; on-box already beats prefill
(+24%) and trails decode ~14.6% (cross-HW Kraken-Point gap). The contract should
be corrected to the 13.65/78.98 v0.9.46 table before chasing 17.48/102.45.

## FLM-version landscape (why the on-box v0.9.46 is not a drop-in)
- v1.0.4 (installed, XRT 2.26.0 ABI): GDN delta-rule baked **bf16** → NaN
  (prefill + decode), prefill GDN disabled-and-broken (`stride_1 out of range`),
  latent `reorder_cpy` OOB.
- v0.9.46 (third_party/FastFlowLM submodule, commit 17a35cb): GDN WORKS (proven
  07-30), xclbins self-consistent, but **closed-source** (no .cpp impl — only
  headers + tests + 23 pre-built .so) and its `xrt::runlist` ABI predates XRT
  2.26.0 → `ERT_CMD_STATE_TIMEOUT` on BOTH XRT 2.21.75 and 2.26.0. Not
  recompilable.
- No v0.9.45/.46 lib built for the current XRT exists on-box. Ideal fetch target:
  a v1.0.0–v1.0.3 lib (working GDN + current XRT ABI), or the v0.9.46 lib +
  its matching XRT/driver (amdxdna 0.7-era).

## Fetch bisect result (2026-09-11, GitHub ROCm/FastFlowLM releases)
FLM releases v0.9.45→v1.0.5 are on GitHub (ubuntu26.04 .debs). Downloaded + tested:
- v1.0.0/v1.0.1 (md5 fc4562dd) → crash during load (latent OOB)
- v1.0.2 (e46e1653) → crash during load
- v1.0.3 (8d76c77b) → NaN or all-zero (non-deterministic; the OOB)
- v1.0.4 (c23bf8eb) → NaN (on-box)

**The bf16-GDN regression predates v1.0.0** — no v1.0.x has a working GDN.
Only v0.9.46 works (07-30 proven, 11.66 tok/s), and its sole blocker is the
runlist ABI: it times out on XRT 2.21.75 AND 2.26.0 because the 07-30 box ran
amdxdna 0.7 (now 0.1). So the fetch path = obtain the v0.9.46-era XRT/driver
(amdxdna 0.7, XRT ~2.2x); v0.9.46 lib + xclbins are already in third_party.
Remaining on-box paths: blind runlist binary-patch of v0.9.46, or the decode
`_gen_sequence`/layer.xclbin rewrite (multi-day).

## Drop-in lib switch (for the fetched v0.9.45/.46 or v1.0.0–v1.0.3)
The engine's FLM lib + xclbin path is now `FLM_ROOT`-env-configurable:
- `build_npu.sh` links `$FLM_ROOT/lib/xrt` (already env-driven).
- `npu_engine_bf16_mm_bridge.cpp::find_xclbin_path()` and
  `flm_prefill_bridge.cpp`'s `FLM_XCLBIN_PATH` now honor `FLM_ROOT`
  (fallback `/home/bcloud/amd-oss/fastflowlm/src`).

To drop in a fetched FLM install: `FLM_ROOT=<install>/src bash engine/npu/build_npu.sh`,
then run with `FLM_ROOT=<install>/src` exported (the libs load via the build rpath,
and the xclbins resolve under `<install>/src/xclbins`).

## Root cause (confirmed, both agents)

`libqwen3_6_moe_npu.so` bakes the GatedDeltaNet (linear_attention) delta-rule
recurrence in **bf16** on the NPU (`GateDeltaNet_prefill.xclbin`, and the
`layer.xclbin` decode path), ignoring `mamba_ssm_dtype: float32`. The SSM
recurrence

```
g    = ssm_a * softplus(alpha·x + dt_bias)     # ssm_a ∈ [-16,0) → g ≤ 0
eg   = exp(g)                                   # decay ∈ [0,1]
S   *= eg                                       # state decay
delta = (v - Σ_i S[i][j]·k[i]) · sigmoid(beta·x)
S   += k ⊗ delta
out  = (S·q)/√head_dim  → gated-RMSNorm × silu(z) → out_proj
```

explodes in bf16 (8-bit mantissa): all 248320 prefill logits NaN, act/SSM-state
BO half-NaN (262144/524288). Weights are clean (ssm_a/ssm_dt F32, others BF16,
all zero-NaN). Independent of XRT version (2.21.75 == 2.26.0, both NaN),
MAX_L, prompt tokens, and `_prefill_with_mv` vs `_prefill_with_mm`.

The float32/float64 math is fully specified and validated:
- `tools/gdn_reference.py` (float64 numpy golden)
- `tools/qwen36_gdn_probe.cpp` (float32 C++, rel RMSE < 1e-3 vs golden)
- `engine/npu/src/npu_engine_universal.cpp::gdn_attn_cpu` (float32, correct)

## The two repair paths

### Path 1 — rebuild GateDeltaNet_prefill.xclbin in float32 (multi-day)
Requires the AIE kernel source, which is NOT in any checked-out tree
(`GateDeltaNetPrefillRuntimeSequence.hpp` is a build-time include; the 228636-B
xclbin is identical across all Qwen3.5/3.6 models). No source → reverse-engineer
the xclbin instruction stream or write a new MLIR-AIE GDN kernel.

### Path 2 — binary patch: run the SSM recurrence on host float32 (tractable)
The lib already ships CPU float32 primitives:
`cpu_func::_elementwise_mac(float*, bf16*, float, int)`,
`cpu_func::_elementwise_mul`, `cpu_func::_f32_to_bf16`,
`cpu_func::_sigmoid/_rms_norm/_gated_norm/_residual_add`.

Injection point mapped this session:
- `qwen3_6_moe_npu::Impl::prefill` (0x77280) → `_prefill_with_mm` (0x76320)
- `_prefill_with_mm` dispatches per layer kind:
  - `qwen3_6_moe_linear_block_prefill_context::forward` (0x83200) ← GDN layers
  - `qwen3_6_moe_attn_block_prefill_context::forward` (0x848e0) ← full-attn
  - `qwen3_6_moe_expert_prefill_context::forward` (0x897f0) ← MoE experts
- `generate_gate_delta_net_prefill_sequence` (0x6f530) is called at 0x833fd
  (inside the linear-block forward) to build the NPU GDN sequence that runs on
  `GateDeltaNet_prefill.xclbin`.

Patch plan (path 2):
1. Decode `generate_gate_delta_net_prefill_sequence`'s 13 args and the sequence
   it emits (conv1d→silu→q/k l2norm→delta rule→gated norm). Determine whether
   the delta-rule is a separable sub-sequence.
2. In `linear_block_prefill_context::forward`, replace the delta-rule NPU ops
   with host `cpu_func::_elementwise_mac`/`_f32_to_bf16` calls in float32,
   keeping conv1d/silu/norm/GEMMs on the NPU (they are bf16-stable).
3. Validate: prefill logits non-NaN, token parity vs CPU reference
   (`qwen36_gdn_probe.cpp` / `gdn_reference.py`), then throughput vs FLM
   published (prefill 102–281 tok/s).

### xclbin is a single fused kernel → sequence rewrite, not a rel32 (partner finding)
`GateDeltaNet_prefill.xclbin` = ONE fused DPU kernel (MLIR_AIE, dpu_kernel_id
0x901) with 8 args: arg1→SRAM (768KB = only 393216 bf16 < 524288 state), args3-7
→HOST DRAM. The delta-rule is baked INSIDE the kernel, so no rel32 swaps just
the recurrence. Path 2 = REPLACE the GDN prefill sequence in the linear-block
forward with `[qkv_proj+z via dequant_mm] → [conv1d+silu + recurrence in host
f32] → [out_proj via dequant_mm]`. The host-float32 recurrence is ~2M f32
ops/token (perf-neutral; GEMMs stay on NPU).

### Confirmed desc field map (qwen3_6_moe_desc::build 0x8b0e0, assertion-validated)
| desc offset | config key | value |
|---|---|---|
| 0x48 | linear_key_head_dim | 128 (HDK) |
| 0x4c | linear_value_head_dim | 128 (HDV) |
| 0x3c | linear_num_key_heads | 16 (NK) |
| 0x40 | linear_num_value_heads | 32 (NV) |
| 0x50 | linear_conv_kernel_dim | 4 (conv_k) |
| 0x2c | = HDK·NK (=KEY_DIM) | 2048 |
| 0x30 | = HDK·NK (=KEY_DIM) | 2048 |
| 0x34 | = HDV·NV (=VALUE_DIM) | 4096 |
| 0x38 | = VALUE_DIM + 2·KEY_DIM (=CONV_DIM) | 8192 |

13-arg signature (128-token chunk) — assertion-validated:
`(8, 4, 2, 32, 128, 128, 128, 128, 128, 128, 0, 12288, 4096)`
= cols=8, rows=4, rep=2, NV=32, HDK=128, raw_tokens=128, rounded=128,
DQ=128, DK=128, DV=128, 0, (conv_k-1)·CONV_DIM/2=12288, tokens·NV=4096.

BD DDR patch table (224 triples, 9060 words) — 4 buffer args:
| arg_idx | patches | offset range | span | role |
|---|---|---|---|---|
| 0 | 96 | [49152, 2113536] | 2064384 B (~2 MiB) | state (f32) |
| 1 | 64 | [0, 16128] | 16128 B (~16 KiB) | convolved qkv[8192] |
| 2 | 32 | [0, 124] | 124 B | norm/control |
| 3 | 32 | [0, 7936] | 7936 B (~8 KiB) | core[4096] |

### Software pipeline: g/beta are next-chunk (not current)
The linear-block forward (0x83200) order: dequant_mm GEMM seqs (qkv/gate/alpha-beta)
→ conv1d seq → GateDeltaNet seq → ONE npu_app run (0x83437) → syncs → sync_to
→ out_proj run (0x834d5) → CPU alpha/beta proj `_cpu_mm_col` (0x83505) →
`_activate_a`(0x83624)+`_activate_b`(0x83637) → sync_to_device 0x3a0(%rbx) (0x8363c).
So g/beta from _activate_a/b are the NEXT chunk's, computed on HOST and synced to the
0x3a0(%rbx) device BO. Current chunk's g/beta live in host buffers from the prior
iteration: a = base+0x5d0, g = base+0x470 (base = *(arg5+0x10), weight_desc =
*(*(desc+0x70)+*(desc+0x8)*8)), beta = 0x10(%rsp), ssm_a/dt_bias = r13. Cleanest
stub: a/b-input variant computing g/beta internally, or one live trace at 0x83624.

### Post-run buffer map (linear-block forward 0x83420–0x83560)
After the GDN `npu_app::operator()` run at 0x83437 (`operator()<buffer<bf16>&,buffer<bf16>&,buffer<unsigned char>&>`, npu_app at `rbx+0xb0`), three `bytes::sync_from_device()` pull:

| addr | buffer | role | size |
|---|---|---|---|
| 0x83443 `0x2f8(%rbx)` | state | read+write (sync_to at 0x834bc) | 1 MiB (524288 bf16) |
| 0x83450 `0x60(*(rbx+0x20))` | core | write-only → out_proj | 4096·T |
| 0x83459 `0x28(%r14)` | conv-state | read+write (sync_to 0x834b0) | ≈16–48 KiB |

- state[32][128][128] = 1 MiB = the xclbin's 0x10000 KB HOST DRAM region; arg1
  SRAM (0xc000 KB = 768 KiB = 24 heads) is per-128-token scratch. State is host-accessible.
- **alpha/beta proj is host-side** (`cpu_func::_cpu_mm_col<bf16>` at 0x83505) — g/beta
  already f32 on CPU; the stub reuses them.
- **out_proj is a separate run** (0x834d5) then sync `0x330(%rbx)` at 0x834e1.
- **partial-token tail**: 0x83359 `jl 838d0` skips the delta-rule emit for <128-token
  chunks — those tails go forward-per-token (existing `_prefill_with_mv` path).
- A 4 MiB memcpy at 0x8347a (src `0x10(%r14)` → dst `0x2e0(%rbx)`, 511·4096·2 bytes)
  is a host-side buffer move (intermediate), not the recurrence.

### GDN prefill is decomposed (confirmed by disasm of linear-block forward 0x83200)
The linear (GDN) block forward emits three separate pieces, so the delta rule
is ISOLATED in `GateDeltaNet_prefill.xclbin` and swappable without touching the
GEMMs or conv1d:
- `gen_dequant_mm_512(seq, …, flm_dtype_t)` (0x94f60) → dequant + GEMM
  (qkv/gate/out projections) — has an explicit dtype param (float32 candidate).
- `gen_seq_conv1d(seq, …)` (0x97f30) → depthwise conv1d (conv.xclbin).
- `generate_gate_delta_net_prefill_sequence(seq, 13×u32)` (0x6f530, called at
  0x833fd) → the delta-rule recurrence + gated norm on GateDeltaNet_prefill.xclbin.

## Native-engine state (for the alternative native hybrid)

`npu_engine_qwen3_6_moe_35b` decode via CPU fallback = **0.6–1 tok/s** (correct,
`gdn_attn_cpu` float32). Profiled 2026-09-11: per-token ≈ 1.7–1.8 s, dominated by
CPU GEMMs at M=1 (qkv_proj [8192×2048], gate_proj, ssm_out_proj, MoE experts
top-8, lm_head 248320×2048 = 508 MMAC). NPU MoE (`NPU_MOE=1`) is launch-bound
at M=1 → 0.32 tok/s. A fused NPU MoE layer kernel is path 1; the lib patch
(path 2) is the shorter route because it reuses FLM's fast NPU GEMM schedule.

## Next steps

- [ ] Decode the 13 args + emitted sequence of `generate_gate_delta_net_prefill_sequence`.
- [ ] Isolate the delta-rule sub-sequence (is it patchable without the kernel source?).
- [ ] Implement the host-float32 delta-rule swap in the lib (rel32 patch, like the
      already-reverted `_prefill_with_mv` redirect at 0x7729c).
- [ ] Validate non-NaN + token parity, then measure decode/prefill/TTFT vs FLM's
      `qwen3.6_results.md` (17.48 tok/s decode, 102–281 tok/s prefill).
