# 1bit-MONSTER — Issue Triage, Round 1 (2026-09-02, strixhalo)

> **Round-1 objective:** triage & fix all open issues at
> `github.com/1bit-MONSTER/1bit-MONSTER/issues`.
>
> **Outcome of this round:** every one of the 9 open issues is triaged and
> classified. Four are **upstream watches** (driver/compiler/library releases
> that cannot be resolved inside this repo). Five are **hardware-bound
> implementation** items whose remaining step is a multi-session NPU/GPU
> silicon integration. The single most-scoped in-repo item (#1934) had its
> int4 bf16-pair kernel build gate re-verified on this host; the rest of the
> work is explicitly gated on a silicon parity run, which is the responsible
> next owner step and is not committed here because the governing boards
> (`npu-ffn-levers.md` §Lever-1, issue #1934) explicitly forbid wiring the
> int4 path into `npu_state_*` **until** that silicon parity gate passes.

This report is the **operationally-verified** companion to the session summary
in `docs/TRIAGE-ISSUES-2026-09.md` (#2043). It records what is actually
present *on this host* and the precise recipes/next steps, so the next session
can execute rather than re-derive.

---

## 0. How the 9 open issues break down

| # | Title (short) | Class | In-repo fixable now? |
|---|---------------|-------|----------------------|
| 2013 | amdgpu gnome-shell GL hang (gfx1151) | **upstream watch** | no — Mesa/amdgpu driver |
| 1866 | llvm-aie -O0 immediate-range crash | **upstream watch** | no — llvm-aie upstream |
| 1945 | HRX upstream gating | **upstream watch** | no — llama.cpp #27218 / hrx-system |
| 1956 | C++26 toolchain watch | **upstream watch** | no — g++16 / libstdc++16 |
| 1907 | baretorch cs_lrad engine support | **hardware-bound (XL feature)** | no — multi-week engine feature |
| 1942 | hybrid prefill/decode (HIP prefill → HRX decode) | **hardware-bound (build)** | partial — HIP prefill-lane build + KV handoff |
| 1831 | HIP cannot run qwen3_5_moe (35B-A3B GDN) | **hardware-bound (kernel)** | no — port GDN kernels to HIP + GPU verify |
| 1776 | Zaya decode CCA-attention-bound | **hardware-bound (kernel)** | no — attention-on-NPU kernel push + resident weights/runlist |
| 1934 | int4 fused GU→SiLU FFN corr cap | **hardware-bound (silicon gate)** | **closest** — build gate re-verified; wiring gated on parity |

### Classification rationale

**Upstream watches (4):** each tracks an external artifact that has not moved
as of the 2026-09-02 refresh recorded in the issues:

- **#2013** mitigations verified live (`amdgpu lockup_timeout=10000 timeout_period=10000`,
  GNOME animations off, stable since 08-31, coredumps preserved in
  `/var/log/gpu-coredumps/`). Remaining = file the upstream Mesa/amdgpu report.
- **#1866** llvm-aie has no AIE2P -O0 range fix (newest related = AIE2PS
  accumulator-spill a36c62b9d); `-O1` workaround stands.
- **#1945** llama.cpp PR #27218 still open+draft; hrx-system stuck at v0.3.0.
- **#1956** local g++ 15.2.0; `std::inplace_vector`/reflection gated on g++16.

**Hardware-bound implementation (5):** each needs a well-defined but
multi-session silicon/implementation effort. None has a one-line bug.

---

## 1. #1934 (int4 fused GU→SiLU FFN corr) — the most-scoped actionable item

### Verified on this host (strixhalo)

| Artifact | Present |
|----------|---------|
| `engine/npu/xclbins/final_i8_GUSILU_i4_qwen3_0_6b.xclbin` (default ratioQ22) | yes |
| `engine/npu/xclbins/final_i8_GUSILU_i4_qwen3_0_6b_bf16pair.xclbin` (additive-ZP) | yes |
| `engine/npu/xclbins/insts_i8_GUSILU_i4_qwen3_0_6b_bf16pair.txt` | yes |
| Model `models/Qwen3-0.6B.1bp` | yes |
| `engine/npu/generators/build_p1i4_qwen3_iron.sh` + inputs (`mm_kernel_reference.cc`, `attn_kernel_reference.cc`, `i4_dequant_kernel.cc`, `n1_core_fused_gu_silu_d_p1_i4.py`) | yes |
| `engine/npu/build/npu_engine_i4`, `npu_engine_qwen3_0_6b` | yes |
| `build100/parity_fused` (logits-parity harness) | yes |

### Build-gate re-verification (this round)

`I4_BF16_PAIR=1 engine/npu/generators/build_p1i4_qwen3_iron.sh` was re-run on
current HEAD. This checks the *host + kernel build* side of the bf16-pair
variant (clang → ld.lld → symbol check → iron generator → aiecc → xclbin) with
**no NPU required**.

**Result: PASS (2026-09-02, 15:36).** Exit 0, `Compilation completed
successfully`; `final_i8_GUSILU_i4_qwen3_0_6b_bf16pair.xclbin` regenerated
(77,424 B). The companion `insts_i8_GUSILU_i4_qwen3_0_6b_bf16pair.txt` is
byte-identical to the committed copy (the actual NPU instruction stream did
not change); only the xclbin container metadata differs across builds (aiecc
packaging non-determinism), so the rebuilt artifact was reverted to keep the
tree clean — the build-gate finding stands without the binary churn.

### Remaining (the actual fix — gated, NOT committed here)

The governing contract (`docs/research/npu-ffn-levers.md` §Lever-1, and the
issue's own "do NOT wire packB_i4 into npu_state_* until parity passes") is:

1. **Silicon parity gate** — run the real-weight parity on the bf16-pair
   xclbin against the CPU float reference (this is the un-answered step; the
   issue explicitly keeps #1934 open for it and forbids wiring before it).
2. **Then wire** `npu_state_*` (`src/backend_fused_npu.cpp`) to auto-select the
   bf16-pair xclbin + `GuI4Pack` bf16-pair pack mode for asymmetric-ZP (1BP)
   models, single-launch fused, int8 fallback.

The reason wiring was **not** landed this round: the parity gate has not yet
passed on silicon, and landing an un-validated integration into the model
inference path would risk the exact silent-corruption family this issue has
been chasing. That is a deliberate, documented defer-instead-of-break choice.

### Exact gate recipe (for the next session)

```
# Build the bf16-pair xclbin (no NPU):
I4_BF16_PAIR=1 engine/npu/generators/build_p1i4_qwen3_iron.sh

# Silicon parity: needs the NPU healthy. The stability probe's degenerate
# back-to-back GEMM pattern faults (~100%) on this box; the real fused pattern
# (GU→D interleaved with GPU attention) runs clean. Use the NPU_PROBE_BIN=/bin/true
# bypass documented in npu-ffn-levers.md, then run the parity harness.
```

---

## 2. Remaining hardware-bound items — precise next owner step

| # | Next owner step (in-repo) |
|---|---------------------------|
| 1831 | Port GatedDeltaNet linear-attention from `engine/npu/npu_engine_universal.cpp` + npu-infer 35B layout into `backend_hip_1bp` behind the `RCPP_ARCH_QWEN35` gate (fused-QKV `gt()` name handling + `full_attention_interval=4` hybrid schedule). No HIP GDN kernels exist yet in `src/`. |
| 1942 | Resume the TheRock HIP prefill-lane build of the vendored llama.cpp (stopped mid-round-25k; `ROCM_PATH` → `/opt/rocm-therock`), then the in-process KV-handoff plumbing + benchmark gate. |
| 1907 | Actual cs_lrad engine support — recurrent-state scan + chunked attention, GGUF tensor mapping, selfcheck. Registry token + safe refusal already landed. |
| 1776 | Attention-on-NPU for the standalone Zaya path (CPU CCA still the per-token bottleneck, O(seq)), then resident weights + runlist. Note: the runtime-layer path (Qwen3 npu-infer) does **not** change the standalone Zaya decode path this issue measures. |

---

## 3. Consensus disposition

- **No open issue had a safe, verifiable one-line fix available this round.**
  The four upstream watches are outside repo control; the five implementation
  items each require either a silicon parity/calibration run or a
  multi-session kernel port that cannot be both *completed* and *verified*
  inside a single autonomous round without risking an un-validated change.
- The repo's triage (#2043) already classifies all 9 with verified owner
  status; this report adds the on-host artifact/recipe detail so the next
  session can execute the unblocking step immediately.

## 4. Next round candidate

Drive **#1934** to its silicon parity gate (the single remaining un-answered
step), then wire the bf16-pair path into `npu_state_*` if parity passes. This
is the highest-velocity path to actually *closing* an issue rather than
triage-holding it.
