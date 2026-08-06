---
bug_id: BUG-001
status: fixed
severity: high
scope: tests/zamba2
title: test_mamba2_kernels_validation red — stale Mamba1-style CPU reference vs true Mamba2 scan kernels
---

# BUG-001: test_mamba2_kernels_validation red

## Symptom (gate)

`test_mamba2_kernels_validation` (ctest label `gpu_required`, golden GPU suite)
fails at HEAD:

```
── Test 3: Selective Scan (fused) ──
  Output err: 2.00e+00  State err: 2.00e+00  ❌ FAIL
── Test 4: Full Mamba2 Decode Block ──
  Output err: 1.87e+00  Conv state err: 4.12e-03  SSM state err: 2.00e+00  ❌ FAIL
```

err 2.00 = max relative error with opposite-sign/zero-vs-nonzero values —
total state mismatch, not precision noise. Tests 1 (GEMV) and 2 (Conv1D) pass.

## Reproduction

```
cd build && ./test_mamba2_kernels   # requires ROCm device
```

## Root cause (4-phase RCA)

1. **Reproduce**: run at HEAD → Tests 3+4 fail (above).
2. **Isolate**: kernel vs test CPU reference disagree on the SSM scan. The
   engine's own CPU path (`src/mamba2_kernels.cpp`) and the HIP kernels
   (`src/mamba2_kernels.hip`) agree with each other; only the test's
   `cpu_selective_scan` diverges.
3. **Hypothesize**: the test reference encodes two stale conventions that the
   kernels deliberately fixed (Aug 4–5) for real-model correctness:
   - `7d50c9065` — A convention: GGUF `ssm.a` already stores `A = -exp(A_log)`
     (#1460); kernels use `A_bar = exp(dt_sp * A_log)` directly. Test still
     re-applies `-expf(A_log)`.
   - `89075c30f` — state semantics: true Mamba2 state is
     `[head][d_state][head_dim]`, each head_dim slice evolves independently
     once per token. Test still uses Mamba1-style shared per-head state
     updated sequentially across head_dim ("decays x[0] into x[1..]"). Same
     commit added the HF `dt_min=0.001` clamp, also missing from the test.
4. **Verify**: official `mamba2.py` `step()` (state shape
   `(batch, nheads, headdim, dstate)`, `dBx = einsum("bh,bn,bhp->bhpn")`,
   one update per (h,p) per token) and vendored llama.cpp
   (`ssm_state` reshaped `[d_state, head_dim, n_head]`) both match the
   kernels. The test's CPU reference matches neither — it was last touched
   `0d2ed2e40` (Jul 30), before the Aug 4–5 kernel correctness fixes.

## Fix approach

Update `cpu_selective_scan` in `tools/test_mamba2_kernels.cpp` to mirror the
kernel: pre-negated A convention, per-(head,head_dim) state slices
(`final_state[h*d_state*head_dim + s*head_dim + hd]`), and the
`dt_sp ≥ 0.001` clamp. Kernels stay untouched — they are correct per
official Mamba2.

## Verify steps

- [x] `cd build && make test_mamba2_kernels && ./test_mamba2_kernels` → all 4 tests PASS
- [x] `ctest --test-dir build -L host --output-on-failure` → all pass
- [x] `ctest -R mamba2` + `test_ssm_a_convention` → green
