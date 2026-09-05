# Upstream report 02 — Reboot-drift phenomenon (deterministic per boot)

**Target:** ROCm/FastFlowLM (upstream of `third_party/FastFlowLM` @ `17a35cb7`)
**Source:** 1bit-MONSTER/1bit-MONSTER issues #2065 (closed), #2083; tracking doc `npu-infer/docs/txn-decode-findings.md` (rounds R40/R41/R66)
**Severity:** correctness / determinism — outputs move across a reboot with byte-identical inputs

## Summary

Byte-identical runtime outputs — identical binary, libs, xclbins, model, and
inputs — **moved across one reboot**:

- Reboot @ 2026-09-02 23:19 (kernel `7.2.0-next-20260821-unstable`):
  logits corr **0.99803** vs the pre-reboot state, argmax **397 → 144370**.
- The **next fresh boot (R41) reverted**: round-37 signature byte-identical
  again.
- Deterministic within each boot. Engine-side piecewise replay (byte-identical
  TXNs) did **not** move.

## What this rules out / suggests

The engine and runtime submit byte-identical TXNs/ELFs on the same device;
only the runtime's output moved. This points at a runtime-flow component
sensitive to kernel/driver state:

- SVA vs identity-IOMMU DMA,
- load-time staging,
- the documented round-27 x0.5 runner-quirk state,

or a genuine arithmetic difference in kernel/firmware execution across the
reboot.

## Repro

- Harness: `npu-infer/tools/capture/run_qwen3_npu`-style driver, tokens
  `1000`+`1001`.
- Procedure: capture on boot A → reboot → same binary/libs/xclbins/model/input
  → capture on boot B → compare (corr / argmax / byte-parity).
- Byte-parity comparison scripts used in R40/R41.

## Environment

- xclbins: Qwen3-0.6B FLM xclbin set
- Kernel `7.2.0-next-20260821-unstable` (and the pre/post-reboot pair it was
  observed on), XRT 2.26.0 runlist stack, amdxdna driver
- rtcap captures: `/home/bcloud/.cache/rtcap/` on the dev box
