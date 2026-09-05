# Upstream report 01 — Batched prefill(ids) RoPE divergence vs N×forward(tok)

**Target:** ROCm/FastFlowLM (upstream of `third_party/FastFlowLM` @ `17a35cb7`)
**Source:** 1bit-MONSTER/1bit-MONSTER issues #2052 (closed), #2083; tracking doc `npu-infer/docs/txn-decode-findings.md` (rounds R39/R67)
**Severity:** correctness — internal inconsistency of the runtime's own decode path

## Summary

`model.prefill(ids)` with a batched token list produces **different KV/rope
state** than N × `model.forward(tok)` over the same prompt. This is an
internal inconsistency of the runtime's own decode path, not a caller bug:
both paths are expected to produce identical state for the same prompt.

## Observed mechanism (captured via interposer)

- Batched prefill **never advances the host-side RoPE (i6) table** — the
  table is the identity table across all dumps.
- `forward()` updates the RoPE table per position as expected.
- The batched kernels then apply RoPE **internally** from an exact-math
  `inv_freq` table that disagrees with the runtime's hardcoded `.rodata`
  `inv_freq` at exactly the **j=25 / j=57** rope classes → bf16 flips →
  argmax flips on near-ties.

## Numbers (Qwen3-0.6B)

- 4-token probe: keys corr **0.9999** — only rope elements 50/51 + 114/115
  differ (growing head set); logits corr **0.9777** with an argmax flip.
- 162-token prompt: corr(mm, seq) = **0.926976**, maxdiff **6.91**.
- Deterministic and bit-reproducible within a boot; stable property of a
  **healthy** runtime (R67).

## Repro

- Harness: `npu-infer/tools/capture/run_qwen3_npu`-style driver, tokens
  `1000`+`1001`.
- Compare (a) `model.prefill(ids)` for the prompt vs (b) N×
  `model.forward(tok)`; dump per-position KV + logits via the rtcap
  interposer (`/home/bcloud/.cache/rtcap/`).
- Byte-parity comparison scripts used in R39.

## Impact / note for us

Per-ctx comparisons must use the `forward()` path (already the documented
methodology in our tracking doc). The upstream question is which of the two
RoPE applications is authoritative and why they disagree.

## Environment

- xclbins: Qwen3-0.6B FLM xclbin set (mm/attn/layer/dequant)
- XRT runlist stack, amdxdna driver (kernel/driver IDs per capture log;
  kernel `7.2.0-next-20260821-unstable` generation)
- FastFlowLM runtime libs (libqwen3_npu.so family), XRT 2.26.0 runlist stack
