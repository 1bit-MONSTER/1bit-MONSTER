---
name: systematic-debugging
description: Use when debugging a bug, hang, OOM, crash, or wrong-output — trace the root cause through evidence instead of guessing. Covers HIP error surfacing, OOM-at-layer-N patterns, stale async errors, and timing "hangs" that are really slow paths.
whenToUse: Any "why is X failing", crash, hang, OOM, or incorrect output investigation.
---

# Systematic Debugging (4-Phase Root Cause)

## Phase 1 — Reproduce with evidence

- Capture the exact command, env, model file, and hardware state (e.g. `zaya_chain_check models/ZAYA1-74B.1bp` on the Strix Halo box).
- Record the failure signature: exit code, first error line, last log line. **The last log line is usually a red herring** — a stale HIP error surfaces at the next `hipGetLastError`/reset call, far from the cause.
- Check failure mode classes first:
  - OOM: which allocation fails, at which layer/size (`hipMalloc` failure messages).
  - >2 GiB copies: `hipMemcpyAsync` takes a signed 32-bit byte count — any tensor over 2 GiB returns `hipErrorInvalidValue` (issue #1715 class).
  - Apparent hang after load: stale async error + first reset HIP check; or slow path (page-cache thrash) that is really "loads but takes 15 min".
  - Garbage output: orientation mismatch (transposed weights), quantization layout mismatch, or expert-indexing bugs.

## Phase 2 — Instrument the suspect path

- Add per-layer/per-tensor timing or progress output at the suspected stage (the issue template asks for "which tensor/layer" — produce exactly that).
- Use `perf`/`rocprofv3` when the box allows; else wall-clock around phases (`std::chrono`).
- For memory pressure: check `free -g` at timeout, note buff/cache vs available.

## Phase 3 — Isolate the root cause

- Binary search: disable halves of the pipeline (packed vs bf16 path via the A/B masks, per-token vs chain, layer ranges).
- Compare against a known-good reference (legacy mask-0 path, smaller model with same format, llama.cpp output for the same model).
- State the root cause as a single mechanism ("the signed 32-bit byte count truncates…"), not a symptom ("load hangs").

## Phase 4 — Fix, then verify it's actually fixed

- Fix the mechanism; then run the *original repro* end-to-end (not a reduced case) and record the evidence (load time, tokens, exit code).
- Keep the evidence: log file path, command, before/after numbers — report them. See `verification-before-completion`.

## Defense in depth (when fixing)

- Make the failure *loud* at the boundary where it happens: check HIP errors where the API is called, not at a later reset.
- Add the regression as a test (see `test-driven-development`) so the class of bug is caught by CI, not rediscovered on a 44.6 GB model.
