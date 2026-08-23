# Single-launch fused decode: core-stream h2 relay (issue #1775 — the real fix)

## Why every other approach fails (verified on strixhalo)

| Approach | Result |
|---|---|
| npu.sync / npu.dma_wait / dma_wait(H2_S) barriers | built + tested on two fresh boots — TCT fires at DMA-accept, not cross-column DDR visibility |
| per-MoE-layer h2 BOs (f8cd4187) | fixes the layer-3+ HANG; not the token flips |
| split two-launch with host h2 sync + drain-read (f4b74476) | works (#1767 partial-fusion milestone) but the shared two-launch path (NPU_FUSED=0) also flips tokens — the handoff race is the S2MM-writeback TCT≠visibility class affecting every NPU output readback |
| BD locks (get_lock_acq/rel, exposed on aiex.npu.writebd) | per-tile resources — the D-phase A2 MM2S on shim[0] cannot wait for shim[1..7]'s locks |
| mem-tile h2 relay (object fifos) | mem-tile DMA channel budget: 3 in / 3 out "at the measured limit" (n1_core_fused_gu_silu_d.py comment); the relay needs 4+ in — over budget |
| core DMA channel relay (A_m mem→core) | core budget 2 in / 2 out — A+A2+B = 3 in, exceeded (measured) |

## The design that fits the budgets: core-to-core STREAM relay

Streams are separate from the DMA channel budget (the AIE switch + core stream
ports). Keep h2 in CORE tile-local buffers (no DDR, no H2_c fifo writeback):

```
GU phase (per column c): A(residual shim0→cores, existing A_c) + B_gu → C1 →
  silu → h2 chunk kept in core[c] tile-local h2buf[c]   (8×256 int8 = 2 KB)
Relay (core-stream, via switchbox config + cascade-style core kernels):
  core[c] sends h2buf[0..c] to core[c+1]; each core keeps received chunks in
  its own h2buf — after the chain every core holds the FULL h2 (8×2048 = 16 KB)
D phase: A2 read directly from the core's own h2buf (no fifo) + B_d → C2
  (existing C2_c/C2_s writeback to DDR, host readback as today)
```

Channel/buffer budget:
- core DMA: A_c + B (2 in), C2 (1 out) — H2 uses the CORE STREAM port, not DMA ✓
- core L1: existing ~43 KB (batch-2 depths) + h2buf 16 KB + relay staging 2 KB ≈ 61 KB < 64 KB — MUST re-verify with aiecc (the old design already shrank depths to 2; expect similar)
- relay traffic: ~144 KB/layer through 128-bit core streams ≈ 10 µs — negligible vs ~300 µs/layer

## Implementation requirements (why this is multi-session)

1. Core-to-core stream ops + switchbox config in the generator. The old dialect
   API (aie.dialects.aie) exposes `switchbox` but the core-stream acquire/produce
   + routing must be verified; the iron API (aie.iron, CascadeFlow) is the
   supported path for worker-to-worker streams — likely a rewrite of the
   generator into the iron paradigm.
2. The relay chain kernel: each core passes chunks downstream while keeping
   copies (a core loop over its stream port + tile-local stores).
3. aiecc build + hardware iteration on strixhalo (clone ~/1bit-fused-verify has
   the toolchain, models, all xclbin variants, the built binary).
4. Determinism gate: 8+ runs of the same binary/model/prompt must produce
   identical tokens; L1 h2 fingerprint sum=1037 must hold.

## Interim (already committed, reliable)

- f8cd4187 per-MoE-layer h2 BOs — kills the hang mode.
- f4b74476 split GU→SiLU + D launches with host h2 sync — correct output, the
  #1767 partial-fusion milestone; the fallback if the stream relay stalls.

## Implementation attempt result (stream relay)

The core-stream relay v2 generator (n1_core_fused_gu_silu_d_v2.py) was
implemented and generates valid MLIR (42 stream ops + 8 flows), but BOTH
the strixhalo mlir-aie and the dump's npu2_40_toolchain REJECT it at the
aiecc verifier stage:

    error: 'aie.put_stream' op expects parent op 'aie.core'

The verifier requires `aie.put_stream`/`aie.get_stream` to have `aie.core`
as their IMMEDIATE parent — stream ops inside `scf.for` loops are not
accepted. The relay needs 128 stream beats per 512 B chunk × ~56 chunk
transfers per core per layer; without loops that is ~7k unrolled stream
ops per core (AIE core code-size limit exceeded). The iron API
(CascadeFlow, aie.iron) is the remaining path — worker-to-worker streams
are a different (dataflow-level) lowering that may route around the
verifier, but it is a complete paradigm rewrite of the generator
(multi-day, uncertain).

CONCLUSION: the single-launch h2 broadcast cannot be built with the
available toolchains' core-stream abstraction. The definitive fix options
are (a) iron-API generator rewrite, (b) a toolchain whose verifier allows
stream ops in loops, or (c) a hardware mechanism outside these
abstractions. The committed two-launch split (f4b74476) remains the
reliable interim; note the shared-path token flips (two-launch included)
still need the S2MM-visibility investigation.
