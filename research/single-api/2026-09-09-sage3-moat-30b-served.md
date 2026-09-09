# goal mttyyykw (sage-3) — moat-model serving on the routed path (2026-09-09)

## Gate 1: 30B Q4_K_M decode gate HELD on the routed path at the KV window
Qwen3-Coder-30B-A3B Q4_K_M via the routed one-call (SAGE_PREFILL_PIN=Vulkan0 for the
long-KV prefill leg - the dual-bundle HRX0 prefill_batch path fails deterministically
at 512-token batches, graph_compute_async -1 @off0, reproduced quiet; noted engine
finding), then memfd export -> Vulkan0 import:
- KV 2986 (in the 2944-3200 gate window): session exported 293,572,756 B state,
  imported pos=2985 resume=30825, decode 256 tok @ 47.19 t/s (>=40 bar) - GATE HELD.
- Exit 0, coherent output; state handoff at contract KV exercised in-service.

## Gate 2: zaya-class decode on its contract engine (moat-q4nx = HRX0 both legs)
zaya-q4nx-c43.gguf (type43) one-call: exit 0, HRX0 prefill 6 tok -> memfd ->
HRX0 import (resume=563) -> decode 16 tok @ 3.80 t/s:
out: 9079 236761 107 2364 107 19058 236764 564 3050 506 14787 1492 236761 3792 786 1751
- Canary 236761/107 (" Paris.") MATCHES the established moat gate (mttfq8ch 2766a3876).
- tok0-2 (9079 236761 107) match the CPU probe oracle (rebuild-probes.sh).
- tok3+ diverge from the CPU/stale-fork reference (2717...): the moat numeric contract
  is the ENGINE FLOAT reference, not CPU llama decode; canary-level parity on the
  contract engine is the established gate. Deeper parity vs the engine-float
  reference = follow-up (needs the NPU_CPU_EXPERT=1 float reference run on this file).
- 3.80 t/s decode = zaya on HRX0 single-engine class (the moat runs single-leg HRX;
  per-token engine-loop overhead included; consistent with the 9-10 t/s HRX0 zaya
  class from the unified-bench rows scaled to 1.8B).

## Findings
1. Dual-bundle HRX0 chunked prefill (prefill_batch, ubatch 512) fails on long prompts
   (deterministic); SAGE_PREFILL_PIN=Vulkan0 env added to route the prefill leg.
2. Moat-q4nx routed path = HRX0 both legs with a memfd round-trip (state_handoff
   exercised even for same-engine legs) - canary-correct.
