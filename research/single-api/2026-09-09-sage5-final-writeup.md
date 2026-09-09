# Single-API engine (SAGE) — final writeup (goal mttyyykw-ggiumm, sage-5, 2026-09-09)

Landing the unified single-API inference engine on the op-class-split architecture.

## Baseline -> final delta
- Before: the served engine routed per-token via BackendManager/strategy (single-leg);
  the phase-split PhaseRouter (prefill -> memfd -> decode) existed only in test mains.
- After: SAGE_PHASE_ROUTE=1 serves policy-class requests through the PhaseRouter in
  the actual unified server: /v1/completions -> generate_completion hook ->
  route_phase_request (cached ServedRouter) -> policy leg -> memfd -> decode leg.
  backend_used reports "phase:hrx0->Vulkan0"; fail-close fallback verified.

## What integrated (commits on goal/sage-single-api)
- serve_router.{h,cpp}: PhasePolicy rows (stock-q4k: HRX0 prefill -> memfd -> Vulkan0
  decode; moat-q4nx: HRX0 both) + two device-pinned Inprocess engines (c7710a6d).
- unified_server hook + BackendManager::config() accessor + SAGE_PREFILL_PIN env
  (dual-bundle HRX0 long-prefill engine limit documented) (c7710a6d, c4f4fe75).
- router_serve_main: one-call contract harness (c7710a6d).

## Gate evidence
- Oracle-exact routed decode 0.6B Q4_K_M: 12095 13 576 6722 315 9625 374 1083 279...
  (router_serve_main); served HTTP path backend_used=phase:hrx0->Vulkan0.
- Token-identity at 30B (sage-2): routed == fork-Vulkan0 direct
  (12095 13 576 6722 315 32961 374 37169) via memfd (293MB @KV2986).
- 30B gate HELD in-service: 47.19 t/s >= 40 @ KV 2986 (tg256); pp2138->tg512
  7.844s e2e decode 65.27 t/s.
- zaya moat canary on the contract engine (HRX0): 236761/107 " Paris." PASS.
- Unified bench (sage-4): routed rows vs Vulkan/HIP/HRX/NPU baselines; engine-loop
  overhead 52-74% of direct llama-bench, amortizing with size.

## Remaining (documented follow-ups, not acceptance gates)
1. Raw t/s parity with llama-bench direct (engine per-token loop tuning).
2. Dual-bundle HRX0 prefill_batch long-prompt failure (engine bug).
3. zaya deep-token parity vs the engine-float reference (moat numeric contract).

## Constraints honored
Dedicated branch (goal/sage-single-api, PR-tracked, auto-pushed); live production
server (:8088, npu-verify checkout) untouched (tests on :8123); shared services
undisturbed; correctness gates green every iteration.
