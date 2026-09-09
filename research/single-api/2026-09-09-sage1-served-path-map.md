# goal mttyyykw (sage-1) — PhaseRouter served-path integration map (2026-09-09)

Branch goal/sage-single-api (worktree ~/wt-sage-single-api off feat/zc-mem-handoff 9b19030c).

## What exists (verified in-tree)
1. engine::PrefillEngine / DecodeEngine interfaces + PrefillResult{state_fd(memfd),...} —
   the zero-copy state contract (phase_router.h).
2. PhasePolicy {model_class, prefill_engine, decode_engine, state_handoff} + PhaseRouter
   (add engines, set_policy, generate(prompt,max_tokens)) — ONE-call phase-split API.
3. Fully-live proof: router_test_main mode 1 (HRX0 prefill -> memfd -> Vulkan0 decode,
   real tokenize+export+import, no files) — 0.6B oracle 12095/13/576/6722/315/9625/374/1083
   PASS (9b19030c); router_duallep_main (both legs per policy, 30B 31.08 t/s = 3.1x HRX0-only);
   bench_contract_main (30B pp2146->memfd->tg512 = 8.015s e2e, decode leg ~80-85 t/s).
4. model->backend route (select_backend_route, model_router.cpp) IS consumed by the served
   path today (backend_manager.cpp 679/1119/1238) but is a single-leg chain (no phase split,
   no memfd handoff).

## The sage-1 delta (router live in the SERVED path)
- Hook: the /api/generate request handler (server.cpp) + backend_manager's route decision.
  When the model's class has a PhasePolicy that SPLITS phases (or state_handoff), the request
  must run ONE generate via PhaseRouter (prefill leg -> state_fd -> decode leg) instead of the
  single-leg chain; policy rows per model class from 841552005 (corrected-decode basis).
- Observability: a trace line per request naming prefill_engine + decode_engine actually used
  (the "phase-routing observable in a bench/server trace" acceptance).
- Gates: oracle-exact per routed class (0.6B/1.7B/4B stock-q4k; 30B gate row; moat-q4nx
  classes stay single-leg HRX per policy).

## Open design points for the implementer
- server.cpp request path: which generate() entry the handler calls today and where the
  PhaseRouter instance (engines registered per model) plugs in; whether engines are per-model
  singletons or per-request.
- Policy precedence vs select_backend_route chains (phase split only when the class row says so).
- Trace format + env gating (route trace to stderr/json like the engine's other logs).
