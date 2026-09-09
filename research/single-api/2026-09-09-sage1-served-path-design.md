# sage-1 design decision — served-path phase routing (2026-09-09)

Branch goal/sage-single-api @ 5949b67f (merged origin/main; 0 behind / 27 ahead).

## Discovered plumbing (verified in-tree)
1. The 1bit engine's served binary = tools/unified_server.cpp ("1bit unified"), NOT
   src/server/* (legacy). generate_completion() drives BackendManager + a strategy
   engine (issue #1271): strategy_engine->route(ctx) -> mgr.select_backend(backend)
   per token — a per-TOKEN routing layer, single-leg, no phase split.
2. The phase-split proof lives in src/router/* mains: PhaseRouter + PhasePolicy
   {model_class, prefill_engine, decode_engine, state_handoff}; both legs are
   hrx::Inprocess instances with DEVICE PINNING (HRX0 | Vulkan0) — the fork-llama
   bundle embedded in the engine runs either engine; state handoff = export_session_mem
   (memfd, session-v9) -> load_session_mem on the decode instance. Verified:
   0.6B oracle-exact (12095/13/576/6722/315/9625/374/1083); 30B dual-leg
   (HRX0 prefill -> memfd -> Vulkan0 decode) 31.08 t/s = 3.1x HRX0-only; 30B
   pp2146->memfd->tg512 = 8.015 s e2e (decode leg ~80-85 t/s).
3. BackendManager's HRX backend imports state via HRX_STATE_MEMFD / HRX_STATE_FILE
   (backend_hrx.cpp ~126-137) but has NO export path; export lives on
   hrx::Inprocess::export_session_mem only.

## Design decision (served hook)
For model classes with a PhasePolicy that SPLITS phases (stock-q4k 30B-class:
prefill engine per 841552005 table, decode Vulkan0; moat-q4nx: single-leg HRX),
generate_completion() delegates the request to a PhaseRouter instance registered
with two device-pinned hrx::Inprocess engines (prefill pin X, decode pin Y) built
from the served model — the router mains' proven construction, promoted into the
server path. Non-policy classes keep the existing BackendManager/strategy path
unchanged. Trace: per request emit "phase_route model=<tag> prefill=<engine>
decode=<engine> state=<memfd|none>" (stderr + response meta). Fail-close: if
either engine fails init or the memfd handoff fails, fall back to the existing
path with a trace line (never a broken request). Correctness gate per routed
class: oracle tokens vs the single-engine reference (0.6B exact; 30B row).

## Implementation order
1. Engine factory: build the two pinned Inprocess engines from the served model
   config (reuse router/hrx_prefill_engine.h + hrx_decode_engine.h).
2. Policy lookup: PhasePolicy rows keyed by model class (extend model_router's
   class detection used by the 841552005 table).
3. generate_completion branch + trace + fail-close fallback.
4. Gates: router_test-style oracle per class (0.6B/1.7B/4B stock-q4k; 30B dual-leg
   row ~80-85 t/s decode; moat single-leg HRX unchanged).
