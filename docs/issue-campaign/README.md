# Issue Campaign — Close-out

Goal `mtubusx1-swek0b` · coordinator @agent-429168 · peers @agent-f85526 (+ @agent-07e844, dropped off mid-run) · 2026-09-09

**Result: 19/19 in-scope issues diagnosed with a confirmed/hypothesized root cause + actionable fix plan.** Diagnosis records: [diagnoses-wave1.md](diagnoses-wave1.md) (#2117/#2115/#2116/#2145/#2147/#2153), [diagnoses-wave2.md](diagnoses-wave2.md) (#2152/#2150/#2159/#2113/#2114/#2102), [diagnoses-wave3.md](diagnoses-wave3.md) (#2105/#2080/#2081/#1866/#2082/#2139/#1942). Full inventory: [FIX-LIST.md](FIX-LIST.md).

## Headline findings

1. **The HRX fork (`fix/hrx-ngl-init-order`) is far ahead of the issue bodies.** The HRX systemic cluster (#2115/#2116/#2117) is largely fixed on the fork; several issues are already resolved there and just need GitHub-side updates.
2. **#2116 (zaya ngl99 corruption) is RESOLVED** for correctness (8df635cb0 oracle-exact); only perf remains (→ #2150/#2159).
3. **#2153 (batched flash) is RESOLVED** (41628ab20 offset-rebind dispatch); residual = a device→CPU lm_head boundary copy.
4. **#1942's stated blocker (cross-backend KV handoff) is RESOLVED** via zero-copy memfd/SCM_RIGHTS + PhaseRouter (branch `feat/zc-mem-handoff`); remaining = #2145 + #2082.
5. **#2159 is a DECISION issue** — HRX-ALONE is capped by a loom/RADV codegen deficit; the dual-engine auto-route + `GGML_HRX_DISABLE=1` already clears all bars but is not "HRX-alone".
6. **#2139's quant lanes already exist** — the 1BP model-loading GATE (not the kernels) refuses `quant2 != 3`.
7. **#1866 reproduced + localized** — peano aie2p CodeGen frame-offset immediate overflow, not an -O miscompile.

## Dependency map (post-diagnosis)

```
#2117 (claim precision) ─► #2116✅, #2147 (op TBD)
#2115 (host path)      ─► #2116✅ (worked around)
#2145 + #2082          ─► #1942 (D2 hybrid acceptance)
#2113 ─► #2114 ─► #2081 (int4 quality-gate data)
#2113 + #2114 + #2070  ─► #2080 (cascade → whole-layer)
#2102 ─► #2105 (permutation vs geometry)
#2150 ─► #2080 (single-launch executor substrate)
```

## Execution model

Round-robin across me + @agent-f85526 (2 agents after @agent-07e844 dropped). Each issue dispatched with a self-contained knowledge packet via `mesh_send(awaitReply:true, block:false)`; group verdict collected via `mesh_wait_all`. Coordinator issues done in-session against fork/repo sources.

## What remains after this campaign (not in scope)

The campaign produced diagnoses + plans only — no code landed, no PRs opened. The plans are staged by priority:
- **P0 (correctness):** #2145, #2147 (needs one repro capture), #2102/#2105 (permutation + geometry), residual #2117 MUL_MAT_ID predicate.
- **P1 (perf/validation):** #2150 (conv kernels → single-launch), #2159 (decision), #2113/#2114 (sweep + D-phase), #2082 (stability matrix), #2152 (concat readback).
- **P2 (feature/decision/toolchain):** #2081 (int4 decision), #2080 (whole-layer authoring), #1942 (hybrid acceptance after #2145/#2082), #1866 (peano upstream or frame shrink), #2139 (gate generalization).
