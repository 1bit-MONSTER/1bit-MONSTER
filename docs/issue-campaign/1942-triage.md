# Issue #1942 — Triage & Status Recommendation

**Triage date:** 2026-09-11 (goal `mtxsovtr-fbqmsj`)
**Issue:** feat(hrx): hybrid prefill/decode policy — HIP large-prefill + HRX warm decode
**State:** open · labels: `performance`, `backend` · 20 comments · author/owner: bong-water-water-bong

---

## 1. Current-state summary (evidence-backed)

### 1.1 Resolved — the stated blocker is gone

The issue body's original blocker — **cross-backend KV handoff** — is **RESOLVED**:

- **State-format round-trip is byte-identical** between the engine's vendored llama.cpp and the HRX bundle's libllama (`LLAMA_SESSION_VERSION` 9/9). Verified 2026-09-02 (comment 5512071871; `docs/research/hybrid-prefill-decode.md` §5.1).
- **Zero-copy memfd/SCM_RIGHTS fd handoff proven** (29.4 MB, token-identical) + **PhaseRouter single-API** E2E on 0.6B and 30B — branch `feat/zc-mem-handoff` (`9eaefa7c5`). (comment 5605667444; `docs/issue-campaign/FIX-LIST.md` headline #4)
- **#2082 CLOSED** — D2 HIP-prefill stability gate passed: ≈1,900 evals across compilers/HIP/flag grids with **zero crashes**; the 09-02 nondeterministic prefill SIGSEGV no longer reproduces on the vendored tree (llama-v040 @ 5266f24). (comment 5568819661; issue #2082 state=closed)
- **Handoff correctness proven on the 30B**: vendored-HIP prefill blob decoded 48/48 token-identical by the fork; no context loss. (`hybrid-prefill-decode.md` §5.2, 2026-09-07)

### 1.2 Landed — silent-context-loss fix (#2203)

The 09-11 §5.2 verdict exposed a failure **worse than a decode error**: on pre-fix `main`, `HrxBackend::reset()` discarded the init-time `HRX_STATE_FILE` import, so the lane returned `finish_reason: stop` while decoding from an **empty KV** — silent context loss, invisible at the call site. (comment 5632420826)

**PR #2203** (merged 2026-09-11, commit `80a8a81eb`) fixes it:
- decode-side guard counts a **resumed** context (`imported_ctx_tokens + prompt_tokens`) against `HRX_MAX_CTX_TOKENS` (default 2048, `0` disables);
- re-imports the state after `reset()` (`HRX_REIMPORT_STATE=0` to opt out) and reports re-import failure instead of decoding from nothing. (PR #2203 body)

### 1.3 NOT met — §5.2 positive clause on the shipped path

The re-scoped §5.2 acceptance (owner Option A, 2026-09-08, comment 5589225170) = **the D2 shipped path delivers CORRECT warm decode on the HRX device** (HIP large-prefill → llama_state handoff → HRX continuation, no context loss). **This is NOT met on the shipped bundle**:

- With #2203's guard on (`HRX_MAX_CTX_TOKENS=2048`): explicit refusal naming the cause (routes to fallback / fails closed).
- With guard off (`=0`): the real ceiling surfaces — `unsupported HRX node 25: FLASH_ATTN_EXT … f16[128,3072,4,1]` → `compute status: -1`.

Root cause is **bundle-side**: the shipped b66 HRX backend **over-claims `FLASH_ATTN_EXT` for any KV > 2048**, so a 2,940-token imported context cannot decode on the HRX device. The local GET_ROWS-capable build decodes the same blob but **segfaults the engine** (split lib layout) — usable only as the `rt_b66` harness lane, not an engine lane. (comment 5632420826; PR #2203 body)

### 1.4 Remaining blockers (open)

| Issue | State | What it gates |
|---|---|---|
| **#2145** | open | b66 bundle fails HRX0-device decode at token 2 on imported ctx (GET_ROWS gap) — blocks the D2 shipped fast path |
| **#1945** | open | upstream watch: llama.cpp PR #27218 (draft) / stable `hrx-system` release / bundle repin — gates any bundle that supports >2048 KV |

### 1.5 Re-open triggers (from the issue body, unchanged)

Either trigger re-activates the matrix (pinned blob `~/hrx-2145/hyp_blob.bin` session v9, model `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M`, `HRX_MAX_CTX_TOKENS=0` to expose the raw lane):

1. **(a)** upstream bundle repin — **#1945**;
2. **(b)** HRX2 decode-ADD coverage (the 09-08 decision's named watch item).

---

## 2. Recommendation

**Keep #1942 OPEN, parked — do not close it.** Narrow the body to its measured state, matching the 09-11 verdict's own disposition ("narrow this issue to its measured state rather than closing it", comment 5632420826).

Rationale:
- The **feature is genuine and unresolved**, not obsolete: HIP large-prefill + HRX warm decode remains the design goal (`hrx-engine-goal.md` §Strategic position item 4), and the value case (small-ctx warm decode) is unaffected by the ≥2k benchmark being HIP-favored.
- The remaining blocker is **upstream-gated** (#1945) — nothing local can move it (GET_ROWS/FLASH_ATTN_EXT ceiling is a proven bundle-side dead end per #1945 body).
- The two re-open triggers are concrete and actionable; closing would lose the parking place that keeps them tracked.

**Concrete next actions (recommended):**
1. **Update the issue body's "Status" header** to state: handoff resolved, #2082 closed, #2203 landed (silent loss removed), remaining = bundle `FLASH_ATTN_EXT` over-claim (#1945-gated) + #2145. This is the single highest-value action — the body currently reads as if the blocker is still the KV handoff.
2. **Optionally post the status comment** in §3 to record the 09-11 state on-thread (the verdict already lives in comment 5632420826, so this is optional; posting only on owner's go-ahead).
3. **Leave #2145 and #1945 open** as the two watch items; do not open new sub-issues.

---

## 3. Ready-to-post status comment draft

> **Status 2026-09-11 — narrowed, parked, kept open (owner's call).**
>
> The stated blocker (cross-backend KV handoff) is **resolved**: state round-trip is byte-identical (`LLAMA_SESSION_VERSION` 9/9), zero-copy memfd handoff is proven, and #2082 is closed. **PR #2203** removed the silent-context-loss failure mode (decode-side ctx guard counting a *resumed* context + re-import after reset).
>
> What remains is **bundle-side and upstream-gated (#1945)**: the shipped b66 HRX backend over-claims `FLASH_ATTN_EXT` above KV 2048, so a 2,940-token imported context cannot decode on the HRX device (guard on → honest refusal naming the cause; guard off → `unsupported HRX node 25: FLASH_ATTN_EXT`). §5.2 positive clause is therefore **not demonstrable on this box as configured**.
>
> **Re-open triggers (either):** (a) upstream bundle repin **#1945**; (b) **HRX2 decode-ADD** coverage. On either, re-run the pinned matrix (`~/hrx-2145/hyp_blob.bin` session v9, `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M`, `HRX_MAX_CTX_TOKENS=0`).
>
> Evidence + write-up: `docs/research/hybrid-prefill-decode.md` §5.2; verdict comment above.

---

## Appendix — evidence index

| Ref | Type | Points to |
|---|---|---|
| issue #1942 body | issue | original acceptance + current Status header + re-open triggers |
| comment 5512071871 | comment | state-format gate passed (byte-identical, 9/9) |
| comment 5568819661 | comment | #2082 matrix clean, crash blocker cleared |
| comment 5578331959 | comment | owner decision (09-08) keep parked (option B) |
| comment 5589225170 | comment | §5.2 re-scope (option A) = correct warm decode |
| comment 5589757236 | comment | shipped-path D2 demo (500 tok via non-default HRX_ROOT build) |
| comment 5605667444 | comment | campaign diagnosis: handoff resolved; remaining #2145+#2082 |
| comment 5632420826 | comment | §5.2 verdict: positive clause NOT met; silent loss found; disposition |
| PR #2203 | PR (merged) | ctx guard + re-import-after-reset fix |
| PR #2146 | PR (merged) | llama_state import shim + HRX_STATE_FILE |
| PR #2156 / #2157 | PR (merged) | 30B RMS_NORM=CPU fix / §5.2 docs re-scope |
| #2145 | issue (open) | imported-ctx HRX0 decode fails token 2 (GET_ROWS gap) |
| #1945 | issue (open) | upstream gating (PR #27218 / stable hrx-system / bundle repin) |
| #2082 | issue (closed) | D2 HIP-prefill stability gate |
| `docs/research/hybrid-prefill-decode.md` | doc | design + §5.1/§5.2 evidence |
| `docs/research/hrx-engine-goal.md` | doc | §Strategic position item 4 (hybrid = next build) |
| `docs/issue-campaign/FIX-LIST.md` | doc | campaign scope + dependency map (#2145+#2082 → #1942) |
