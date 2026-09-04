# 1bit-MONSTER — Issue Triage + Consolidation Report (2026-09-02)

> **Session summary:** 13 open issues triaged; 4 resolved (1 fixed + merged, 3
> closed with evidence), 9 kept open with verified owner status / upstream
> escalation. Repo-wide consolidation: remote branches 61 → ~9, worktrees 8 → 4,
> two stranded site features ported to main, one CI-blocking workflow bug fixed.
> Research: Round 39 (runtime batched-prefill ≠ sequential-forward rope
> divergence) recorded in `npu-infer/docs/txn-decode-findings.md`.

## 1. Repo consolidation (scope drift → clean state)

The shared checkout had accumulated a large superseded Downloads-page change
set, ~50 stale merged-PR remote branches, dead worktrees, and a secrets
backup inside the tree. Resolved:

| Item | Before | After |
|------|--------|-------|
| Remote branches | 61 | ~9 (live: `main`, `feat/hrx-gfx1151-build`, 2 open-PR heads, worktree branches + daily `seo/daily-*`) |
| Worktrees | 8 | 4 (`feat/hrx` research, `-attach`, `-iso-build`, `-lemonade-v1170`) |
| Downloads-page dirt on the research branch | uncommitted (superseded) | dropped — canonical content was already on the merged PR (#2032) |
| `integrations/discord-support-bot/.env.bak-*` | in-tree secret | moved to `~/secret-archive/` |
| Superseded local branches (`context7-*`, zaya-round posts, `fix/analytics-*`, `main`, …) | 8+ local refs | deleted (merged content safe in `main`) |
| Stale no-PR branches (`experiment/compiler-ab`, `fix/npu-1799-fp`, `wip/stash-archive`) | remote + local | archived locally as `backup/*`, remote deleted |

## 2. Merged PRs this session

| PR | What | Commit (main) |
|----|------|---------------|
| **#2032** | Downloads page — release packages served from the website (tarball `.tar.xz`, `.deb`, AppImage + SHA256SUMS + manifest), `/downloads` redirects, `make package-site` | `56f0c82a` |
| **#2035** | Light/auto/dark theme modes — port of the work stranded on the research branch to main (`site_theme_modes.py`, `theme.js`, 42 pages restamped incl. Downloads page) | `329b1651` |
| **#2019** | GHCR OCI package publishing (replaces tag-triggered GitHub Releases; `latest` + `sha-*`) + `scripts/download.sh` | `3d9f5984` |
| **#2038** | Census registry tokens for 5 newly uncovered HF classes (issue #2031) — see §3 | `a91af3da` |

CI-blocking workflow bug found and fixed en route: the Scope Guard script's
`ALLOWED` list comment contained an apostrophe ("the **site's**") that closed
the single-quoted string early — the shell then tried to execute the
`*.deb/*.AppImage/*.tar.xz` glob (exit 127), blocking every PR touching
`.gitattributes`. Fixed on the downloads branch (`97782e44`, merged via #2032).

## 3. Issue dispositions

### Resolved this session

| Issue | Sev | Status | Where |
|-------|-----|--------|-------|
| #2031 (census: new HF classes) | med | ✅ fixed + merged | 5 classes tokenized → `RCPP_ARCH_BREEZE_TTS=993`, `RCPP_ARCH_HYV4=994`, `RCPP_ARCH_BANANAMIND21CODER=995`, `RCPP_ARCH_BANANAMIND21LITE=996`, `RCPP_ARCH_CONCEPT_DOMINANT_GPTBERT=997` in `bitnet_model.h`, wired into the generic loader's safetensors + GGUF refusal guards. Class names verified against live HF configs (breeze = TTS; hy_v4 = Gated-MLA text LM, PICO-family candidate for bananamind 2.1). Census gate passes. PR #2038 → `a91af3da`. |
| #2006 (per-shape insts) | med | ✅ closed (superseded) | The hand-rolled mm-in-isolation validation was overtaken by the runtime-layer path (`NPU_RUNTIME_LAYERS=1`), byte-identical to the FastFlowLM runtime at 1000-token decode (rounds 36–38). Evidence: `npu-infer/docs/txn-decode-findings.md`. |
| #2015 (mm weight-BO closed reorder) | med | ✅ closed (superseded) | Exhaustive negative documented (`captures/README.md`); engine reproduces the runtime byte-for-byte through its own packing (`npu_pack_layer_bo`), so host-side derivation of the dequant formula is moot. |
| #2008 (64-byte nondeterministic mm region) | low | ✅ closed | Artifact of the superseded hand-rolled path; the validated runtime path passes strict byte gates (0 ULP at 1000 ctx). Hypothesis recorded (uninitialized L1 at the 64-row lane boundary) for a future revival. |

### Kept open — verified owner status added

| Issue | Sev | Status |
|-------|-----|--------|
| #2013 (gnome-shell GL hang, gfx1151) | high | Mitigations verified live: `amdgpu lockup_timeout=10000 timeout_period=10000` active, GNOME animations off, **stable since 08-31** (no new coredumps/hangs across reboots). Coredumps preserved (`/var/log/gpu-coredumps/amdgpu-20260831-*`). Remaining: upstream Mesa/amdgpu report. |
| #1942 (hybrid prefill/decode) | med | State-format gate **answered** (round 25j): vendored llama.cpp ↔ HRX bundle full-state blob round-trips byte-identically (0.6B + 30B-A3B Q4_K_M). Remaining: HIP prefill lane build (TheRock, stopped at round 25k) + KV handoff plumbing + benchmark. |
| #1776 (Zaya decode CPU-attention-bound) | med | Status corrected: the runtime-layer path (Qwen3 npu-infer) does **not** change the standalone Zaya path — CCA-attention-on-NPU remains the open lever. |
| #1831 (HIP cannot run qwen3_5_moe 35B) | high | Scoped: GDN math has NPU-side references (`npu_engine_universal.cpp`, npu-infer 35B layout work) to port into `backend_hip_1bp` behind the qwen35 gate (fused QKV + GatedDeltaNet linear attention; `full_attention_interval=4` schedule). |
| #1945 (HRX upstream watch) | med | No signal: llama.cpp #27218 still draft (08-31), hrx-system stuck at v0.3.0 (May). |
| #1866 (llvm-aie -O0 immediate crash) | med | Upstream fix not landed (fetched 09-02; newest = AIE2PS accumulator-spill only). -O1 workaround stands. |
| #1956 (C++26 toolchain watch) | med | No signal: local g++ 15.2.0, `std::inplace_vector`/reflection still gated on g++16/libstdc++16. |
| #1907 (baretorch cs_lrad) | med | Unchanged — XL engine feature (registry token + safe refusal already landed). |
| #1934 (int4 fused FFN corr cap) | high | Host contracts all merged (#1978/#1983/#1984/#1985/#1986); remaining = `npu_state_*` production wiring of the single-launch i4 fused path (auto-select + GuI4Pack bf16-pair packing for asymmetric-zp models) + silicon parity gate — multi-session NPU integration, still open. |

## 4. Research: Round 39 — runtime batched prefill ≠ N×forward (rope divergence)

Recorded in `npu-infer/docs/txn-decode-findings.md` (commit `1819bc9d`,
feat/hrx-gfx1151-build). The FastFlowLM runtime's batched `prefill(ids)` does
not produce the same KV/rope state as N sequential `forward(tok)` calls for
the same prompt: keys diverge only at rotary elements 50/51 + 114/115 (rope
freq classes j=25/j=57 — the .rodata-vs-exact-math classes from Round 38),
final logits argmax flips, and greedy continuations diverge from the 3rd
token. Root cause: the batched prefill never advances the host-side RoPE
(i6) table (identity at every position), while `forward()` updates it per
position — the path the engine replicates (byte-identical, 1000 ctx).
Implication: per-ctx validation must use the runtime's RT_TOKENS/forward
mode; engine == runtime-seq is reaffirmed; runtime's own prefill is a
divergent rope variant of its decode path.

## 5. Open escalations / next owner steps

- **#2013** — file the Mesa/amdgpu report with the preserved coredumps.
- **#1831** — port GDN linear attention from the NPU-universal reference into `backend_hip_1bp`.
- **#1942** — resume the TheRock HIP prefill-lane build (`ROCM_PATH` → TheRock root; round-25k config fix), then the KV-handoff + benchmark gate.
- **#1934** — `npu_state_*` i4-fused production wiring + silicon parity gate.
- **#1866/#1945/#1956** — re-probe when upstream moves (llvm-aie range fix / llama.cpp #27218 out of draft / g++16).
