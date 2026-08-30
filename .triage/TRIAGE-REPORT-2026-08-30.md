# 1bit-MONSTER — Issue Triage Report Round 2 (2026-08-30)

> **Scope:** all 57 open issues on `1bit-MONSTER/1bit-MONSTER` (27 from round 1 +
> 30 new since 2026-08-28). Each new issue was read in full (body + comments) and
> its repo claims checked against the working tree, `origin/main`, and the open
> PR #1941 branch (`feat/lse-backend`).
>
> **Round-2 completion (2026-08-30): 43 of 57 issues fixed/closed.**
> - Round-1 fixes were **merged to `origin/main` via PR #1917** (2026-08-29) —
>   the 21 round-1 issues whose fixes it carries were verified and closed.
> - PR #1941 (C++26/#embed/HRX engine) **merged** — unblocked the HRX items.
> - 19 issues fixed in **PR #1969** (`fix/issue-round-2`): #1958, #1918, #1946
>   (HW-verified), #1954, #1957, #1959, #1867, #1961, #1965, #1963, #1960,
>   #1920, #1948, #1952, #1953, #1964, #1962 + docs. Host-side Discord ops
>   (#1947/#1950/#1951) fixed on strixhalo.
> - **14 remain open**, all genuinely long-running: deep NPU kernel work
>   (#1934 per-group-scale int4, #1935 Qwen3 GU h2, #1919 cascade — documented
>   PROVEN BLOCKED, #1839 gs-tile), XL features (#1831 HIP GatedDeltaNet+MoE,
>   #1907 baretorch, #1942 hybrid prefill/decode), resource-gated HRX
>   verification (#1943/#1955), upstream watch (#1944 llama.cpp PR #27218,
>   #1945, #1956 libstdc++16, #1866), env-blocked runlist (#1776 XRT<2.25).
>
> **Second finding (dependency):** PR #1941 (`feat/lse-backend`, open) is the
> carrier for C++26/#embed, the HRX engine, and several NPU mitigations. The
> HRX issues (#1942–#1959) reference files that exist **only on that branch**
> (`src/hrx_inprocess.*`, `docs/research/hrx-engine-goal.md`,
> `third_party/lemonade/tools/gen_hrx_model_entries.py`, 44 `*-HRX`
> `server_models.json` entries, `tools/npu_stability_probe.cpp`,
> `tools/parity_fused.cpp`, `docs/bug-reports/CASCADE-SINGLE-LAUNCH-ISSUES.md`).
> They are **gated on #1941 landing**, not independent work.

## Round-1 completion status (27 issues — fixes unmerged)

| Issue | Sev | Status (2026-08-30) | Where |
|-------|-----|---------------------|-------|
| #1838 | med | ✅ fixed + silicon-verified | fix/triage-round (`03f5044b`) |
| #1834 | med | ✅ fixed | 1b4f468b |
| #1913 | med | ✅ fixed | 1b4f468b |
| #1908 | med | ✅ fixed | 1b4f468b |
| #1909 | med | ✅ fixed | 1b4f468b |
| #1910 | med | ✅ fixed | 1b4f468b |
| #1911 | med | ✅ fixed | 1b4f468b |
| #1843 | med | ✅ fixed | 1b4f468b |
| #1836 | med | ✅ fixed (close-out) | 1b4f468b |
| #1835 | med | ✅ fixed (close-out) | 1b4f468b |
| #1869 | high | ✅ fixed (close-out) | 1b4f468b |
| #1870 | med | ✅ fixed | 30646f0a |
| #1837 | high | ✅ fixed | 57b2e9f2 |
| #1864 | high | ✅ fixed | 1b4f468b + I4_SCALAR_C1 default (`0f893b1e`) |
| #1865 | med | ✅ fixed + NPU-verified | da9bbd54 + 954dc298 |
| #1872 | high | ✅ mitigated | 11274a39 (direct-vector dequant) |
| #1874 | high | ✅ mitigated | 0f893b1e (scalar-C1 default) |
| #1832 | high | ✅ fixed + live-verified | 7d118368, e41a9acc, b385d43c, 9a3a2ed3, 03f5044b |
| #1831 | high | 🔶 interim (CPU fallback wired) | dafee20c + 27bc2d3d; full HIP port XL |
| #1878/#1912 | med | ✅ harness merged | 1747ca23, 62236189 |
| #1882 | high | ✅ tracked | e2881230, 954dc298 (status table + reproducers) |
| #1776 | med | ⏳ env-blocked | create_runlist() gated XRT≥2.25 (`3985f14f`) |
| #1866 | med | ⏳ escalate upstream | -O0 range crash; -O1 workaround only |
| #1867 | med | ⏳ no fix landed | build_llvm_aie.sh recipe never committed (any branch) |
| #1907 | med | 🔶 deferred (XL) | baretorch cs_lrad engine |
| #1839 | med | ⏳ open | gs-tile delivery — only comment is the #1882 tracker ref; no code fix |

**Blocking action: push `fix/triage-round` → merge to `origin/main` (PR).**
Without it, 14 of the 27 remain open on GitHub despite completed fixes.

## New issues since round 1 (30) — summary

| Issue | Sev | Effort | In-repo fix (what to land) | Dependency |
|-------|-----|--------|----------------------------|------------|
| #1958 | **high** | S | unified_server `-m`: if arg is an existing path / ends `.gguf`, use it as `model_path`; make "not found" a hard error (verified: silent fallback at `tools/unified_server.cpp:1454`) | none |
| #1933 | **high** | M | Merge the `npu_stability_probe` + `USE_NPU_FFN` gate from `feat/lse-backend` (probe is NOT on main); escalate XRT/amdxdna GP fault upstream | PR #1941 |
| #1934 | **high** | L | Per-group-scale int4 kernel restructure (C1 carrying per-group scales); **do NOT wire `packB_i4` until landed** | none (kernel work) |
| #1918 | med | S | Add `dflash2draft`/`lowonmind`/`oxmini` registry mappings in `include/rocm_cpp/bitnet_model.h` (none present; only `dflash`/`dflashdraft`/`dflashlaguna` mapped) + rerun `Testing/census_coverage.py` | none |
| #1919 | med | M | Driver-side FIXED (amdxdna 0.17.0 TDR, verified on HW); remaining: compute deadlock (C2 never written) — separate correctness item; docs live on PR #1941 | PR #1941 docs |
| #1920 | med | S | Obviated by driver 0.17.0 `aie2_hw_reset` (no `/dev/mem`); update/retire `aie-reset` doc | driver upgrade (done) |
| #1935 | med | M/L | Iron generator GU h2 redistribution (Qwen3 1:6 vs Zaya 2:1); `build_iron_cascade_qwen3.sh` exists only on PR #1941 | PR #1941 |
| #1942 | med | XL | Hybrid prefill/decode — cross-backend KV handoff (llama state export/import); design project | PR #1941 + upstream |
| #1943 | med | M | Per-entry embedding-quant annotation + representative pull+serve of 43 `*-HRX` entries (multi-GB downloads, resource-gated) | PR #1941 |
| #1944 | med | — | GET_ROWS fail-closed on non-Q4_K embeddings — **upstream-only** (llama.cpp PR #27218); all local workarounds dead ends | upstream |
| #1945 | med | S | Watch item: re-probe/benchmark on stable hrx-system release or PR #27218 moving | upstream |
| #1946 | med | M | Port test_zero_copy proof to Vulkan dma-buf idiom (hipHostRegister rejects XRT BOs on TheRock) | none |
| #1947 | med | S | Ops: point `discord-traffic-digest.py` CHANNEL at #traffic-report (host file, not in repo) | none (host) |
| #1948 | med | S | Decision: upstream mlir-aie npu2_40 patches vs documented backup; write decision into repo | none |
| #1957 | med | M | `#embed` freshness: CMake dep on embedded files + build-time stale warning | PR #1941 |
| #1959 | med | M | Harden `HRX_ROOT` change-across-instances (static path cache / reject) — file on PR #1941 only | PR #1941 |
| #1960 | med | S | Ops: broaden CLOUDFLARE_API_TOKEN (Zone:DNS Edit) or fail loudly in workflow (#1939 merged; DNS still manual) | none (secrets) |
| #1961 | med | S/M | `/docs` guild-scoped → global registration or per-guild sync | none |
| #1963 | med | M | Fix 56 Context7 parse failures (bad markdown/front-matter) + re-index | none |
| #1965 | med | M | Secrets: move from hand-written `.env` to secret manager or validate-on-install | none |
| #1949 | low | S | Likely **already resolved** by #1938/#1940 merge (dir now tracked on origin/main) — verify + close | none |
| #1950 | low | S | Ops: move/delete stale digest message `1542832357118189608` | none (host) |
| #1951 | low | S | Ops: `systemctl --user restart docsbot` to pick up corrected env | none (host) |
| #1952 | low | S | Docs: correct the ~175 tok/s spec → in-process ~80-87; keep re-bench recipe | none |
| #1953 | low | S | Doc close-out: fork B closed (IREE HAL no dma-buf); zero-DMA stays Vulkan/SharedBO | none |
| #1954 | low | S | Bump 7 legacy `-std=c++17/20` pins (CMakeLists:866,880,978,1562,1581,2078,2083) after #1941 | PR #1941 |
| #1955 | low | M | Representative pull+serve of HRX entries (resource-gated); deployment preference decided | PR #1941 |
| #1956 | low | S | Watch: libstdc++16/g++16 unlock for P2996/senders/inplace_vector; adopt incrementally | upstream |
| #1962 | low | M | Deploy `/docs` bot off the dev host + watchdog/alerting | none (ops) |
| #1964 | low | S | Decision + doc: publish internal docs section or document the CURATED exclusion | none |

## Summary metrics

| Metric | Count |
|--------|------:|
| Open issues | 57 |
| Round-1 (previously triaged) | 27 — 22 fixed/close-out **but fixes unmerged to origin/main**; 2 escalate (#1866, #1867); 1 XL (#1907); 1 open (#1839); 1 env-blocked (#1776) |
| New this round | 30 |
| — high | 3 (#1958, #1933, #1934) |
| — medium | 17 |
| — low | 10 |
| Fixable in-repo now (no dependency) | ~13 new (#1958, #1918, #1920, #1946, #1947, #1948, #1950, #1951, #1952, #1953, #1960, #1961, #1963, #1964, #1965, #1949-verify) |
| Gated on PR #1941 (feat/lse-backend) | #1933(mitigation), #1935, #1942, #1943, #1954, #1955, #1957, #1959 (+ #1919 docs) |
| Upstream-only (escalate; no repo change) | #1944 (llama.cpp PR #27218), #1945 (watch), #1956 (libstdc++16), #1866, #1867 |

## Ranked fix order (recommended)

| Prio | Issue(s) | Sev | Effort | Action |
|------|----------|-----|--------|--------|
| 1 | **merge round-1** | — | M | Push `fix/triage-round` and open a PR to `origin/main`; close the ~14 round-1 issues whose fixes it carries (#1838, #1834, #1913, #1908-#1911, #1843, #1836, #1835, #1869, #1870, #1837, #1864, #1865, #1872, #1874, #1832, #1878/#1912, #1882) |
| 2 | #1958 | high | S | `tools/unified_server.cpp`: file-path resolution for `-m` + hard error on no match (verifiable in ~1 h) |
| 3 | #1918 | med | S | Add 3 census aliases to `bitnet_model.h` + rerun coverage |
| 4 | #1920 | med | S | Doc: driver 0.17.0 `aie2_hw_reset` replaces `aie-reset`/`/dev/mem`; note in docs |
| 5 | #1949 | low | S | Verify dir tracked on main → close as resolved-by-merge |
| 6 | #1947+#1950+#1951 | med | S | Discord ops trio (host script channel, stale message, service restart) |
| 7 | #1960 | med | S | Cloudflare token scope or fail-loud in `provision-docs-domain.yml` |
| 8 | #1953+#1964+#1952 | low | S | Doc close-outs (fork B, docs-site exclusion, benchmark spec correction) |
| 9 | #1961+#1963+#1965 | med | M | Docs-bot hardening: global `/docs`, fix Context7 parse failures, secret handling |
| 10 | #1946 | med | M | Port test_zero_copy to Vulkan dma-buf idiom |
| 11 | #1948 | med | S | mlir-aie patches decision (upstream PR vs backup-as-canonical) |
| 12 | #1934 | high | L | Per-group-scale int4 kernel restructure (the int4 production blocker) |
| 13 | #1935 | med | M/L | Iron generator GU h2 1:6 fan-out redesign |
| 14 | #1933 | high | M | Merge probe+gate from #1941 branch; upstream escalation for XRT GP fault |
| 15 | **merge PR #1941** | — | L | Lands HRX engine + C++26/#embed + NPU mitigations; unblocks #1942-#1945, #1954, #1955, #1957, #1959 |
| 16 | #1942 | med | XL | Cross-backend KV handoff design (after #1941) |
| 17 | #1944+#1945+#1956 | med | S | Upstream watch items (llama.cpp PR #27218, hrx-system release, libstdc++16) |
| 18 | #1943+#1955 | med | M | Representative HRX pull+serve + per-entry quant annotation (resource-gated) |
| 19 | #1962 | low | M | Off-host `/docs` bot deployment + watchdog |
| 20 | #1919 (compute) | med | M | Zero-DMA cascade deadlock — generator/fifo fix distinct from driver (track in #1882-style docs) |
| 21 | #1839 | med | M | gs-tile delivery — still open; fold into #1882 tracker close-out |
| 22 | #1866/#1867 | med | S | Upstream reproducer for -O0 crash; commit the llvm-aie build recipe |

## Per-issue detail (new issues)

### High
- **#1958 (high, S)** — `1bit unified -m <path>` silently loads a *different* model. Verified: `tools/unified_server.cpp` resolves `-m` only against `discover_models()` names (exact → ci → prefix) and on no-match prints `** Model '<arg>' not found -- using first available.` (line 1454) then loads `discovered.front()`. Fix: if the arg is an existing file (or contains `/` / ends `.gguf`), set `cfg.model_path` directly; make no-match a hard error. No dependency — highest-value small fix.
- **#1933 (high, M)** — repeated AIE GEMMs GP-fault/heap-corrupt in `libxrt_driver_xdna.so.2.21.75` and wedge the NPU. Root cause is the XRT/amdxdna userspace driver (upstream). The mitigation (`tools/npu_stability_probe.cpp` out-of-process gate + `USE_NPU_FFN` in `src/backend_fused.cpp`) exists **only on `feat/lse-backend`** — the `USE_NPU_FFN` gate is present on this branch but the probe exec is not. Merge the mitigation with #1941 (or cherry-pick) and escalate upstream.
- **#1934 (high, L)** — fused GU→SiLU int4 caps FFN corr at ~0.972 with K-uniform scale; per-group-scale kernel restructure required. Explicit "do NOT wire `packB_i4` into `npu_state_*`" gate until landed. `engine/npu/generators/i4_pack.h` + `test_i4_pack.cpp` already verified on this branch — the restructure is the remaining piece.

### Medium
- **#1918 (med, S)** — census-watch: `dflash2draft` (GLM-5.3-Flash-DFlash2), `lowonmind`, `oxmini` unmapped in `include/rocm_cpp/bitnet_model.h` (verified: only `dflash`/`dflashdraft`/`dflashlaguna` present). Add aliases or real impls + rerun `Testing/census_coverage.py`. 5 gated repos are unverifiable without a token (retried next run) — not actionable.
- **#1919 (med, M)** — zero-DMA single-launch cascade hang: **driver-side FIXED** (upstream amdxdna 0.17.0 with `tdr_timeout_ms=2000` + SMU power-cycle; verified: `run.wait()` returns `state=8` in 2.0 s, second launch succeeds, no reboot). Remaining: the compute itself deadlocks (C2 never written) — a distinct correctness item now surfaced as a timeout. Docs (`AGENT-COORDINATION.md`, `CASCADE-SINGLE-LAUNCH-ISSUES.md`) exist only on `feat/lse-backend`.
- **#1920 (med, S)** — `aie-reset` `/dev/mem` mmap fails under strict devmem/lockdown. Obviated by driver 0.17.0's `aie2_hw_reset()` (SMU power-cycle + firmware reload, no `/dev/mem`), per the #1919 comments. Recommend: document the driver path as the reset mechanism; optionally retire/rewrite `aie-reset`.
- **#1935 (med, M/L)** — Qwen3 GU (K=1024 → N_GU=6144, 1:6) violates the iron generator's `K == N_GU/2` h2-broadcast assumption (Zaya is 2:1, silicon-verified). Needs GU h2 fan-out redesign; `build_iron_cascade_qwen3.sh` is on `feat/lse-backend` only. Distinct from #1919.
- **#1942 (med, XL)** — hybrid HIP-prefill + HRX-warm-decode; blocker is cross-backend KV handoff between vendored llama.cpp and the HRX bundle's libllama (state-format compat). Design project; acceptance = correct continuation + beats either backend alone.
- **#1943 (med, M)** — verify+annotate 43 `*-HRX` registry entries: per-entry token-embedding quant (determines serve-on-HRX vs llamacpp fallback) + representative pull+serve. Resource-gated (multi-GB GGUFs). `gen_hrx_model_entries.py` + entries only on `feat/lse-backend`.
- **#1944 (med, upstream)** — in-process HRX decodes only Q4_K-embedding models; q5_0/q8_0/Q4_K_S/IQ2XXS fail-closed at GET_ROWS. Every local workaround proven dead (n_gpu_layers; tensor_buft_overrides → scheduler keeps HRX split). Fix must come from llama.cpp PR #27218; identical limit in the subprocess path, so nothing is lost.
- **#1945 (med, S)** — upstream watch: re-engage on stable ROCm/hrx-system release or PR #27218 past draft; recipe in `docs/research/hrx-engine-goal.md §P2` (PR branch).
- **#1946 (med, M)** — `test_zero_copy`'s `hipHostRegister` on XRT-mapped BOs fails on TheRock 7.16; port the CPU↔GPU no-copy proof to the Vulkan dma-buf idiom (`test_vk_attn_slice` passes fresh). File present in this tree.
- **#1947 (med, S)** — ops: traffic-digest cron posts to #general; belongs in #traffic-report. Script is `~/.local/bin/discord-traffic-digest.py` (host file, not in repo) — fix `CHANNEL` constant + docstring/footer.
- **#1948 (med, S)** — decision: local mlir-aie npu2_40 patches (AIELowerDynamicBDPool/BdLowering) committed locally + backed up 2026-08-29; upstream PR vs backup-as-canonical. Write the decision + recovery steps into the repo.
- **#1957 (med, M)** — `#embed` (P1967) freshness: CMake dependency so touching an embedded xclbin/insts rebuilds the embed TU + build-time stale warning. `engine/npu/src/npu_embedded.h` exists only on `feat/lse-backend` (PR #1941 introduced the embeds).
- **#1959 (med, M)** — `HRX_ROOT` changed across backend instances → fresh dlopen → ggml static-init abort (no-dlclose constraint). Harden: resolve bundle path once per process / reject differing HRX_ROOT. File on PR #1941 only.
- **#1960 (med, S)** — `CLOUDFLARE_API_TOKEN` lacks Zone:DNS scope → `provision-docs-domain.yml` (#1939, on main) cannot self-provision DNS; currently hand-added CNAME. Broaden token or fail loudly when the zone can't be resolved.
- **#1961 (med, S/M)** — `/docs` slash command is guild-scoped (`tree.copy_global_to(guild)` + guild sync); not available on a second server. Options: global registration (up to 1 h propagation), per-guild sync on join, or document single-server support.
- **#1963 (med, M)** — Context7 library: 56 parse failures (37 → 56 as snippets grew 1896 → 1923). Identify offending markdown (front-matter/oversized/unsupported) and fix; `ctx7 refresh` after. `ctx7` CLI present on this host.
- **#1965 (med, M)** — `install-docsbot-service.sh` requires hand-written `.env` with 3 live secrets; no secret management. Recommend secret manager / validate-on-install / read from existing `~/.secrets` locations.

### Low / close-out
- **#1949 (low, S)** — untracked `integrations/discord-support-bot/`. **Likely resolved**: PRs #1938/#1940 merged the dir into `origin/main` (verified: `integrations/discord-support-bot/{bot.py, context7.py, docs_slash.py, llm.py, docsbot.service, install-docsbot-service.sh, …}` are tracked on main). Verify the working tree is clean and close.
- **#1950 (low, S)** — stale 2026-08-28 digest message `1542832357118189608` in #general; move/delete (already ingested into `~/data/discord-inbox.md`).
- **#1951 (low, S)** — `docsbot.service` still running with pre-fix `DISCORD_KNOWN_CHANNEL`; `systemctl --user restart docsbot` + verify via journalctl.
- **#1952 (low, S)** — benchmark doc correction: in-process ~80-87 tok/s vs documented subprocess ~175 (warm-server best case, not reproducible fresh — fresh measured 38.2). Correct the spec; keep the re-bench recipe.
- **#1953 (low, S)** — doc close-out: fork B (IREE HAL dma-buf import) infeasible — unimplemented TODO in `runtime/src/iree/hal/allocator.h`; zero-DMA stays on Vulkan/SharedBO. Decision recorded; close.
- **#1954 (low, S)** — 7 legacy targets still `-std=c++17/20` (CMakeLists:866 `spec_decode`, 880 `zaya_logit_check`, 978 `test_tokenizer_logprobs`, 1562 `qwen36_moe_probe`, 1581 `gen_npu_insts`, 2078 `zaya_codec`, 2083 `test_zaya_codec`). Bump to C++26 after PR #1941 merges; note any target that genuinely needs the old standard.
- **#1955 (low, M)** — 43 `*-HRX` entries landed + deployment preference (engine-native primary) decided; remaining is representative pull+serve (resource-gated).
- **#1956 (low, S)** — C++26 headline features (P2996 reflection, P2300 senders, P0843 inplace_vector, P2072 text) gated on libstdc++16/g++16; watch upstream, adopt incrementally (BackendType name mapping first).
- **#1962 (low, M)** — `/docs` bot is a single dev-host systemd service; deploy off-host (VPS/container) + watchdog + move secrets.
- **#1964 (low, S)** — docs site (`scripts/gen_docs_site.py` CURATED list) excludes `docs/research|archive|plans|goals|bug-reports`; decide publish-vs-document-exclusion explicitly in `docs/README.md`.

## Round-1 issues still genuinely open (no fix in any branch)

| Issue | Sev | Status | Action |
|-------|-----|--------|--------|
| #1866 | med | escalate | upstream llvm-aie -O0 immediate-range reproducer; -O1 workaround documented |
| #1867 | med | no fix | commit the pinned `build_llvm_aie.sh` recipe (libunwind in runtimes) — never landed |
| #1907 | med | XL | baretorch cs_lrad engine: registry token (S) then layer math (XL) |
| #1839 | med | open | gs-tile delivery — only tracker ref; needs the #1882 close-out treatment |
| #1776 | med | env | runlist batched launch — code gated on XRT≥2.25, box has 2.21.75 |

## Round-2 completion status (2026-08-30, PR #1969 + host fixes)

| # | Issue | Status |
|---|-------|--------|
| #1958 | unified_server `-m` path resolution | ✅ committed ff621030 + closed |
| #1918 | census aliases | ✅ committed 5e8874e5 + closed |
| #1946 | test_zero_copy → Vulkan dma-buf bidirectional | ✅ committed 8df2f251 + closed (**HW-verified**) |
| #1954 | 7 C++17/20 pins → C++26 | ✅ committed d02ca272 + closed |
| #1957 | #embed freshness | ✅ committed db281033 + closed |
| #1959 | HRX DSO path pinning | ✅ committed 79b8fb82 + closed |
| #1867 | build_llvm_aie.sh | ✅ committed 23c98f5f + closed |
| #1961 | /docs global registration | ✅ committed 838cad2b + closed |
| #1965 | install script secret assemble/validate | ✅ committed 838cad2b + closed |
| #1963 | docs markdown lint | ✅ committed 5e49839d + closed |
| #1960 | CF token fail-loud | ✅ committed 733e5571 + closed |
| #1920 | NPU driver reset doc | ✅ committed 5cdf1733 + closed |
| #1948 | mlir-aie patch decision | ✅ committed 830024f4 + closed |
| #1952 | HRX bench spec | ✅ committed 5cdf1733 + closed |
| #1953 | fork B close-out | ✅ committed 5cdf1733 + closed |
| #1964 | docs-site curation policy | ✅ committed 5cdf1733 + closed |
| #1962 | /docs off-host runbook | ✅ committed 379ff65e + closed |
| #1947 | traffic-digest channel | ✅ fixed on strixhalo host + closed |
| #1950 | stale digest message | ✅ already gone (404) + closed |
| #1951 | docsbot restart | ✅ fixed on strixhalo host + closed |
| #1949 | untracked dir | ✅ verified tracked on main + closed |
| #1933 | XRT GP fault mitigation | ✅ on main via #1941 (probe+gate) + closed |
| round-1 ×21 | #1832,#1834,#1836,#1837,#1838,#1843,#1864,#1865,#1869,#1870,#1872,#1874,#1878,#1882,#1908,#1909,#1910,#1911,#1912,#1913,#1835 | ✅ fixes on main via #1917, closed 2026-08-30 |

## Remaining 14 open — all long-running (see header)

- **NPU kernel engineering (strixhalo):** #1934 (int4 per-group-scale restructure),
  #1935 (Qwen3 GU h2 fan-out), #1919 (single-launch cascade — PROVEN BLOCKED,
  p1/p2 two-launch is production), #1839 (gs-tile delivery).
- **XL features:** #1831 (HIP GatedDeltaNet+MoE), #1907 (baretorch cs_lrad), #1942
  (hybrid prefill/decode KV handoff).
- **Resource-gated verification:** #1943, #1955 (HRX pull+serve, multi-GB downloads).
- **Upstream watch:** #1944 (llama.cpp PR #27218 GET_ROWS), #1945 (hrx-system
  release), #1956 (libstdc++16/g++16), #1866 (llvm-aie -O0 range).
- **Env-blocked:** #1776 (runlist needs XRT≥2.25; box has 2.21.75).
