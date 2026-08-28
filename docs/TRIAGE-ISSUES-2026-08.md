# 1bit-MONSTER — Issue Triage Report (2026-08-27/28)

> **Status: fixes for 14 issues landed** in commits `1b4f468b`, `e038c9cb`,
> `dafee20c`, `3985f14f` (main).
> Remaining: HW-verification-dependent kernel changes, upstream escalations,
> and XL engine features (see "Completion status" below).

27 open issues analyzed, each grounded in the repo checkout (main @ 9d59a82a) by an
independent agent: issue body + comments read, tree grepped, git history checked,
upstream-vs-in-repo ownership determined.

## Completion status (2026-08-28, commit 1b4f468b)

| Issue | Sev | Status | Where |
|-------|-----|--------|-------|
| #1838 | med | ✅ fixed + silicon-verified | KERNEL_STATIC .data; final fused kernel ELF has .data@0x7f60c/0x7f610 (g_i4_call, call counter), ZERO .bss; CPU gates ALL PASS (2026-08-28) |
| #1834 | high | ✅ fixed | mm_binary_q1.cc union bit-casts + g_counter .data (HW-verify rebuild) |
| #1913 | med | ✅ fixed | check_chess_aietools.sh guard + USE_XCHESSCC=1 + docs §2 |
| #1908 | med | ✅ fixed | run_aiesim.sh 2025.2 default + 2026.1 warning |
| #1909 | med | ✅ fixed | empty-aiesol loud warning in run_aiesim.sh |
| #1910 | med | ✅ fixed | TXN-replay harness ported to main + docs §7d |
| #1911 | med | ✅ fixed | hand-built ps.so + loud-fail check in run_aiesim.sh |
| #1843 | med | ✅ fixed | version-pinned constraint catalog in ws09 README |
| #1836 | med | ✅ fixed (close-out) | fixed-point silu documented; escalate pipeliner |
| #1835 | med | ✅ fixed (close-out) | int32 ratioQ22 documented; escalate soft-float |
| #1869 | high | ✅ fixed (close-out) | v66 workaround at HEAD; escalate reproducer |
| #1870 | med | ✅ fixed | fix_toolchain.sh LLVM-version match gate + libclang_rt check |
| #1837 | high | ✅ fixed | build_p1i4.sh call-site arg-setup guard (NPU_STRICT_1837 opt-in) |
| #1864 | high | ✅ fixed | #error guard + I4_SCALAR_C1_ACK_1864 in build_zaya_fused.sh |
| #1832 | high | ✅ fixed + live-verified | NPU universal backend loads 35B-A3B q4nx on real NPU ("worker ready" ~35s); handshake timeout 10s→300s (model pack); manifest format fix; zaya→FLM diversion; onebin links amdclang++ |
| #1878/#1912 | med | ✅ harness merged | bench_compiler_ab.sh + README-COMPILER-AB.md on main (regression test for upstream fix) |
| #1831 | high | ✅ interim | qwen3next CPU engine wired (CMake + backend_manager + router route for qwen35moe); full HIP port still XL |
| #1776 | med | ✅ header gate | create_runlist() gated behind XRT>=2.25; runlist impl still env-blocked |
| #1865 | med | ✅ fixed + NPU-verified | h2 via delivered arg, pC zeroing via arg, zero_c1 removed, #1842 pins retired; C2gate corr=1.0 bad=0/2048 BYTE-IDENTICAL on strixhalo |
| #1907 | med | 🔶 deferred (XL) | baretorch token WITHOUT cs_lrad engine would silently mis-execute (registry comment forbids); full engine is XL |
| #1866 | med | ⏳ escalate | -O0 immediate range — upstream llvm-aie; -O1 workaround documented |
| #1874 | high | ✅ mitigated | I4_SCALAR_C1 is now the production default (verified corr 1.0); mmul path opt-in via I4_USE_MMUL=1 |
| #1872 | high | ✅ mitigated | #1874 flip removes Bb round-trip from production; I4_DIRECT_VECTOR_DEQ register path for mmul (arithmetic-verified 512000/512000); NPU gate pending |
| #1776 | med | ⏳ env | runlist needs XRT>=2.25 (box has 2.21.75); code path is version-gated |
| #1831 | high | ⏳ XL | HIP GatedDeltaNet+MoE; interim: wire qwen3next_engine.cpp as CPU fallback |
| #1882 | high | ✅ tracked | sub-item status table + upstream reproducer list in docs/aiesim-debugging.md §8 |

## Summary

| Metric | Count |
|--------|------:|
| Open issues triaged | 27 |
| critical | 0 |
| high | 10 |
| medium | 17 |
| Fixable in this repo (code/scripts/docs) | 23 |
| Upstream-only (escalate; no repo change) | 4 (#1912, #1869, #1866, #1835) |
| Already worked around in tree (close-out only) | ~7 (#1836, #1835, #1869, #1834, #1878-partial, #1843-partial) |

## Ranked fix order (recommended)

| Prio | Issue | Sev | Effort | In-repo fix (what to land) |
|------|-------|-----|--------|------------------------------|
| 1 | #1838 | med | S | Force .data placement for 41 zero-init probe statics in mm_kernel_reference.cc; ELF .bss lint in build_p1i4.sh |
| 2 | #1834 | high | S | Replace remaining __builtin_memcpy bit-casts in mm_binary_q1.cc with unions; close as worked-around |
| 3 | #1913 | med | S | Pre-flight guard: --xchesscc requires Vitis aietools root; warn/fail in check_mm_kernel_2x4.sh + docs |
| 4 | #1908 | med | S | aiesim toolchain-version guard (2026.1 broken) + doc correction in docs/aiesim-debugging.md |
| 5 | #1910+#1909+#1911 | med | M | Port TXN-replay harness (run_aiesim.sh + txn_replay_main.cpp) from experiment/compiler-ab to main; empty-aiesol + missing-ps.so loud-fail guards; docs |
| 6 | #1843 | med | S | Version-pinned re-validation table in research/ws09-int4-grouped/README.md |
| 7 | #1836/#1835/#1869 | med/high | S | Doc close-out: workaround landed (int32 ratioQ22, silu_pair_q22, v66 pointer-arith), escalate upstream, keep #1882 as tracker |
| 8 | #1870 | med | M | aiecc<->peano LLVM-version match gate + libclang_rt check in fix_toolchain.sh |
| 9 | #1837 | high | M | Build-time guard in build_p1i4.sh: fail loudly if 3-arg extern call lacks p1/p2 setup in emitted asm |
| 10 | #1864 | high | M | Compile-time guard so I4_SCALAR_C1 cannot silently emit broken scalar RMW; escalate upstream |
| 11 | #1867 | med | M | scripts/build_llvm_aie.sh pinned build recipe (libunwind in runtimes) + docs |
| 12 | #1865 | med | M | Migrate hardcoded tile-local addresses (zero_c1@0xE000, h2w@0x7F000) to extern-call args; retire #1842 pins (HW-verify) |
| 13 | #1832 | high | M | Server: register npu_engine_universal worker as InferenceBackend; format-aware routing for .q4nx |
| 14 | #1878 | med | M | Merge experiment/compiler-ab A/B harness + unpack_i4_sx shim to main; document --xchesscc recipe |
| 15 | #1912 | med | M | Upstream escalation (mlir-aie chess arg delivery) + merge A/B harness as regression test |
| 16 | #1882 | high | M | Tracker upkeep: close worked-around sub-items, package ISS repros for upstream |
| 17 | #1872 | high | M | Direct-vector dequant (no Btmp memory round-trip) — HW-verify on strixhalo |
| 18 | #1776 | med | L | runlist batched launch in zaya_decode.cpp gated on XRT>=2.25 (env-blocked until driver upgrade) |
| 19 | #1874 | high | L | Validate mmul C-store transpose hypothesis in ISS; flip I4_SCALAR_C1 to default until proven |
| 20 | #1907 | med | XL | baretorch cs_lrad: registry token first (S), then layer math + GGUF mapping + selfcheck |
| 21 | #1831 | high | XL | HIP 1BP: GatedDeltaNet + fused QKV + gated MoE; interim: wire qwen3next_engine.cpp as CPU fallback |
| 22 | #1866 | med | S | Escalate upstream (-O0 immediate range) with reproducer; keep -O1 workaround |

## Per-issue detail

### Already worked around in tree (fix = close-out + escalate)
- **#1834 (high, S)** — memcpy() libcall clobbers r0/r1. Workaround landed 5cdc89bd (union bit-casts in fold stash + dequant). Residual: mm_binary_q1.cc:45,55 still `__builtin_memcpy` bit-casts. → replace, add kernel-authoring rule, escalate to peano/llvm-aie.
- **#1836 (med, S)** — float silu loop miscompiled at p>=1. Superseded: silu_pair_q22 fixed-point (v70 h2 byte-identical, corr 1.0). → close, escalate pipeliner bug upstream, keep #1882 link.
- **#1835 (med, S)** — soft-float (sf*0.0625)/scc NaN. Workaround: int32 ratioQ22 dequant, no float libcalls in shipped kernels. → close as resolved-by-workaround; escalate upstream.
- **#1869 (high, S)** — scalar pointer arithmetic miscompiles j>=8. Workaround at mm_kernel_reference.cc:786 (v66, validated). → escalate with reproducer; no code change.
- **#1843 (med, S)** — codegen constraint catalog. Mostly documented in ws09 README; missing: exact toolchain versions per constraint. → add version table, then close into #1882.

### In-repo fix (script/doc guards — no hardware needed)
- **#1913 (med, S)** — --aietools must be Vitis root for --xchesscc. → pre-flight check + doc warning.
- **#1908 (med, S)** — 2026.1 aiesimulator segfaults. → harness pins 2025.2; correct doc's "re-verified on both" claim.
- **#1910+#1909+#1911 (med, M)** — aiesim needs TXN-replay ps.so; empty aiesol; silent ps.so skip. → port harness from experiment/compiler-ab @ d65c2359; add loud-fail guards; docs.
- **#1870 (med, M)** — LLVM-23 aiecc vs LLVM-21 peano mismatch. → version-match gate in fix_toolchain.sh.
- **#1837 (high, M)** — 3-arg extern call drops p1/p2. → build-time asm guard in build_p1i4.sh + escalate.
- **#1864 (high, M)** — scalar RMW += miscompile. → compile-time guard on I4_SCALAR_C1; escalate with aiesim repro.
- **#1867 (med, M)** — llvm-aie runtimes build failure. → pinned build recipe script.
- **#1865 (med, M)** — hardcoded tile-local addresses unreliable. → migrate to extern-call args; HW-verify.
- **#1878 (med, M)** — chesscc flag mismatch. → merge A/B harness; document working recipe.
- **#1882 (high, M)** — consolidated tracker upkeep.
- **#1776 (med, L)** — runlist launch batching, XRT>=2.25 gated. Env-blocked.

### In-repo fix (server/engine code)
- **#1832 (high, M)** — zaya never routes .q4nx to npu_engine_universal. detect_backends() registers only FLM backend; src/backend_npu.cpp implements the right protocol but is not wired into zaya. → port to InferenceBackend, format-aware routing, HW-verify.
- **#1907 (med, XL)** — baretorch cs_lrad new arch. Registry token first (S), then engine layer.
- **#1831 (high, XL)** — HIP qwen3_5_moe unsupported. Interim: wire validated qwen3next_engine.cpp as CPU fallback (currently an orphan, not in CMake).

### Upstream-only (escalate)
- **#1912 (med)** — chess external-func arg delivery. Merge A/B harness as the regression test.
- **#1869 (high)** — pointer-arith miscompile (workaround in tree).
- **#1866 (med)** — -O0 immediate range crash.
- **#1835 (med)** — soft-float NaN (workaround in tree).
