# Issue Campaign — Fix List & Knowledge Packets

Goal `mtubusx1-swek0b` · generated 2026-09-09 · coordinator @agent-429168 · peers @agent-f85526, @agent-07e844

Scope: **19 actionable bug/perf issues**. Out of scope: #2070 (parked), #1945 (upstream watch), #2166 (bot census), the 12 open PRs.

Done definition per issue: **confirmed root-cause diagnosis + actionable fix plan posted to the issue** (or a bounded repro/experiment plan where root cause is not confirmable without multi-day hardware work). No code is landed by this campaign.

---

## Priority map

| Pri | Meaning |
|---|---|
| P0 | Correctness blocker — systemic defect or blocks another shipped path |
| P1 | Perf / validation / stability gate |
| P2 | Feature, product decision, or toolchain bug with a workaround |

## Issue → assignment (round-robin, 3 agents × ~6–7 each)

Wave 1 (me + peers): #2117, #2115, #2116, #2145, #2147, #2153
Wave 2: #2152, #2150, #2159, #2113, #2114, #2102
Wave 3: #2105, #2080, #2081, #1866, #2082, #2139, #1942

(Exact per-agent assignment decided at dispatch time; waves are the contract.)

---

## Summary table

| # | Title (short) | Pri | Cluster | Root-cause status | Depends on |
|---|---|---|---|---|---|
| 2117 | ggml-hrx over-claims ops → hard errors | P0 | HRX systemic | Known (claim predicates) | — |
| 2115 | HRX host-execution path corrupts CPU-mixed graphs | P0 | HRX systemic | Known (host path not CPU-equiv) | #2117 |
| 2116 | zaya ngl99 decode corrupt (mixed-split boundary) | P0 | HRX systemic | Narrowed (not one op) | #2115, #2117 |
| 2145 | llama_state-imported ctx fails HRX0 @ token 2 | P0 | HRX device | **Root-caused + fix in PR #2203** | #1942, #2082 |
| 2147 | qwen3moe-30B fails HRX decode | P0 | HRX device | Open (#2117-class) | #2117 |
| 2153 | no batched FLASH_ATTN_EXT (ne3>1) | P1 | loom kernel | Known (ne3==1 hard-require) | — |
| 2152 | CONCAT dim-0 loom kernel corrupts + faults | P1 | loom kernel | WIP, 2 hypotheses eliminated | — |
| 2150 | zaya HRX launch-bound 6.4 t/s | P1 | HRX perf | Measured (~640 launches) | #2080 |
| 2159 | HRX-ALONE can't reach Vulkan-class | P1 | HRX perf | Characterized (codegen deficit) | — |
| 2113 | single-launch cascade perf tuning | P1 | NPU cascade | Known (12.3 vs 4.65 ms) | #2078✅, #2114 |
| 2114 | full-decode cascade validation sweep | P1 | NPU cascade | Thin (1 point validated) | #2113 |
| 2102 | mm.xclbin within-tile layout permutation | P0 | fused-dequant | Open | — |
| 2105 | fused-dequant NaN (93696) | P0 | fused-dequant | Likely tile alignment | #2102 |
| 2080 | Zaya whole-layer per-ctx ELF (M3/M4) | P2 | NPU feature | M1 done, M2 in progress | #2113, #2114, #2070 |
| 2081 | int4 vs int8 dense-FFN product decision | P2 | decision | Open (needs data) | #2114 |
| 1866 | aie2p -O0 backend crash | P2 | toolchain | **Upstream-only watch (2026-09-11): nothing in-repo needs -O0** | #1155/#1276 |
| 2082 | D2 HIP-prefill stability gate | P1 | hybrid | Re-probe clean | #1942 |
| 2139 | qwen35moe 1BP fast-path + quant lanes | P2 | GPU feat | Open (gaps listed) | #2138✅ |
| 1942 | hybrid prefill/decode (KV handoff) | P2 | hybrid design | **Handoff resolved; §5.2 positive clause blocked by the bundle's KV ceiling (#1945)** | #2145, #2203 |

---

# Per-issue knowledge packets

Each packet is send-ready (fits one mesh message). `verify` is the per-issue verification contract. `repo`/`branch` locate the work; "fork" = `bong-water-water-bong/llama.cpp` (HRX forks on strixhalo: `~/hrx-ws/amd-hrx-graph`, `~/hrx-slice/`, `~/hrx-gfx1151/`).

---

## #2117 — ggml-hrx over-claims ops its kernels cannot run (P0)

- **Status:** Root cause known. `eager_capability_declared()` over-claims op families with partial loom-kernel coverage; scheduler commits nodes to HRX, kernels reject shape at graph build → hard "unsupported HRX node" error, no CPU fallback.
- **Confirmed triggers (all on zaya port):** GET_ROWS (262272-vocab embedding), ROPE (partial-head n_rot=64<128), SOFT_MAX (17-slot router), ARGSORT (full 16-expert), GLU (gate/up-along-ne2), MUL_MAT (>262144-row lm_head + f16×BOTH sides), empty (0-element) tensors.
- **Local mitigations already committed** (fork `fix/hrx-ngl-init-order`): pattern-precise predicates + empty-tensor guard + 0-length binding tolerance; qwen3-regression-clean.
- **Repo/branch:** `bong-water-water-bong/llama.cpp` fork, `fix/hrx-ngl-init-order`. Key files: `ggml/src/ggml-hrx/ggml-hrx.cpp` (`eager_capability_declared`, `supports_op`), dispatch matchers, kernel corpus `ggml/src/ggml-hrx/kernel-corpus/`.
- **Fix plan target:** implement the in-code TODO — split placement capability from exact graph-execution capability so `supports_op` reports actual kernel capability and unhandled shapes split to CPU cleanly. Verify each of the 7 trigger shapes splits (not errors) and qwen3 stays regression-clean.
- **verify:** each of the 7 op/shape triggers splits to CPU instead of hard-erroring; zaya `-ngl 99` no longer aborts on an unhandled shape; qwen3 dense decode unchanged.

## #2115 — HRX host-execution path corrupts/errors CPU-mixed graphs (P0)

- **Status:** Root cause known. ggml-hrx host-execution path (HRX0_HOST buffer) is not a correct CPU equivalent: stream/command-buffer state bug ("command buffer is not in a recording state") + numeric divergence for CPU-mixed graphs (ngl0 or CPU-forced ops).
- **Local fix committed:** `7ac6a8e` clears GPU/IGPU lists at `n_gpu_layers==0` → pure CPU. Underlying host path still broken for fine-grained CPU↔HRX mixes.
- **Repo/branch:** fork `fix/hrx-ngl-init-order`. Key files: `ggml/src/ggml-hrx/ggml-hrx.cpp` (host execution path, stream/command-buffer lifecycle).
- **verify:** qwen3-0.6B `-ngl 0` with HRX registered decodes without error; zaya `-ngl 0` decodes == CPU oracle ("Paris"/9079); no "not in a recording state".

## #2116 — zaya decode on HRX (ngl99) still corrupt (P0)

- **Status:** Narrowed. Executes end-to-end (prompt ~25–30, gen ~7–8 t/s) but output garbage. Corruption is NOT one op (gating RMS_NORM/MUL_MAT/MUL_MAT_ID/flash-attn/weights singly+jointly never recovers oracle while HRX is live). CPU-forced zaya ops (partial-head rope, 17-slot router, GLU, recurrent state) bounce through broken host path (#2115). Remaining HRX-claimed layout ops (VIEW/PERMUTE/RESHAPE/SET_ROWS) cannot be CPU-forced without scheduler abort.
- **Env knobs committed (default-off):** `GGML_HRX_DISABLE`, `GGML_HRX_CPU_OPS=<list>`, `GGML_HRX_NO_FLASH_ATTN`, `GGML_ZAYA_DEQUANT_F16`. Isolation matrix: `research/flm-parity/T3-ZAYA-HRX-DEVICE-SPEC.md` round 14.
- **Depends:** #2115, #2117.
- **verify:** `zaya-q4nx-c43.gguf -ngl 99 -p "The capital of France is"` decodes greedy 9079 = "…Paris."; NaN=0.

## #2145 — llama_state-imported ctx fails HRX0 decode @ token 2 (P0)

- **Status (2026-09-11, goal `mtwqm7qx-hlc0ht`):** root-caused and fixed for review. The bundle's HRX backend **over-claims `FLASH_ATTN_EXT` above KV 2048** (not a KV/reserve mismatch): with the guard disabled the graph emits `unsupported HRX node 25: FLASH_ATTN_EXT … 35:f16[128,3072,4,1]` → `compute status: -1`. **And the pre-fix engine never even reached it** — `HrxBackend::reset()` discarded the init-time `HRX_STATE_FILE` import, so the lane decoded from an EMPTY KV and returned `finish_reason: stop`: **silent context loss reported as success.** PR **#2203** adds a decode-side guard that counts a *resumed* context (`HRX_MAX_CTX_TOKENS`, default 2048) and re-applies the import after reset — the lane now decodes from the KV it was handed or refuses with a named cause. Bundle-side over-claim remains upstream-gated (#1945).
- **Repro (current):** shipped bundle `~/hrx-slice/hrx-llamacpp/out/llama-hrx-b66` (`LD_LIBRARY_PATH=$B/lib`, `GGML_HRX_CPU_OPS=RMS_NORM`), blob `~/hrx-2145/hyp_blob.bin` (288,963,676 B, session v9), model `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`, `HRX_MAX_CTX_TOKENS=0` to expose the raw lane. Logs: `/tmp/mx_*` on strixhalo; measurements on #2145 (comment 5632416879).

## #2147 — Qwen3-Coder-30B-A3B (qwen3moe) fails HRX decode (P0)

- **Status:** Open. Every HRX decode fails `graph compute -1` incl. 5-token prompt; CPU oracle (GGML_HRX_DISABLE=1) decodes 8/8. Fork's validated roster = qwen3 DENSE ≤4B + zaya; 30B-A3B is uncovered. Old b59/b66 served 30B-A3B in-process (~80 tok/s) → coverage regression/gap, not impossibility.
- **Repo/branch:** fork `fix/hrx-ngl-init-order`. Repro: `/tmp/m2/rt_H2 nat <Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf> /tmp/m2/x.bin "The capital of France is" 8`, env `LD_LIBRARY_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_devel/lib:<fork>/build/bin RT_NC=512 RT_NGL=99`.
- **Depends:** #2117 (claim precision).
- **verify:** qwen3moe-30B decodes 8/8 on HRX; matches CPU oracle.

## #2153 — no batched FLASH_ATTN_EXT dispatch (ne3>1) (P1)

- **Status:** Root cause known. Flash dispatch matchers hard-require `ne[3]==1`; multi-seq packs active sequences into ne[3], no dispatch → unclaimed → CPU → V-cache transpose on device-resident tensor → abort. Every alternative route empirically closed 2026-09-08.
- **Fix contract captured:** `research/flm-parity/SETROWS-GATE-CONFIRMED.md` (bb14eb5a0) + `BATCHED-FLASH-IMPL-PLAN.md` (efe770683) — stream-loop extension of the decode-split wmma kernel for ne3>1.
- **Repo/branch:** fork `fix/hrx-ngl-init-order`. Files: `ggml/src/ggml-hrx/.../dispatch-flash-attention.cpp` (+ kernel corpus).
- **verify:** `llama-server -np 4` batched decode fires batched flash without abort; multi-seq tokens correct. (CPU-path multi-seq ~33 t/s = verified interim.)

## #2152 — CONCAT dim-0 loom kernel corrupts cols-6 + faults conv_input (P1)

- **Status:** WIP. Kernel built through full pipeline (manifest→dispatch-registry→DispatchRegistration→root-param match) and FIRES, but decode corrupt for cols=6 (QKraw [1024,6]+[256,6]); conv_input (cols=1280, CPU-pinned recurrent src0) AMDGPU-faults. Two layout hypotheses eliminated (flat-copy; ne0-fastest strided view). Remaining suspects: non-plain dim-0 semantics in zaya reshape chain; dispatch matcher rows/cols derivation; concat-output HRX→CPU handoff.
- **Repo/branch:** fork `fix/hrx-ngl-init-order`. WIP (untracked): `kernels/loom-libs/ops/concat_f32.loom` + `dispatch_registration/common/dispatch-concat.{cpp,h}`. Gate: `GGML_HRX_CONCAT_COLS`.
- **verify:** cols-6 + conv_input concat decode matches CPU numpy reference of one captured concat (device readback or CPU-forced placement diff). Blocks conv-speed path alongside #2151.

## #2150 — zaya HRX decode launch-bound 6.4 t/s (P1)

- **Status:** Root cause measured. ~640 HRX program launches/token × ~0.17 ms ≈ 110 ms of 157 ms/token; compute ≈ 50 ms (ceiling ~20 t/s). Programs cache-hit but every submit+sync ~0.17 ms; sched splits at CPU-island boundaries into ~640 dependent programs/token. Adding kernels is net-negative (each claimed op adds a program).
- **Required fix (structural, absent):** record each token's full program sequence as ONE iree command graph with event dependencies (CUDA-graph style), submit once/token. Floor ~50 ms → ~16–20 t/s.
- **Repo/branch:** fork `fix/hrx-ngl-init-order`. Evidence: `research/flm-parity/{GOAL-STATUS,KERNEL-ROUND,RESULTS-TASK5-FINAL,CHECKPOINT-2026-09-08-ec8072}.md`.
- **Depends:** #2080 (single-launch executor substrate).
- **verify:** single-launch/token decode reaches ≥16 t/s; oracle-exact token stream (9079/236761/107/…), NaN=0.

## #2159 — HRX-ALONE decode can't reach stock-Vulkan class (P1)

- **Status:** Root cause characterized. Ceiling 231/123/58 t/s (0.6B/1.7B/4B) vs Vulkan bars 350/168/76 (llama-bench pp512/tg128, ngl-1). Correctness bit-exact; throughput is the blocker. Constant ~1.3× achieved-BW shortfall (115 vs 144 GB/s @0.6B), unchanged by any kernel change. HIP probe: same algorithm under hipcc = 134 GB/s at decode granularity, 220+ at scale. Mechanism: loom emits 3× `s_delay_alu` density (27% vs hipcc 10%) — dependent VALU ops at tight distances; memory-bound so scheduler-ILP fixes have bounded payoff.
- **Three constraint-change options (issue body):** (1) permit hand-written HIP/amdgcn decode kernels (~290–350 t/s at the bar boundary); (2) sanction loom scheduling surgery (deep, fleet-risk); (3) accept dual-engine auto-route + `GGML_HRX_DISABLE=1` (already clears bars: 360/172/77) — but that is NOT HRX-alone (audit rejected).
- **Evidence:** `research/launch-collapse/2026-09-08-*.md` (commits acb413559..f508ae2da).
- **verify:** HRX-alone reaches ≥ Vulkan bars (350/168/76) OR one of the 3 options documented as the accepted decision.

## #2113 — single-launch cascade perf tuning (P1)

- **Status:** Known. Correctness-first single-launch fused GU→SiLU→D cascade ≈12.3 ms (12.4–14.9 incl. per-token pack/fold) vs tuned production split (P1 GU+silu→h2 + P2 D) = **4.65 ms** (M1 profile). Cascade is NOT yet a decode win. Remaining: E1/E3 D-launch tuning; wire as default MoE FFN executor once < split; confirm fold-element AB stream doesn't regress batch/all-ones (`fused_ab_probe`).
- **Baseline (M1, 2-token decode):** attn 0.71 ms/layer, moe 38.69 ms/layer; [CAS] 12.4 ms incl. pack.
- **Depends:** #2078✅ (resolved, corr 0.578→0.999346), #2114.
- **verify:** cascade single-launch < 4.65 ms split; wired as default MoE FFN; batch/all-ones probe still bad=0.

## #2114 — full-decode cascade validation sweep (P1)

- **Status:** Thin. Validated on ONE point (layer-1, routed expert e=7, 2-token, real residual): corr 0.999346 vs CPU float, 0.999827 vs 2-launch, maxabs 0.0208. NPU_CASCADE_DECODE proven only as layer-1 FFN.
- **Needed:** (1) sweep [CAS] across all 40 MoE layers × top-2 routed experts (no degenerate fold gsG/gsU near-zero columns or scale outlier); (2) multi-prompt decode gate (per-layer logits corr + final-token argmax parity); (3) regression batch/M=8 all-ones (`fused_ab_probe bad=0/16384` with fold-element generator + new xclbin); (4) quantize residual gap (0.999346 vs int8-mirror CPU, not float).
- **Artifacts:** `zaya_decode.cpp` [CAS]/NPU_CASCADE_DECODE probes; dumps + scripts on ryzen `~/zaya2078-analysis/dump` (FINDINGS-20260905.md).
- **Depends:** #2113.
- **verify:** 40-layer × top-2 sweep clean; multi-prompt argmax parity; batch probe bad=0/16384.

## #2102 — mm.xclbin fused-dequant within-tile layout permutation (P0)

- **Status:** Open. Kernel executes + fused-dequantizes (C[256,8192] nonzeros=93888, sum=192) but corr(C, CPU-dequant W) = 0.000772. Kernel reads quantized tiles in its own within-tile reorder (`reorder_cpy` — `8*(ir/8)+2*tc+(tr%2)` interleave), not natural tensor order.
- **Repo/files:** `tools/mm_gemm_oracle.cpp`; refs `docs/research/fastflowlm-analysis/Q4K_METADATA_CRACK.md` §13–14, `npu-infer/tests/test_npu_gemm.cpp`, `npu-infer/tools/gen_mm_insts.cpp`.
- **verify:** C correlates with W_gt (corr ≥ 0.999) via correct within-tile permutation.

## #2105 — fused-dequant output has 93696 NaN (P0)

- **Status:** Likely cause = A/B geometry (M=256 K=2048 N=8192) / tile ordering not aligned with kernel's within-tile layout → dequant reads invalid scale/mantissa bytes → NaN. Closely related to #2102. NaN scattered (C[0]=0, C[1]=0, C[2]=nan…); kernel executes (93888 nonzeros, sum=192).
- **Needed:** determine NaN cause (dims vs alignment); harness reports NaN (m,n) pattern to localize misaligned slice; drive with known-good geometry before scaling to q_proj.
- **Depends:** #2102.
- **verify:** NaN=0 across full C[256,8192]; high corr vs W_gt.

## #2080 — Zaya whole-layer per-ctx ELF decode (M3/M4) (P2)

- **Status:** M1 done (MoE-bound 124/156 ms/tok; runlist not applicable per-op). M2 in progress (single-launch cascade, N_D=2048 ROWS=4 built; numerics open). M3/M4 = author whole-layer kernels (co-place attn + fused MoE, fold residual/norm/cca_prep/RoPE/silu/router on-device, KV→device BO) + `RuntimeLayerEngine`-for-Zaya `forward()` (one `xrt::runlist`/token). Weeks of AIE2P bring-up.
- **Why authoring not capture:** FLM ships no Zaya model → no ELFs to interpose-capture.
- **Building blocks (verified separate):** `attn.xclbin` (corr ≥0.999), fused GU→SiLU→D (Qwen3 exact; Zaya numerics open), per-ctx layer-ELF + runlist via #2063 (`RuntimeLayerEngine`, `npu-infer/src/runtime_layer.cpp`).
- **Artifacts:** `~/wt/zaya-m1/engine/npu/docs/M3-ZAYA-WHOLE-LAYER.md`.
- **Depends:** #2113, #2114, #2070 (device-VA, parked).
- **verify:** whole-layer ELF + one runlist/token; numeric parity vs CPU-float; byte-determinism; ≥25–40 tok/s.

## #2081 — product decision: fused int4 vs int8 dense FFN (P2)

- **Status:** Decision needed. (a) fused int4 primary (`NPU_FUSED_USE=1` default-on after decode-quality gate): 3.15 MB/layer GU DMA vs 6.3 MB, zaya i4 corr 0.999336 (beats int8 0.9985); cost = 4-bit-vs-8-bit representation delta. (b) keep int8 baseline, int4 env-gated experiment.
- **Acceptance for (a):** default-on after token-parity/quality gate vs int8 baseline (qwen3-0.6b + zaya1-8b); `npu_state_*` wiring of `packB_into_fused_i4`; ms/layer + tok/s before/after.
- **Depends:** #2114 (decode-quality gate data).
- **verify:** recommendation with token-quality gate data + perf record.

## #1866 — aie2p -O0 backend crash (P2)

- **Status:** **Upstream-only watch (re-verified 2026-09-11, goal `mtwoc3im-zdxsxb`).** The crash is real — on `main` @ `6963fc694` with llvm-aie `91977805fa`, `-O0` still dies with "immediate operand value -33216 is out of range [-32768, -64]" (−O1/−O2 compile). **But nothing in-repo requires `-O0`:** the kernel builds pin `-O2` (`build_c1b_iron.sh`) and `-O1` (`build_c1b_iron_o1.sh`), the generators never ask for it, and every remaining `-O0` reference that concerns aie2p is prose about this bug. The impact recorded here ("-O2 miscompiles scalar RMW (#1864); -O1 same miscompile") is **stale**: #1864 was fixed in-tree on 2026-08-29 and closed.
- **Repro:** `<peano>/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O0 -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED -I<aietools>/include -c engine/npu/generators/mm_kernel_reference.cc`. *(On strixhalo the include root is `~/Xilinx2025/2025.2/Vitis/aietools/include`; the `~/Xilinx/2025.2/...` path in `build_p1i4.sh` does not exist there.)*
- **verify:** re-run the repro when llvm-aie PRs **#1155** ("[AIE2P] Range-check the spill immediate offset in eliminateFrameIndex") or **#1276** land; expect `-O0` to compile via the indexed-addressing fallback. Until then `-O1`/`-O2` are the paths and nothing waits on this. Verification comment: [#1866 comment](https://github.com/1bit-MONSTER/1bit-MONSTER/issues/1866#issuecomment-5631463570).

## #2082 — D2 HIP-prefill lane stability gate (P1)

- **Status:** Re-probe clean. 09-02 direct-`llama_decode` SIGSEGV (2-token, CPU+HIP, nondeterministic) suspected build race/UB; 09-03 ~300 llama-bench pp2/tg2 evals CLEAN (clang build-hip + gcc); no upstream threadpool fix in window; REPL artifact = invocation issue.
- **Tasks:** (1) locate/rebuild round-13 direct-harness source; (2) context-flag matrix (n_ubatch 2/16/512 × n_ctx 512/4096/8192 × KV F16/AUTO × flash on/off × threads 1/4/8/16, ≥20 runs/cell); (3) LD_PRELOAD interposer A/B (clean-env control); (4) stability gate Nx200 pp2/tg2 clean (both compilers); (5) then #1942 D2 acceptance (§5.2).
- **Depends:** #1942.
- **verify:** stability gate Nx200 clean; 09-02 crash reproduced+root-caused or declared benign.

## #2139 — qwen35moe 1BP fast-path + quant lanes + hipGraph (P2)

- **Status:** Open (follow-up to #1831✅ / #2138✅ merged). Gaps: (1) hipGraph capture for q35 decode (eager today); (2) packed per-tensor fast paths (decode launches generic `launch_q4nx` per tile class — top-8 expert slices, shared expert, lm_head/embed); (3) wire ROCmFP4 / F16/F32 / TQ2NZ quants (1BP gate refuses everything but ONEBP_Q4NX, `quant2 != 3`).
- **Repo/files:** `engine/npu/.../backend_hip_1bp.cpp` q35 block; `launch_q4nx` / `launch_rocmfp4` / `launch_tq2nz` exist.
- **Acceptance:** corr ≥ M3-baseline per newly-enabled quant (same-file corr methodology as #2138); q35 hipGraph decode > eager (gfx1151, #2138 bench driver).
- **verify:** per-quant corr gate + hipGraph > eager.

## #1942 — hybrid prefill/decode policy (HIP prefill + HRX warm decode) (P2)

- **Status (2026-09-11, goal `mtwqm7qx-hlc0ht`):** the cross-backend handoff **is resolved** (state round-trip byte-identical, PR #2146 shim + zero-copy memfd work on #2161). What remains is the §5.2 acceptance's positive clause, and it is blocked **bundle-side**: the shipped b66 HRX over-claims `FLASH_ATTN_EXT` above KV 2048, so a 2,940-token imported context cannot decode on the HRX device — the engine now refuses it explicitly (PR #2203) instead of silently decoding from an empty KV. No engine-loadable bundle on the box supports >2048 KV. Re-scoped measurement + the re-open triggers are in `docs/research/hybrid-prefill-decode.md` §5.2.
- **Acceptance (re-scoped, Option A 2026-09-08):** the D2 shipped path delivers CORRECT warm decode on the HRX device — measured 2026-09-11 as **not met on the shipped bundle** (KV ceiling), with silent context loss removed as the intermediate win.
- **Depends / triggers:** #2145 (fix in PR #2203) and the upstream bundle repin **#1945** (or HRX2 decode-ADD coverage) before the positive clause can be re-run.
- **verify:** re-run the §5.2 matrix (pinned blob + model, `HRX_MAX_CTX_TOKENS=0`) once an HRX with >2048 KV lands; token-parity vs the CPU oracle.

---

## Cross-issue dependency map

```
#2117 (claim precision) ──► #2116, #2147, (unblocks #2153/#2152 clean splits)
#2115 (host path)      ──► #2116
#2145 + #2082          ──► #1942 (D2 hybrid)
#2113 ──► #2114 ──► #2081 (quality-gate data)
#2113 + #2114 + #2070  ──► #2080 (cascade → whole-layer)
#2102 ──► #2105 (NaNs)
#2150 ──► #2080 (single-launch executor substrate)
```

Priority order to dispatch: **P0 first** (#2117, #2115, #2116, #2145, #2147, #2102, #2105), then P1, then P2. Waves preserve dependency direction (a dependee is dispatched before/with its dependents).
