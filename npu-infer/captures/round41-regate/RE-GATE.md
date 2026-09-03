# Round 41 re-gate receipts — runtime drift REVERTED on the 2026-09-03 fresh boot (2026-09-03)

Fresh real-runtime capture (run_qwen3_npu harness, XRT path, amd-oss libs,
model Qwen3-0.6B-NPU2, tokens 1000+1001 = 2 forwards) on the **2026-09-03
10:37 ADT boot** (kernel `7.2.0-next-20260821-unstable-ogc-g2a559b27-1`),
captured with the round-35 wait-hook interposer
(`cap_interposer.so`, runlist::wait actpost/kvpost hooks). This is the
direct re-run of the Round-40 pre-reboot reference baseline
(`/home/bcloud/.cache/rtcap-postreboot-20260903/`, taken 10:31 on the
drifted boot) and the round-37-era artifacts
(`/home/bcloud/.cache/rtcap/`, Sep-1/2).

## Setup

    # harness + interposer rebuilt from feat/hrx-gfx1151-build sources (this branch)
    g++ -O2 -std=c++20 -include climits run_qwen3_npu.cpp utils_stub.cpp -o run_qwen3_npu \
      -I/home/bcloud/amd-oss/fastflowlm/src/include \
      -I/home/bcloud/amd-oss/fastflowlm/src/include/npu_utils \
      -L/home/bcloud/amd-oss/fastflowlm/src/lib/xrt \
      -lqwen3_npu -lq4_npu_eXpress -lgemm -ldequant -lmha -llm_head \
      -L/usr/local/lib -laiebu -lxrt_coreutil -lxrt_core \
      -Wl,-rpath,/home/bcloud/amd-oss/fastflowlm/src/lib/xrt
    g++ -O2 -fPIC -shared cap_interposer.cpp -o cap_interposer.so -ldl -lxrt_coreutil
    LD_PRELOAD=$PWD/cap_interposer.so CAP_DIR=<out> ./run_qwen3_npu /home/bcloud/Qwen3-0.6B-NPU2 2
    # -> actpost_002/004 (acts after runlist waits), bo_from_0199 (ctx-1 logits),
    #    bo_from_0229 (ctx-2 logits), 850+ weight/TXN BO captures

Captured files here (all 1 MiB BO dumps, this boot):
`actpost_002_563cb3f93dc0_1048576.bin` (ctx-1 acts),
`actpost_004_563cb3f93dc0_1048576.bin` (ctx-2 acts),
`bo_from_0199_1048576.bin` (ctx-1 logits, first 303872 B = 151936 bf16),
`bo_from_0229_1048576.bin` (ctx-2 logits).

## Result — today's runtime is BYTE-IDENTICAL to the round-37-era runtime

References: round-37 era = `/home/bcloud/.cache/rtcap/` (Sep-1 23:46 – Sep-2
00:06); pre-reboot drifted = `/home/bcloud/.cache/rtcap-postreboot-20260903/`
(Sep-3 10:31, 6 min before this boot). bf16 decoded as `u32<<16`.

| measurement | TODAY (fresh boot) | R37 (Sep-1 era) | PREREB (drifted) |
|---|---|---|---|
| actpost_002 std | 8.6037 | 8.6037 | 8.3435 |
| actpost_004 std | 0.8758 | 0.8758 | 0.8682 |
| logits ctx-1 argmax/max | 397 @ 12.8125 | 397 @ 12.8125 | 144370 @ 13.25 |
| logits ctx-1 std | 3.3801 | 3.3801 | 3.3509 |
| logits ctx-2 argmax/max | 88 @ 13.75 | 88 @ 13.75 | 88 @ 14.125 |
| logits ctx-2 std | 2.2089 | 2.2089 | 2.2368 |

| comparison | actpost_002 | actpost_004 | logits ctx-1 | logits ctx-2 |
|---|---|---|---|---|
| TODAY vs R37 | corr **1.00000000**, **byte-identical** | corr **1.00000000**, **byte-identical** | corr **1.00000000**, maxdiff 0, **byte-identical** | corr **1.00000000**, maxdiff 0, **byte-identical** |
| TODAY vs PREREB | 0.99788541 | 0.99776887 | 0.99803301 | 0.99454453 |
| PREREB vs R37 | 0.99788541 | — | 0.99803301 | — |

The TODAY-vs-PREREB correlations reproduce the Round-40 documented drift
magnitudes exactly (act 0.997891, logits 0.998033 / 0.9945), confirming
methodology parity. The drift signature (argmax flip 397→144370 on ctx-1,
act std shift, ~0.2–0.5 logit systematic delta) is entirely absent today.

## Interpretation — closure of issue #2065's re-gate

1. **The round-37-era device state is back.** Acts AND logits, ctx-1 and
   ctx-2, are byte-for-byte identical to the Sep-1 runtime artifacts. The
   runtime arithmetic is per-boot stateful, not permanently shifted.
2. **Round-37's engine↔runtime corr 1.000000 reproduces on this boot**:
   the engine/replay (stable, deterministic 397 @ 12.8125) now equals the
   runtime byte-for-byte again — the engine IS the runtime reference on a
   clean boot.
3. **The "a fresh boot does NOT restore it" observation (Round 40 / FINDINGS)
   was one transition, not a law**: the Sep-2 23:19 boot drifted; the
   Sep-3 10:37 boot restored the round-37-era signature exactly. The drift
   is reversible per boot — consistent with a per-boot device/firmware
   init-state variable (SVA/IOMMU or AIE tile config), not a code or
   arithmetic change in the closed runtime.
4. **Rounds 35–39 runtime-side references are re-validated**: they were
   captured in the round-37-era state, which is the state on this boot.
   Engine-side claims (== round-37-era runtime) hold unchanged.

## Caveats

- This measures ONE fresh boot. The drift's boot-to-boot variance (which
  boots drift, which restore) is still unexplained; the root-cause thread
  (#2065: kernel/driver-dependent execution, SVA vs identity IOMMU,
  firmware arithmetic config) remains open.
- act std here (8.60) is over the full 1 MiB bf16 buffer; the Round-40
  figures (194.46 / 188.62) were computed on a content span — the byte
  identity and correlation numbers above are the loader-independent
  receipts.
