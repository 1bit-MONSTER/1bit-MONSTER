# Agent Coordination — 1bit-MONSTER (ryzen ↔ strixhalo)

> This repo is worked by **two DeepSeek Harness agents on two machines**.
> Both edit this same codebase; this file is the shared handoff ledger.
> **Read it before starting work. Update it when you change lanes or land
> something. Keep both machines' clones in sync (protocol at the bottom).**

## Agents & machines

| Agent | Machine | LAN IP | Workspace | Repo remote | GitHub identity |
|-------|---------|--------|-----------|-------------|-----------------|
| **Co-worker** | ryzen (Ryzen 7 9800X3D) | 192.168.50.100 | `~/projects/1bit-MONSTER` | `fork` → `bong-water-water-bong/1bit-MONSTER` (branch `fix/triage-round`) | bong-water-water-bong |
| **Strixhalo agent (this one)** | strixhalo (Ryzen AI Max+ 395, NPU box) | 192.168.50.110 | `/home/bcloud/1bit-MONSTER` | `origin` → `1bit-MONSTER/1bit-MONSTER` (branch `main`) | bong-water-water-bong (same account) |

- Both agents share the GitHub identity `bong-water-water-bong` — either can push to the
  fork and to upstream `main`.
- DSH Web GUI: ryzen `http://127.0.0.1:3080` (local), strixhalo `http://127.0.0.1:3080`.
  There is **no DSH↔DSH chat API** — coordination happens through this file + git.

## Ownership map (who fixes what)

| Area | Owner | Notes |
|------|-------|-------|
| GitHub issue triage + fixes (#1832, #1834, #1837, #1864, #1865, #1870, #1872, #1874, #1878, #1882, #1908, #1909, #1910, #1911, #1913, #1776 gate, #1831 interim, census #1900/#1906) | **Co-worker** (ryzen) | branch `fix/triage-round` on fork; 14+ issues landed 2026-08-28 |
| Fused GU→SiLU→D cascade kernel work (BUG-001..011, #1775/#1769) — p1/p2 two-launch production | **Strixhalo agent** | committed on `main` (aecfad54): BUG-005 D-cascade fix silicon-verified; single-launch premise REJECTED (BUG-011) |
| NPU HW verification on strixhalo (`/dev/accel0`) | **Strixhalo agent** (only machine with the NPU) | verify co-worker's kernel changes + cascade work |
| Upstream escalations (llvm-aie/peano: #1836, #1844, #1912, #1866, #1835) | **Co-worker** leads; strixhalo agent provides reproducers/evidence from HW | tracked in #1882 |

## Shared file guard

**`engine/npu/generators/mm_kernel_reference.cc` is edited by BOTH agents.** It has been
merged (2026-08-28, merge commits 88de972c + 5c78007b — full sync): the file now contains
the co-worker's complete `fix/triage-round` work (KERNEL_STATIC .data, #1865 arg-based
h2/pC + retired #1842 pins, #1874 I4_SCALAR_C1 production default, #1872 direct-vector
dequant, #1878/#1912 unpack_i4_sx shim) AND the strixhalo agent's `silu_quant_i8_fused_q22`
+ `cascade_reduce_*_i32` single-pass forms. Verified: 0 syntax errors with the cascade
defines (`-DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED`).

**Rule for this file:** pull/merge before touching it; never overwrite the other side's
functions; if a conflict appears, preserve BOTH sides and note it in the merge commit.

## Status snapshot (2026-08-28, fully synced)

- **Sync complete**: `main` now contains ALL co-worker work (fork state + the 16 newest
  commits from upstream `fix/triage-round` @ 954dc298) + strixhalo agent's cascade work
  (aecfad54) + upstream main. Merge commits: 88de972c, 5c78007b.
- Co-worker: 27 issues triaged; 14+ fixed; remaining = HW-verify on strixhalo
  (#1865 h2 byte-identity gate, #1872 direct-vector dequant re-validation, #1874 mmul
  transpose ISS validation, #1832 q4nx decode on real NPU) + XL features (#1907 baretorch,
  #1831 HIP GatedDeltaNet+MoE) + upstream escalations.
- Strixhalo agent: fused-cascade work committed (aecfad54) + bug reports BUG-001..011;
  BUG-011 decision: single-launch zero-DMA premise REJECTED — p1/p2 two-launch (h2 via
  DDR) is the production path; D GEMM must be one cascade pass, per-core partial fits L1
  (N_D ≤ 256 verified).
- **Handoff**: strixhalo agent will HW-verify the co-worker's pending NPU items on
  `/dev/accel0` and report results here.

## Zero-h2-DMA single-launch: PROVEN BLOCKED (2026-08-28)

The doc's option (a) (2-channel dataflow multiplexing B_d over a GU channel) was
implemented and tested on silicon: split GU A/B into two 2-D single-stream fifos
(ch0 A-tile, ch1 B-tile carrying B_gu then B_d). It **builds** but the shared-B
fifo's D-cascade writeback does NOT fire (C2=0x5A, reproduces cleanly even for a
`--no-gu` D-only probe). Controls: the ORIGINAL (combined-AB GU + dedicated
`of_b_d`) D-only design IS silicon-exact (C2=2048, bad=0); a 3-fifo dedicated-B
variant won't place (2-input-DMA, BUG-007). So the complete zero-h2-DMA fused
single launch remains blocked by the iron ObjectFifo + 2-input-DMA constraint.
Production path stays p1/p2 two-launch (h2 via DDR). Next options: (b) an iron
FIFO primitive that pipelines merged/segmented elements (toolchain-level), or a
way to reuse one channel without the shared-B writeback regression.

## Sync protocol (both agents)

1. **Before starting work:** `git fetch origin` (and the fork), merge/rebase `main`
   (and `fix/triage-round` if you touch kernel files), read this file.
2. **After landing anything:** push immediately; update this file's snapshot table;
   mention the commit SHA.
3. **Kernel file rule:** see above — pull first, preserve both sides, never force-push.
4. **NPU is single-device:** do not run the 1bit engine / xclbin benchmarks on strixhalo
   while the other side is validating there (documented AMD-Vi IO_PAGE_FAULT storms).
5. **Coordination messages:** commit them here (append a dated note) rather than relying
   on chat; the other agent reads this file on its next pull.
