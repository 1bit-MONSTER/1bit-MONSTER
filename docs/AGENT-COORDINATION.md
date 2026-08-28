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
merged once (2026-08-28, merge commit 88de972c): the file now contains the co-worker's
`KERNEL_STATIC` (.data) + #1865 arg-based h2/pC + #1874 I4_SCALAR_C1 default + #1872
direct-vector dequant AND the strixhalo agent's `unpack_i4_sx` shim + `silu_quant_i8_fused_q22`
+ `cascade_reduce_*_i32` single-pass forms. Both sides are syntax-verified.

**Rule for this file:** pull/merge before touching it; never overwrite the other side's
functions; if a conflict appears, preserve BOTH sides and note it in the merge commit.

## Status snapshot (2026-08-28)

- Co-worker: 27 issues triaged; 14+ fixed; remaining = HW-verify on strixhalo
  (#1865 h2 byte-identity gate, #1872 direct-vector dequant re-validation, #1874 mmul
  transpose ISS validation, #1832 q4nx decode on real NPU) + XL features (#1907 baretorch,
  #1831 HIP GatedDeltaNet+MoE) + upstream escalations.
- Strixhalo agent: fused-cascade work committed (aecfad54) + bug reports BUG-001..011;
  BUG-011 decision: single-launch zero-DMA premise REJECTED — p1/p2 two-launch (h2 via
  DDR) is the production path; D GEMM must be one cascade pass, per-core partial fits L1
  (N_D ≤ 256 verified).

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
