# NPU recon inventory — strixhalo (2026-09-15)

Goal `mu35shsg-i3hlyi`, task-recon. Read-only pass over the box: hardware, the
home-directory prior work, the harnesses, the artifacts, and the *live* lanes
that were already running when this goal started. Every line carries the path it
came from.

## 1. Hardware (as measured, not from the docs)

| item | value | source |
|---|---|---|
| APU | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, 32 CPUs, 1 socket, 32 NUMA-0 CPUs | `lscpu` |
| Memory | 122 GiB total, 55 GiB used, 11 GiB free, 44 GiB shared | `free -h` |
| NPU | `/dev/accel/accel0` (crw-rw---- root:render), amdxdna module 0.1 | `ls -l /dev/accel/`, `modinfo amdxdna` |
| XRT | 2.21.75 | `xrt-smi examine` |
| NPU firmware | 1.1.2.65 | `xrt-smi examine` |
| BIOS | 1.09 | `xrt-smi examine` |
| VRAM carve-out | 1 GiB (1073741824 B) — UMA carve small, as documented | `/sys/class/drm/card*/device/mem_info_vram_total` |
| GTT | 120 GiB (128849018880 B) | `/sys/class/drm/card*/device/mem_info_gtt_total` |
| Kernel | 7.2.0-next-20260821-unstable-ogc-g2a559b27-1 | `uname -r` |
| Disk | 1.9 T NVMe, 1.4 T used, 366 G free (80%) | `df -h /home` |
| `flm` | `/opt/fastflowlm/bin/flm` — **not on the non-login ssh PATH**; call by absolute path | `command -v flm` empty, path used in the peer's harness |

## 2. Engine checkouts and the dirty state

| path | branch | HEAD at recon | note |
|---|---|---|---|
| `~/1bit-MONSTER-goal` | `goal/runlist-decode-wire` | `2103a2b10` → `4929ae2b1` during recon | the active lane (see §5) |
| `~/1bit-MONSTER` | `fix/qwen3-4b-split-g-artifacts` | `57511b39d` | separate lane |
| `~/1bit-MONSTER-npu` | — | — | engine tree, not the goal lane |
| `~/wt/family-head-block` | `family/head-block-loop` | `4929ae2b1` | created by this goal for the family lane |

**Unexplained-dirty-state resolution.** `git -C ~/1bit-MONSTER-goal status` shows four
classes of uncommitted entry, none of them mine and none touched:

| entry | mtime | disposition |
|---|---|---|
| `engine/npu/xclbins/attn_mha_1024_nh20_hd128.elf` (177728 → 154528 B) | 00:40 | earlier session's rebuild from the withdrawn L2 nh20 chase (`benchmarks/RESULTS-attention-c2-regression-2026-09-15.md:324,353` call it a real capture artifact but not the one the bf16 arm can use) |
| `engine/npu/xclbins/final_bf16_{D,G,O,QKV,U}_*.xclbin` (84752 B each) | 14:08 | the five Nanbeige bf16 projection builds from the same lane |
| `engine/npu/xclbins/insts_bf16_*.txt` | 14:08 | their instruction streams |
| `tools/q35_full_golden.py` | Sep 14 21:46 | Qwen3.5 golden-reference script, predates today's sessions |

They are left exactly as found: the live peer goal `mu34scbf-4ffm1o` states as a
binding rule "Do not commit or discard another session's in-flight work", and the
same artifacts survive ~20 of its commits, which is the evidence that its commits
are scoped rather than `git add -A`. My own work is on `family/head-block-loop` in
`~/wt/family-head-block` so nothing of mine enters that worktree.

## 3. Prior work recovered (goal IDs and the docs that carry it)

| goal | lane | artifact |
|---|---|---|
| `mttxt22c-a6rv75` | long-context parity, declared complete 2026-09-15 | `~/1bit-MONSTER-goal/benchmarks/LEVERS-register-2026-09-15.md` |
| `mtunui03-ekarlt` | runlist whole-layer decode | `~/npu-ab/` + register §2 |
| `mtuhp2fy-c8yfgb` | dense decode wiring — speed already met, superseded by accuracy | `benchmarks/RESULTS-0_6b-dense-decode-wiring-2026-09-15.md` |
| `mu34scbf-4ffm1o` | **live** correctness/accuracy | `benchmarks/RESULTS-oracle-accuracy-0_6b-2026-09-15.md`, `~/.pi/goals/active_goal_2026091517351951_mu34scbf-4ffm1o.md` |
| — | family coverage (prior) | `RESULTS-remaining-families-parity-2026-09-11.md`, `RESULTS-coverage-multifamily-2026-09-13.md`, `RESULTS-family-attention-shape-2026-09-14.md` |

Harnesses that exist and are reused rather than rebuilt:
`~/npu-ab/npu_ab.sh` (A/B yardstick with the I1–I5 invariants and the native gate
truth table; `~/npu-ab/BASELINE-2026-09-10.md`), `benchmarks/oracle_accuracy_0_6b.sh`
(20 prompts, FLM oracle), `benchmarks/flm_parity.sh` (**knows to be I1-violating** —
see §4), and the device-claim convention `~/.dsh/scratch/mesh/`.

The Vitis install repair is real and still in place: `~/Xilinx/2025.2 ->
~/Xilinx2025/2025.2` (also in `/home/bcloud/1bit-MONSTER` and `/home/bcloud/wt/*`),
licence at `~/.Xilinx/Xilinx.lic`.

## 4. Finding added by this recon: the tokenizer defect is every family, not one

`npu_ab.sh:157-163` records that `engine/npu/tokenizer/tokenize` drops BPE merges and
that this was measured on Qwen3-0.6B. Measured today for all nine family variants
(`/tmp/fam/ids1024_<Model>.txt` vs `/tmp/fam/ids_engine_<Model>.txt`, same source
`~/npu-ab/README.md`):

| model | HF tokenizers ids | engine tool ids | agreement |
|---|---:|---:|---|
| Nanbeige4.1-3B | 1024 | 2555 | DISAGREE |
| Phi4-mini-Instruct | 1024 | 2535 | DISAGREE |
| Gemma3-1B | 1024 | 1052 | DISAGREE |
| Gemma3-4B | 1024 | 1052 | DISAGREE |
| Gemma4-E2B-IT | 1024 | 1052 | DISAGREE |
| Gemma4-E4B-IT | 1024 | 1052 | DISAGREE |
| LFM2-1.2B | 1024 | 2716 | DISAGREE |
| LFM2-2.6B | 1024 | 2716 | DISAGREE |
| Qwen3.5-4B | 1024 | 2587 | DISAGREE |

So any harness that pipes the engine tool's output into one arm (the register's
finding 3) violates I1 on **every** family, not only Qwen3. The family id sets here
are built with HF `tokenizers` 0.23.1 for that reason.

## 5. Live lanes at recon (why this goal could not start writing)

`~/1bit-MONSTER-goal` is an active multi-agent worktree. During recon:
`mu34scbf` had 5 `npu_engine_*` processes resident and, in 20 minutes, landed the
special-token finding, the detokenizer fix, the "dense is broken" refutation, the
1.7B/4B/8B extension and the retirement of the corr ≥ 0.998 criterion (details in
the goal file and `RESULTS-oracle-accuracy-0_6b-2026-09-15.md`). The L1 attention
lane is claimed at 14:35 by another session
(`~/.dsh/scratch/mesh/device-claim-goal-lane-2026-09-15.txt`) with no handback yet.
The parked production `flm serve qwen3.6-moe:35b-a3b` (PID 2438, port 8098) holds
the device baseline and was not touched.

Reachability: this agent (ryzen, alias `agent-741b58`) cannot message the box's
agents — the ryzen↔strixhalo mesh relay has produced no frames since 2026-09-11
(`~/.mesh-relay/relay.log`), so the box's file convention is the only channel.

## 6. Family-lane blockers, as the code states them

| family | nh | nkv | hd | what blocks it | citation |
|---|---:|---:|---:|---|---|
| Nanbeige4.1-3B | 20 | 4 | 128 | head-block loop (nh > n_aie_cols), and `nkv` is hardcoded 2 | `n1_core_attn.py:72-94`, `:101` |
| Phi4-mini | 24 | 8 | 128 | head-block loop | same |
| Qwen3.5-4B | 16 | 4 | 256 | head-block loop **and** PV N-split (hd > n) | `n1_core_attn.py:56-70` |
| Gemma3-1B/4B | 4/8 | 2 | 256 | PV N-split only | same |
| Gemma4-E2B/E4B | — | — | — | not yet characterised in the recovered docs | — |
| LFM2-1.2B/2.6B | — | — | — | **no engine binary exists** (`engine/npu/build/` has no `npu_engine_lfm2_*`) | `ls engine/npu/build/` |

`RESULTS-attention-c2-regression-2026-09-15.md:959-1000` already sketches both fixes
(head-base + outer pass loop; PV head-dim tiling) and fixes the order: **PV N-split
first** (Gemma3, self-contained), then the head-block loop (Nanbeige, Phi4), with
Qwen3.5-4B needing both. The guard for any such change is named there too:
`attn_insts.txt` must stay `f3d0a132bde24a60` for the hd128/nq8 build.
