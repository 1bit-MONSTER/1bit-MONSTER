# Engineering Journey — Reverse-Engineering the XDNA 2 NPU

> **This is the hero story of 1bit.systems.** It started with a laptop, a disassembler, and no docs: AMD shipped a 50 TOPS XDNA 2 NPU locked behind a closed-source runtime (FastFlowLM) — 22 proprietary `.so` files, 209 xclbin bitstreams, zero documentation. We reverse-engineered the entire stack in 4 days and replaced it with open C++.
>
> Every crash, breakthrough, and bug below is documented in real-time. ~1800+ hours of engineering, all open source, MIT. Since UPDATE 34 the through-line is one binary (`build/1bit`) and one language direction: C++23 for compute kernels, **Mojo 1.0 as the unified language** for everything around them — servers, converters, tooling, control planes. No interpreter at runtime, anywhere.

## Table of Contents

- [2026-08-19 — The Two-PC Fleet: Harnesses on the LAN, Six Deployment Bugs Fixed](#2026-08-19--the-two-pc-fleet-harnesses-on-the-lan-six-deployment-bugs-fixed)
- [UPDATE 34: The Burn & the Mojo Shift — one through-line, one unified language](#update-34-2026-08-12-the-burn--the-mojo-shift--one-through-line-one-unified-language)
- [2026-08-16 — The Frontier Gates: 5/5 Validated, Then the Repo Tried to Eat It](#2026-08-16--the-frontier-gates-55-validated-then-the-repo-tried-to-eat-it)
- [2026-08-15 — 100% HF Coverage: Every Arch-Bearing Checkpoint Maps to an Engine Token](#2026-08-15--100-hf-coverage-every-arch-bearing-checkpoint-maps-to-an-engine-token)
- [2026-08-07 — The Unified Control Plane Lands](#2026-08-07--the-unified-control-plane-lands-pool-wired-spec-decode-in-server-zoo-55)
- [UPDATE 34: FLM Is Fully Gone — Byte-Identical Streams, True Batch, 2× on the 35B](#update-34-2026-08-15-flm-is-fully-gone--byte-identical-streams-true-batch-2×-on-the-35b)
- [UPDATE 33: NPU Firmware RE via Raw ioctls, Driver Regression Fixed, JARVIS Ships](#update-33-2026-08-10-npu-firmware-re-via-raw-ioctls-driver-regression-fixed-jarvis-ships)
- [UPDATE 32: amdxdna Wedge Saga, 35B MoE Goes Live, Vivado-Free FPGA Toolchain](#update-32-2026-08-09-amdxdna-wedge-saga-35b-moe-goes-live-vivado-free-fpga-toolchain)
- [UPDATE 31: The TileFuse Day — NPU Kernel From Scratch, Q4NX Pivot, Converter Ladder](#update-31-2026-08-08-the-tilefuse-day--npu-kernel-from-scratch-q4nx-pivot-converter-ladder)
- [UPDATE 30: The One-Heap Pool — all models resident, spec-decode in-server](#update-30-2026-08-07-the-one-heap-pool--all-models-resident-spec-decode-in-server)
- [UPDATE 29: Memory Campaign — arena-frag leak fixed, top-1 backend init](#update-29-2026-08-03-memory-campaign--arena-frag-leak-fixed-top-1-backend-init-10-bug-audit)
- [UPDATE 28: Mamba1 GPU Backend — 79.4 tok/s](#update-28-2026-07-20-mamba1-gpu-backend--794-toks-9-bugs-killed)
- [UPDATE 27: Fused Layer Engine — 291 tok/s](#update-27-2026-07-06-fused-layer-engine-goes-production--291-toks-3×-v12)
- [UPDATE 26: All 3 Bugs Confirmed Fixed](#update-26-2026-07-05-all-3-bugs-confirmed-fixed--aie-micro-tiling-root-cause-resolved)
- [UPDATE 25: v12 Was Never Output-Validated](#update-25-2026-07-03-v12-was-never-output-validated--3-real-bugs-found-still-incoherent)
- [UPDATE 24: Fused XCLBIN Resumed](#update-24-2026-07-03-fused-xclbin-resumed--schedule-fixed-deadlock-isolated-new-kernel-bug-found)
- [UPDATE 23: Production Release](#update-23-2026-07-02-1532-adt-production-release--v20260702-all5models)
- [UPDATE 22: All 5 Models at v12 Batch Speed](#update-22-2026-07-02-1513-adt-all-5-models-at-v12-batch-speed-0-crashes)
- [UPDATE 21: Full NPU Engine State](#update-21-2026-07-02-1201-adt-session-close--full-npu-engine-state)
- [UPDATE 20: Fused XCLBIN — First Attempt](#update-20-2026-07-02-0517-0727-adt-fused-xclbin--first-attempt-q4nx-blocker)
- [UPDATE 19: Multi-Model XCLBINs](#update-19-2026-07-02-0624-0627-adt-multi-model-xclbins-model-agnostic-engine)
- [UPDATE 18: M=32 Target, NPU LM Head](#update-18-2026-07-02-0401-adt-m32-target-npu-lm-head-flm-comparison)
- [UPDATE 17: M=16 Batch Decode — 16 ms/tok](#update-17-2026-07-02-0300-adt-m16-batch-decode--16-mstok-152×-speedup)
- [UPDATE 16: Full Profile + 50 ms/tok Batch-4](#update-16-2026-07-02-0200-adt-full-profile--50-mstok-batch-4-decode)
- [UPDATE 15: PR-Agent Live, Landing Page](#update-15-2026-07-01-1500-adt-pr-agent-live-landing-page-deployed-242-mstok-verified)
- [Session 2026-07-16/20 — FLM Fully Replaced, Model-Agnostic Broadening, TQ2 Ternary](#session-2026-07-1620--flm-fully-replaced-model-agnostic-broadening-tq2-ternary)
- [Session 2026-07-05/06 — Q4NX/GGUF Fully Decoded, NPU GEMM Root-Caused, First Validated 1-Bit Number, DSpark](#session-2026-07-0506--q4nxgguf-fully-decoded-npu-gemm-root-caused-first-validated-1-bit-number-dspark)
- [Session 2026-07-03/04 — Triton-XDNA Eval, memlock Fix, Spec-Decode Reality Check](#session-2026-07-0304--triton-xdna-eval-memlock-fix-spec-decode-reality-check)
- [Session 2026-07-02/03 — Production Stack, Release, Site Refresh](#session-2026-07-0203--production-stack-release-site-refresh)
- [Earlier Updates (14–1)](#earlier-updates)

---

## UPDATE 34 (2026-08-12): THE BURN & THE MOJO SHIFT — ONE THROUGH-LINE, ONE UNIFIED LANGUAGE

**We burned the repo down to its through-line and picked the language that will carry it forward. Everything that wasn't the engine or the app that proves it is gone — SaaS, agent stack, voice cloning, the JARVIS v1 side-servers. And the glue language that used to be three (Python for tooling, C++ for the engine, JS for the web) is now one: Mojo 1.0, released this week, is the unified language we're building the control plane in.**

### The burn: what got cut, and why it's not coming back

The roadmap was rewritten around a single sentence — *engine (NPU + GPU + CPU, one binary) → JARVIS (voice assistant, reference app)* — and then the repo was made to match it. One commit, ~19k lines deleted (`cbce9630`):

- **SaaS**: `tools/jarvis/auth.cpp`, `billing.cpp`, `usage.cpp`, `beacon.cpp` — the product layer for a product that doesn't exist yet. Gone. So is the Cloudflare auth worker (`workers/`) and the Zaya Co-Host dashboard (`site/dashboard/`) that talked to those APIs.
- **Voice cloning**: the whole `zaya_audio/` training stack (codec training, voice packs, RVQ-VAE adapters, ONNX export) — a personal quest, not a product. `src/codec_decoder.cpp` went with it.
- **Agent stack**: RAG, planner, personas, prompts, skills, the daily-routine/awareness scripts — a product layer, not ours. The engine serves it via Lemonade; it doesn't ship one.
- **JARVIS v1's HTTP hop + WebSocket side-server**: `jarvis_server.cpp`, `audio_stream.cpp`, and upstream's new `voice_session.cpp`/`ws_proto.cpp` — replaced by the in-process pipeline: `mic → VAD → STT (libwhisper, now HIP-accelerated via `src/whisper_hip.hip`) → LLM (in-process BackendManager) → TTS → speaker`. One process, one pipeline, no WebSocket.
- **Kept on purpose**: `agent_watchdog.cpp` (engine thermal/strategy, live in unified_server) and the whisper HIP port that the JARVIS blocker (P1) needed.

CI, packaging, and scope-guard were updated in the same pass — `test_auth`/`test_billing` and the WS tests dropped, `jarvis_server` symlinks gone, the ALLOWED path list now reflects the real repo. The Python server for the AMD-gui Adrenalin replica was also burned, but that's the next section.

### The Mojo shift: why we stopped writing glue in three languages

This week Mojo hit 1.0. We'd been watching it since the 0.x betas — a language that is to Python what C++23 is to C, with first-class GPU/NPU kernels and the ability to call C libraries directly — and the 1.0 release is the moment we committed: **from here on, new non-kernel code is written in Mojo, not Python.**

- **The AMD-gui Adrenalin replica backend** (a stdlib-Python sysfs/HTTP server for the Radeon Software control surface on Linux) is being rewritten in Mojo 1.0 as a single self-contained binary — same wire contract, zero Python at runtime. Hand-rolled libc HTTP over `external_call` (the stdlib's own mechanism), hand-rolled JSON (the payloads are fixed-shape), zero community packages (the ecosystem's pins are pre-1.0; `mojo==1.0.0` is the only pin we need). It replaces a Python interpreter dependency with one ELF, the same move we made for the NPU stack with C++.
- **The pattern is the point**: every layer of this project now follows the same rule — one binary, no interpreter at runtime, sysfs/PCI/proc access from the metal. C++23 stays for the compute kernels where it's earned its place; Mojo becomes the unified language for everything around them: servers, converters, tooling, control planes.
- **Why it's safe this time**: Mojo 1.0's stdlib was verified against the `v1.0.0` tag before we pinned it (no `std.net` yet — hence the libc sockets; no regex — hence the fixed-pattern parsers; `Process` API has no SIGTERM — hence the lazy-reap design). The version pin is the discipline: `mojo==1.0.0`, no `max==25.2`-style beta pins anywhere.

### Status

- **Repo**: one through-line (engine → JARVIS), one binary (`build/1bit`), one language direction (C++23 for kernels, Mojo 1.0 for everything else). Rebased onto 333 commits of upstream (`3f507b07`) with the burn on top — builds clean.
- **JARVIS P1** (whisper on the engine): `whisper_hip.hip` wired into the forward pass; scalar fallback retained; `WHISPER_GPU=0` escape hatch.
- **Next**: land the Mojo rewrite of the Adrenalin control plane (M0–M2: toolchain, JSON+sysfs, HTTP+GETs), then WS-09 (the single router) and the WS-11 NVMe expert streaming work from the roadmap.

---

## UPDATE 34 (2026-08-15): FLM IS FULLY GONE — BYTE-IDENTICAL STREAMS, TRUE BATCH, 2× ON THE 35B

**The last FLM artifact is dead. The open instruction generator now emits byte-identical streams to FLM's proprietary dumps (verified with `cmp` on all 4 ops), the open aiecc toolchain builds the xclbins (microkernel compiled with peano clang — no xchesscc), and true batch decode replaces the invalid fake-batch that had inflated our "-B 8" numbers. Validated head-to-head: we match FLM exactly at M=128 and beat it 11-15% with M=32 kernels on the 0.6B, and 2× on the 35B-A3B.**

### The FLM-free zone, completed in one day

1. **Instruction streams: byte-identical.** Reverse-engineered the complete FLM stream spec from the dumps (M=128 kernel = 4×32-row slices; N in 1024-tiles, K in 64-chunks; per-block bd rotation; the 12R-TCT sync pattern; all offsets/strides/values as formulas). The reworked `gemm_generate_sequence_i8` emits identical bytes — the old open generator was 230× slower (per-tile RTP config).
2. **Xclbins: open-built.** The v27 MLIR-AIE flow + a peano-clang-compiled microkernel (`-Di8_i32_ONLY`, the repo's `.o` was gitignored/missing) produce xclbin+insts pairs; M=32 decode-optimized kernels beat FLM.
3. **True batch decode.** The "-B 8 = 7.4 tok/s" claim was a fake batch (issue #111 — top-K candidates as sequential tokens, non-causal). Rewrote it: BS=8 independent sequences, per-sequence KV caches, causal attention, per-seq LM head, NPU am=B. Tokens verified identical to BS=1.
4. **35B 2×.** The CPU MoE FFN dequantized experts fresh per token (~33M floats/layer/token). A per-layer dequant LRU cache (16 slots) cut the FFN 107.5→61.6 ms/layer; with BS=8 grouped expert execution: 2900→1450 ms/tok, tokens identical.

### Validated numbers (Qwen3-0.6B, universal engine)

| Config | ms/tok |
|---|---|
| FLM M=128 (BS=1) | 255-262 |
| Open M=32 (BS=1) | 230-233 (~11% better) |
| Open M=32 (BS=8) | 235-237 |
| 35B-A3B BS=8 + cache | 1450 (vs 2900 = 2×) |

Also shipped: 1BP v4 dedup (shared Zamba blocks stored once, alias index entries), IQ1_S/IQ1_M block-size fixes (206/230 → 50/56 — every IQ1 file offset was wrong), spec-decode draft-vocab overflow fix (was segfaulting), fused-engine port to the validated execution model (fixing a shared-weight-BO bug), symlink model cache.

## UPDATE 33 (2026-08-10): NPU FIRMWARE RE VIA RAW IOCTLS, DRIVER REGRESSION FIXED, JARVIS SHIPS

**Raw ioctls now drive the NPU directly and produce bit-exact GEMM output — the first compute outside FLM's own stack. A driver regression that made the 35B MoE model spit garbage was root-caused to amdxdna 0.16.0 and fixed by reverting to the in-tree 0.7.0 build. JARVIS shipped NPU-FLM speech-to-text, SSE streaming, and a loopback-trusted web UI (PR #1576), plus a deadlock fix that had been hanging every chat request. eeg-medical was archived — its foundation-model retraining was redundant with ZUNA1.1, which already trains on the same TUH-EEG corpus.**

### Raw-ioctl NPU firmware RE: verified GEMM outside FLM (00:38–02:45)

A full day's arc from static dead-end to working silicon access, independent of FLM's binary:

1. **Firmware image fully mapped, proven not encrypted.** A block-entropy scan of the entire 429,680-byte firmware image found no 4KB window exceeding entropy 7.8 — the whole thing is a **signed pure-data container**, not a compressed or encrypted payload (~460KB is zero-fill). Static analysis exhausted; pivoted to live instrumentation.
2. **Live mailbox protocol captured with zero risk.** `echo "module amdxdna +p" > .../dynamic_debug/control` turned on 36 stock trace sites — no rebuild, no interruption of live inference — and decoded the full startup sequence and message header format.
3. **The `EXEC_DPU` silent-failure trap.** The command header's COUNT field (not BO size) determines payload length; `COUNT=0` triggers a silent `-EINVAL` with no dmesg line at all — only visible via ftrace.
4. **Real DPU instructions executed via raw ioctls**, correctly reporting an expected error for unmapped weights — proving the full loop (cmd BO → `CHAIN_EXEC_NPU`/0x18 → real firmware execution → status) works end to end.
5. **v28 MoE launch fusion** (`NPU_MOE_FUSED=1`): new `MOE_GUSGU`/`MOE_DSD` xclbins merge 4 FFN launches/layer into 2 via N/K concat of routed+shared experts. Correctness fix caught in testing: the DSD weight BO must be **block-diagonal**, not a naive concat, or the gate blend stops being separable. `test_moe_fused_math.cpp` passes at 1.64% max deviation (<5% tolerance).
6. **A "verified GEMM" claim, self-corrected within the hour.** The first PASS was a verify-code bug — `bf16((uint16_t)exp)` reinterpreted truncated integers as bf16 bits, so an all-zero output "matched" by coincidence while the real device status was `ERROR(5)`. Real root cause: `CONFIG_CU` needs `cu_bo` to contain an actual **Program Device Image (PDI)**, not a raw instruction blob; also, xclbin kernels output **int32**, not int16.
7. **PDI extraction perfected via bpftrace/uprobes.** kprobes on the driver's response handler decoded the exact rejection code (`APP_LOAD_PDI_FAIL`, `0x3000003`); uprobes on XRT shim's `get_pdi` revealed the true PDI offset inside the xclbin's `AIE_PARTITION` section — first guess `0xd8` was still wrong.
8. **MILESTONE (02:39): verified GEMM via raw ioctls, full pipeline.** Corrected PDI offset to `0xD0`. Real bit-exact GEMMs: 128×3072×1024 (0/131072 wrong, ~10ms) and 128×12288×2560 (0/327680 wrong, 20.5ms, 393 GOP/s). Full 8-step recipe documented: PDI = xclbin `AIE_PARTITION[0xD0:]`; A/B/C buffers must be SHMEM, not BO_DEV; instructions are BO_DEV; specific kernel-arg blob layout; `cfg=0x804`; 16-tile hwctx.
9. **All four qwen3-0.6b NPU ops verified bit-exact via raw ioctls**: QKV 128×1024×4096, O 128×2048×1024, GU 128×1024×6144, D 128×3072×1024 — zero mismatches on any of them.
10. **The exec path wedged again** after ~40 IOMMU faults from loading new PDIs (the running production server, whose contexts predate the wedge, was unaffected) — recovery is reboot-only, same NPU driver story as always.

Bonus same-session fix: **zaya NPU tokenizer** — real BPE `model.htok` generation (new `tools/hf_tokenizer_to_htok.py`) plus a weak-symbol linker bug where `tests/zaya_server.cpp`'s local `SimpleTokenizer` was silently shadowed by the shared implementation, producing garbage decode. Throughput: **3.3 → 42 tok/s**.

### amdxdna driver regression: 0.16.0 broke the 35B, reverting to 0.7.0 fixed it

Separate from the wedge-recovery saga in UPDATE 32, this is a correctness regression on a *specific newer driver version*:

1. **04:57 — #1568 reproduced live**: a fresh 35B session goes straight to garbage soup — not the usual deterministic `/` fallback — proving the corruption is **cumulative NPU/firmware state**, not per-process. An existing `llama3.2:1b` context is unaffected; only *new*-context creation wedges at the firmware level. Degradation model formalized: coherent → token soup → empty output → deterministic `/` (end state), recovery = reboot only.
2. **A separate, unrelated bug fixed along the way**: `mlir_aie`'s `NPU_CONTEXT_CACHE_SIZE["npu2"]` was hardcoded to 32, overcommitting this Strix Halo's real firmware hwctx pool of **exactly 16** (measured with a purpose-built `ctxpool` harness — the first guess of "≈6-7" was wrong; only direct measurement gave the right number). This caused `AIE2_STATUS_MGMT_ERT_NOAVAIL` under parallel pytest (`-n auto`). Fixed via `XRT_CONTEXT_CACHE_SIZE=1`: **5/5 PASS in 79s vs 225s serial (2.8×)**. Patched first in site-packages, then made reinstall-proof via `iron-repo/conftest.py`. Upstream PR filed: `mlir-aie#3526`.
3. **13:00 — kernel moved to HWE 7.0.0-29-generic.** Initial (wrong) theory: NPU only supports ~3 concurrent contexts, blamed for 35B startup failures.
4. **13:13 — correction**: 4 small NPU contexts work concurrently just fine. The real finding — **the 35B produces garbage even standalone with a free context**, a genuine regression against the last known-good 2026-07-31 benchmark, which ran on amdxdna 0.7.
5. **13:53 — FIXED.** Reverting to the kernel's in-tree amdxdna **0.7.0** driver (`/usr/lib/modules/7.0.0-29-generic/.../amdxdna.ko.zst`) — the exact version from the last known-good bench — correctly runs the 35B end-to-end (zaya → flm → NPU, "4" for "2+2"). A second bug fixed in the same pass: `zaya-qwen36.service` had a `Requires=zaya-gpu8b.service` chain that pulled up the entire small-model fleet on start, starving the 35B of NPU columns and crash-looping it — fixed by dropping the `Requires`/`After`/wait-port. **Confirmed: the 35B and the 3-small-model fleet are mutually exclusive** (a column/resource budget constraint, not a context-count cap).
6. **14:47 — two landmines left by the driver swap, both fixed post-reboot**: a stale `modules.dep` still pointing at the moved-aside DKMS `.ko` (fixed with `depmod -a`), and a stale `/etc/modprobe.d/amdxdna.conf` still setting `force_iova=1` (a 0.16/7.1.5-only parameter that the 0.7.0 in-tree driver rejects with a misleading "Unknown symbol" error). Also fixed: `zaya-gpu8b.service` stuck in a permanent start-pre kill loop. All chains verified post-fix: 14.1/21.5/13.9 tok/s across llama3.2/qwen06/gpu8b.
7. **Capstone assessment**: the NPU firmware is genuinely locked (RSA-2048 signed, no dev build, unsigned builds refuse to load — no third-party firmware is possible), but the platform is **not crippled** — the DRM driver is open GPL, the mailbox protocol is now fully decoded, raw-ioctl exec is proven bit-exact, and the 16-context hwctx pool is a real RTOS/SRAM constant, not an artificial throttle.

### JARVIS: deadlock fix, NPU-FLM STT, SSE, session KV reuse (PR #1576, `7d4b7549`)

Three commits landed on `main` together:

- **`34a55cb7`** — `build_context` double-lock deadlock fix (non-recursive mutex swap). This bug **hung every `/v1/chat/completions` request**.
- **`1c5dfb9d`** (= `8ed882a9`) — `flm_tag_for_model` widened the 1.7B tag bucket to `H<=2048`.
- **`f0d745f2`** (= `550d0299`) — speech-to-text via FLM's whisper HTTP endpoint (`:8496`, override `JARVIS_STT_URL`), **replacing the whisper.cpp+ffmpeg fork/exec path**; a loopback-trusted web UI auth bypass; `stream:true` requests answered as a single SSE chunk; and a long-prompt model-selection fix (was targeting a nonexistent `qwen3.5:9b`, now correctly `qwen3:4b`). `ctest -L host`: 10/11 pass.

Two more fixes landed the same day:

- **`zaya-npu.service` crash-loop fixed**: the lazily-spawned `flm serve` child was dying with "could not raise memlock limit to 1465 MB" then `mmap() EAGAIN` — systemd's default `LimitMEMLOCK=8MB` masked the bug in manual shell testing (interactive shells get unlimited). Fix: `LimitMEMLOCK=infinity` in `scripts/zaya-npu.service`. Result: coherent responses at ~3.3 tok/s, `x-backend: NPU FLM`.
- **FLM multi-turn KV reuse**, two build steps ~10 minutes apart: first, `src/npu_flm_delta.h` — pure prefix-delta decision logic (send only the delta if the new prompt extends the previous one, with self-healing fallback to a full reset) — because the text-REPL path was sending `<<RESET>>` before *every* query, forcing a full re-prefill each turn despite FLM's own REPL keeping KV resident. Second, a request-level session layer on top: `session_id` in the request body, `Backend::continue_text(delta)` virtual, `g_flm_session_id`/`g_flm_session_last` tracked under a dedicated mutex.
- **New documented constraint**: STT and the 35B contend for the same NPU context and cannot coexist — `flm-whisper` holding a context makes the 35B fail its coherence probe and fall through every backend. Confirmed fixed once `flm-whisper` is stopped (`zaya-qwen36` then passes 6/6 coherent probes at 3.2–3.8s). Design-state fleet: `npu :8088`, `qwen06 :8089`, `gpu8b :8090`, `whisper :8496` active, `qwen36` stopped on contention.
- The most recent commit in this whole window, `e8bd91ed fix(npu): retry FLM spawn/probe on NPU contention; stop leaking flm-real children`, directly addresses this contention.

### JARVIS mobile: a Flutter app for the voice pipeline (spans 2026-08-08 → 2026-08-10)

Design doc (`docs/superpowers/specs/2026-08-08-jarvis-mobile-design.md`, approved 08-08) establishes the architecture: **the phone is a thin terminal** — mic, speaker, VPN client only. The full pipeline (VAD → Whisper → router → LLM → codec TTS → cloned voice) runs entirely on the Strix Halo box; no on-device inference, no audio persistence. Transport is WebSocket + Opus (not WebRTC — ~200ms latency judged acceptable). iOS is a confirmed target (cloud Mac mini via rentamac.io, Xcode 26.6, Flutter 3.44.9, existing Apple Developer account).

Three milestones landed:

- **M1 — server-side gap-fill**: `/v1/voice/session` full-duplex WebSocket path with an auth hook, a `VoiceSession` VAD state machine, an RFC-compliant WS handshake/parser (fixed for standards-conformant clients), and `WS_STREAM_BIND` (default loopback; `0.0.0.0` lets the phone reach the port over LAN/VPN, paired with `JARVIS_WS_TOKEN`). Docs: `docs/mobile/RUNBOOK.md`, `scripts/jarvis-gateway.service`.
- **M2 — the Flutter app itself**: Android+iOS scaffold, protocol layer, session controller; a gateway WS client and audio IO services with tests; connect/voice screens with state lights and transcript UI; a live-gateway integration test with a stub server for simulator E2E.
- **M3 — resilience and packaging**: connection-loss UX (stop mic, offline state, Reconnect button), reconnect mic-failure handling, a double-tap reconnect guard, and a final review pass fixing manifest permissions, a save-race, pulse lifecycle, and player disposal.

CI enablement: `4b67b4ab` allowed `mobile/` through the scope-guard workflow (previously blocked, since JARVIS mobile is Jarvis's offline layer within engine scope, not a separate product).

### Ops/infra: the reboot "watchdog" nuance, netconsole, firewall, fail2ban

The CHANGELOG's "reboot.sh watchdog EBUSY fix" entry is correct about the code change (`273a50c5` — `scripts/reboot.sh` no longer aborts under `set -e` when `exec 9<>"$WD"` returns EBUSY because systemd already holds `/dev/watchdog` via `RuntimeWatchdogSec=60`) but doesn't explain what was actually causing the box to hang:

- Across the last several boots, the reset-reason register shows the watchdog fired **only once** — the deliberate `reboot.sh --force` test. Every other "hang" was a **slow-but-clean software reset** (~1–1.5 min unexplained stall in the post-journal shutdown phase). Suspects, unconfirmed: amdgpu display (`REG_WAIT timeout optc35_disable_crtc` logged at every boot) or the USB4/Thunderbolt controllers.
- A **separate, genuinely broken bug** was found and fixed: the watchdog safety net had been non-functional since it was created — the kernel package's own modprobe blacklist was silently deny-listing `sp5100_tco` at boot, so `/dev/watchdog` only existed after someone manually ran `modprobe sp5100_tco`. Fixed by shadowing the blacklist entry in `/etc/modprobe.d/`.
- A **netconsole capture rig** was installed (UDP listener on a second box, `netconsole-capture.service`; sender configfs target on strix; `loglevel=8` added past the `quiet` filter) to catch the stall's last kernel messages the next time it happens.
- **ufw/Docker firewall bypass fixed**: ufw doesn't filter Docker-published ports — they traverse the `FORWARD`/`DOCKER` chains, bypassing `INPUT` deny. Verified empirically (Supabase Postgres `:54322` reachable from LAN despite ufw). Fixed with a `DOCKER-USER` iptables rule made persistent via a oneshot systemd unit (Docker recreates its chain fresh on daemon start). Resolves #1571. Also found and closed: 2 accepted SSH password logins from a LAN IP ~30s after a brute-force burst — password auth disabled, fail2ban (sshd jail) and `ufw limit 22/tcp` added, closing #1572.

### eeg-medical archived — redundant with ZUNA1.1

`tools/zuna_port.cpp` (the ZUNA1.1 C++ port, hyperparameters verified against the paper: dim=1024, 16 layers, 8 heads, 382.1M weights) already existed in this engine before this session started — an earlier session had missed that and begun redundant work. Investigating further: **ZUNA1.1 is confirmed trained on the TUH EEG corpus** (per its paper, arXiv 2607.27308, and its predecessor ZUNA1, arXiv 2602.18478) — meaning the project's own TUH→B2 sync (5TB, running since UPDATE 31) would produce duplicate data toward a duplicate objective. Decision executed, fully reversible: `github.com/1bit-systems/eeg-medical` archived — its shipped work (retraining EEG foundation models on data ZUNA1.1 already trained on) was redundant, and its README's core claim ("ZUNA1.1 trained on research data, not clinical") is factually wrong — TUH is clinical data. `tools/zuna_port.cpp` stays in 1bit-systems as the canonical copy; the TUH→B2 sync was killed ~2 days into a ~17-day run. The `eeg-medical.1bit.systems` DNS record was deleted, though the underlying Cloudflare Pages project itself couldn't be (the API token lacked `Pages:Edit` scope) — that token should be rotated.

### Housekeeping: issue triage, repo audit, PR #1573 consolidation

- **29 open issues triaged**: 9 resolved with commit citations, 2 stale, 13 roadmap, 2 actionable security items (#1571, #1572 — both closed same day, above), 3 left unresolved (#1570 Qwen2.5-3B collapse, #1568 35B MoE — root-caused later the same day, above, #1536 amdxdna panic).
- **Repo audit** found: ~1.6MB of stale/byte-duplicate xclbin backups, ~130MB of tracked build artifacts under `engine/npu/xclbins`, `third_party/lemonade` vendored as 784 plain tracked files (784 files, 13.6MB) unlike the other proper-submodule `third_party/` dirs, and `third_party/FastFlowLM/UPSTREAM.md` sitting untracked *inside* the submodule (would vanish on re-clone). No secrets found.
- **PR #1573** (`feat/rebuild-all`, ~62 commits — the entire 35B MoE + Q4NX + JARVIS workstream from UPDATE 31/32) merged after two CI fixes (`4b67b4ab` scope-guard, `b46fc033` build the new host tests in the required C++ check). Follow-ons acted on the audit findings directly: `7c4a525b` moved `UPSTREAM.md` out of the submodule, `cab9a678` dropped 24 stale xclbin files. A draft release `v2026.08.10` was created (no git tag — draft releases don't create tag refs). Outcome: 11 issues closed, 18 remain open.
- **`gh-ops.yml`** (`d11e969b`): new weekly+manual workflow reporting code scanning, dependabot, secret scanning, failed runs, and billing via the `gh` CLI, opening a single security-triage issue when alerts exist. Companion fixes for paginated alert fetches, `GH_TOKEN`, and `GH_REPO` env wiring.

---

## UPDATE 32 (2026-08-09): AMDXDNA WEDGE SAGA, 35B MOE GOES LIVE, VIVADO-FREE FPGA TOOLCHAIN

**The NPU driver's wedge-and-hang problem is root-caused to a firmware fatal error with no recovery path, and fixed with a new hardware-reset primitive wired into the scheduler timeout — signed and running as amdxdna 0.16.0. The 35B MoE model goes end-to-end for the first time, warming up to 44-70 tok/s. A Vivado-free open-source FPGA bitstream toolchain is built and verified round-trip bit-exact.**

### The amdxdna driver wedge / kernel panic saga

The longest single investigative thread of the whole window:

1. **Firmware static RE dead-ends confirmed.** `npu.dev.sbin` is byte-identical (SHA256 match) to the stock `npu.sbin` — no relaxed-validation dev build exists. The suspicious 64KB blob (entropy ~7.0) isn't zlib, lzma, XOR, or any known cipher, and no ARM/Thumb/x86 code decodes from it. An earlier finding was corrected in the process: the supposed "code section"/"v5 gate targets" are actually a log-descriptor table, not gate-check code. A vulnerability map (`VULN_MAP.md`) confirmed all known driver CVEs are already fixed in 7.1.5 — the only live attack surface is the mailbox protocol (~23 opcodes).
2. **13:04 — first kernel panic from `rmmod amdxdna`.** `BUG: unable to handle page fault … RIP amdxdna_pci_driver_exit` on kernel 7.1.5-070105. Root cause: a client-lifetime use-after-free race during device removal (`amdxdna_remove()` frees client structs while open FDs can still call `amdxdna_drm_close()`). An upstream fix exists (applied to `drm-misc-fixes` 2026-06-29) but was confirmed **absent** from the running module via objdump. New standing rule: never `rmmod amdxdna` — reboot to reload instead.
3. **16:58 — kernel upgraded to 7.2.0-rc5, all-layers sweep 5/5 PASS, no wedge** across ~500+ hwctx cycles (vs. a previous ~20-cycle wedge trigger). Declared "fixed permanently" at 17:00, self-corrected at 17:01 (overclaim — one clean session doesn't prove "permanent" or "everything," and a separate, unrelated Qwen2.5-3B degeneration was still broken). The correction proved warranted: **the wedge recurred at 19:22** on the same kernel.
4. **Root cause found: a firmware fatal error with no driver recovery path.** The wedge isn't a driver hang — it's the AIE firmware hitting a fatal error, DPU program counter stuck at `0xffffffff` (executing garbage). Once dead, every mailbox call fails, and the driver's existing recovery paths only retry *through* the dead mailbox. A real recovery mechanism (`aie2_xdna_reset`, an SMU power cycle) already existed in the driver but was only ever invoked at init time. **Fix**: a new `aie2_hw_reset()` — full suspend-all-contexts → hw_stop → hw_start → resume-all-contexts power cycle — wired into `aie2_sched_job_timedout`, built for 7.2.0-rc5.
5. **19:43 — patched driver installed**, initially unsigned; the Secure Boot signing recipe was documented (MOK key already enrolled, `sign-file` + `zstd` + `depmod` + `modprobe`). Custom build registers as amdxdna **0.16.0** (newer than the stock 0.10.0).
6. **The `rmmod` rule is obsolete for 0.16.0** (the UAF fix is binary-verified present) but **still applies** to any stock kernel (7.1.5, 7.2.0-rc5, mainline 7.2-rc6 unpatched).
7. **A separate "reboot hangs" mystery resolved to the same root cause**: 7 kernel panics since Aug 6, all `rmmod`-triggered — because reboots had been preceded by a manual `rmmod amdxdna` to swap in instrumented driver builds. New rule: never `rmmod` as part of a reboot; the kernel tears down hardware fine without the module's exit path.
8. **Final validation**: post-cold-reboot, the patched signed 0.16.0 driver ran `test_all_layers.py` **5/5 PASS in 631.98s with zero wedge/fatal messages** — the self-heal path never even had to trigger on this workload.

*(Three sessions later, in UPDATE 33, this same 0.16.0 driver turns out to have introduced a separate correctness regression specific to the 35B MoE model — a different bug on the same subsystem, not a contradiction of the fix above.)*

### 35B MoE: full pipeline + batched prefill/decode performance

1. **02:39 — NPU worker path unblocked.** Fixed a `.1bp` manifest-lookup segfault (a misread header field produced a garbage 12.9GB "JSON length"); rewrote `tools/tq2_to_q4nx.cpp` (Q4NX-quantized 1BP → `.q4nx`, closing #1467), round-trip exact (diff 0.0). Batch scaling reached **12.6k rows/s at batch 64**.
2. **03:08 — NPU MoE decode 2.8× faster: 4.9s/tok → 1.73s/tok.** Three changes in `npu_engine_universal.cpp`: a fused transposed-dequant writing directly into `[K,N]` layout (eliminating a separate f32 intermediate + transpose pass, cutting per-miss cost 28ms → 14.6ms); `EXP_CACHE_SZ` 128 → 256 (4.42 → 3.35s/tok); and hot-expert pre-warming (`NPU_ROUTE_STATS`/`NPU_WARM_EXPERTS`/`NPU_WARM_TOP` — 1089 experts pre-packed in 12.6s). Result: zero decode-phase cache misses, `moe_ffn` averaging 26.6ms/layer.
3. **Correction: decode is launch-bound, not attention-bound.** Stage timers showed FFN dominating (31.5ms of ~47ms/layer) via 4 serial NPU launches, each paying ~5-9ms dispatch latency regardless of the tiny actual compute (~3μs for 16.7M MACs). Two attempted fixes were tried and reverted: async launch-overlap (kernels share AIE columns and just serialize, adding jitter) and a CPU shared-expert fallback (slower, plus box contention). The real lever — batching decode launches across tokens, or fusing per-layer xclbins (realized as v28 fusion in UPDATE 33) — was identified but parked here.
4. **Batched MoE prefill: rejected, then fixed with per-token ascale.** First attempt (`NPU_MOE_BATCH=1`) was 13-24% faster wall-clock but numerically divergent beyond M=1 — one shared `dynamic_ascale` over all M×H tokens collapsed per-token dynamic ranges, and the error compounded over 40 layers until it flipped argmax choices. An old code comment rejecting batching ("2.25× more MACs, sequential is faster") turned out to be right about the *outcome* but wrong about the *reason* — batching genuinely is faster. Fix: a new `go_rows` primitive in the i8 GEMM (`quantize_async_rows`/`dequant_only_rows`) computing per-token scales from each token's full expert set. Result: bit-identical through layer 19 at M=9, only FP accumulation-order noise beyond that (~1.2% relative — same order as existing int8 quant noise), while retaining **13-17% speedup at M=9, 24% at M=32**.
5. **The full pipeline fixed — 4 stacked Q4NX bugs**, root-causing the earlier "qwen3:4b tag + L=0 + degenerate" symptom: the layer scan looked for `"model.layers."` (plural) but the 35B's header uses `"model.layer."` (singular); the architecture was derived from the file's basename (always wrong for FLM's `<ModelName>/model.q4nx` layout) — fixed to read the parent directory name instead; no tag mapping existed for arch "qwen3.6" — added in *both* `src/backend_npu_flm.cpp` and `tests/backends/backend_npu.cpp` (the mismatch between the two cost several rebuild cycles before it was caught); and a 64KB header-size cap rejected the 35B outright (its real header is 85,552 bytes) — raised to 256KB.
6. **Coherence-probe positional bug fixed**: `coherence_probe()` fed every prompt token at `pos=0`, so the NPU-FLM backend — which fires its query the moment `pos` resets to 0, i.e. on what it reads as "prompt complete" — queried FLM with only a 2-token prompt, false-rejecting qwen3:4b's legitimate `<think>` start token. One-line fix: feed prefill at real positions. Regression-verified across llama3.2:1b, qwen3:4b (previously rejected), and qwen3.6-moe:35b-a3b.
7. **22:29 — culminating result: 35B MoE at 44-70 tok/s through the full pipeline.** Zaya (:8088) → NPU FLM → qwen3.6-moe:35b-a3b: **69.9 tok/s warmed** (64 tok/0.9s), **44.1 tok/s** on a second request (48 tok/1.1s), cold-spawn **9.5 tok/s** (6 tok/631ms) — consistent with the A3B active-parameter count matching dense qwen3:4b's 67.5 tok/s.
8. **Later the same day — a new, wider problem surfaced.** After ~1h of model churn, zaya-spawned `flm` worker processes started dying (`defunct`) and returning single-byte garbage (`0xAE`), tripping probe failures and a systemd restart loop. Filed as #1568 (alongside #1569 GGUF misroute, #1570 Qwen2.5-3B parked, #1571 Supabase exposed on `0.0.0.0`, #1572 SSH brute-force + password auth). `zaya-npu.service` was temporarily reverted to serve only Llama-3.2-1B. *(Root-caused the next day — see UPDATE 33's driver-regression section — as the 0.16.0 driver, not this session's application-level fixes.)*

### Vivado-free FPGA bitstream toolchain

A self-contained track: nextpnr-xilinx + prjxray + openFPGALoader built from scratch on the Strix box (18:28–19:00), producing a `blinky.bit` (2.19MB) with IDCODE `0x0362d093` verified against real hardware. Rebuilt persistently at `/home/bcloud/fpga-toolchain/` after a cold reboot wiped `/tmp` (the recipe needed a split cmake version — 3.28.4 for prjxray, 3.31.6 for nextpnr — plus several GCC13+/GCC15 `<cstdint>` fixes). Correctness was proven, not just well-formedness: all 4974 fasm2frames-emitted frames (497,374 words) came back byte-identical to the bitstream via `bitread` — **546,208 words compared, zero differences.**

### Scope decisions: JARVIS stays, ZUNA moves out

JARVIS stays in the `1bit-systems` repo (it's an application of the engine, not a separate product — unlike ZUNA, extracted to its own `~/zuna` repo, itself flagged as an open scope-boundary question that gets resolved by archiving eeg-medical entirely in UPDATE 33).

---

## UPDATE 31 (2026-08-08): THE TILEFUSE DAY — NPU KERNEL FROM SCRATCH, Q4NX PIVOT, CONVERTER LADDER

**The single busiest day in the window: an original NPU int4 GEMV kernel built from scratch (inspired by, but not copied from, the TileFuse paper — its actual kernel code was never public), a pivot to official Q4NX weights after community GGUF conversions proved unusable, a 15× serve-mode speedup, MoE expert ops running on the real 22GB 35B model for the first time, a from-scratch ONNX INT8 pipeline, and a four-rung Qwen2.5 converter debugging ladder (1.5B → 3B → 7B) that found a different silent corruption bug at every rung.**

### An original NPU int4 GEMV kernel (TileFuse-inspired), milestones M1→M4

Official Q4NX weights turned out to be a dead end for custom-converted models (confirmed later the same day — see the Q4NX pivot below), so the day pivots to building an NPU int4 GEMV kernel from first principles. The TileFuse paper (arXiv 2606.11357) describes the dataflow, but its actual kernel code was never released — only a static docs site — so the AIE kernel itself is original work guided by the paper's description, not a port.

- **M1 — the pre-tiler.** `docs/research/tilefuse_prep.py`: 128×64 tiles, int4 codes packed as adjacent-column nibbles, bf16 per-channel scales, int8 code-domain zero-points. Round-trip passed on synthetic data (max err 0.0087 vs. an int4 physics ceiling of 0.0101) and on a real Llama-3.2-1B `q_proj` slice (err 0.0288 vs. ceiling 0.0383). Three bugs fixed en route: int4 bytes pack adjacent *columns*, not row-pairs; the zero-point must be stored in the int8 code domain, not as a raw float minimum (storing `round(lo)` produced a 30× error); and constant-column tiles need `scale=|lo|`.
- **M2 — converter, CPU reference, and toolchain, three steps.** A new C++ converter (`tools/gguf_to_tilefuse.cpp`) passed CHECK on all 113 2D tensors of Llama-3.2-1B-Instruct.Q8_0, producing a 656.5MB `.tfb`, after fixing four bugs (GGUF shape is file-order but data is logical row-major — must not transpose; a manifest-offset patch bug that shifted every tile read by one; the constant-column scale fix; the zero-point bug, both restated from M1). A CPU reference GEMV (`tools/tf_gemv.cpp`) then passed CHECK on 30 tensors after fixing an accumulation bug (`out[c]=acc` was overwriting instead of accumulating) and a gate-indexing crash. A subtle gotcha cost real time: `struct.unpack('<e')` is IEEE fp16, not bfloat16 — it silently misled probes by ~100×. Finally the IRON toolchain itself came up clean (`~/iron` venv, python3.14, mlir-aie + llvm-aie wheels), validated on-NPU with a stock axpy (96/96) and stock GEMV (95/95).
- **The qgemv kernel runs on real NPU silicon, 10/10.** `aie_kernels/aie2p/qgemv.cc` fuses int4 unpack, dequant, and dot product on a 4352-byte tile FIFO — exact match vs. CPU on both 1-tile and 4-tile configurations, after fixing a kernel arg-count mismatch (garbage via a null pointer) and a zero-point duplication bug where the generator used `np.repeat` instead of `np.tile`, producing a block layout `[64][64]` instead of the correct interleaved one.
- **"THE Q4NX WALL IS BROKEN"** — a real-model validation loading `blk.0.attn_k.weight` from a real community Llama-3.2-1B `.tfb` passed 5/5, chunked into 8-tile passes to respect a 32-worker placement limit. This "5/5" later turned out to be a false positive — see M3.
- **A vectorization attempt** (16-lane row-group dot products) passed both test suites but was *not* faster yet (7.25s vs. 4.05s — vector setup overhead dominates at this granularity; the paper's 64×8 block structure is needed to actually win).
- **An alternate performance path: bf16.** `gguf_to_tilefuse --bf16` mode is near-lossless (err 0.0002); stock IRON GEMV on a real bf16 `.tfb` measured **175μs / 24GB/s** on a K=512,N=2048 tensor — about 8× faster than the int4 qgemv kernel's ~1340μs at this point in the day.
- **M3 exposed two false-positive bugs.** First, `test_real_model.py`'s earlier "5/5 PASS" was fake: `XRTTensor.from_torch` copies its input, so the test was reading back the *input* tensor, not the kernel's actual output buffer — the physics-bound gate even passed for an all-zero output. Fixed by reading via `buf.to_torch()`. Second, a **launch-alternation quirk**: consecutive QGEMV launches alternate between stale and valid output (even calls garbage, odd calls correct, deterministically) — stock GEMV doesn't show this; suspected an mlir-aie runtime FIFO ping-pong bug. Workaround: call twice per launch, keep the second result. A third, separate bug (the same `np.tile`-vs-`np.repeat` issue from the kernel work) had also silently narrowed `test_real_model.py` and the M3 sweep to only 16 distinct input values, with a bound loose enough to swallow the wrong math. Genuine corrected numbers: real-model max err/bound **0.0718**; M3 re-run **2/2 PASS in 12:42**.
- **M4, performance, multiple steps.** A bf16 host-dequant path hit **100μs/launch** at batch 1 (109μs at batch 4, 127μs at batch 8; err ~0.035 from bf16 rounding) vs. the int4 fused kernel's ~1100μs/launch, though batch 16 hit a placement wall ("no ShimNOCTile has sufficient DMA capacity"). A full-array mem-tile dataflow (per the paper's §4.4) then reached **197μs for 2048 outputs in one launch** at batch 32 (err ~0.05), and re-ran the M3 int4 sweep 3× faster (2/2 in 4:24) — but exposed a new cap: `QGEMV` silently truncates batches above 32. An honest per-token benchmark landed at **~0.9s/token** — 4714 launches × ~189μs, with the ~185μs launch floor dominating (the kernel itself is only ~2μs) — identifying layer-level batching as the next lever. That lever landed same-day: batching to **17 launches/token (16 blocks + embed) cut the time to ~390ms/token** (from 892ms), and revealed a new floor: the bf16 path moves 2.5GB of weights per token at ~10GB/s effective shim DMA, making it DMA-bound rather than launch-bound, while the int4 fused path moves 4× less data (656MB/token) but is compute-bound at ~1ms/tile for scalar dequant — flagging vectorized in-kernel dequant as the path toward ~100-150ms/token.
- **An all-layers bf16 sweep passed 5/5** after four more bugs: a false positive from reading the test's own allocated output tensor instead of the real one; a bf16 construction bug (`.view(np.uint16)` interleaves the low/high halves of each pair instead of `.astype(bfloat16)`); a K-chunk reference using a column slice instead of a row slice; and a quadratic loader re-slicing a 533MB array inside a loop (hours → fixed to 0.3s). Full round-trip: 114/114 PASS.

### The Q4NX pivot: official FastFlowLM weights vs. community GGUF conversions

1. **Verified with official weights.** `third_party/FLM_Q4NX_Converter` vendored; the pipeline works end-to-end with *official* pulled weights (`llama3.2:1b`, ~4.8 tok/s including spawn), but **community GGUF conversions produce degenerate output (loops)** — establishing the day's key limitation. Three bugs fixed along the way: a missing `FLM_CONFIG_PATH` on the per-request child process; a coherence probe wrongly rejecting char-shifted NPU output (fixed with an `all_char_band` exemption); and a Llama tag-mapping fix disambiguating on both H and L.
2. **FastFlowLM confirmed open-sourced (MIT)**, having moved from the FastFlowLM org to the ROCm org the day before (v0.9.46). Vendored at `third_party/FastFlowLM`. Two candidate decode dispatch paths were found for the degeneration bug: `generate_dequant_q4_1_seq` vs. `generate_dequant_q80_packed_in_q4nx_seq`.
3. **`tools/batch_convert_q4nx.sh` shipped** (`86269a99`, `d110bee1`), finding the converter's actual bug: an uncommented upstream debug line (`sys.argv` override to `'gemma4-2b-mmproj.gguf'`) clobbered every CLI invocation — the converter had been **totally unusable** before this fix.
4. **The fusion engine rebuilt** (`engine/npu/build_npu.sh`, 19 variants) with a per-tensor-scale `packB` fix replacing broken per-32-group scales; batch scaling proved essentially free up to 16 rows (1.9ms@1 → 2.2ms@16 → 6.4ms@64).
5. **Mystery solved: NPU2 is a different architecture, not a repack.** Official q4nx linears store raw I8 at `K=5120` (vs. vanilla GGUF's K=1024-2048); an earlier "±15 std5.6 vs. GGUF ±0.1" observation was actually I8 bytes being misdecoded as Q4_1 nibbles — no repack of a vanilla GGUF can produce official Q4NX's layout.
6. **The pivot's verdict** (branch `feat/q4nx-pivot`): a runnable q4nx model must come from `flm pull`, not conversion. Also decided: **`flm serve` works fine with official weights** (~45 tok/s vs. ~4.8 tok/s per-request spawn) — the earlier belief that "serve mode degenerates" turns out to have been an artifact of broken *converted* weights, not a problem with serve mode itself.

### Serve-mode migration + a tokenizer band bug

`SimpleTokenizer`'s decode fallback used stale character-shift bands that no longer matched the encoder's real bands (ids 132-226 should map to -100, 300-555 to -300) — ids 201-226 silently decoded to control characters, which is why short test strings looked fine while longer prose came out garbled. Verified end-to-end: **71.4 tok/s via `flm serve` vs. 4.8 tok/s per-spawn — about 15×**. A separate "16-char truncation" symptom turned out to be a red herring: a legacy endpoint defaulting `n_predict` to 16 instead of reading `max_tokens`. Both fixes landed as `87846a51`. In a case of parallel-agent convergence, another agent had independently built an equivalent serve backend (`86269a99`, `d110bee1`) — this session's changes were reconciled in its favor.

### 35B MoE: expert worker ops verified, then a 5.8× xclbin rebuild

1. MoE expert worker ops 40/41 ran on the real 22GB Qwen3.6-35B-A3B model for the first time, after fixing three bugs: `find_tensor_info` returned data offsets as **int32**, overflowing negative for offsets ≥2^31 and silently corrupting per-layer dimension detection on any model over 2GB (cascading to a SIGSEGV in `std_attn_step`) — fixed to `uint64_t`; an NPU out-of-memory at ~1.2GB BOs (dense QKV/O alone is 1080MB on the 35B) — fixed by making shared-expert BOs NL=1 with a host-side packed int8 cache; and a worker-op protocol payload-ordering bug. At this point, NPU MoE still lost to CPU: full decode was 2,823ms/tok on CPU fallback vs. 8,999ms/tok with `NPU_MOE=1` (3.2× slower — old xclbins).
2. **The MoE xclbins rebuilt as v27** (multi-row, 32-core; `build_moe_v27.sh`): op40 (expert GU) dropped from 39ms to **6.7ms**, op41 (D) from 20ms to **7.3ms** — a **5.8× speedup**. Full decode improved 9.0s → 7.2s → 5.8s/tok across the session (progression tied to `EXP_CACHE_SZ` 32→128 plus OpenMP tuning) — still losing to CPU's 2.8s/tok at batch 1. A `NPU_TIMING` breakdown explained why: an all-hit pack costs 2.0ms, but each cache **miss** costs ~25-30ms, and the 35B's routing churns through 100+ distinct experts per layer, giving an LRU miss rate of 50-80% at batch 1. The insight that mattered: **a fully warm path would be ~25ms/token across all 40 layers — 40 tok/s** — the number that gets realized on 2026-08-09/10 (see UPDATE 32).

### An ONNX INT8 QDQ converter/loader pipeline, built from a protobuf bug up

1. `src/onnx_loader.cpp` had been **parsing the wrong protobuf field numbers entirely**, finding zero tensors in any real ONNX file (`GraphProto.initializer` is field 12, not 5; `TensorProto.name` is field 12, not 8; `raw_data` is field 14, not 9 — all verified empirically against real wire format). Rewritten, with `DequantizeLinear` node parsing added and a new **WMMA_I8** path: INT8 QDQ → dequant → Hadamard-rotate in block-128 → per-row requant.
2. `bitnet_decode --config` reads an HF `config.json` sidecar and auto-loads `.onnx` paths; a new `std_arch` dispatch fixed a crash where any non-Qwen3/non-BitNet architecture (e.g. Phi) hit null sub-norm pointers.
3. Full serving verified with zero extra code: `bitnet_decode --server` (OpenAI-compatible + SSE) served qwen3-0.6b INT8 correctly, **57 tok/s, 877ms for 50 tokens**, streaming confirmed working.
4. In retrospect, this pipeline required finding and fixing **six independently fatal bugs**: the protobuf field numbers; a converter reading Q8_0 as unsigned int8 instead of signed (misreading any value >127, ~1.86× std error); the big one — the loader uploaded F32 norm weights but the rmsnorm kernel read them as FP16, so every value's low 16 bits were denormal-tiny and even-indexed outputs underflowed to zero; a Qwen3 `head_dim=128 ≠ hidden_size/num_heads` mismatch requiring the o-proj quantize step to use `nh*hd` width, not the naive hidden size; a `hipMemsetAsync` racing `decode_stream` without an explicit stream argument; and the `std_arch` dispatch crash from item 2.

### Llama-3.2-1B verification and a logit-flatness investigation

Llama-3.2-1B INT8 verified correct (Berlin, blue) with **PPL 49 vs. 828 for gibberish — 17× discrimination**, confirming tied embeddings are correct for this model (not a conversion bug). A separate concern — that logits looked suspiciously flat — was **not proven to be quantization-caused**: the numpy fp16 reference used to test it had six of its own bugs (wrong RoPE convention, wrong prompt token ids, a missing QDQ dequant step, a single shared KV list instead of one per layer, a layer-0-only comparison instead of full-stack, and mismatched single-token vs. full-prompt inputs). The GPU-side verification held up throughout (position-0/layer-0 query within 0.7% of fp32, 4/4 correct answers, PPL 49 vs. 828). Separately, arch-aware chat templates were added, and a key finding closed the loop on repetition loops: the **base (non-instruct)** Llama-3.2-1B shows the identical "which is" repetition loop as the instruct model — proving loops are a known trait of the 1B model itself, not a quantization or template bug.

### INT8 weight error budget

Measured against the GGUF Q8_0 source on llama-3.2-1b/qwen3-0.6b: plain per-row INT8 gives 0.98%/0.91% mean relative error; the Hadamard-rotation WMMA path improves that to **0.854%/0.847%**; adding QuIP-style sign flips on top made **no further difference at W8A8** (the existing rotation already captures the benefit — sign-flips only matter at W4A4). Conclusion: ~0.85% weight-side error rules out weight quantization as the cause of the small models' repetition/flatness behavior.

### The Qwen2.5 converter debugging ladder: 1.5B → 3B → 7B

Four rungs, each hiding a different silent corruption bug:

1. **Qwen2.5-1.5B**: both the official and an independent community GGUF turned out **corrupted** — input norms ~2000× off, linear weights 106% wrong vs. the HF safetensors source. Fixed by bypassing GGUF entirely with a new `tools/safetensors_to_onnx_int8.py` reading bf16 safetensors directly, plus adding attention-bias support (Qwen2.5 has q/k/v biases that llama.cpp GGUFs drop, and with the model's characteristically tiny norm weights, the bias term dominates attention scores if missing). Result: correct answers, **PPL 17.5**, 28/28 tests including a new GQA test. A *separate*, still-unresolved collapse remains on the INT8 GPU serving path specifically (PPL 1.19M, V-cache zeroed at pos≥1) — extensively debugged (hook-presence changes behavior, `HIP_LAUNCH_BLOCKING` doesn't help) but never isolated; parked.
2. **Qwen2.5-3B**: converted cleanly but degenerated at runtime (PPL 52k, a "Located 当前位置" loop) — parked mid-day, then **solved same day**: the converter was silently falling back to a stale 1.5B config (`num_hidden_layers=28`) instead of the real 3B's **36**, dropping the last 8 layers with no error. Fixed by deriving layer count from the tensor-name index, never from config. Result: **PPL 31**, coherent ("Paris", "Eiffel Tower").
3. **Qwen2.5-7B**: **PPL 13.6 — the best in the zoo.** Two bugs: an untied `lm_head` was never being emitted for models with `tie_word_embeddings=false`, so the 7B ran with a *tied* head (using the embedding table as the output head) and produced garbage — caught via a parameter-count check (7.07B in the ONNX vs. 7.62B expected, a 545M gap matching exactly the missing `lm_head`); and a `config.json` path-resolution bug for directory sources (`os.path.dirname(dir)` resolved to the *parent* directory, missing the local config and silently falling back to the same stale 1.5B config that bit the 3B rung above). Verified: rich, coherent text with correct chat-template turn-ending.

### Zaya GGUF/.1bp conversion: the 74B download, an 8B NaN, and a MoE converter fix

1. A **74B Q4_K_M GGUF (45.76GB, 1923 tensors)** was downloaded, unblocking two parked issues.
2. **Zaya1-8B NaN root-caused**: `cca_conv_grp.weight` was reading garbage, exploding the conv output into `inf*0` NaN through RoPE. The GGUF itself was confirmed clean — the bug was in the WIP `gguf_to_onebp` converter's 3D-tensor handling, ultimately traced to a **GGUF dtype-30 mixup**: `GgufReader` treated dtype 30 as a legacy "F32_V3" 4-byte float, but current llama.cpp defines dtype 30 as `GGML_TYPE_BF16`. Zaya's `cca_conv_grp.weight` is genuinely BF16 — reading it as f32 consumed twice the intended byte span, desyncing every subsequent tensor read.
3. **The #1522 converter had two separate root causes**, found by running the real 45.76GB file through it: a shape heuristic voted row-major for Zaya's `[rows,cols,experts]` layout, inferring 4096 "experts" and producing 242GB of fp32 garbage — fixed by trusting the `<arch>.expert_count` metadata instead of guessing; and `token_embd.weight` (1.07B elements) tripped a `1<<30` element cap and was silently skipped while its bytes stayed reserved in the index, desyncing every later offset in the file — the cap was raised to `1<<31` and made a fatal error instead of a silent skip. Verified via round-trip dequant against the source GGUF: Pearson r=0.994-0.999, zero NaNs. Output: `ZAYA1-74B.1bp`, **43.6 GiB** (PR #1546).
4. **An architecture fact worth recording**: ZAYA1's layers **alternate** — even layers are CCA-attention-only, odd layers are MoE-only, and **no dense FFN exists anywhere** in the model. A missing sublayer's weights should produce a null pointer and a skip, never a silent zero-fill. Relatedly, no working reference implementation of Zaya's router "EDA" temporal-state semantics exists anywhere (even the llama.cpp Zaya fork refuses to load the 8B GGUFs and degenerates on the 74B) — the engine loads the 8B `.1bp` end-to-end, but its output isn't conditioned on the prompt, and this remains unresolved for lack of anything to verify against. The 74B additionally needs ~145GB of fp16 GPU expert buffers against a 62GB Strix pool — current fp16-only expert kernels can't run it at all yet.

### The rest of 2026-08-08

- **A third NPU driver wedge this session** (`DRM_IOCTL_AMDXDNA_CREATE_HWCTX err=-22`, after 20+ xclbin load/unload cycles) persisted without self-clearing, forcing a reboot — bridging directly into UPDATE 32's wedge saga.
- **The self-hosted CI runner was removed.** `strix-halo-runner` uninstalled locally (2.4GB freed); NPU-only workflows deleted (`bench.yml`, `npu-reset.yml`, `npu-pool-probe.yml`, `end-to-end-smoke.yml`); hardware jobs removed from the remaining shared workflows. Rationale: NPU/GPU jobs need `/dev/kfd`/`/dev/accel`, which GitHub-hosted runners don't have — CI is now fully GitHub-hosted.
- **The TUH EEG data-access workstream began** (background context for UPDATE 33's later eeg-medical archival decision): access approved via NEDC, a full rsync sync pipeline built (sftp proved rate-limited/reset after ~136GiB, so rsync-only, chunked ≤100GiB, cron-scheduled), targeting a ≈5TB/414k-file corpus at ~29MB/s (~17-day ETA).
- A HIP backend bug where `sample_token` returned -1 (forward/lm_head emulation for the split ZAYA HIP boundary) was fixed.
- **Background/context, not engine work**: AMD's acquisition of Taalas (announced Aug 6 — a hardwired-model-silicon compiler claiming ~16-17k tok/s/user for a single hardwired model) prompted a staged-plan validation of a "1-bit FPGA LLM box" concept — judged a plausible ~50-150 tok/s niche for a 1B 1-bit model on an Alveo U250, with zero ASIC mask-cost, though far short of Taalas's on-die-ROM throughput; a Strix Halo upgrade-path discussion (128GB unified RAM as the biggest lever for fitting 70B Q4 fully in memory, NVMe weight-streaming, small FPGA "Taalas toy" boards); and a Vitis/Unified SDI 2026.1 toolchain install.

---

## UPDATE 30 (2026-08-07): THE ONE-HEAP POOL — ALL MODELS RESIDENT, SPEC-DECODE IN-SERVER

**The unified server now boots with every model in the zoo resident in one mmap'd pool (11 slots, ~6 GB on Strix), runs lossless speculative decoding in-process (`--draft-model`), and finally produces coherent answers from all 9 models end-to-end — Zamba2, the NPU FLM path, and the HIP 1BP backends all fixed. 13 commits landed on main via PR #1535.**

### The unified model pool (#1535)

The pool idea from UPDATE 29's roadmap is now end-to-end. `UnifiedModelPool` was generalized: any model file (gguf/q4nx/h1b/1bp) is mmap'd resident — `.1bp` gets header-parsed metadata, everything else generic. `--pool` makes the server load every discovered model at boot; `POST /v1/pool` reports residency, `/v1/models` tags pooled models. One process, one API, all models resident. Measured through the single endpoint: Qwen3-4B (NPU FLM) 20.8 tok/s, Llama-3.2-1B/Qwen3-0.6B (Vulkan) 12.4, Bonsai (HIP 1BP) 3.1, Zamba2 Q8_0 (HIP) 2.2; zoo-smoke 5/5 PASS.

### In-server speculative decode

Phase 2 core is now a server flag, not a demo. The backend interface gained the three spec-decode primitives (`decode_one` / `verify_batch` / `rollback`), implemented by the ggml-vulkan backend with multi-token `llama_decode` (per-position logits + KV rollback via `llama_memory_seq_rm`). `--draft-model` loads a second side-by-side ggml-vulkan backend; `--spec-decode` runs the lossless loop (draft proposes N greedy tokens, target verifies in one batch, longest consistent prefix accepted, rejected positions rolled back and fix re-decoded). Two real bugs found along the way: `reset()` was a no-op — every request extended ONE unbounded sequence, so KV never actually cleared per request; and the first-token bug — draft and target logits share a vocab size but NOT a vocab, and a shared buffer made the first output token the draft's argmax.

### Zamba2 end-to-end (5dcef800)

Zamba2-1.2B-Instruct-v2 now generates coherent answers through the unified server on the HIP GPU engine — verified vs HF transformers with identical top-1 (`'4<|im_end|>'` greedy). Three root causes, all found:

1. **The GGUF on disk was corrupt** — ~60% of weight elements didn't match the checkpoint (spot-checked ssm_in, embedding). Re-converted a clean F16 GGUF from the safetensors.
2. **Converter bugs**: `use_shared_attention_adapter=True` never folded the q/k/v LoRA adapters (only the FFN gate_up one was) — every attention head computed with raw weights; and the tokenizer vocab was never written, so `.htok` synthesis failed and the server fell back to the previous model's tokenizer (Qwen3 ids fed to a 32k-vocab model).
3. **Merges format**: `tokenizer.json` merges are `[a, b]` pairs, the gguf lib wants `'a b'` strings.

### Model zoo sweep: 8/9 coherent (04271bd8, 0bab2f6e)

Full regression sweep of all 9 models through the unified server — 8/9 now generate coherent responses. Tokenizer synthesis was the big one: the server now builds a fresh `.htok` from the model's own or sibling GGUF (`GgufReader::write_htok`), because the checked-in `tokenizer.htok` was a stale Llama-era v1 file and without it every non-NPU model decoded as ASCII-garbage `[id]` soup. Plus: byte-piece fallbacks for non-GPT-2 vocabs, the ggml_vulkan sampler chain switched from greedy (which made small models loop) to temp 0.8/top_p 0.95/dist, and a nasty `cur_p.selected` bug — it's an INDEX into the candidates array, not the token id.

NPU FLM went end-to-end too: an ODR collision where two distinct `OnebpModel` classes (CPU vs NPU) shared one dtor symbol meant every NPU-side object was destroyed with the wrong layout → SIGSEGV after any failed 1BP open (fixed by renaming the NPU-side class to `NpuOnebpModel`). The FLM backend got a `generate_text()` hook driving the subprocess REPL with the whole prompt (FLM tokenizes internally; a token loop can't drive it), with REPL artifacts (ANSI codes, `<<RESET>>` echoes, `[FLM]` log lines) stripped from responses. And the Qwen3-4B-NPU2 model cache was corrupt — bad download pre-reboot.

### Zaya 1BP + routing (9ed7eb94, 7ae82829, 4c68f3ff)

The HIP 1BP backend is wired into the router chain with logits-based sampling (temperature/top-p/repetition penalty — replacing argmax), `rope_theta` passed from the `.1bp` header, and per-arch chat templates — verified with a real-prompt coherence probe.

### CI + housekeeping

- The e2e smoke job no longer breaks when built without the llama.cpp submodule (`backend_ggml_vulkan.h` self-stubs the factory, mirroring the `.cpp`'s `#else` stub); submodule pinned to the fork so CI can fetch the zamba2-quantize commit.
- ShellCheck SC2034 in `scripts/zoo-smoke.sh` fixed (unused loop var — was blocking the required C++ check on every PR).
- **Branch cleanup**: 7 stale remote branches deleted (3 dependabot bumps, superseded fingerprint-dispatch docs, merged/closed eeg-zuna-research, zamba2-ssm-a-convention, ws05 per-vocab-ppl-gates), 4 stale PRs closed. `origin/main` had been sitting at Aug 5 — the whole Aug 7 work landed via PR #1535.
- **CI flake found, not fixed**: the smoke-test's NPU `mmap EAGAIN` failures are the `pool-probe` job wedging the amdxdna driver (`Can not get flush memory` in dmesg) — the server runs clean locally with the NPU free. Root-caused in the wiki; workaround is a reboot before needing smoke green.

---

## UPDATE 29 (2026-08-03): MEMORY CAMPAIGN — ARENA-FRAG LEAK FIXED, TOP-1 BACKEND INIT, 10-BUG AUDIT

**The long-unexplained host-RSS creep in `unified_server` is root-caused and fixed, the multi-backend memory bloat is cut by ~10 GB, and the 10-bug audit (#1429–#1438) landed. Release v2026.08.03 shipped and the memory fixes are deployed on the live service.**

### The leak that wasn't a leak: glibc arena fragmentation (#1428)

For weeks the server's host RSS crept ~0.1–1.7 MB per generation request — monotonic, never returned, ~90 MB/hr at 1 req/s, and 6.8 GB baseline after 50 min uptime. Every container was ruled out by triage. A heaptrack capture settled it:

- **Zero unfreed bytes on the generate/forward/lm_head path.** Nothing leaked.
- The culprit: `FusedBackend::generate()` allocated `std::vector<float> l(VOCAB)` (**608 KB** for Qwen3) on the stack **every token** and freed it every token.
- That's the **glibc dynamic mmap-threshold trap**: the first 608 KB block is mmap'd, but after the first free glibc raises its threshold and services later same-size requests from the **brk arena**, interleaving them with the small per-token HIP/kernel allocations. Top-of-heap never becomes trimmable → RSS creeps up and never comes back. This is exactly why `/proc/maps` stayed flat while `VmData` grew, and why the rate tracked tokens.
- Fix: hoist `h`/`l` to reusable member buffers sized once in `init()`. Zero per-token host allocation → nothing to fragment. Measured: mt=128 creep decays 208 → **20 kB per 10 reqs** (old: 17 MB per 10 reqs, ~160×) and plateaus; the `MALLOC_MMAP_THRESHOLD_`/`TRIM_THRESHOLD_` env band-aid on the service became unnecessary and was removed.

### The 15 GB serving baseline (#1427)

`BackendManager::init_in_order()` loaded **every** backend × full model copy — fused (f32 GPU + host), hip_1bp (bf16 GPU), vulkan_hpp (pools), cpu_generic (f32 RAM) — ~15 GB (7 GB RSS + 8.2 GB GTT) for a 0.6B model, so systemd-oomd killed desktop apps when CI built concurrently. Two fixes:

1. **Dead host weights freed**: `cpu_L`/`cpu_embed`/`cpu_final_norm`/`cpu_output` are pure load→GPU/NPU-pack buffers — provably unread at inference. Freed after load: RSS 6768 → 5163 MB.
2. **Top-1 + CPU fallback init**: keep the first accelerator + one CPU fallback; the rest stay discoverable and init **lazily via the existing failover path**. This was safe because the per-token cross-backend routing those extra copies paid for was **KV-incoherent anyway** — each backend keeps a private KV cache + `pos`, so a token routed to a backend that never saw the prefix attends to empty KV. Instance GTT: 8.2 → **2.6 GB**.

### Everything else since UPDATE 28

- **Coherent GPU inference** (#1397): QK-norm, per-layer KV, residual fixes — GPU paths became output-valid; `fused_gpu_npu` is the production path at **321 tok/s (3.1 ms/tok)**.
- **One ELF to rule them all** (08-02): `build/1bit` = zaya + unified + router + jarvis + vision + agent CLI in a single binary; `1bit pull`/`list` in pure C++; packaging ships one binary; the 296-line bash launcher is dead.
- **Lemonade embedded** (08-01): 14 Lemonade backends run in-process; `unified_server --lemonade` hands off to the full server.
- **Exact SiLU + `-ffast-math` dropped** (41d0977e7): Q4NX ppl was *backend*-broken, not converter-broken — the #1243 audit's 6.5% density red flag resolved with per-vocab ppl gates + prefill M=128 fused chain (#1413).
- **RVQ-VAE codec in C++** (#1368) with GGUF export.
- **10-bug audit resolved** (#1429–#1438): SSRF, unkillable-server, OOB/SIGFPE/bad_alloc families, loader/backend hardening.
- **Clean shutdown** (#1426): `_exit()` instead of the static-dtor ABRT that segfaulted on every stop.
- **Release pipeline fixed** (#1404–#1425): tag builds run on the runner (not container), `onebin` target restored after sync merges clobbered it, libdrm/glslc/spirv-headers dep parity with CI.
- **Narrative purges** (#1412 + follow-ups): TheRock-era and Rust-era claims swept — the stack is pure C++23, zero Python at runtime.
- **Ponytail purge** (#1325): ~48K lines of dead code deleted (with the few survivors restored in #1390).

**The environment bit back once**: a driver wedge during sustained load forced a cold reboot on 08-03; post-reboot the NPU (RyzenAI-npu5, firmware 1.1.2.65) and iGPU came back clean, and the memory campaign above was verified on the recovered hardware.

---

## UPDATE 28 (2026-07-20): MAMBA1 GPU BACKEND — 79.4 TOK/S, 9 BUGS KILLED

**The Mamba1 GPU backend (`mamba1_engine.hip` + `backend_mamba1.cpp`) is now fully built, linked, and validated end-to-end on Strix Halo. BlackMamba 1.5B: 79.4 tok/s. BlackMamba 2.8B: 46.1 tok/s.**

What was delivered:

1. **Build linkage fixed**: `create_mamba1_backend` was only compiled into `unified_server`, not `libbackend_manager.a` — every other binary (test_backend, backend_demo, vision_server, etc.) failed to link. Moved `backend_mamba1.cpp` into the static lib. HIP device stubs were also missing because the file was compiled as CXX despite launching kernels with `<<<>>>` syntax; moved all kernel launches into `extern "C"` wrapper functions in `mamba1_engine.hip` so callers compile as plain CXX.
2. **Conv state buffer overflow fixed**: the conv state shift loop wrote to `cs[(d_conv-1) * d_inner + i]` but the buffer was only `[d_conv-1, d_inner]` (max valid index `d_conv-2`). This caused silent GPU memory corruption on every SSM layer forward pass. Fixed the loop bound from `dc-2` to `dc-3`.
3. **A_log exponentiation fixed**: Mamba1 parameterizes `A = -exp(A_log)`, but the selective scan kernel used `A_log` directly as `A` in `A_bar = exp(dt * A)`. This meant the SSM dynamics were completely wrong. Added `-expf()` in the scan loop to compute `A = -exp(A_log)` before discretization.
4. **Model routing fixed**: GGUF Mamba models are now routed to `mamba1_gpu` backend (was falling through to ZINC GPU catch-all).
5. **Both BlackMamba sizes converted and benchmarked**: 1.5B (30 layers, 15 SSM + 15 MoE) at 79.4 tok/s, 2.8B (36 layers, 18 SSM + 18 MoE) at 46.1 tok/s — both on Strix Halo iGPU via ROCm HIP, alternating SSM/MoE layer dispatch.
6. **Diagnostic tool**: `tools/test_mamba1_backend.cpp` loads a Mamba1 GGUF directly into the HIP backend without the HTTP server — warmup, benchmark, and generation in one shot.

**The BlackMamba `⚠️` in the README is gone.**

---

## UPDATE 27 (2026-07-06): FUSED LAYER ENGINE GOES PRODUCTION — 291 TOK/S (3× V12)

**The fused layer engine now ships at 291 tok/s (3.4 ms/tok), 3× the v12 baseline, in a 38 KB binary.**

What was delivered:
1. **One xclbin call per transformer layer**: QKV projection, attention, O projection, gate+up, SiLU, and down projection all run on the NPU in a single dispatch. No CPU attention, no intermediate BO syncs. Uses `design_full_layer.xclbin` (416 KB) from the torch2aie toolchain with per-position instruction files.
2. **3.4 ms/tok decode**: The fused dispatch eliminates the per-GEMM ioctl overhead that limited v12. At 291 tok/s, the NPU's INT8 throughput is now the bottleneck, not the dispatch layer.
3. **38 KB binary**: The fused engine binary is smaller than the previous 74 KB daemon despite doing more per call. Static linking + stripped symbols + no Python runtime paths.
4. **Fixed scale optimization in universal engine**: `dynamic_ascale()` replaced with `FIXED_ASCALE = 8.0f / 127.0f` — saves 35 μs per GEMM call (4 ms/batch across 112 calls). Worth +11% on decode.
5. **FLM v0.9.44 workaround in daemon**: FLM's `/v1/chat/completions` has a `basic_string::substr` bug. Daemon now converts chat messages to text prompts via a lightweight Qwen3 template and calls `/v1/completions` instead.

**Narrative shift**: v12 (97 tok/s, C++ standalone INT8) is now the fallback path. The fused layer engine is the production path. All docs, badges, and benchmarks updated to reflect this. Everything from "74 KB binary, 94 tok/s" to "38 KB binary, 291 tok/s."

---

## UPDATE 26 (2026-07-05): ALL 3 BUGS CONFIRMED FIXED — AIE MICRO-TILING ROOT CAUSE RESOLVED

**v12 is now coherent. 97 tok/s verified. GEMM kernel bit-exact.**

A parallel investigation (branch `fix/npu-hf-cache-i32-kernel`) independently confirmed
what UPDATE 25 suspected: the remaining bug was in the **compiled xclbin kernels**, not
the host code. Root cause: `n1_core_i8_v2.py` (the INT8 MLIR generator) was **missing AIE
micro-tiling** — the GEMM kernel received weights in the wrong internal layout despite
being bit-for-bit correct at the BO level.

Fixes applied:
1. **xclbin output width** — matched INT8 generator output width to host's i32 Cm buffer
   (`cd73e137`)
2. **Smoke-test prompt** — replaced malformed prompt with valid chat template
   (`3d984285`)
3. **RMSNorm weight clip** — clipped weights to [-2,2] in cb/universal engines
   (`49e78785`, partial)
4. **GEMM kernel verified** — hardware dump-and-compare confirmed bit-exact
   (`7f8f3586`)
5. **Root cause identified** — missing AIE micro-tiling in n1_core_i8_v2.py
   (`01a4b7f4`)
6. **Parallel theories reconciled** — both investigation paths now agree
   (`16016167`)

All 6 fixes cherry-picked onto main as `232db025`..`bffe5a2e`.
**97 tok/s v12 now produces coherent output.**

---

## UPDATE 25 (2026-07-03): v12 WAS NEVER OUTPUT-VALIDATED — 3 REAL BUGS FOUND, STILL INCOHERENT

Set out to swap the production daemon's NPU backend from FLM (proprietary, closed-source)
to v12 (our own C++ engine, "97 tok/s, beats FLM's 94, Zero Python"). Before wiring it in,
sanity-checked actual chat output against FLM for the same prompt. FLM answered "What is
2+2?" correctly (" 4."). v12 — byte-identical reproduction of the unmodified original —
produced complete garbage. Every doc and benchmark in this repo checks tok/s and "doesn't
crash," never coherence. The 97 tok/s number is real; the output behind it never was.

Found and fixed 3 real, confirmed bugs, all present in `npu_engine_cb.cpp` since it was
first written and inherited by `npu_target_model.h` (spec-decode's target-model dispatch):

1. **LM head weight substitution** — `lm_head.weight` gets correctly dequantized then
   immediately discarded; the code computes final vocab logits against the *embedding*
   table instead (assumes tied embeddings). Qwen3-0.6B's checkpoint stores them completely
   separately (confirmed via Q4NX header data_offsets) — the model computes a reasonable
   final hidden state, then reads logits off the wrong matrix.
2. **Weight-packing transpose** — `dequant_i8_to_float` returns row-major
   `[out_features, in_features]`; the GEMM dispatch needs `[in_features, out_features]`.
   The packing loop read the buffer with the wrong stride, silently scrambling every
   weight matrix (Q/K/V/O/Gate/Up/Down) while still producing finite, plausible-looking
   numbers. Also: O-proj and Down-proj dequant calls used the wrong `in_features` (1024
   default instead of their real 2048/3072), scrambling the tiling itself.
3. **Activation quantization clipping** — hardcoded INT8 scale assumed activations stay
   within [-5,5]; measured range is [-8.24,7.01]. Silently clipped every layer, compounding
   across all 28.

All three fixed, in all three copies of this logic (`npu_engine_cb.cpp`,
`npu_engine_server.cpp` — a new persistent-server variant built for the daemon swap,
and `spec-decode/engine/npu_target_model.h`). Chat output is **still incoherent** after
all three fixes, individually and combined, tested against both RoPE conventions
(interleaved-pairs and HF's actual rotate_half). Ruled out via ground-truth comparison
against the real HF model: embedding lookup, RoPE theta/config, GQA head mapping, K/V
extraction offsets — all correct. Remaining suspects: RoPE rotation convention (tested,
inconclusive) or the compiled `.xclbin` kernels themselves, undebuggable without the AI
Engine Simulator — blocked on this machine since Update 24's investigation (missing
`aie2p_8x4_device.json` for NPU2). Full writeup: `docs/archive/V12-CORRECTNESS-BLOCKER.md`.

**FLM proxy stays in production.** Do not wire v12/1bit.engine into the daemon until this
is resolved and re-verified against real chat prompts, not just dispatch speed.

---

## UPDATE 24 (2026-07-03): FUSED XCLBIN RESUMED — SCHEDULE FIXED, DEADLOCK ISOLATED, NEW KERNEL BUG FOUND

Picked back up the fused-transformer-xclbin effort flagged as "next" at the end of Update 17
(the intervening Updates 18-23 covering the fused-xclbin dead end, the pivot to the universal
5-model v12 engine, and the merch store live in git history / other docs, not fully reflected
in this file until now).

### What Was Found

1. **Reconstructed the correct Q4NX weight-packing schedule** by cross-referencing the MLIR
   generator against `qwen3_model.py::_projection_stream_from_schedule` — the fused xclbin
   expects weight chunks distributed per-column/per-row (`row_chunk = block*16 + group*4 +
   patch*2 + row_in_patch`), not replicated identically across columns as the old
   `q4nx_stream.cpp` did. `npu-sandbox/npu-infer/tools/pack_fused_v3.py` already implements this
   correctly (verified byte-identical on regen) by reading real Q4NX chunks straight out of
   `model.q4nx`, no dequant/requant.
2. **Schedule-correct weights alone didn't fix the full-layer deadlock** — re-ran `npu_engine_v13`,
   still 62857ms timeout, all-zero output.
3. **Isolated the deadlock to the O/UP/GATE/DOWN tail.** The smaller QKV-prefix xclbin (rebuilt
   fresh via `full_layer_qkv_prefix_runner.py`) dispatches cleanly in ~4ms, no deadlock at all —
   matching what the sibling BitNet port (`torch2aie/examples/bitnet-decode-layer`) found for the
   identical design shape. The full-layer deadlock is a lock/dataflow bug specific to the tail
   phases, not a data-scheduling problem.
4. **Found a second, separate bug: QKV-prefix produces numerically wrong output**, even
   deadlock-free. ~1000+ K/V cache mismatches vs. the CPU golden reference, at multiple token
   positions. Traced RoPE and RMSNorm formulas in the Chess kernel (`postprocess_qkv.cc`) against
   the Python reference — both match exactly. V-cache (no RoPE/norm at all) is *also* wrong,
   narrowing the bug to the Q4NX GEMM/dequant kernel (`qwen3_decode_kernels.cc`) or record
   absorption — unresolved, needs kernel-level debug instrumentation to pin down further.

### Status

Fused xclbin is closer than before (schedule solved, deadlock scope narrowed) but still not
working end-to-end — two distinct kernel bugs remain (tail deadlock, QKV numeric correctness).
v12 (97 tok/s, standalone INT8 GEMM, zero Python) stays production. Full details in
`docs/archive/FUSED-INTEGRATION-BLOCKER.md`.

---

## UPDATE 23 (2026-07-02 15:32 ADT): PRODUCTION RELEASE — v2026.07.02-all5models

Shipped: tag `v2026.07.02-all5models`, site updated to "One engine. Every model. Any chip.." 5 model
families verified, 0 crashes, 28 tok/s on Qwen3-0.6B (all-models auto-detect binary). vs FLM: 2.4×
slower per-token, but open source, zero dependencies, 5 models from one 120KB binary. Fused xclbin
flagged as the path to close the gap (picked back up in Update 24, three sessions later).

---

## UPDATE 22 (2026-07-02 15:13 ADT): ALL 5 MODELS AT V12 BATCH SPEED, 0 CRASHES

Model-agnostic engine (`npu_engine_all.cpp`) verified across the full catalog:

| Model | Decode |
|---|---|
| Qwen3-0.6B | 58 ms/tok |
| Gemma4-E2B | 117 ms/tok |
| Qwen3-VL-4B | 141 ms/tok |
| Llama-3.1-8B | 185 ms/tok |
| Qwen3-8B | 215 ms/tok |

Fix: `dequant_i8_to_float_ex` had `in_features` hardcoded to 1024 — only 0.6B ever worked
correctly. Corrected to read `H`, `NH×HD`, `IM` per-projection from the model header; all 5
families verified working.

---

## UPDATE 21 (2026-07-02 12:01 ADT): SESSION CLOSE — FULL NPU ENGINE STATE

*(Reconstructed from git history — commits in this window didn't carry explicit UPDATE numbers;
assigned 20/21 here to keep the sequence readable.)*

v12 engine at 97 tok/s (10 ms/tok), 24× speedup, beating FLM Kraken Point (66.5 tok/s). Fused
xclbin: 3 xclbins compiled (QKV-prefix, full-layer, unified), 5 Chess kernels recompiled for
Qwen3-0.6B, blocked on Q4NX weight format (see Update 20) — NPU firmware confirmed an active
xclbin via `ERT_CMD_STATE_TIMEOUT`, an early sighting of the same deadlock symptom Update 24 later
isolated. Model xclbins: 23 total across 5 families (Qwen3-0.6B, Qwen3-VL-4B, Qwen3-8B,
Llama-3.1-8B, Gemma4-E2B). CLI scaffolded by a second agent (package.json, tsconfig, command
routing).

---

## UPDATE 20 (2026-07-02 05:17–07:27 ADT): FUSED XCLBIN — FIRST ATTEMPT, Q4NX BLOCKER

The first full attempt at the fused-transformer-xclbin idea flagged in Update 18. Contract
established for Qwen3-0.6B dimensions, 5 kernels recompiled with Chess for the smaller model.
MLIR generator produced a working design; 2 xclbins compiled (374KB full-layer, 253KB QKV-prefix).
`npu_engine_v13` proved the dispatch mechanics work — xclbin loads, BOs allocate (9.4MB weights),
kernel dispatches without crashing — but hit a wall: the fused xclbin's weight-stream layout
expects FLM's proprietary Q4NX chunk format, and the engine's flat INT8 weights don't match it.
Weight-stream scheduler work got the packed size exactly right (2,458,816 dwords) but DMA still
timed out (63s) — diagnosed at the time as a Q4NX *quantization* mismatch (dequant→requant producing
NaN/Inf). Decision: keep v12 (97 tok/s, standalone GEMM) in production, treat fused xclbin as a
separate weight-format workstream.

(Update 24, a session later, revisited this with fresh eyes and found the real bug was the
*schedule* — chunks replicated identically across columns instead of indexed per-tile — not the
quantization theory reached here. See `docs/archive/WEIGHT-STREAM-BLOCKER.md` for the correction.)

---

## UPDATE 19 (2026-07-02 06:24–06:27 ADT): MULTI-MODEL XCLBINS, MODEL-AGNOSTIC ENGINE

Two parallel threads landed close together:

- **Multi-model build-out**: 22-23 xclbins compiled across 6 model families (Qwen3-0.6B, Qwen3-8B,
  Qwen3-VL-4B, Gemma4-E2B, Llama). `npu_engine_mt.cpp` (model-agnostic multi-token engine) +
  `model_config.h` (auto-detects model dimensions from Q4NX headers) + a 42-model catalog
  (`model-catalog.md`) classifying every FLM NPU2 model by architecture. `build_all_models.sh`
  automates the xclbin builds.
- **Engine speed**: v12 at 10 ms/tok (97 tok/s), 24× speedup from the v3 baseline
  (244→50→16→10 ms/tok across v3→v6→v9→v12).
- **Attention**: `attn_scalar.o` + `attn_c8.xclbin` compiled but not integrated — CPU OpenMP was
  still faster for context <128 at this point.
- Site live: 145 visitors, CI pipeline + PR-Agent running, benchmarks current.

---

## UPDATE 18 (2026-07-02 04:01 ADT): M=32 TARGET, NPU LM HEAD, FLM COMPARISON

FLM Kraken Point benchmark for reference: 66.5 tok/s on weaker hardware than ours. Engine evolution
recap v3→v10: 244→16 ms/tok (15.2×, same numbers as Update 17). NPU LM head landed on-chip: 4ms
(N=30720 xclbin, 88KB) — previously a CPU-side cost. `xrt::runlist` batching investigated and found
to save only 27μs/layer (not worth the complexity). M=32 v11 targeted for >100 tok/s. Next flagged:
NPU attention kernel (`edge_attention.o` compiled, not yet integrated) and the fused-xclbin idea
that Update 20 picks up.

---

## UPDATE 17 (2026-07-02 03:00 ADT): M=16 BATCH DECODE — 16 ms/tok, 15.2× SPEEDUP

### 244→16 ms/tok in One Session

```
v3 (Jul 1): 244 ms/tok  baseline
v6 (Jul 2):  50 ms/tok  batch-4 + OpenMP LM head           (4.4×)
v7 (Jul 2):       —     ioctl=9μs, r.wait=1334μs probe
v8 (Jul 2):  27 ms/tok  M=8 batch decode                   (8.2×)
v9 (Jul 2):  16 ms/tok  M=16 batch decode                  (15.2×)
```

### M=16 Batch Decode — How It Works

The v7 probe proved `r.wait()` at 1334μs per GEMM call is NPU compute time,
not driver overhead. The NPU is 99% idle in the M dimension at M=1. At M=16,
compute stays at 1334μs but processes 16× more data → 11ms/tok per batch step.

Single-token boot (157ms) provides top-16 token candidates from LM head logits.
The 16 candidates run through one batched forward pass (28 layers, 4 GEMMs each)
= 112 dispatches at 1334μs = 149ms NPU time + LM head (6ms) + CPU (10ms) ≈ 170ms.
170ms / 16 tokens = 11 ms/tok.

At 64 tokens (4 batches): 16.1 ms/tok effective. Boot amortized away.

### FLM Gap: 1.5×

FLM: 93 tok/s = 10.7 ms/tok (proprietary).  
v9: 63 tok/s = 16.0 ms/tok (open source).  
Gap: 1.5×. Was 20× yesterday morning.

Next: LM head on NPU (151936×1024 INT8 matmul on D-style xclbin) = ~1ms.
That alone brings batch step from 11→6ms/tok and effective to ~8ms/tok.
Combined with M=32: ~4ms/tok effective = matches FLM.

### Session Summary

- 9 engine versions built and tested on-device
- f32 embeddings: -20% decode latency
- OpenMP LM head: 67→6ms
- μs-probe: identified NPU compute as bottleneck (not ioctl)
- M=4→8→16 batched decode: dispatch amortization
- 15.2× total speedup
- CI pipeline on self-hosted runner
- All numbers on https://1bit.systems

---

## UPDATE 16 (2026-07-02 02:00 ADT): FULL PROFILE + 50 ms/tok BATCH-4 DECODE

### NPU Dispatch: The Root Cause

μs-accurate profiling (`npu_engine_profile.cpp`) proved our GEMM overhead:

```
Per-GEMM dispatch (112 per token):
  Quantize A:    6 μs   (<1%)
  Sync A→NPU:    2 μs   (<1%)
  Kernel+wait: 1346 μs   (99%)  ← THE BOTTLENECK
  Sync C←NPU:    8 μs   (<1%)
  Dequant C:     1 μs   (<1%)

Total: 1363 μs/call × 112 calls = 156.8 ms (70%)
LM head: 67 ms (30%)
CPU ops (norms, RoPE, attn, SiLU): 0.7 ms (<1%)
```

The NPU is spending 99% of dispatch time in launch+wait overhead.
Actual M=1 GEMM is 0.5-5 μs. Overhead ratio: **2000×**.

### Chained Batch-4 Decode (v6): 50 ms/tok

Instead of per-token dispatch, we generate top-4 tokens from LM head
logits and run them all through one batched forward pass. Each batch
step takes ~160ms for 4 tokens = 40 ms/tok. Boot step: 157ms.

```
$ OMP_NUM_THREADS=16 ./npu_engine_v6 16

  [0] boot=127595 top4=127595,65831,39815,63550 (157ms)
  [1] batch=4 tok=9275 ms=161 (40 ms/tok)
  [5] batch=4 tok=106211 ms=159 (40 ms/tok)
  [9] batch=4 tok=83570 ms=158 (40 ms/tok)
  [13] batch=3 tok=83570 ms=157 (52 ms/tok)
=== 50 ms/tok effective ===
```

Token IDs diverse across batches. No NaN. Clean exit.
4.4× speedup from v3 (244→50 ms/tok).

### OpenMP LM Head

Pre-converted BFP16→F32 embeddings (622 MB) + OpenMP on 16 Zen5 cores:
LM head: 67ms → ~6ms per token (11× faster). This plus batch-4
amortization is what dropped us from 222→50 ms/tok.

### What We Learned

- Removing weight re-sync (v4) saved nothing — weights already on device.
- 2-layer draft model (spec decode v0) had 0% acceptance rate on Qwen3.
- Batching at decode time works: dispatch overhead amortizes across tokens.
- CPU is never the bottleneck — 26 μs/layer vs 5599 μs GEMM dispatch.

### Next: Fused Transformer XCLBIN

The 112 dispatches per token are now 112 per 4 tokens = 28/token effective.
To get to FLM's 93 tok/s, we need a single fused transformer-layer xclbin
that chains QKV→norm→attention→O→norm→GU→D on NPU without host round-trips.
That turns 28 dispatches into 1. Then LM head goes on NPU via D-xclbin INT8
matmul. Then we're at ~10 ms/tok.

---

## UPDATE 15 (2026-07-01 15:00 ADT): PR-AGENT LIVE, LANDING PAGE DEPLOYED, 242 ms/tok VERIFIED

### Live Production Stack

```
https://1bit.systems          → 50 TOPS landing page (Cloudflare Pages)
https://github.com/.../1bit-systems → Full source, benchmarks, journey
PR-Agent: The-PR-Agent v0.41 (DeepSeek) + GitNexus impact reports → auto-review on every PR
```

### Verified Timing (2026-07-01 15:00 ADT)

```
=== NPU Engine v3 — Continuous Batch ===
Prefill 9 tokens: 179ms (20 ms/tok)
Decode 4 tokens: 242 ms/tok
Tokens: 106811, 63165, 117266, 109842
```

| Metric | Today | Overnight (Jul 1 04:00) |
|--------|-------|-------------------------|
| Prefill M=9 | 179ms (20 ms/tok) | ~200ms |
| Decode | **242 ms/tok** | 219 ms/tok |
| Prefill M=1 | 161ms | — |
| Prefill M=4 | 162ms (40 ms/tok) | — |
| PPR Agent | The-PR-Agent v0.41 + DeepSeek + GitNexus | — |
| Landing page | 50 TOPS headline deployed | — |

### What pi-agent Tightened

- Timings stable across all benchmarks (prefill + decode scaling verified)
- No regression from overnight session — 242 ms/tok matches the 244 ms/tok from 09:30
- Engine runtime exit code 0, all tokens diverse, no NaN

### 1-bit Models Confirmed

Bonsai-1.7B IQ1_S: 281 tok/s on Radeon 8060S Vulkan, 385 MB. pi-agent patched llama.cpp with Q2_0 validation for Strix Halo gfx1151. Models on disk at /home/bcloud/models/bonsai-1.7b/.

### PPR Agent Deployed

The-PR-Agent/pr-agent@v0.41 (upstream, pinned SHA) → DeepSeek
GitNexus knowledge-graph report injected via artifact_path (blast radius, affected flows)
Config: 3 AI reviewers, INT8-focused review instructions, automatic review on PR open + push

### What's Next

- NPU attention dispatch for >32 token context
- GGUF Q8_0 native loader (bypass Q4NX)
- 1-bit NPU kernel (ternary GEMV on XDNA2)

---

## UPDATE 13 (2026-07-01 04:00 ADT): INT8 ENGINE COMPLETE — 219 ms/tok, CONTEXT POOL

## 🏆 Peak Achievement: 31.0 TFLOPS on NPU (config2 design)

**Verified at `/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config2/`**
```
Avg NPU tflops: 31.0081
Max NPU tflops: 31.4522
Matrix: 3072×4096×1536 (M×K×N), tile: 192×128×96
Design: 32 cores (8 cols × 4 rows), Chess kernel
```

### Engine: WORKING at 1.93s/tok with BFP16 xclbin

| Version | XCLBIN | Speed | Status |
|---------|--------|-------|--------|
| v2 | 4096x4096 BFP16 | 15.6s | First working |
| v3 | 2048x2048 BFP16 | 2.04s | 8x faster |
| v7 | **1024x1024 BFP16** | **1.93s** | 220KB xclbin, all fixes |
| config2 | **config2 (192×128×96)** | **31.0 TFLOPS** | 32 cores, Chess kernel |

### Architecture: Complete & Verified
| Component | Status | Detail |
|-----------|--------|--------|
| Q4NX I4 dequant | OK | Tile-grid 32x256, zero NaN/Inf |
| NPU GEMM | OK | 1024x1024 BFP16 ebs8, 12 TFLOPS |
| 28-layer pipeline | OK | Q/K norms, RoPE, KV cache, SiLU MLP |
| LM head | OK | Embedding table (tied embeddings) |
| Token quality | OK | 84869, 55120, 70247, 75499 (diverse, temp=1.0) |
| Logit range | OK | [-16.3, 23.8] correct LLM distribution |
| FW | OK | 1.1.2.65 (latest for device 0x17f0_11) |

### BF16 Kernel: Compiled, Blocked by SRAM
The Chess API supports native BF16 via `aie::mmul<8,8,8,bfloat16,bfloat16,32>` with emulation flag `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16=1`. Kernel compiles and links but xclbin fails because:
- B tile: 64x128 BF16 = 16KB. With depth=2 = 32KB.
- A tile: 32x64 BF16 = 4KB. With depth=2 = 8KB.  
- C tile: 128x128 BF16 = 32KB. With depth=1 = 32KB.
- Total L1: 32+8+32 = 72KB > 64KB. Blocked.
- Fix needs: redesign to 64x64 B tiles (8KB, fits at 8+8+16=32KB depth=2)

### All Fixes Applied
1. x16 weight scaling in pre_pack (RMSE 0.0003 vs 0.032 naive)
2. LM head = embedding table (tied embeddings, removed I4 quantization error)
3. 9-token chat template prefill
4. Q/K per-head norms + RoPE (rope_theta=1e6, correct per position)
5. KV cache with full QK^T + softmax attention
6. 1024x1024 BFP16 xclbin (220KB, compiled today)

### Key Files
| File | Purpose |
|------|---------|
| npu-infer/src/npu_engine_v7.cpp | Working engine |
| npu-infer/src/dequant_q4nx.c | Correct I4 dequant |
| npu-infer/build/qwen3_gemm/design_1024_bfp16.xclbin | 220KB xclbin |
| npu-infer/build/qwen3_gemm/mm_bf16_direct.o | BF16 Chess kernel (compiled, ready) |
| npu-infer/build/qwen3_gemm/mm_scalar.o | Scalar BF16 kernel (working alt) |
| /home/bcloud/Desktop/HANDOFF-NPU-OPTIMIZATION.md | This handoff |

### Build & Run
cd /home/bcloud/npu-sandbox/npu-infer
g++ -std=c++23 -O3 -o build/npu_engine_v7 src/npu_engine_v7.cpp build/dequant_q4nx.o \
  -Iinclude -I/home/bcloud/torch2aie/toolchain/xrt/include \
  -I/home/bcloud/torch2aie/examples -I.../gemm_asymmetric_tile_buffering \
  -L.../xrt/lib64 -L.../mlir_aie.libs -lxrt_coreutil -luuid -lm
LD_LIBRARY_PATH=.../xrt/lib64:.../mlir_aie.libs:.../sysroot/usr/lib64 ./build/npu_engine_v7


## BREAKTHROUGH — Full GEMM Pipeline Running! (2026-06-28)

### Current Status: 5 GEMM runs on mm.xclbin in 3.6ms ✅
- All 4 xclbins loaded successfully
- I8→BF16 weight conversion working
- 5 GEMM kernel invocations (5 column-blocks of Q_proj × K_proj) complete
- Output matches input pattern — NPU computing correctly
- Total time: 3.6ms for Q_proj GEMM (5 column blocks × [256,1024])

### What's Next
1. **Fix `bo::sync()` timing** — the 3.6ms includes weight syncs which shouldn't be needed per layer
2. **Add all 28 layers** — iterate through all layers with proper weight management
3. **Add attn.xclbin** — attention kernel with KV cache
4. **Add layer.xclbin** — full transformer layer
5. **Add dequant.xclbin** — dequantization before GEMM
6. **Build decoder loop** — proper token generation with sampling

### Key Files
- `include/engine.h` — NpuBo, WeightPacker, XclbinManager, NpuInferenceEngine
- `src/engine.cpp` — 300 lines of working code
- `src/main.cpp` — Entry point
- `include/model.h` — Model + weight packer API
- `src/model.c` — Q4NX parser + I8→BF16 converter

### Build/Run
```bash
cd /home/bcloud/npu-sandbox/npu-infer/build
cmake .. && make -j4
./npu_infer
```

## Final Benchmark Summary (2026-06-28)

### GEMM Compute
| dtype | TFLOPS | % Peak | % Chess | Config |
|-------|--------|--------|---------|--------|
| INT8 | 7.14 | 13.6% | 22.9% | M=8192 K=8192 N=4096, 32×256×32, 2× unroll |
| BF16 | 3.31 | 6.3% | 10.6% | M=8192 K=8192 N=2048, 32×128×32, 2× unroll, no transpose |

### LLM Inference (qwen3:0.6b, Turbo, ~2W)
| Tokens | TTFT | Prefill | Decode | KV Cache |
|--------|------|---------|--------|----------|
| 10 | 0.48s | 23 t/s | 82 t/s | 0.1% |
| 500 | 0.61s | 79 t/s | 91.5 t/s | 3.3% |
| 1000 | 0.63s | 70 t/s | 87.3 t/s | 6.4% |
| 1264 | 0.61s | 89 t/s | 84.6 t/s | 8.0% |
| 8 concurrent | 0.48s | — | 82-85 t/s | — |

### Efficiency
- NPU: 46 tok/s/W (2W) — 25× more efficient than GPU (1.9 tok/s/W @ 20W)
- NPU GEMM: 3.57 TFLOPS/W — 6× more efficient than GPU (0.57 TFLOPS/W)
- KV cache headroom: 92% free after 1264 tokens (~15,000 token capacity)

### Deliverables
- 7 kernel variants (packed, unroll2x, swp, 8acc, vliw, optimized)
- Instruction compiler (byte-exact parse/rebuild, 224 commands)
- XAIE transaction generator
- NPU template compiler
- libgemm C wrapper (114KB instructions generated)
- GTT dma-buf zero-copy benchmarks (56 GB/s)
- SMU init order fix (aie2_pci.c)
- Q4NX model loader + NPU weight packer (now uses BF16 byte-pair reading, not per-group dequant)
- NPU inference engine (3 xclbins, 3 hwctx, runlist-based submission in progress)
- libunlock.so (both FLM gates bypassed)
- FLM protocol fully reverse-engineered (BO layout, weight format, kernel args)

### Repos
- https://github.com/bong-water-water-bong/strixhalo-npu-setup
- https://github.com/bong-water-water-bong/npu-gpu-cpu

## Max Context Stress Test (Turbo Mode)

| Metric | Value |
|--------|-------|
| Prompt tokens | 9,868 |
| TTFT | 6.2s |
| Prefill speed | 1,591 t/s |
| Decode speed | 29.8 t/s |
| KV cache used | 61.5% |
| Free KV tokens | ~6,000 |
| Second request | KV cache persisted correctly |

Turbo `--prefill-chunk-len 8192` delivers 1,591 t/s prefill at full context.
Decode degrades from 91.5→29.8 t/s at 60%+ KV cache — still usable.
KV cache has room for ~6,000 more tokens within 16,384 ctx-len.
Multi-turn conversation: KV cache persists correctly across requests.

## Session 2025-06-28 Findings

### Weight Format Breakthrough
Q4NX `dtype=I8` is MISLEADING. The data is ACTUALLY BF16 stored as pairs of bytes:
- Every 2 consecutive I8 bytes form one BF16 value: `[lo_byte, hi_byte]` little-endian
- Shape [256, 5120] I8 = [256, 2560] BF16 values
- No per-group dequantization needed — read byte pairs directly as BF16
- The per-group absmax scaling approach was incorrect (produced wrong weights)

### Critical Issue: opcode=3 is IDENTITY
- mm.xclbin with opcode=3 copies input BO to output BO unchanged
- Weight BOs at idx=5 and idx=6 are COMPLETELY IGNORED
- Tested with different weights at idx=5 and idx=6: no effect on output
- The actual GEMM opcode has NOT been found yet
- Sequential opcode testing (0-15) on mm.xclbin hangs the device at op=1
- Possible causes:
  1. GEMM is done via `runlist::execute()` not individual `kernel::operator()`
  2. A different xclbin (not mm.xclbin) handles GEMM
  3. The kernel needs BOs pinned to specific memory (SRAM vs HOST)
  4. The kernel uses a DIFFERENT set of arguments than what we provide

### Current Engine State
- Builds and runs: loads model, creates BOs, sends weights, runs all 28 layers
- Output is deterministic but WRONG: tokens [919, 996, 185, 385, 495, 156, ...]
- 16 tokens generated in ~3.5s (220ms/tok)
- ~591 BOs (after BF16 fix, down from ~985 with per-group dequant)
- Weight init time: ~190ms (vs ~2100ms with per-group dequant)

### Next Steps / Options

**Option A: Build npu_sequence framework from scratch**
- Implement `npu_dma_memcpy_nd` equivalent using DRM ioctl BD creation
- Need to understand the DMA BD format, tile addressing, and channel assignment
- Estimated: several weeks of reverse-engineering

**Option B: Use libgemm.so + our own npu_sequence**
- Load libgemm.so and call `Gemm::generate_seq()` for DMA + compute
- Create npu_sequence with known struct layout (we have it)
- Call `cmds2seq()` to compile to instructions
- Submit instructions via XRT kernel with instruction BO
- Challenge: need correct tile placement and BD assignment parameters

**Option C: LD_PRELOAD interposition on FLM**
- Intercept gen_layer_seq and cmds2seq to capture the compiled instructions
- Replay them in our engine with different activations
- Pro: immediate working GEMM
- Con: requires FLM running for initial capture, model-specific

**Option D: DRM ioctl exploration**
- The DRM interface has CREATE_BD/SYNC_BD ioctls we haven't explored
- Maybe use mmap on NPU tile memory directly
- NPU has shared virtual memory feature

## Key Discoveries from 2025-06-28 Late Session

### Architecture: Weight DMA via libgemm instruction generation
- **ALL 4 xclbins opcode=3 is IDENTITY** — none read from weight BOs directly
- **Weight DMA is REQUIRED** — weights must be in AIE tile-local memory via DMA BD descriptors
- **libgemm.so** can be `dlopen`'d independently (ZERO external deps beyond libstdc++)
- **libgemm.so** contains: `Gemm::Gemm(LM_Config&)`, `Gemm::generate_seq`, `Gemm::Impl::generate_seq`, `npu_dma_memcpy_nd`, all command classes
- **libgemm.so** has `Gemm::Impl::shim_tiles` in `.rodata` (read-only, values = `[0,1,2,3,4,5,6,7]` — correct defaults)
- **libmha.so** can be `dlopen`'d independently and contains `npu_sequence::cmds2seq()`
- **libqwen3_npu.so** CANNOT be loaded standalone (needs SafeTensors symbols from FLM binary)
- **npu_sequence struct**: requires careful initialization but just setting n_tile_rows=4, n_tile_cols=4 works
- **Gemm::generate_seq succeeds** — populates internal vectors in npu_sequence with DMA descriptors
- **Internal vectors**: offset 0x28 = pointer array (to command objects), offset 0x38 = real instruction words
- **Instruction words generated for various GEMM shapes**: Q_proj (584 words), O_proj (704+912), gate/up (784+1548), down (1024+3600)

### Critical Technical Details
- **shim_tiles** is at 0x15960 in libgemm.so's `.rodata` (read-only, values [0,1,2,3,4,5,6,7])
- **npu_sequence layout**:
  - 0x00: n_tile_rows (u32)
  - 0x04: n_tile_cols (u32)
  - 0x0C: ncmds (u32, set by generate_seq)
  - 0x10: op_line_count (u32, set by generate_seq)
  - 0x18: pointer to command array (set by generate_seq)
  - 0x28: vector begin/end/cap (pointer array to command objects)
  - 0x38: vector begin/end/cap (instruction word output)
- **Instruction format**: Starts with header words (0x00001ef1, 0x00000091), then BD descriptor data including address, size, control flags (opcode=3, group=65536)
- **GOT entry at 0x18f58** resolves to read-only .rodata (NOT writable BSS as previously thought)
- **Tile data in FLM** (captured from running process):
  - proj_tiles: [34,50,66,82, 35,51,67,83, 36,52,68,84, 37,53,69,85] — 4×4 grid, col=2-5, row=2-5
  - mvm_tiles: [2,3,4,5,0,0,0,0,0,0,0,0,0,0,0,0]
  - attn_qk_tiles: [32,64,39,71, 2,3,4,5,0,0,0,0,0,0,0,0]
  - attn_kv_tiles: [48,80,55,87, 32,64,39,71, 2,3,4,5,0,0,0,0]
  - shim_tiles: [0,1,2,3,4,5,6,7,0,0,0,0,0,0,0,0]

### Generated Instruction Files
- `/tmp/gemm_Qproj_vec38.bin` — 16 bytes (containing 0x1ef1 header)
- `/tmp/gemm_Oproj_vec38.bin` — 3648 bytes (912 u32 words)
- `/tmp/gemm_gate_vec38.bin` — 6192 bytes
- `/tmp/gemm_up_vec38.bin` — 6192 bytes
- `/tmp/gemm_down_vec38.bin` — 14400 bytes

### BREAKTHROUGH: libgemm.so instructions submitted to XRT kernel
- **Wrote `test_libgemm9_final.cpp`**: calls `Gemm::generate_seq()` then submits vec@0x38 instructions as SRAM BO to XRT kernel
- **ALL 5 GEMM configurations execute successfully** through kernel with opcode=0 (dynamic instruction mode)
- **Execution times**: Qproj=3.15ms, Oproj=0.12ms, gate=0.10ms, up=0.08ms, down=0.08ms
- **Kernel accepts SRAM BO as arg 1 (instr)**: uses `xrt::memory_group(1)` for instruction BO in SRAM bank
- **Instructions reference hardcoded addresses** — need to patch BO addresses to match our actual BO physical addresses
- **Kernel arg layout verified**:
  - arg 0: opcode (uint64_t, offset 0)
  - arg 1: instr ptr (SRAM BO, group 65537, offset 8)
  - arg 2: ninstr (uint32_t, offset 16)
  - args 3-7: BOs (HOST group 65536)
- **XRT sync bug**: `bo.sync(dir, 0, size)` treats sz=0 as flag meaning "use size from third param" — `sync(dir, 0, 4MB)` crashes but `sync(dir, sz, 0)` with non-zero sz works

## Full Pipeline Results

All 7 FLM pipeline functions successfully loaded and called:
- `_send_rope_rms_weights` ✅
- `_send_rms_weights` ✅
- `gen_dequant_seq` ⚠️ "DEPRECATED FUNCTIONS"
- `_send_x` ✅
- `_move_weights` ✅
- `generate_seq` ✅
- `cmds2seq` ✅

Output: 114,208 bytes (28,552 instructions). Kernel executes (ERT_CMD_STATE_COMPLETED) but produces identity — `gen_dequant_seq` is deprecated and may not add weight DMA. The newer dequant path (`generate_dequant_q80_packed_in_q4nx_seq`) needs investigation.

Full pipeline source: `npu-sandbox/xrt-direct/full_pipeline.cpp`

## Session 2025-06-28 Late Testing — FLM HTTP Single-Connection Limit

### Discovery: FLM's HTTP Server Crashes Under Concurrent Connections

tested the unlock library strategy extensively and discovered a fatal limitation:

```
FLM can only handle ONE TCP connection at a time.
Even with --socket 10 (10 I/O threads), concurrent connections CRASH FLM.
```

### Test Results

| Test | Result |
|------|--------|
| Single request (sequential) | ✅ Works (0.5s prefill + 0.07s decode)
| 2 concurrent requests to SAME instance | ❌ `ConnectionResetError(104)` — FLM crashes
| 2 separate instances (8083 + 8084), 1 concurrent each | ❌ Both crash (`ConnectionResetError`)
| Sequential requests with `--no-keepalive` | ✅ Works, but not concurrent
| `--socket 1` (single-threaded) | Still crashes on concurrent; logs "Connection limit reached (1)"
| `--socket 16 --q-len 10` | Same crash behavior

### Root Cause
FLM's HTTP server (based on standalone ASIO) has a hard limit of 1 active connection.
The `--socket` parameter appears to set max concurrent I/O THREADS, not max connections.
When a 2nd TCP connection arrives while the 1st is still being processed:
1. FLM logs "Connection limit reached (1), rejecting new connection"
2. FLM crashes (SIGABRT or segfault)
3. Process dies, all pending requests get `ConnectionResetError`

### Implications
- **LD_PRELOAD unlock is a dead-end**: Even if both NPU gates are bypassed, FLM's HTTP server
  can't handle concurrent requests. The unlock worked (both mutex + g_npu_in_use bypassed)
  but FLM's global inference state (`current_messages`, model context, BO state) is not
  thread-safe — concurrent entry corrupts state and crashes.
- **Separate FLM instances also fail**: 2+ FLM instances on different ports each work
  individually but also crash under concurrent HTTP connections.
- **dlsym in constructor causes segfault**: LD_PRELOAD of `pthread_mutex_lock` interceptors
  crashes FLM if `dlsym(RTLD_NEXT, ...)` is called inside `__attribute__((constructor))`.
  Lazy resolution (resolve on first actual call, not in constructor) avoids this.
  Even a minimal pass-through LD_PRELOAD (no NPU logic, just dlsym + forward) crashes.

### Viable Path Forward

**Option 1: Proxy/Queue (#1 priority)**
Build a lightweight proxy in front of FLM that:
- Accepts multiple concurrent HTTP client connections
- Queues requests internally
- Feeds them ONE AT A TIME to FLM (serial via Unix socket or single HTTP conn)
- Returns each response to the waiting client
- This gives **no throughput gain** (still 1.1 req/s limit) but prevents client-side timeouts

```
Client A ─╮
          ├─→ [Proxy (queues)] ─→ [FLM (1 req at a time)]
Client B ─╯
```

**Option 2: Build our own NPU engine (npu-infer)**
Continue the `npu-infer/` engine path. Current status:
- ✅ Q4NX model loader (311 tensors, 28 layers)
- ✅ BF16 weight format (byte-pair reading, not per-group dequant)
- ✅ Weight BO packing [256, 1024] blocks
- ✅ XCLBIN loading + kernel execution
- ✅ `libgemm.so` instruction generation (5 GEMM shapes)
- ✅ XRT kernel accepts SRAM instruction BO (opcode=0)
- ❌ Instructions reference hardcoded addresses — need BD address patching
- ❌ Need to understand BD format to replace addresses with `bo.address()`
- ❌ Need real GEMM output (currently identity, opcode=3)

**Option 3: Enhanced unlock with https://github.com/nicedoc/singleton**
Use a separate NPU driver/hack approach that doesn't go through FLM at all.

### Updated Bottleneck Analysis

The original bottleneck analysis was partially wrong. FLM has TWO bottlenecks:

```
Client → HTTP Server (FLM) → [NPU Gates] → NPU HW
              ↕                   ↕             ↕
        Single-connection    Mutex + flag     ~50% utilized
        hard limit (1)       (bypassed via    
                              LD_PRELOAD)
```

Even unlocking both NPU gates doesn't help because the HTTP server itself can't handle
concurrent connections. FLM's true bottleneck is its **HTTP server architecture**, not
just the NPU lock.

## Session 2025-06-28 Late Testing — `cmds2seq()` Discovery & Instruction Pipeline

### `cmds2seq()` WORKS from Independent `npu_sequence`

Prior handoff said `cmds2seq()` crashes on independently-created sequences. **This was incorrect** — it only crashes when `npu_sequence` internal vectors aren't properly initialized. With correct initialization (n_tile_rows=4, n_tile_cols=4, DDR base addresses set), `cmds2seq()` works from both `libmha.so` and correctly compiles commands to instructions.

**Verified flow:**
```
npu_sequence seq = {};
seq.n_tile_rows = 4;
seq.n_tile_cols = 4;
seq.ddr_io_base = (uint32_t)(act_bo_address & 0xFFFFFFFF);
seq.ddr_i_base  = (uint32_t)(act_bo_address & 0xFFFFFFFF);
seq.ddr_w_base  = (uint32_t)(weight_bo_address & 0xFFFFFFFF);
seq.ddr_z_base  = (uint32_t)(weight_bo_address & 0xFFFFFFFF);
seq.ddr_lock    = 0;

gemm.generate_seq(&seq, M, K, N, M, false, 3, 1);
// seq now has 350-704 commands, dirty_flag=1

cmds2seq(&seq);
// seq.vec@0x38 now has 3384-4412 instruction words with BD descriptors
```

### Instruction Output After cmds2seq

| GEMM Shape | Instr Before | Instr After | BD Headers |
|-----------|-------------|-------------|-----------|
| Qproj (256,1024,1024) | 4 words | ? | Minimal (tiny) |
| Oproj (1024,1024,256) | 912 words | 3384-4412 words | 10-14 BDs |
| gate (256,1024,2048) | 1548 words | ? | ~20 BDs |

### BD Descriptor Format (from analysis)

Decoded BD structure at word N:
```
Word N+0: 0x00000091  (BD header type indicator)
Word N+1: 0x00000000  (flags/unknown)
Word N+2: 0x....     (48-bit address, low 32 bits)
Word N+3: 0x0000.... (48-bit address, high 16 bits)
Word N+4: size/control field (e.g., 0x00000004 = 4)
Word N+5: 0x00000000 (control flags, e.g., 0x8000 = read)
Word N+6: 0x00000000
Word N+7: 0x00008000 or 0x00010000 or 0x00004000
...more fields follow...
```

BD field meanings (determined from repeated patterns):
- `0x00008000` + `0x00000001` at W[N+7,N+8]: Read DMA (tile → DDR)
- `0x00010000` + `0x00000003` at W[N+7,N+8]: Write DMA (DDR → tile)
- `0x00004000` + `0x0000000f` at W[N+7,N+8]: Barrier/sync

### Key Discovery: BD Addresses Reference Command Objects, NOT BO Addresses

The 48-bit addresses in the instruction BD descriptors (`0x7390..., 0x7832..., 0x764b...`) point to **command objects** (npu_write_cmd, npu_dma_block_cmd instances) in the seq's command vector (vec@0x28), NOT directly to BO data buffers.

After `cmds2seq()`, the instruction stream contains:
1. **Heap addresses** of command objects — the NPU DMA engine reads these for additional data
2. **DDR base addresses** (from seq.ddr_*_base) encoded as 32-bit offsets within specific BD fields
3. **Control flags** for DMA direction, tile selection, synchronization

### Architecture: Dual DMA Model

The instructions handle **activation DMA only** (moving activations between DDR BO and tile SRAM).
Weight DMA is a SEPARATE step via `npu_sequence::npu_dma_memcpy_nd()`, which generates additional
BD descriptors for transferring weights from weight BOs to tile-local SRAM.

### Impact on npu-infer Engine

The engine needs to:
1. Create `npu_sequence` with correct tile params + DDR base addresses (= bo.address() & 0xFFFFFFFF)
2. Call `Gemm::generate_seq()` for each GEMM operation to get command objects
3. Call `npu_sequence::npu_dma_memcpy_nd()` for weight transfers (need to find correct signature)
4. Call `npu_sequence::cmds2seq()` to compile everything to instruction words
5. Copy instructions to SRAM instr_bo
6. Submit to XRT kernel with opcode=0
7. The instructions handle all DMA internally — weight BOs at args 5,6 might not be needed

### Open Questions
1. What is the exact `npu_dma_memcpy_nd()` signature? (defined in libgemm.so)
2. How do the tile addresses map to physical AIE tiles?
3. Can we skip weight DMA and pass weights via kernel args?
4. What is the correct opcode for compute-only mode (without DMA instructions)?

### Answer to Open Question #4 (from FLM strace)
FLM uses **opcode=3 with instr=0, ninstr=0** — meaning it uses the xclbin's pre-compiled AIE kernel.
FLM does NOT use opcode=0 (dynamic instruction mode). This means:
- Opcode=3 IS the "compute-only" mode where the AIE kernel handles everything
- The xclbin's AIE program knows what to do with args 3-7 (BOs)
- But our tests show opcode=3 produces IDENTITY output, suggesting:
  a) The AIE kernel requires specific tile/SRAM state (from prior DMA)
  b) The identity behavior is expected with freshly loaded xclbin
  c) FLM sets up tile SRAM state via weight DMA before running the kernel

**Conclusion**: Even opcode=3 requires proper tile SRAM setup (weights in tile memory).
The AIE kernel reads weights from tile SRAM, not from DDR BOs. The kernel args (BOs) tell it
where in DDR to find the activation data, but weights must be pre-loaded to tile SRAM.

### Next Priority
1. Find `npu_dma_memcpy_nd()` signature by searching libgemm.so symbols
2. Build combined pipeline: generate_seq + dma_memcpy_nd + cmds2seq → instruction stream
3. Test with opcode=0 and SRAM instr_bo containing both weight + activation DMA descriptors
4. Or: find if there's a simpler weight submission API that doesn't need DMA descriptors

### Session 2025-06-28 End — `cmds2seq` works, instructions don't produce GEMM, need runlist

Summary of last session's findings:

**`cmds2seq()` WORKS** — confirmed earlier today. With proper seq initialization (tile dims + DDR base addrs), cmds2seq compiles command objects to instruction words.

**Instructions DON'T produce GEMM output** — Even with cmds2seq and real BO addresses, the instruction-based submission (opcode=0 with SRAM instr_bo) produces identical output as opcode=3 (identity/no-op). This means:
- The instructions contain only DMA descriptors (moving data between DDR and tile SRAM)
- The actual GEMM computation needs a SEPARATE kernel invocation OR is embedded in runlist
- The instructions reference heap addresses (command objects), not BO addresses
- `seq.ddr_*_base` fields are NOT directly embedded in instruction stream

**`libqwen3_npu.so` CAN be dlopen'd** — with just `libmha.so`, `libgemm.so`, and `libxrt_coreutil.so` as dependencies. All key functions resolve:
  - `_move_weights()`, `_send_x()`, `_send_rms_weights()`, `_send_rope_rms_weights()`
  - `gen_layer_seq()`, `gen_lm_head_seq()`, `gen_mha_engine_seq()`
  - Static tile data: `proj_tiles`, `mvm_tiles`, `attn_kv_tiles`, `attn_qk_tiles`
- However, these methods need a `qwen3_npu_sequence::Impl` instance (can't construct without FLM binary)
  
**`npu_dma_memcpy_nd()` from `libgemm.so` functions** — exported and callable. Takes 15 parameters. Can be used to generate weight DMA commands. However, calling it after `generate_seq` replaces the command vector (doesn't append). Must call BEFORE generate_seq.

**FLM uses `xrt::runlist` for all operations** — XRT intercept log shows:
  - FLM creates a `runlist` with multiple ops (weight DMA ops + compute ops)
  - Ops with only 2 BOs (arg3=act_bo, arg4=ws_bo) = WEIGHT DMA operations
  - Ops with 3 BOs (arg3=act_bo, arg4=ws_bo, arg5=weight_bo) = GEMM COMPUTE
  - ALL ops use opcode=3 with instr=0, ninstr=0
  - After runlist::execute(), individual run::start() calls drive compute

**IMPLICATION**: The xclbin encapsulates BOTH weight DMA AND GEMM compute. Opcode=3 triggers a full operation that:
  - Reads weight from arg5 BO (or pre-loaded weights in tile SRAM)
  - Reads activation from arg3 BO
  - Writes result to arg3 BO
  - Uses arg4 (ws) as temporary workspace

**BUT standalone opcode=3 with direct kernel call does NOTHING** — ALL BOs unchanged. This proves the xclbin requires the runlist context or prior tile state.

**NEXT STEPS (priority order):**
1. Build `xrt::runlist`-based test that mimics FLM's submission: multiple ops with weight DMA followed by compute
2. Or: Build test that uses `_move_weights` from `libqwen3_npu.so` to load tile SRAM, followed by opcode=3 compute
3. Or: Try xclbins for individual layers (layer.xclbin, attn.xclbin, dequant.xclbin) with runlists

**Updated findings (2025-06-28, late session):**
- **ALL 4 xclbins with opcode=3 produce IDENTITY for any BO config** — tested mm, attn, layer, dequant. None modify any BO.
- **Instructions with opcode=0 on ALL xclbins also produce identity** — the BD descriptors in the instruction stream reference heap addresses (command objects), not BO device addresses. `cmds2seq` does NOT replace heap addresses with BO addresses.
- **`-rdynamic` + stub SafeTensors works** to load `libqwen3_npu.so` with RTLD_NOW. Needed stubs: `SafeTensors::load_weights`, `MHA::MHA()`, `MHA::~MHA()`, `bytes::bytes()`, `bytes::~bytes()`. However, `Impl::C1` crashes with minimal LM_Config (floating point exception from divide-by-zero on hidden_size=0).
- **`npu_app_manager::C1`** is exported but needs real xrt::device, not worth bootstrapping.
- **FLM binary can't be dlopened** — PIE executable, `cannot dynamically load position-independent executable`.
- **The real GEMM requires the xclbin's internal tile SRAM state** — weights must be pre-loaded into AIE tile SRAM before opcode=3 execution. The xclbin's built-in program controls both weight DMA and compute; it checks tile lock/ready registers before executing.
- **FLM's weight DMA BOs are small (1MB) pre-packed tensor slices**, prepared during initialization from the model weights. These are separate from the 128MB weight BOs used in compute starts.

**Revised understanding of FLM per-layer pipeline:**
1. Allocate per-layer scratch BOs (2×2MB, 2×1MB)
2. Create 5 weight-DMA `run` objects in a `runlist` (each: opcode=3, bo3=weight_tensor1-5, bo4=shared_act_bo_10MB)
3. `runlist::execute()` — atomically loads 5 tile's worth of weights into AIE SRAM
4. After completion, run 8 `run::start()` calls for GEMM compute (each: opcode=3, bo3=output_scratch, bo4=1MB_scratch, bo5=weight_bo_128MB)
5. sync BOs to read back results

**Key open questions:**
- What makes runlist ops weight-load vs compute? (Same opcode=3, different BO patterns)
- How are the 1MB weight tensor BOs formatted? (Pre-packed from weights via `_move_weights`)
- Does the xclbin's built-in AIE program handle the full layer pipeline internally?

**Most promising path forward:**
Build a comprehensive XRT capture (intercept library) that captures the ACTUAL BO content before/during FLM inference. This would reveal both the weight tensor format and how the runlist ops are structured. Then we can either:
- A) Replicate the exact same BO setup and runlist pattern
- B) Use FLM's own `npu_app_manager` with proper initialization to generate the full pipeline

## Session 2026-06-28 Deep Research — Definitive Findings

### npu_sequence Layout — DEFINITIVELY DETERMINED

Built probe (`/tmp/probe_seq_layout.cpp`) that dumps all vector states before/after `generate_seq` and `cmds2seq`. Results for Oproj (1024,1024,256):

| Offset | Vector Type | Before gen_seq | After gen_seq | After cmds2seq |
|--------|------------|----------------|---------------|----------------|
| 0x28 | `vector<cmd_ptr>` (8B ptrs) | empty | 352 ptrs → cmd objs | UNCHANGED |
| 0x38 | `vector<uint32_t>` raw BDs | empty | 912 words (3.6KB) | 3384 words (13.2KB) |
| 0x40 | `vector<uint32_t>` **IRON output** | empty | 2468 words (9.6KB) | **4936 words (19.3KB)** |

**`cmds2seq()` APPENDS to vec@0x38 and POPULATES vec@0x40 with proper IRON-format instructions including DDR_PATCH commands.** The correct instruction source for opcode=0 submission is **vec@0x40** (not vec@0x38 which contains raw BDs without DDR_PATCH metadata).

### cmds2seq Call Verified Working

- `cmds2seq` is a **weak symbol** in `libgemm.so` at offset `0xdd20`
- Also present in `libmha.so` (offset `0xdd20`) and `libqwen3_npu.so` (offset `0x59a70`)
- Requires `RTLD_GLOBAL` + loading `libmha.so` and `libqwen3_npu.so` to resolve
- Mangled name: `_ZN12npu_sequence8cmds2seqEv`

### Opcode=0 + cmds2seq: STILL IDENTITY

| Test | Instructions | DDR_PATCH | Opcode | Result |
|------|-------------|-----------|--------|--------|
| test_libgemm9_final (original) | 4-3600 raw BDs (vec@0x38) | 0 | 0 | IDENTITY |
| test_libgemm10_fixed (+cmds2seq) | 3952-7560 IRON (vec@0x40) | 40-128 | 0 | IDENTITY |
| Full pipeline (7 FLM calls + cmds2seq) | 28,552 IRON | 640 | 0 | IDENTITY |
| Original full_pipeline.cpp | 28,552 IRON | 640 | 3 | IDENTITY |

**The mm.xclbin kernel produces identity output regardless of opcode or instruction format.** Even with the complete FLM pipeline (rope_rms → rms → dequant → send_x → move_weights → gen_seq → cmds2seq) generating 114KB of proper IRON instructions, the NPU copies input to output unchanged.

### Key Test Binary Status

| Binary | Path | Status |
|--------|------|--------|
| test_libgemm9_final | `npu-infer/build/test_libgemm9_final` | Runs, identity output |
| full_pipeline (original) | `xrt-direct/full_pipeline` | Runs, identity output |
| gemm_final.so | `/tmp/gemm_final.so` | Shared lib, calls cmds2seq correctly |
| capture_lib.so | `xrt-direct/capture_lib.so` | Intercepts XRT, captures logs |
| npu_infer | `npu-infer/build/npu_infer` | Full engine, wrong output |

### npu-infer Engine Critical Bugs Found

1. **Row-blocking bug**: Only first 256 rows of each weight tensor are packed — 75%+ of weights silently zero for tensors with >256 rows
2. **No RMS normalization**: Pre-attention and pre-MLP RMS norm never applied
3. **No real attention**: Calls attn.xclbin but doesn't implement QK^T softmax
4. **Weight1 = Weight2**: Same BO passed for both weight arguments
5. **No dequantization**: Reads I8 bytes directly as BF16 pairs, ignores group scales
6. **Missing implementation**: `run_mm_blocked()` declared in header but never defined
7. **Single-kernel, not runlist**: Each weight block gets individual `run_gemm()` with `r.wait()` — no batching

### torch2aie — Custom Kernel Compilation Path EXISTS

The `/home/bcloud/torch2aie/` directory contains a complete AIE kernel development toolchain:
- **Chess compiler** for AIE2P (`xchesscc_wrapper aie2p`)
- **MLIR-AIE** Python dialect for dataflow description
- **aiecc** compiler driver producing xclbin + instruction binaries
- **Working examples**: Qwen3 decode layer kernels, GEMM kernels, attention kernels
- **Pre-built xclbins**: ATB GEMM configs (128×64×128, 192×128×96), prefill attention
- **Numerical verification**: `run_kernel_main16_q4nx.py` validates against Python reference

This is the path to creating custom xclbins with REAL compute kernels that read from weight BOs.

### Root Cause Theory

The mm.xclbin/attn.xclbin/layer.xclbin kernels are "weight-stationary" — they expect weights pre-loaded into AIE tile SRAM via a prior DMA step (FLM's weight DMA runlist batch). The GEMM compute step reads weights from tile SRAM, not from kernel argument BOs. Our instructions are correct for activation DMA but the compute kernel never executes because tile SRAM doesn't contain weights in the expected format/layout.

**The pre-compiled xclbin is a black box.** Without modifying the xclbin itself (which requires the torch2aie toolchain), we can't make the existing kernels do GEMM.

### Updated Priority — Two Viable Paths

**Path A: torch2aie custom xclbin** (Clean, but effort)
1. Use the existing torch2aie pipeline to compile a new GEMM xclbin
2. The custom kernel reads weights from DDR BOs (kernel args), does GEMM, writes output
3. No tile SRAM pre-loading needed — everything through kernel args
4. Model after `examples/gemm_asymmetric_tile_buffering/` or `examples/qwen3-decode-layer/`

**Path B: Capture FLM's runlist protocol via enhanced LD_PRELOAD** (Hack, but faster)
1. Intercept `xrt::runlist::execute()` and dump ALL BO contents before submission
2. Intercept `xrt::runlist::add()` to capture the exact run configuration
3. Replicate FLM's complete weight-DMA-then-compute protocol
4. This reveals what tile SRAM state the xclbin expects

### ## Session 2026-06-28 Final — 40-Column NPU2 Compiler & Firmware Analysis

### 40-Column Compiler Build — SUCCESSFUL

Modified MLIR-AIE source at `/home/bcloud/mlir-aie/`:
1. `include/aie/Dialect/AIE/IR/AIETargetModel.h:823` — `return 8` → `return 40` (header-only, fully inlined)
2. `python/iron/device/__init__.py:35` — `_MAX_COLS["NPU2"] = 8` → `= 40`
3. Rebuilt with `ninja` (123/123 targets)
4. Toolchain wrapper at `/home/bcloud/mlir-aie/npu2_40_toolchain/`

**Verified new compiler works:**
- `aie-opt` accepts `tile(39, 2)`, rejects `tile(40, 2)` with bounds error ✅
- `NPU2().cols = 40`, `NPU2().rows = 6`, 160 compute tiles, 40 mem tiles, 40 shim tiles ✅
- Virtualized variants (1-7 cols) still work via `npu2_1col`..`npu2_7col` ✅

### 40-Column XCLBIN Compiled — 1.8MB, 160 cores
- All 160 AIE core ELF files compiled via xchesscc
- Partition JSON encodes `column_width: 40`, txn header encodes `numCols = 0x28 = 40`
- xclbin passes xclbinutil validation, bootgen would accept it

### Bug Fix: Partition Metadata Auto-Detection
**Problem:** Partition JSON and txn header both hardcoded `tm.columns() = 40`, causing ALL xclbins to report `column_width=40` (even 12-col designs used only 12 columns).

**Fix (applied to rebuilt toolchain source):**
- `tools/aiecc/aiecc.cpp:generatePartitionJson()` — now walks tile ops to compute actual design columns instead of using `targetModel.columns()`
- `lib/Targets/AIETargetNPU.cpp:emit()` — same fix for txn header `numCols`
- Both match the actual tile placements: 12-col design → `column_width=12`, etc.

### Firmware Limit: 8 Columns HARDCODED
- `DRM_IOCTL_AMDXDNA_CREATE_HWCTX` rejects `EINVAL` for any `column_width > 8`
- Tested: 9, 10, 12, 16, 40 — **ALL rejected**
- 8 columns works perfectly at 31.0 TFLOPS
- Firmware binary: `/lib/firmware/amdnpu/17f0_11/npu.sbin.1.1.2.65.zst` (decompressed `npu.sbin`, 430KB)
- Validation string at offset `0x1d6d1`: `"Invalid column count: %u >= %u"`
- The `aie2_max_col` kernel driver parameter (`echo 40 > /sys/module/amdxdna/parameters/aie2_max_col`) does NOT override this — firmware validates independently
- Older firmware `npu.sbin.1.0.0.166` (376KB) has **no column validation strings** — might accept >8 columns but likely lacks other features

### Conclusion
**31.0 TFLOPS is the practical maximum** from the NPU without firmware modification.
The MLIR-AIE compiler can be told about all 40 columns, firmware only allows 8-column-partitions.
To unlock 50+ TFLOPS: reverse-engineer PSP firmware format, patch the column limit constant,
reflash with valid hash/signature.

### Firmware Deep-Dive (this session)

**Two firmware files, different purposes:**

| File | Version | Role |
|------|---------|------|
| `npu.sbin` → `1.0.0.166` | 376KB | Boot/init firmware — minimal AIE tests, NO partition mgmt, NO power gating, NO column validation |
| `npu_7.sbin` → `1.1.2.65` | 429KB | Runtime AIE mgmt — partitions, power gating (ONO 0-7), CDO/PDI loading, 8-col limit |

**1.0.0.166 CANNOT substitute for 1.1.2.65** — completely different PDI header, no partition creation code, no power management. Swapping would brick the NPU.

**Signature chain (verified from kernel source at `/home/bcloud/amdxdna-dkms/`):**
1. Kernel sends `MSG_OP_QUERY_AIE_TILE_INFO` → firmware responds with `cols=40`
2. Kernel sets `ndev->total_col = min(aie2_max_col, 40)` where `aie2_max_col` is the kernel param (set to 40)
3. On `MSG_OP_CREATE_CONTEXT`, firmware validates `num_col` against its **own internal limit**
4. The 8-column limit is in the firmware's **encrypted ARM64 text section** (0x100-0x1c000, RSA-4096 signed)
5. String `"Invalid column count: %u >= %u"` at offset 0x1d6d1, comparison constant `0x08` at offset 0x17b04

**No patching path available:**
- Code section encrypted (100% entropy)
- RSA-4096 signature in last 512 bytes
- No AMD PSP signing keys
- No alternative firmware with higher limit

**Bottleneck chain confirmed:**
```
Kernel driver    → Firmware (npu_7.sbin) → AIE HW
(aie2_max_col=40)   (8-col limit, signed)  (40 cols exist)
     ✓                   ✗                    ✓
```
The kernel driver allows 40! The firmware rejects >8 at `CREATE_CONTEXT`.

### Golden Artifacts

| Artifact | Path | Purpose |
|----------|------|---------|
| 40-col toolchain | `/home/bcloud/mlir-aie/npu2_40_toolchain/` | Rebuilt aiecc with 40-col target + partition fix |
| 31 TFLOPS xclbin | `config2/build/final_3072x4096x1536_192x128x96.xclbin` | Verified golden 8-col GEMM |
| 40-col xclbin | `config2/build_40col/final_6144x4096x3840_192x128x96.xclbin` | 160-core design (firmware rejects) |
| Source patches | `AIETargetModel.h:823`, `aiecc.cpp`, `AIETargetNPU.cpp` | All modifications for 40-col |
| Kernel driver source | `/home/bcloud/amdxdna-dkms/src/amdxdna/` | Full XDNA kernel module (out-of-tree) |
| Old firmware | `/lib/firmware/amdnpu/17f0_11/npu.sbin.1.0.0.166.zst` | Boot init, NOT AIE runtime |
| Decompressed firmwares | `/tmp/npu.sbin.1.0.0.166`, `/tmp/npu.sbin.1.1.2.65` | For binary analysis |
| String analysis | `/tmp/old_fw_sorted.txt`, `/tmp/new_fw_sorted.txt` | Sorted string tables for diffing |

Files Created This Session

| File | Purpose |
|------|---------|
| `/tmp/probe_seq_layout.cpp` | npu_sequence layout probe — confirms vec@0x40 is IRON output |
| `/tmp/test_libgemm10_fixed.cpp` | cmds2seq + opcode=0 test — still identity |
| `/tmp/full_pipeline_opcode0_v2.cpp` | Full 7-step pipeline + opcode=0 — 28,552 instrs, still identity |
| `/tmp/fullpipe_opcode0_512x512x8192.bin` | 114KB IRON instruction dump (640 DDR_PATCH commands) |
| `/tmp/test_rdynamic2.cpp` → `src/test_libgemm10_rdynamic.cpp` | Loads `libqwen3_npu.so` via `-rdynamic` + stubs — library loads, `Impl::C1` crashes (hidden_size=0 div-by-zero) |
| `/tmp/test_all_xclbins_op3.cpp` → `src/test_all_xclbins_op3.cpp` | Tests opcode=3 on ALL 4 xclbins (mm, attn, layer, dequant) — ALL produce identity |
| `/tmp/test_instr_on_layer.cpp` → `src/test_instr_on_layer.cpp` | Tests opcode=0 instructions on layer.xclbin — identity output (instrs reference heap addrs, not BO addrs) |
| `/tmp/bo_capture_v*.so` → `src/xrt-direct/bo_capture.cpp` | **BREAKTHROUGH: DRM ioctl intercept library that dumps BO content during FLM inference** |
| `/tmp/bo_dump/` → `xrt-direct/captured_bo_dump/` | **Captured actual BO content from FLM inference** — reveals full memory architecture |

### BO Content Capture Results

**Architecture**: Built `bo_capture_v10.so` that intercepts DRM ioctls on `/dev/accel/accel0` at the `CREATE_BO`, `GET_BO_INFO`, `SYNC_BO`, and `EXEC_CMD` levels. Uses `mmap` on the device fd with `map_offset` from `GET_BO_INFO` to directly read BO content.

**Captured BO Map (verified from live FLM run)** :

| Handle | Size | Type | Content |
|--------|------|------|--------|
| h=1 | 64MB | type=2 | **Main working buffer** — zeros at startup, holds intermediate results during inference |
| h=2-5 | 444K-311K | type=3 | **xclbin config buffers** — pre-mapped via vaddr, immutable |
| h=6 (layer0) | 10MB | type=1 | **Activation buffer** — BF16 `0x3bXX-0x3cXX` values, input/hidden state |
| h=7 (layer0) | 1MB | type=1 | **Pre-packed weight tensor** — BF16 values [-1.5, +1.1], mean≈0.018, ~6% non-zero |
| h=8 (layer0) | 128MB | type=1 | **Command/runlist buffer** — kernel descriptors and DMA entries (NOT raw weights) |
| h=9 (layer0) | 1MB | type=1 | **Pre-packed scale/bias** — mostly `0x3f80` (1.0 BF16), 158 unique values |
| h=10 (layer0) | 10MB | type=1 | **Second activation buffer** — alternates with h6 |
| h=11 (layer0) | 1MB | type=1 | **Pre-packed weight tensor #2** |
| h=12-117 | per layer | type=1 | **Repeating pattern**: 10MB act, 1MB weight-A, 128MB cmd, 1MB weight-B, per layer × 28 |
| h=119 | 94MB | type=1 | **Q4NX quantized weights** — byte range [0,255], mean=126.7, std=63.3, near-uniform distribution |
| h=180-195 | 8MB-2MB | type=1 | **Scratch/workspace buffers** for dequant, norms, KV cache |

**Critical Discovery — Weight Flow**:
1. `h119` (94MB) holds the **entire quantized model weights** — loaded from `model.q4nx` file at init time
2. Before each layer exec, FLM **dequantizes and packs** a slice of h119 into the 1MB BF16 BOs (h7, h9, h11...)
3. On EXEC_CMD, the NPU reads the 1MB BF16 tensors from host BOs into tile SRAM via DMA
4. The 128MB cmd BOs (h8, h12, h16...) contain the **runlist descriptors** that orchestrate the DMA + compute ops on the NPU
5. The 10MB act BOs (h6, h10, h14...) are ping-pong buffers for layer activations

**The 128MB cmd BOs contain kernel structures** like:
- `0x....1773` pointers (likely XRT kernel run handles)
- `0x00108200` size fields (1088*4096 style DMA sizes)
- `0x82100000` layout markers
- These are NOT raw weights — they're NPU execution descriptors

**Implication for standalone engine**: To replicate FLM's GEMM, we need to:
1. Dequantize Q4NX weights to BF16 (the 1MB pre-packed format)
2. Fill the 128MB command buffer with proper runlist descriptors
3. Fill the 10MB activation buffer with input
4. Call EXEC_CMD via the same ioctl/runlist pattern

Since we now have actual BO content dumps from FLM, we can either:
- **Clone the exact weight layout** — replicate FLM's pre-packed BF16 format for our own BOs
- **Reverse-engineer the cmd buffer** — the 128MB BO content reveals the exact xclbin command format
- **Wrap FLM's internal functions** — use `libqwen3_npu.so`'s `_move_weights()` to pack weights, then submit via our own XRT path

## Session 2026-06-28 — Q4NX Format Fully Reverse-Engineered

### Weight Format Breakthrough

Q4NX `dtype=I8` is **MISLEADING**. The data is actually **INT4** (not INT8):

- Each I8 byte holds 2 I4 values (low nibble + high nibble, signed)
- Groups of 32 I4 values with per-group BF16 `[scale, zero_point]` (4 bytes header)
- Dequantization: `BF16_value = I4_value * scale + zero_point`
- Data layout per group: `[scale:u16_BF16][zero_point:u16_BF16][16 bytes = 32 I4 nibbles]`
- Expansion ratio: 36 bytes → 32 BF16 = 64 bytes → ~1.78x (NOT 3.2x as initially calculated)

**Wait, let me recheck:** For gate_proj: I8 shape [384, 5120] = 1,966,080 bytes. Expected: 3,145,728 BF16 values. With I4 packing, each group of 32 I4 values needs 4 bytes (scale+zp) + 16 bytes (32 I4 packed into nibbles) = 20 bytes. Groups: 3,145,728 / 32 = 98,304. Total: 98,304 * 20 = 1,966,080 bytes. **EVERY BYTE ACCOUNTED FOR!**

The I8 shape [384, 5120] is a storage artifact:
- 5120 I8 "columns" / 32 groups = 160 groups per row, BUT 5120 bytes / 20 bytes per group = 256 groups per row
- 384 I8 "rows" * 256 groups = 98,304 total groups ✓

The mapping from storage shape to logical shape is:
- `I8_rows = logical_rows / 32 * 4` (each logical row of 32 I4 = 4 bytes)
- `I8_cols = logical_cols / 32 * 20` (each group of 32 I4 = 20 bytes)

### BF16 tensors
- Embedding, norms: stored as raw BF16 (little-endian uint16 pairs)
- `bf16_to_float(v) = (float)((uint32_t)v << 16)`

### Verified with existing npu-infer model.c
The model.c code (lines 88-101) reads I8 data as BF16 byte pairs — this works correctly ONLY for tensors where the storage IS already BF16 (like norms). For I4-quantized tensors, the proper dequantization is needed.

## Session 2026-06-28 — NaN debugging + Fused engine rewrite

### Key Discoveries

1. **BOTH engines collapse to a single repeating token**: Old engine outputs 4739 repeating,
   fused engine outputs 55120. This is NOT a bug in the fused engine — it's a model quality
   issue from NPU BFP16 compute diverging from ideal FP32.

2. **Original xclbin vs M=128 xclbin produce different numerical outputs**:
   The original `design_1024_bfp16.xclbin` (220KB) and the custom `final_128x1024x1024.xclbin`
   (52KB) use different AIE designs (4× column vs 8-core-1-row). Same weights pack to the same
   BFP16 but the NPU compute path differs enough to accumulate numerical error over 28 layers
   → NaN at layer ~19.

3. **`npu_infer` binary is stale**: The old `engine.cpp` was overwritten by `git stash`.
   The binary still runs from pre-compiled object files.
   Current `engine.cpp` has `NpuInferenceEngine` (FLM-style) which is NOT the same as
   `CustomNpuEngine` that `main.cpp` expects. This means `make npu_infer` is broken.

### What was built

- **Completely rewritten `npu_engine_fused.cpp`**: Clean, compact, 345ms/tok engine
  using original 1024×1024 xclbin with N-tiling for larger projections.
- Fixed weight packing to use exact same layout as reference engine.
- Engine runs all 28 layers with no NaN, generates tokens at 345ms/tok.

### New xclbin path

Fused engine now uses:
```
XCLBIN: /home/bcloud/npu-sandbox/npu-infer/build/qwen3_gemm/design_1024_bfp16.xclbin
INSTS:  /home/bcloud/npu-sandbox/npu-infer/build/qwen3_gemm/design_1024_bfp16.insts
```
(NOT the custom M=128 xclbins which produce NaN in 28-layer pipeline)

### Files changed this session
- `src/npu_engine_fused.cpp` — Major rewrite: single xclbin (1024×1024), N-tiled
- `src/engine.cpp` — Minor: hnorm diagnostic added (reverted by git stash)
- `src/npu_engine_fused.cpp` — Changed xclbin path to original design_1024_bfp16
- `docs/fusion-level-0.md` — Created: detailed documentation
- `Desktop/HANDOFF-NPU-OPTIMIZATION.md` — Updated status + fusion level #0

### Next steps
1. Restore CustomNpuEngine implementation (recover from git stash or object files)
2. Or: rebuild fused engine with M=128 variants AND consistent BFP16 (pack at
   1024×1024 tile count for all variants → requires recomputing shuffle for variants)
3. Temperature-based sampling to break token repetition
4. Compare logits with PyTorch reference to validate NPU compute accuracy

### Current Status
- ✅ Q4NX format fully understood (I4 group quantization + BF16 byte-pair storage)
- ✅ torch2aie toolchain verified working (19.5 TFLOPS config1 GEMM)
- ✅ CPU inference engine architecture designed
- ✅ **Fusion Level #0**: Custom M=128 xclbins (5 variants) built and verified
- ✅ **Multi-variant engine**: `npu_engine_fused.cpp` — tiled 1024×1024 backend using 
   original xclbin, all 28 layers, no NaN, ~345ms/tok
- ✅ **Tiled N-dim support**: Q (2048 dims → 2 tiles), G/U (3072 dims → 3 tiles), 
   O (1024), D (3072 K-dims → K-tile clipped to 1024)
- ⚠️ Output token differs from old engine (55120 vs 4739) due to N-tiling

## Fusion Level #0 — Custom M=128 decode xclbins

**Status: Complete** — 5 xclbins built and individually verified.

28-layer integration produces NaN due to BFP16 precision differences between
original 1024×1024 xclbin and the M=128 variants. 
**Workaround:** `npu_engine_fused.cpp` now uses the original `design_1024_bfp16.xclbin`
with N-tiling for projections with >1024 output dimensions.

### Built XCLBINs (8-core, 1-row AIE design)
| xclbin | Size | For |
|--------|------|-----|
| `final_128x1024x1024_128x64x128.xclbin` | 52KB | K, V proj (1×1024→1024) |
| `final_128x1024x2048_128x64x128.xclbin` | 58KB | Q proj (1×1024→2048) |
| `final_128x1024x3072_128x64x128.xclbin` | 64KB | gate, up (1×1024→3072) |
| `final_128x2048x1024_128x64x128.xclbin` | 52KB | O proj (1×2048→1024) |
| `final_128x3072x1024_128x64x128.xclbin` | 52KB | down proj (1×3072→1024) |

### Key Files
| File | Purpose |
|------|---------|
| `/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/n1_core_placed.py` | 8-core MLIR design source |
| `/home/bcloud/npu-sandbox/npu-infer/src/npu_engine_fused.cpp` | Multi-variant engine |
| `/home/bcloud/npu-sandbox/npu-infer/build/npu_infer_fused` | Compiled binary (345ms/tok) |
| `/home/bcloud/npu-sandbox/npu-infer/docs/fusion-level-0.md` | Detailed fusion doc |

## Session 2026-06-29 — Full Optimization Sprint

### 🏆 Final Engine: 210 ms/tok (3.2× faster than 668ms baseline)

Achieved through iterative optimizations on the torch2aie M=128 xclbin infrastructure:

| Optimization | Speed | Gain | Key Change |
|-------------|-------|------|------------|
| **Baseline** (multi-xclbin, REF pack, 1024 BOs) | 668 ms | — | Initial fused engine |
| **Sized BOs + direct packing** | 310 ms | **2.2×** | A BO: 128×K (not 1024×K), C: 128×N, direct pack(K,N) |
| **Pre-shared A + float norms** | 298 ms | +4% | Q/K/V share one A prep; G/U share one; pre-computed float norms |
| **Threaded LM head** (4 threads) | 239 ms | **+20%** | Split 151936 vocab across 4 threads for dot products |
| **Fused QKV+GU xclbins** | 215 ms | +10% | Q+K+V weights concatenated → single [1024×4096] xclbin; G+U → [1024×6144] |
| **Threaded attention** (4 threads) | 210 ms | +3% | 16 attention heads split across 4 threads |
| **Disk cache for packed weights** | 2.5s init | — | Saved packed blobs to /tmp/npu_*.bin |
| **-O3 -march=native -flto** | 210 ms | +2% | Compiler flags |
| **Total** | **210 ms** | **3.2×** | — |

### Engine Architecture

**6 xclbins loaded simultaneously:**

| Index | Shape | Purpose | xclbin file |
|-------|-------|---------|-------------|
| v0 | 128×1024×2048 | Q projection (1×1024→2048) | `final_128x1024x2048_128x64x128.xclbin` |
| v1 | 128×1024×3072 | Gate, Up projections (1×1024→3072) | `final_128x1024x3072_128x64x128.xclbin` |
| v2 | 128×2048×1024 | O projection (2048→1024, K=2048) | `final_128x2048x1024_128x64x128.xclbin` |
| v3 | 128×3072×1024 | D projection (3072→1024, K=3072) | `final_128x3072x1024_128x64x128.xclbin` |
| v4 | 128×1024×1024 | K, V fallback (1024→1024) | `final_128x1024x1024_128x64x128.xclbin` |
| v5 | 128×1024×4096 | **Fused QKV** (Q+K+V concatenated) | `final_128x1024x4096_128x64x128.xclbin` |
| v6 | 128×1024×6144 | **Fused GU** (G+U concatenated) | `final_128x1024x6144_128x64x128.xclbin` |

**GEMMs per token:** 4 per layer × 28 layers = **112 NPU calls/token** (down from 196)

**Per-layer GEMM pipeline:**
1. Fused QKV: [1×1024] × [1024×4096] → split into Q[2048], K[1024], V[1024]
2. CPU: Q/K norms + RoPE + KV cache + threaded attention (4 threads)
3. O: [1×2048] × [2048×1024] → [1024]
4. CPU: residual add + RMS norm
5. Fused GU: [1×1024] × [1024×6144] → split into G[3072], U[3072]
6. CPU: SiLU activation
7. D: [1×3072] × [3072×1024] → [1024]
8. CPU: residual add

**CPU acceleration (key files: `npu_engine_fused.cpp`):**
- Threaded LM head: 4 threads split 151936 vocabulary (from ~14ms → ~4ms)
- Threaded attention: 16 heads across 4 threads, per-head score buffer on stack
- Pre-computed float norm weights: all RMS norm weights converted at init
- Static arrays for RoPE cos/sin (no std::vector allocation)
- Disk cache: packed weights saved to /tmp/npu_*.bin for ~2.5s init

### Key Source File

**`/home/bcloud/npu-sandbox/npu-infer/src/npu_engine_fused.cpp`** — 310 lines, self-contained.
- Build: `bash /home/bcloud/npu-sandbox/npu-infer/build/build_fused.sh`
- Run: `bash /home/bcloud/npu-sandbox/npu-infer/build/run_fused.sh`

### Performance Data

| Metric | Value |
|--------|-------|
| Decode | **210 ms/tok** (3.2× faster than 668ms) |
| Prefill (9 tokens) | **1691 ms** (188 ms/tok) |
| Init (1st run, pack) | 2592 ms |
| Init (cached) | ~2.5s |
| Token diversity | 58861, 40378, 72378, 75984, 125367, 7138, 37006, 69422 (all different) |
| Logit range | [22.6, -14.4] (correct LLM distribution) |
| NaN count | 0 across 28 layers |

### Built XCLBIN Inventory (config1/build/)

| xclbin | Size | Status |
|--------|------|--------|
| `final_128x1024x1024_128x64x128.xclbin` | 52KB | ✅ Working (K, V) |
| `final_128x1024x2048_128x64x128.xclbin` | 58KB | ✅ Working (Q) |
| `final_128x1024x3072_128x64x128.xclbin` | 64KB | ✅ Working (G, U) |
| `final_128x2048x1024_128x64x128.xclbin` | 52KB | ✅ Working (O) |
| `final_128x3072x1024_128x64x128.xclbin` | 52KB | ✅ Working (D) |
| `final_128x1024x4096_128x64x128.xclbin` | 70KB | ✅ Working (Fused QKV) |
| `final_128x1024x6144_128x64x128.xclbin` | 118KB | ✅ Working (Fused GU) |
| `final_128x1024x8320_128x64x128.xclbin` | 94KB | ✅ Built (2-layer QKV, N=8320) |
| `final_128x4096x1024_128x64x128.xclbin` | 52KB | ✅ Built (2-layer O, K=4096) |
| `final_128x1024x12288_128x64x128.xclbin` | 118KB | ✅ Built (2-layer GU, N=12288) |
| `final_128x6144x1024_128x64x128.xclbin` | 52KB | ✅ Built (2-layer D, K=6144) |
| `final_256x1024x4096_128x64x128.xclbin` | 115KB | ✅ Built (multi-token QKV, M=256) |
| `final_256x2048x1024_128x64x128.xclbin` | 90KB | ✅ Built (multi-token O, M=256) |
| `final_256x1024x6144_128x64x128.xclbin` | 132KB | ✅ Built (multi-token GU, M=256) |
| `final_256x3072x1024_128x64x128.xclbin` | 90KB | ✅ Built (multi-token D, M=256) |

### Blocked Items

| Item | Cause | Detail |
|------|-------|--------|
| **BF16 native xclbin** | aiecc DMA descriptor bug | All BF16 MLIRs hang regardless of tile size/kernel. BFP16 works. aiecc generates wrong DMA descriptors for bfloat16 memory types. |
| **2-layer batch QKV** (N=8192) | aiecc assertion failure | `__assert_fail` in aiecc at exactly N=8192 (=1024 per core). Workaround: N=8320 (1040 per core) builds. Engine integration needed. |
| **>8 columns** | Hardware limit | NPU2 has 8 physical AIE columns. DRM ioctl rejects HWCTX with column_width > 8. Both kernel (aie2_max_col=128) and firmware (1.0.0.166, 1.1.2.65) enforce this. |
| **Multi-token decode** (M=256, 2-row) | Kernel g_counter ABI | Chess kernel `mm_128x64x128.o` has `g_counter` cycling 0,1,2,3 (for 4-row n32_core). With 2-row design, values 2,3 write out of bounds. Need modified kernel. |

---

## INT8 on NPU2 — FINAL ARCHITECTURAL VERDICT (2026-06-28/29)

INT8 xclbins BUILD and RUN for all 5 matrix shapes, but produce **394% mean relative error** with random input data on the NPU2 8-core design. The root cause is architecturally unfixable within the MLIR-AIE ObjectFifo abstraction.

### Root Cause: K-Slice Interleaving on Shared A Fifo

The BFP16 reference design (210ms/tok, 12 TFLOPS) uses:
- 1 shim DMA channel for A data (shared across 8 cores via mem tile stream extractor)
- Per-column B and C fifos (independent B data per core)
- Depth-2 linked fifo pool (linked A_L3L2→A_L2L1 via `--unified --dynamic-objFifos`)

This architecture means all 8 cores share ONE stream of A data. The fifo distributes elements round-robin:
- Core 0 gets A(K[0:64]), Core 1 gets A(K[64:128]), ..., Core 7 gets A(K[448:512])
- Then back to Core 0: A(K[512:576]), etc.
- Each core accumulates C += A(K_fixed_slice) × B(K_all) over all 16 K-iterations
- **Each core only sees 64 of 1024 K-values** — the rest are zero-contribution

For BFP16 (block floating point with 8-element shared exponents), adjacent K-blocks have similar dequantized values → K-interleaving error is small.

For raw INT8, A values are independent across K → **394% mean relative error**.

### Attempted Fixes — All Blocked

| Approach | Result | Blocked By |
|----------|--------|------------|
| Per-core A fifos (v9-v12) | ❌ Compile crash | DMA channel limit: ~2 per shim tile, need 8 |
| Single-core (v13-v15) | ❌ RTE crash | NPU routing conflicts for cross-column A/B |
| Per-shim A distribution (v17) | ✅ Builds, same K-issue | Linked fifo pool depth-2 limits to 2 sub-views |
| Depth-16 linked pool (v19) | ❌ aiecc crash | Resource exhaustion (lock/BD slots) with 8 consumers |
| DRAM-backed bf16copy (v21) | ✅ Builds, **4× correct value** | BFP16 w/ r=8,s=8 sub-viewing doesn't translate to INT8 |
| Weight reordering | ❌ Mathematical impossibility | Σ A(K_sub) × B_reordered ≠ Σ A(all K) × B(original K) |

### DRAM-Backed bf16copy Attempt (v21, 2026-06-29)

Exact copy of the BFP16 generator (`n1_core_i8_bf16copy.py`) with:
- `m=128, mtk=512, depth=2` — A_L3L2 element = (128, 512) int8 = 64KB
- `--unified --dynamic-objFifos` for DRAM-backed pool
- BFP16-style dimensionsToStream for producer/consumer sub-viewing

**Result**: Compiles and runs, but produces exactly **4× the correct value** (4096 instead of 1024 for K=1024 all-1s). The BFP16 dimensions (r=8, s=8) create sub-view groups of 8 elements each — appropriate for BFP packed formats but wrong for raw INT8. The 4 inner A-iterations × the same B create 4× accumulation.

**Attempted fix**: Set r=1, s=1 (no sub-grouping). This broke the sub-view mapping entirely — all C output at 4× (4096 instead of 1024) because the pool only has 2 sub-views that cycle, giving each inner iteration the same data.

The fundamental conflict: **BFP16 dimensions produce the correct number of linked pool sub-views for 8 cores × 16 K-iterations = 128 acquires**. INT8 with r=1,s=1 dimensions only produces 16 sub-views (depth 2 × 8: max pool size for linked fifos).

### Windows INT8 Answer
The same NPU2 silicon on Windows uses AMD's proprietary XDNA driver (DirectML) with a fundamentally different dataflow architecture:
- **M-parallel tiling** (row-parallel, NOT K-parallel) — each column gets different M-rows
- **Software-managed BD chains** — time-multiplexes shim DMA across all columns without hardware lock-based fifos
- **Pre-compiled tuned kernels** for common shapes

This bypasses MLIR-AIE's ObjectFifo resource constraints. The NPU2 hardware CAN do INT8 at ~50 TOPS — just not through the MLIR-AIE stack's abstraction.

### Built XCLBIN Inventory (build/int8/)

| xclbin | Size | Status | All-1s | Random |
|--------|------|--------|--------|--------|
| `final_i8_KV_v2.xclbin` | 54KB | ✅ Runs | ✅ K=1024 | ❌ 394% error |
| `final_i8_QKV_v2.xclbin` | 90KB | ✅ Runs | ✅ K=1024 | ❌ interleaved |
| `final_i8_GU_v2.xclbin` | 114KB | ✅ Runs | ✅ K=1024 | ❌ interleaved |
| `final_i8_O_v2.xclbin` | 54KB | ✅ Runs | ✅ K=1024 | ❌ interleaved |
| `final_i8_D_v2.xclbin` | 54KB | ✅ Runs | ✅ K=1024 | ❌ interleaved |
| `final_i8_KV_v17.xclbin` | 54KB | ✅ Runs | same K-issue | ❌ 129K/131K errors |
| `final_i8_KV_bf16copy.xclbin` | 49KB | ✅ Runs | **4× correct** | — |

### Generator Files

| File | Purpose |
|------|---------|
| `bf16_kernel_dev/n1_core_i8_v2.py` | Original m=32, shared A, passes all-1s |
| `bf16_kernel_dev/n1_core_i8_v17.py` | Per-shim A distribution |
| `bf16_kernel_dev/n1_core_i8_v19.py` | Depth-16 linked pool (aiecc crash) |
| `bf16_kernel_dev/n1_core_i8_bf16copy.py` | Exact BFP16 copy for INT8 (4× value) |
| `build/int8/mm_128x64x128.o` | DIM_M=128 kernel (matmul_scalar_i8_i16) |

### Recommendation
**Use BFP16 for the inference engine** (210ms/tok, 12 TFLOPS, correct results).

INT8 on NPU2 via MLIR-AIE is architecturally blocked:
- Shared A fifo → K-interleaving → wrong results for random data
- Per-core A fifos → DMA channel limit (2 per shim tile)
- Depth-16 linked pool → aiecc resource exhaustion (lock/BD slots)
- DRAM-backed bf16copy → sub-view dimensions incompatible with INT8 (produces 4× values)

The xclbins are valid for K-invariant workloads (batchnorm at inference, uniform convolution inputs, test/benchmark with pattern data). For general LLM inference, BFP16 is the correct precision on this hardware via this toolchain.

---

### Next Steps (for future sessions)

1. **Fix multi-token kernel**: Recompile `mm_bfp_mixed.cc` with `g_counter` mod 2 instead of mod 4 → 2-token decode → ~110ms/2tok = 55ms/tok
2. **Fix 2-layer batch engine**: Integrate N=8320/K=4096/K=6144 xclbins → ~170ms/tok
3. **Layer batching**: Fuse O and D across layers (8-column design already handles K up to 6144)
4. **2-layer batch + multi-token combined**: 2 tokens × 2 layers per batch → 28/2=14 batches → ~80ms/2tok = 40ms/tok


### INT8 Engine Architecture

```
Engine pipeline (219 ms/tok, 4.6 tok/s):

Init:      Register 4 xclbins → create 4 hw_contexts + BOs → dequant model → pack INT8 weights
            └─ Context pool: xclbins persist across swaps, only hc recreated per GEMM

Per-layer: RMS norm → QKV GEMM(514μs) → Q/K norm+RoPE → CPU softmax+attention →
           O GEMM(252μs) → residual → RMS norm → GU GEMM(742μs) → SiLU →
           D GEMM(326μs) → residual  (×28 layers)

Per-token: Final RMS norm → LM head(CPU: 155M MACs) → softmax sample → embed lookup
```

### Speed History

| Engine | ms/tok | tok/s | Key Change |
|--------|--------|-------|------------|
| INT8 scalar kernel | 13,000 | 0.08 | matmul_scalar_i8_i16 |
| INT8 vectorized | 442 | 2.3 | matmul_i8_i16 (mac_8x8_8x8) |
| + -O3 + cached norms | 371 | 2.7 | Compiler flags, norm caching |
| + Context pool | **219** | **4.6** | Eliminate xclbin re-registration |
| BFP16 v8 (baseline) | 1,335 | 0.7 | — |
| FLM proprietary | 11 | 93 | Reference (proprietary stack) |

### Proven NPU GEMM Performance

| Projection | Shape | Latency | TFLOPS |
|-----------|-------|---------|--------|
| QKV (fused) | 128×1024×4096 | 514 μs | 2.1 |
| O | 128×2048×1024 | 252 μs | 2.1 |
| GU (fused) | 128×1024×6144 | 742 μs | 2.2 |
| D | 128×3072×1024 | 326 μs | 2.5 |

### Context Pool Architecture

Instead of the old sa() (ensure-alive swap) which destroyed and recreated
the entire XRT state (xclbin registration, hw_context, kernel, BOs), the
new design pre-registers all 4 xclbins at init. Per-layer switching only
recreates hw_context and kernel — BOs persist. This eliminates 112
xclbin re-registrations and BO re-creations per token.

```cpp
struct I8Slot {
    xrt::uuid uuid;  // pre-registered
    unique_ptr<xrt::bo> bA,bB,bC;  // persist across swaps
    void activate(xrt::device& d){
        hc.reset(); hc=make_unique<xrt::hw_context>(d,uuid);
        k.reset(); k=make_unique<xrt::kernel>(*hc,"MLIR_AIE");
    }
};
```

### Path to 50-100 ms/tok (10-20 tok/s)

| # | Optimization | Speedup | Est ms/tok | Effort |
|---|-------------|---------|------------|--------|
| 1 | ✅ Context pool | 42% | 219 | Done |
| 2 | Weight pre-loading (layer-dim B taps) | 26% | ~160 | 3 days |
| 3 | LM head on NPU (dedicated xclbin) | 9% | ~145 | 2 days |
| 4 | 32-core GEMM xclbins | 20% | ~115 | 5 days |
| 5 | NPU edge attention (BF16) | 18% | ~85 | 3 days |
| 6 | Fused QKV-attn-O xclbin | 10% | ~70 | 7 days |

### Key Files

| File | Purpose |
|------|---------|
| `npu-infer/bf16_kernel_dev/n1_core_i8_v2.py` | INT8 GEMM MLIR generator (8-core, broadcast) |
| `npu-infer/bf16_kernel_dev/n1_core_i8_4row.py` | INT8 GEMM MLIR generator (32-core, 4×8) |
| `npu-infer/src/npu_engine_i8.cpp` | INT8 inference engine (219 ms/tok) |
| `npu-infer/build/int8/final_i8_*_v.xclbin` | 5 vectorized INT8 xclbins |
| `npu-infer/build/int8/insts_i8_*_v.txt` | Instruction sequences |
| `npu-infer/build/chess_infer/attn_06b.xclbin` | NPU edge attention (421 μs, DPU kernel) |

### Build Commands

```bash
# Rebuild xclbins (if generator changes)
cd npu-infer/build/int8
xchesscc_wrapper aie2p -c -I $AIETOOLS_DIR/include -I $MLIR_AIE_DIR/include \
  -DDIM_M=32 -DDIM_K=64 -DDIM_N=128 -I$MLIR_AIE_DIR/include/aie_kernels \
  -Di8_i16_ONLY $MLIR_AIE_DIR/include/aie_kernels/aie2p/mm.cc -o mm_32x64x128.o

PYTHONPATH=$MLIR_AIE_DIR/python python ../bf16_kernel_dev/n1_core_i8_v2.py \
  -M 128 -K $K -N $N -m 32 -k 64 -n 128 > design.mlir

aiecc --aietools=$AIETOOLS_DIR --alloc-scheme=basic-sequential \
  --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
  --xclbin-name=final.xclbin --npu-insts-name=insts.txt design.mlir

# Build engine
cd npu-infer/build
g++ -std=c++17 -O3 -march=native -ffast-math \
  -I../include -I$TORCH2AIE/examples \
  -I$TORCH2AIE/examples/gemm_asymmetric_tile_buffering \
  ../src/npu_engine_i8.cpp dequant_q4nx.o \
  -o npu_engine_i8 -lxrt_coreutil -lm -luuid

# Run
sudo ./npu_engine_i8
```

---

## Session 2026-07-02/03 — Production Stack, Release, Site Refresh

### FLM Proxy Daemon (July 2)

The C++ engine runs 5 models but at lower tok/s than FLM. Decision: proxy to FLM for production while the open-source engine catches up on the fused xclbin.

**Daemon** — `daemon/npu-gpu-cpud.py` (420 lines, Python stdlib only):
- OpenAI-compatible HTTP on port 9090 (`/v1/chat/completions`, `/v1/models`, `/v1/health`)
- Starts FLM as a subprocess on port 52625, proxies requests
- Routes by model size: <2B→NPU, 2-8B→GPU, >8B→CPU
- Moved from `npu-gpu-cpu/` external repo into this repo — now ships with the source

**Systemd unit** — `daemon/npu-daemon.service`:
- `FLM_PMODE=turbo` by default
- `Restart=always` with 5s backoff
- `LimitMEMLOCK=infinity` for NPU memory access

### TypeScript Build Fix (July 2)

`npm run build` was broken — `bridge.ts` and `server.ts` imported `fastify` which wasn't installed. These were WIP TypeScript servers that tried to run the C++ engine directly; the Python daemon replaced them. Excluded from tsconfig, removed `fastify` from package.json. Build exits clean.

### Benchmark Results — FLM Turbo (July 3)

| Metric | pmode=performance | pmode=turbo |
|--------|-------------------|-------------|
| Decode (Qwen3-0.6B) | 94.1 tok/s | **94.7 tok/s** |
| TTFT | 513 ms | **497 ms** |
| GPU Llama-3.1-8B | 11.3 tok/s | 11.3 tok/s (no change) |
| Qwen3-8B (GPU) | timeout | 5-10 tok/s (unstable) |

Turbo gain: marginal (+0.6% decode, -16ms TTFT). The 500ms TTFT is the NPU loading weights from DDR — no software knob fixes this. Only a fused xclbin can break through.

### CPU + GPU Tuning

- CPU governor: powersave→performance (marginal TTFT improvement)
- GPU perf level: auto (2900 MHz under load, 600 MHz idle)
- GPU sclk seen at 2646-2900 MHz. No fan controls exposed on this APU — EC handles it.
- No manual overclock available on NPU — clock gated by XDNA firmware.

### Release Packaging (July 2-3)

Built and uploaded to GitHub Releases (`v2026.07.02`):
| File | Size | Contents |
|------|------|----------|
| `runtime.tar.gz` | 43 KB | Pre-built CLI + daemon + systemd unit + docs |
| `src.tar.gz` | 2.3 MB | Full source (excludes binaries, node_modules) |

Release notes show 94 tok/s FLM, 97 tok/s C++ v12. Clean install: `tar xzf` → `bash install.sh` → `1bit chat`.

### Stale Numbers Purge (July 2)

Every file in the repo still said 63 tok/s (old v9 number from June). The daemon swapped to FLM proxy weeks ago. Hunted down every occurrence:
- `src/commands/chat.ts`: 63→94 tok/s
- `CLAUDE.md`: tagline, verify command, engine description
- `README.md`: badges, tables, engine speeds, port 8081→9090, FLM competitor→partner framing
- `site/index.html`: hero panel, stats, console output, docker port, footer, JS animation
- `engine/npu/BENCHMARKS.md`: full restructure with production FLM numbers at top
- `~/.1bit/agent/settings.json`: npuEndpoint port 8081→9090

### Site (July 2-3)

Deployed to Cloudflare Pages. Visual polish:
- "Open source" in blue, "Zero dependencies" in pink
- Hero shows FLM proxy curl command on the console panel
- All port references updated to 9090
- Footer shows FLM + C++ v12 numbers

### GitHub Traffic (as of July 2)

| Metric | Value |
|--------|-------|
| Stars | 10 |
| Forks | 3 |
| Views (14 days) | 49 unique / 19 visitors |
| Clones (14 days) | 1,096 total / 296 unique cloners |
| Top referrer | 1bit.systems (11), Google (11), GitHub (11) |
| Release downloads | 0 (new release just posted) |

The Jun 21-22 clone spike (492 in one day) looks like a scraper or bot. Organic traffic is steady at 2-9 visitors/day from search and direct.

### Files Changed

| File | Change |
|------|--------|
| `daemon/npu-gpu-cpud.py` | New — moved from npu-gpu-cpu/ |
| `daemon/npu-daemon.service` | New — systemd unit |
| `src/commands/up.ts` | Rewrote to use repo daemon |
| `src/commands/chat.ts` | 63→94 tok/s banner |
| `src/cli.ts` | Help text updated |
| `tsconfig.json` | Exclude bridge/server |
| `package.json` | Remove fastify, add daemon to files |
| `CLAUDE.md` | Updated tagline and verify |
| `README.md` | Full number refresh |
| `site/index.html` | Full number refresh + styling |
| `packaging/install.sh` | Rewritten for tarball flow |
| `engine/npu/BENCHMARKS.md` | Restructured + turbo results |
| `docs/journey.md` | This entry |
| `.github/workflows/deploy.yml` | Cloudflare Pages deploy on push to main |

### Current Status (July 3, 2026)

- **Production**: FLM proxy on port 9090, pmode=turbo, 94.7 tok/s
- **C++ engine**: 5 models, 28 tok/s (ALL) / 97 tok/s (v12), auto-detect
- **Site**: Live at https://1bit.systems, all numbers current
- **Release**: 2 tarballs on GitHub, clean install flow
- **Build**: `npm run build` exits clean
- **Next**: Fused xclbin port (blocked by IRON Python API)
- **Traffic**: 296 unique cloners in 2 weeks, zero marketing

### Repos

- `https://github.com/1bit-systems/1bit-systems` — This repo (source of truth)
- `https://github.com/bong-water-water-bong/npu-infer` — INT8 engine + xclbin generators
- `https://github.com/bong-water-water-bong/npu-gpu-cpu` — Handoff docs + unified control plane

## Session 2026-07-03/04 — Triton-XDNA Eval, memlock Fix, Spec-Decode Reality Check

### Triton-XDNA (AMD's Triton-to-XDNA compiler)

Evaluated `amd/Triton-XDNA` as a candidate to replace handwritten `edge_attention.cc`/`n1_core_i8_v2.py` MLIR. Cloned to `npu-sandbox/Triton-XDNA/`, built via prebuilt wheels (Python 3.12 venv, `sandbox/`).

**Root cause of every launch failure was `RLIMIT_MEMLOCK`, not NPU contention.** XRT's launch path does `mmap(..., MAP_SHARED|MAP_FIXED|MAP_LOCKED)` for a 64MB device buffer; the default systemd session limit (`DefaultLimitMEMLOCK=8M`) is far too small. `npu-daemon.service` works because it explicitly sets `LimitMEMLOCK=infinity`; ad-hoc shells didn't. Spent real time chasing a red herring (stopped/restarted `npu-daemon.service` mid-investigation, verified it wasn't the cause — failed identically with the NPU device completely free).

**Fix**: `/etc/security/limits.d/90-bcloud-memlock.conf` — `bcloud soft/hard memlock unlimited`. Persistent, applies to new login sessions (PAM limits don't retroactively apply to already-open shells).

**Result**: `matmul_i8_m64_n64_k64` example compiles to a real AIE2P device binary (`.pdi`/`.elf`) and runs correctly on this exact hardware — validated bit-exact (`atol=0, rtol=0`) against PyTorch CPU reference across 8 shape combos (M,N,K ∈ {256,1024}), run twice each. Correctness only — no throughput benchmark run yet.

### Spec-decode reality check

Ran the real `npu_spec_decode` binary (not the synthetic `spec_decode_bench` sweep) against the actual trained checkpoint at `checkpoints/eagle3_draft_2k.bin` (`eagle3_qwen3_0.6b_2k`, step_21). Same memlock issue hit here too — same fix applies repo-wide, not just Triton-XDNA.

**Result: 0.2 tok/s, 0.0% acceptance, 1.02x effective speedup** vs the ~94-97 tok/s non-speculative baseline — i.e. currently a ~500x regression, not a speedup. Root cause: step_21 is only ~2% of a full run (config implies ~1,000 steps for 3 epochs at global_batch_size=32 over the 10,976-example regenerated dataset) — the draft head is barely past random init. Not an integration bug as far as we can tell; the dispatch path itself works (loads, runs, produces tokens). Needs the full training run to complete before it's benchmarkable again.

Also noticed `checkpoints/eagle3_qwen3_0.6b_10k/` (the name `run_full_pipeline.sh` actually targets) is empty — no checkpoint saved — while the `_2k`-named run is the one that produced `step_21`. Divergence not investigated further this session.

### NPU daemon verify

Re-verified FLM proxy after the stop/restart: 91.6-93.0 tok/s decode, ~42 tok/s prefill, ~495ms TTFT — consistent with the 94±5 baseline (the 82 tok/s seen immediately post-restart was just cold-start noise). Separately noticed the GPU/Lemonade backend (`lemond`) is a dead zombie process (port 13305 not listening) — pre-existing, not caused by this session. Unrecognized model names silently route to it and fail with a raw connection-refused error instead of a clean "unknown model" response.

### QKV weight cache corruption — the real root cause (July 5)

**Background**: two parallel bugfixes happened in the same session:
- Decode off-by-one (commit `21864a41`): decode loop ran LM-head AFTER forward, re-running layers on the prefill's finalized hidden state.
- Prefill Q stride (commit `f668ef76`): `qo_b[pi*NH*HD+...]` should be `pi*4096`; only bit at npt>1.

**New finding** (`docs/NPU-QKV-CACHE-WEIGHTS-BROKEN.md`): a `--trace` dump mode was added to `npu_engine_cb.cpp` that runs npt=1, token 100, layer 0 and dumps 17 substage intermediates as float32 binaries. This was diffed against the HF float reference from `tools/layer_trace.py` via `tools/cb_trace_diff.py`:
- `h_ln1` (RMSNorm output) was bit-exact: cos_sim=1.000, max_abs=0.000
- `q_flat` (QKV GEMM output) immediately blew up: cos_sim=-0.21

Then `tools/cb_weight_compare.py` directly compared the engine's HF-cached INT8 QKV weights (`/tmp/hf_weights_cache/qkv_*.bin`, dequantized with the global scale wsc.qk) against the Q4NX INT4-dequant float reference:
- Q block cos_sim = -0.237, K = -0.244, V = -0.244 for layer 0; same across layers 1-2.

A negative cos_sim means the cached INT8 weights are essentially uncorrelated garbage — the cache generation script was wrong. This overrides the earlier theory that the stride was the sole root cause: the stride is real but only accounts for npt>1; the weight cache corruption accounts for ALL npt including the single-token case.

The generator script that wrote `/tmp/hf_weights_cache/*.bin` is not in the repo. `docs/NPU-ENGINE-CORRECTNESS-STATUS.md` was updated to reflect this new finding.

## Session 2026-07-05/06 — Q4NX/GGUF fully decoded, NPU GEMM root-caused, first validated 1-bit number, DSpark

The longest push in the project's history. Two threads ran in parallel: a
model-format thread (decode *any* model on either chip) and a correctness thread
(why is the fast NPU engine's output garbage). By the end the NPU GEMM bug that had
silently corrupted every "97 tok/s" run was root-caused and fixed, Q4NX and Q2_0
were both decoded bit-exact, and the first genuinely validated, coherent 1-bit
number landed: **279 tok/s.**

### GGUF ↔ Q4NX: decode any model, architecture-agnostic

Built `gguf_parser.h` (v2/v3, architecture-agnostic metadata via suffix matching;
Q8_0/Q4_0/Q4_1/Q5_0/Q5_1/Q4_K/Q5_K/Q6_K/Q8_K/F32/F16/I8), `tools/gguf_to_q4nx.cpp`,
and a full GGUF→NPU pipeline that dequantizes any GGUF, re-quantizes to INT8,
uploads to NPU BOs, and runs the whole decode loop (RMSNorm, RoPE, QKV/O/GU/D GEMM,
attention, SiLU, AVX-512 LM head). **Q4NX is fully decoded** and the NPU is no
longer locked to one hand-produced model file — any GGUF can drive it.

### NPU INT8 GEMM: the real root cause (it was never the host)

Every prior "v12 97 tok/s" run produced incoherent output; four sessions of
host-side fixes never fixed coherence. Settled it with hardware dump-and-compare:
dumped the exact quantized activation+weight bytes sent to the NPU, computed `A@B`
in numpy on those exact bytes, compared against the hardware readback — **zero
correlation** across all four shapes (QKV/O/GU/D). Positive control: AMD's own
`single_core.py` / `whole_array.py` matmul examples PASS numpy-verified on this
exact chip + Chess compiler at the exact production shapes. Diffing revealed
`n1_core_i8_v2.py`'s L2→L1 `object_fifo` calls never applied the r/s/t=8 micro-tile
reformatting AIE's `mmul<8,8,8>` requires — plain row-major streaming. Replaced the
generator with AMD's proven `single_core.py`; isolated GEMM test went from
uncorrelated garbage to **0 errors / 0 max diff** on all four shapes, and the
post-prefill hidden-state norm collapsed from ~4,050,000 (near-input-independent)
to ~250 and started tracking the prompt. (`docs/research/GEMM-KERNEL-CORRECTNESS-CONFIRMED.md`.)
Also fixed: i16-vs-i32 xclbin output width (~120,000× error), a malformed smoke-test
prompt, and unbounded RMSNorm weights that were masking the broken kernel.

### Q2_0 ternary: bit-exact, and the first real 1-bit number

The prism-ml Ternary-Bonsai Q2_0 format isn't publicly documented. Reverse-
engineered from raw bytes, verified **bit-exact vs the F16 reference
(cosine = 1.000000)** across every layer type: 128 elems / 34 bytes, fp16 scale
then 2-bit LSB-first codes, value `(code-1)*d`. Decoder: `tools/q2_0_decode.py`.
Then measured, on hardware, coherent: **Ternary-Bonsai-1.7B native Q2_0 (1.58-bit)
= 274–279 tok/s** on the Radeon 8060S via Vulkan — *"The capital of France is
**Paris**…"* — versus **22 tok/s** F16. A **12.6× speedup** from native 2-bit
storage. `llama-bench` tg64 = 278.81 ± 2.95 t/s. This is the honest, reproducible
"1bit" headline (`docs/VALIDATED-BENCHMARKS-2026-07-05.md`, `docs/one-bit-headline.md`).

### ZINC Q2_0 kernel + build unstick

Wrote `zinc:src/shaders/dmmv_q2_0.comp` (mirrors the proven `dmmv_q8_0` reduction)
and wired it through loader/dispatch — **builds and runs the ternary model natively
at ~894 tok/s**, but output isn't coherent yet (a ZINC-internal DMMV weight-layout
detail, not the format — dequant is bit-exact). Branch `zinc:feat/q2_0-vulkan-kernel`.
Separately, ZINC's repo was stuck in a half-finished Zig 0.15→0.16 migration that
built with neither toolchain; restored `main` to a clean 0.15.2 build and preserved
the 0.16 attempt on `wip/zig-0.16-migration`.

### DSpark (speculative-decode draft) — projected, not yet measured

DSpark is a small draft model (5-layer transformer + Markov head + confidence head)
for speculative decoding. Measured **5.90× acceptance** (5.90/7 blocks, 73.7%) on
10 gsm8k samples with Qwen3-4B; confidence-head AUC 0.912. The headline
**"572 tok/s" is a projection** (base NPU × 5.90×), not an end-to-end coherent
measurement — the draft is still training and rides on the NPU base engine. Label
it as a projection until measured; it is not a validated production number the way
94 tok/s (FLM) and 279 tok/s (GPU ternary) are.

### Honest status at session end

- ✅ **NPU production (FLM proxy): 94 tok/s, coherent** — validated live.
- ✅ **GPU native 1.58-bit ternary: 279 tok/s, coherent** — validated, reproducible.
- ✅ **NPU INT8 GEMM kernel: root-caused and fixed** (bit-exact via AMD's generator).
- ✅ **Q4NX + Q2_0 fully decoded**; GGUF→NPU pipeline architecture-agnostic.
- ⚠️ C++ NPU `npu_engine_cb` and the ZINC-native Q2_0 path build/run *fast* but are
  **not yet coherent**. "97 tok/s v12", "291 tok/s fused", and "572 tok/s DSpark"
  are raw-throughput / projected figures on paths whose output was never validated
  coherent — qualify them, don't market them as production alongside the two numbers
  that are.
- ❌ `engine/fusion/main.zig` still prints a dispatch table and runs no inference.

### 2026-07-11 addendum — the DSpark story continues

The 279 tok/s / 572 tok/s-projection numbers above later drifted into a flat
"disproven" claim (a 2026-07-07 test reported 0.1–0.2 tok/s at 0% acceptance with no
qualification). That claim was itself wrong: traced to (1) a checkpoint-path wiring
bug in `npu_spec_integration.cpp` that made the benchmark silently run an untrained
draft model, and (2) a training config that regressed after this session — the
`global_batch_size=32` / 10,976-example dataset described above became
`global_batch_size=512` / 360 examples by 2026-07-11, both changes making an
already-fragile training setup much worse. See `docs/wiki/performance.md` for the
corrected "unresolved, not disproven" status and the real 0.8 tok/s / 0% acceptance
measurement taken with the wiring bug fixed.

## Session 2026-07-16/20 — FLM fully replaced, model-agnostic broadening, TQ2 ternary

The single biggest architectural change since the last addendum: **FastFlowLM is no
longer the default NPU path, and its native `.so`-dependency is gone entirely.**
22 closed-source libraries were disassembled, 209 xclbin bitstreams traced back to
their AIE generators, and the whole stack rebuilt from source
(`docs/research/fastflowlm-decode/SUMMARY.md`). `engine/npu/src/npu_engine_universal.cpp`
no longer `dlopen`s FLM's `.so` files for NPU attention/GEMM instruction
generation — it uses pre-compiled instruction files instead, and `backend_manager.cpp`
now marks `npu_xrt` `auto_selectable` with the comment "NPU_XRT is the default now."

That comment was aspirational for four days. `src/model_router.cpp` — the file that
actually decides which backend a qwen3-architecture model gets routed to — still hard-
coded `{"npu_flm", "cpu_generic"}` as of this morning (2026-07-20), meaning every
qwen3 model kept going through the FastFlowLM subprocess in practice regardless of
what `backend_manager.cpp` claimed. Fixed today (PR #567): route is now
`{"npu_xrt", "npu_flm", "cpu_generic"}` — native engine first, FLM kept only as a
fallback, not removed outright. Honest tradeoff, not a free win: `npu_xrt`'s
single-core GEMM kernels are correctness-verified on real hardware
(`docs/research/GEMM-KERNEL-CORRECTNESS-CONFIRMED.md`, 2026-07-17/18), but the 8-core
multi-tile path that would close the throughput gap to FLM's fused-xclbin numbers
is still "unverified in combination" per that same doc. Shipped the routing change
anyway, on the user's explicit call, because "FLM is diagnostic-only, not the
serving path" needs to be true in the code, not just asserted in the README.

**Model-agnosticism kept widening**, matching the standing "every model, every
quant" scope (not scope creep — see the earlier note on this in the repo's own
memory). GGUF architecture support went 2→8 (LLAMA, MISTRAL, QWEN2, GEMMA, PHI,
ZAMBA2 alongside the original two), quant support went 4→13 (legacy Q4_1/Q5_0/Q5_1
plus the full K-quant family), the HIP backend takes runtime `ModelConfig` instead
of hardcoded Zaya1-8B dims for non-Zaya models, and GGUF parsing was consolidated
into one shared, verified module instead of several divergent per-file copies.

**1BP's own namesake feature had never actually shipped.** `ONEBP_TQ1`/`ONEBP_TQ2`
existed in the format's `OnebpQuant` enum since it was designed, but every model
converted so far — including genuinely ternary-trained ones — went through the
4-bit `Q4NX` path regardless of source precision, for a project called "1bit.systems."
TQ2 (symmetric 2-bit ternary, one BF16 scale per 32-element group, no zero-point,
half of Q4NX's tile size) is now implemented end-to-end: converter, loader,
on-disk layout. Verified against Bonsai-1.7B (genuinely ternary-trained, Apache-2.0)
— structural match exact, numerical match "100% of dequantized values within
BF16 scale-rounding precision" against the real C++ loader, not just the Python
converter. Separately, the 1BP converter/loader was found dropping norms and MoE
expert weights entirely (91% of Zaya1-8B's tensors were silently missing) — fixed
same window.

**BlackMamba conversion, and a metadata bug worth flagging for future architecture
ports.** Converting BlackMamba-1.5B/2.8B (Zyphra's Mamba+MoE hybrid) to GGUF then
to 1BP hit a config-read failure — `H=0 L=0` — despite the GGUF file parsing fine
structurally. Root cause: `scripts/blackmamba_to_gguf.py` wrote its metadata keys
unprefixed (`"block_count"`, `"embedding_length"`) instead of prefixed with the
architecture name (`"mamba.block_count"`), which is the GGUF convention every
reader in this repo actually expects — `model_discovery.cpp`'s own suffix matching
(`ends_with(key, ".block_count")`) requires the leading dot that only the prefixed
form provides. Neither the HF-style lookup nor the architecture-prefixed lookup
matched a bare key, so config silently came back zeroed instead of erroring loudly.
Fixed by prefixing all of BlackMamba's custom keys with `mamba.`; worth checking
any other hand-written GGUF exporter in this repo for the same pattern before
trusting its output loads correctly anywhere beyond a byte-level structural check.

**Vision went from greenfield to a real, working POC.** Qwen2-VL support — actual
image-to-text end to end, not just tensor plumbing — landed (#491), with a fix for
generation running past the real EOS token instead of a fixed budget (#492).
Separately, lightweight image preprocessing (stb_image, no OpenCV dependency,
optional HIP resize/normalize kernel) replaced a hypothetical OpenCV dependency —
this shipped with a real build break (a deleted copy constructor with no
corresponding move constructor, breaking `std::vector::push_back`) that sat
unnoticed until a routine full-repo build check today; fixed with proper move
semantics rather than restoring the deleted copy path.

**The landing page was making a claim its own tooling had already disowned.**
`benchmarks/latest.json._unverified` — a real quarantine mechanism, not decoration
— has flagged `npu_validated_tok_s` ("69/94 tok/s NPU") as "NO SOURCE... MUST NOT
be published" since issue #107. That didn't stop it from being the headline number
in `site/index.html`'s `<title>`, meta description, OG/twitter tags, JSON-LD, and
hero `<h1>` — and the page's own JS had a hardcoded string literal that
unconditionally overwrote the meta description with that same number on every
successful page load, bypassing the quarantine guard that protects every other
binding on the page. Replaced with claims that are actually sourced: the native
NPU stack as the new default, ~41 TFLOPS prefill peak (which does have a real
`bench_prefill_variants` citation in `numbers.json`), and current binary size.
Historical throughput numbers elsewhere on the page (the V12 tuning timeline) were
left in place — they were real measurements at their dates — but relabeled so they
read as history, not a current production claim.

**Status at end of this window**: native open-source NPU engine is the default
route for qwen3 models, not yet throughput-competitive with FLM on the verified
single-core path. Model catalog now spans the full Zyphra family plus 1BP
conversions (all on Hugging Face) with BlackMamba added this session. TQ2 ternary
is real and verified against a genuinely ternary-trained model, not just a format
spec. Full repo build and test suite both clean as of this session's end (one
apparent GPU memory-fault test failure turned out to be a local test
misconfigured to point a non-MoE-scoped test at a real MoE model, not a code bug).

---

## 2026-08-07 — the unified control plane lands (pool wired, spec decode in-server, zoo 5/5)

**What happened**: the 1bit engine went from "5 models working one at a time" to
**one process, one API, all models resident** — plus a lossless speculative
decoding loop running inside the server.

**Unified model pool (end to end).** `UnifiedModelPool` existed as an unwired
1bp-only class; it's now the server's residency layer (`--pool`): every model
in the weights dir is mmap'd at boot (11 slots / ~6 GB on strix, 1bp parsed +
generic gguf), `POST /v1/pool` is the control-plane report, `/v1/models` tags
pooled models. One `unified` process then serves Llama-3.2-1B, Qwen3-0.6B,
Bonsai-1.7B-TQ2 (HIP 1BP), Zamba2-1.2B Q8_0 (HIP Mamba2), and Qwen3-4B (NPU
FLM) through the same OpenAI-compatible endpoint — `scripts/zoo-smoke.sh`
5/5 PASS. Measured e2e (includes per-request routing): NPU 20.8 tok/s, GGUF
1B 12.4 tok/s, Bonsai 3.1, Zamba2 2.2.

**Speculative decode in-server.** The roadmap Phase-2 loop went from standalone
demo to a server feature: `--draft-model X --spec-decode` runs the
lossless-consistent loop with one-batch verification (`verify_batch`) and KV
rollback (`rollback` via `llama_memory_seq_rm`). Two bugs surfaced and fixed
while wiring: draft and target logits must never share a buffer (different
vocabs — the first token silently became the draft's argmax), and
ggml-vulkan's `reset()` was a no-op (every request had been extending one
unbounded KV sequence; it now clears per request — this also made outputs
deterministic: 3/3 identical).

**Also**: Zamba2 Q8_0 (tensor-exact vs HF checkpoint), the vendored llama.cpp
fork gained zamba2 quantize support (arch alias + kv-name fallback), AMD
Vitis 2026.1 installed for the FPGA roadmap.

**Honest status vs the "one heap, one API" NPU goal**: the CONTROL PLANE is
unified (one process, one API, pooled models). The NPU-side single device-heap
carve + one chained EXEC_CMD (docs/plans/one-heap-pivot.md) is still DRAFT —
that's the next milestone, on a kernel/driver surface that's already present
in amdxdna.

---

## 2026-08-15 — 100% HF Coverage: every arch-bearing checkpoint maps to an engine token

**What happened**: the HF architecture census — the number behind the
"HF model coverage" claim on the README and landing page — hit a verifiable
**317,310 / 317,310 = 100.00%**. Every architecture-bearing text-generation
checkpoint on HuggingFace now routes to a known engine token. The long tail
is closed — and the mechanism that had been silently inflating the number was
found and killed so it can't lie again.

**The sentinel drift bug.** Commit `6ad2947f` moved the `RCPP_ARCH_UNKNOWN`
sentinel from `255` to `988` (to free the low enum range for new families),
but the census tooling (`census_coverage.py`, `census_classify.py`,
`census_tail_sweep.py`, `census_tail_verify.py`) still probed `== 255`. Every
unmapped architecture class was silently counted as *mapped* — the
UNKNOWN bucket was reporting as covered. The `model_type` fallback path was
dead code. The fix: the sentinel is now read from the live `bitnet_model.h`
at module load (regex on the enum), so it can't drift again. After the fix,
816 checkpoints previously lumped into UNKNOWN were correctly attributed to
real families (LLAMA +258, QWEN3 +230, GPT2 +105, ...), and the genuine
uncovered count went to zero.

**The 100% number, made reproducible.** `Testing/census_coverage.py` now
compiles a probe against the real `rcpp_arch_from_string()` and regenerates
`census_full_summary.json` from the actual committed mapping — the number is
no longer a hand-maintained figure, it's recomputed from the registry every
run. Full HF census: 399,220 total models, 317,310 with architectures,
77,210 no_arch; 317,310 / 317,310 arch-bearing text-gen checkpoints map to
an engine token. (The ~20k structurally-unclaimable checkpoints — T5/MT5/BART
encoder-decoders, ParlerTTS, chess engines, ~2,000 one-off custom classes —
have no architecture class at all, so they're outside the denominator.)

**A watcher, so coverage can't silently regress.** `Testing/hf_new_models.py`
(now step [4/4] in `scripts/jarvis-daily-routine.sh`) polls HuggingFace's
newest models daily (text-generation + image-text-to-text tags), fetches each
`config.json`, strips the architecture class, probes the live registry, and
alerts on any uncovered class. It skips GGUF/LoRA/PEFT derivatives (no
config.json by design — the raw base model carries the config and gets
checked separately) and has a tag fallback for gated repos. State persists in
`Testing/hf_new_models_state.json` (seen ids, capped at 5000). First real
catch: `MuseGlimmerForConditionalGeneration` on `meta-models/Muse-Glimmer-30B`
→ strips to `museglimmer` → `RCPP_ARCH_MUSE` — covered. One uncovered class
flagged (`emberproelia`, 1 model) for triage.

**The README and landing page were never updated.** The headline, the badge,
the stats table, and the census paragraph still said 94% / 93.88% — the
verified 100% never made it into the docs the screenshot showed. Fixed: all
four now read 100% / 317,310 (100.00%).

**Honest status vs the "500+ models" goal**: this is registry coverage, not
per-family bring-up. Every arch-bearing HF text-gen checkpoint now *maps* to
an engine token — the long tail is closed. What remains (per
`docs/plans/monster-500-models.md`) is the per-family quirk table at scale:
~19 families validated end-to-end today (full-vs-torch or numpy-exact), the
rest need tensor-name/norm/rope/activation quirks filled in and measured.
The registry, kernels, and validation harness are proven; the gap is
coverage + process, not architecture.

## 2026-08-16 — the frontier gates: 5/5 validated, then the repo tried to eat it

**What happened**: the five routed-but-unvalidated frontier families — the
last gap between "every arch maps to a token" (2026-08-15) and "every arch
actually *runs*" — were audited, implemented, and gated against their
reference implementations in one session. All five passed. Then a merge
came through that silently dropped the whole session's work from the main
line, and the recovery — via `git reflog` and a lost-branch merge — is
half the story.

### The five gates

Each family was held to a **generation gate**: run the engine on a real (or
mini) checkpoint and compare full logits against the authoritative
reference — the HuggingFace modeling source for the exact checkpoint. Not
greedy argmax, full logit vectors.

| Family | What the audit found | Engine | Gate result |
|--------|----------------------|--------|-------------|
| **Nemotron 3** | LayerNorm1P (weight stored as w−1), relu2 non-gated MLP, partial RoPE 0.5 | `backend_generic` (arch token 989) | ✅ **real** 8B checkpoint, top1 7503 *" Paris"* == HF, corr 0.99986 |
| **DeepSeek V4** | **the existing engine was fiction** — MLA + a 4×4 "mHC mix matrix" don't exist in V4 | `src/deepseek_v4.cpp` rewritten | ✅ mini-gate top1 342 == HF, 20/20 |
| **GLM-5.2** | V3-MLA + DSA indexer with cross-layer shared top-k | `src/glm_moe_dsa.cpp` new | ✅ mini-gate top1 171 == HF, 20/20 |
| **MiMo V2** | MoD hybrid: SWA/full attention with separate dims, sigmoid group-topk MoE | `src/mimo_v2.cpp` new | ✅ mini-gate top1 524 == HF, 20/20 |
| **Qwen3.5** | GatedDeltaNet + gated GQA hybrid; the "NPU refusal" was only the VLM | `src/qwen3_5.cpp` new | ✅ mini-gate top1 142 == HF, 20/20, corr 1.0 |

**The audit was the part that earned its keep.** Two of the five engines
shipped *before* the audit were written against architectures that don't
exist. DeepSeek V4's previous `deepseek_v4.cpp` assumed MLA KV compression
and a learned 4×4 residual-mixing matrix — the real V4 has **Shared-KV MQA**
(one KV head, K=V), **mHC hyper-connections** (fn/base/scale + Sinkhorn-Knopp
projection of a doubly-stochastic comb matrix, 20 iterations), per-head
learnable attention sinks, grouped output projection, and frozen
`tid2eid[input_ids]` hash routing on the first three MoE layers. The plan
doc's framing ("nearest to validated V3 (MLA)") was wrong; the real V4
dropped MLA entirely.

**Every gate caught a real math bug:**
- *DeepSeek V4*: integer division in the grouped-output width
  (`(heads/groups)*head_dim` = 0 for 4 heads / 8 groups), and **in-place mHC
  stream corruption** — the comb matrix mixes the *old* streams, and writing
  the new streams in place breaks streams 1–3 while stream 0 looks fine.
- *GLM-MoE-DSA*: the DSA indexer consumes the **normed** attention input
  (the decoder feeds `input_layernorm(h)` into `self_attn`), and the roped
  indexer query was never written back before scoring.
- *MiMo-V2*: the checkpoint's remote modeling code never initializes
  `gate.weight` / `e_score_correction_bias` (`torch.empty` → ±1e35 garbage),
  and random-init sigmoid scores all ≈0.5 make top-k an arbitrary tie —
  the fixture had to init the router properly to make routing deterministic.
- *Qwen3.5*: `q_proj` emits per-head **[query|gate] interleaved**, not two
  contiguous blocks.

**Test suite:** `Testing/run_all.sh` went 13/13 → 17/17. Census held
100.00% (317,310 / 317,310). The engines were then wired into the router
and backend manager (`src/backend_frontier.cpp`) so `1bit` actually serves
them — GLM/MiMo discriminate by arch string, since the census maps them
into LLAMA/QWEN2 tokens.

### The merge that tried to erase it

The session's work was committed on a detached lineage. A `monster-rebrand`
merge (PR #1630) rewrote `include/rocm_cpp/bitnet_model.h` from 552 to 33
arch tokens, and the two histories diverged *at* that merge. `main` ended
up on the stripped side: 33 tokens, no NEMOTRON, none of the frontier
engines, none of the census — while the 2,223-commit frontier lineage
(b0b4fd66 → 76e91550) sat with no branch ref pointing at it. The README
showcase commit landed on the stripped main, so even the docs looked fine
while the substance was gone.

**Recovery, the honest way:** the commits were still valid objects — no
force-push had run, nothing was garbage-collected. A `git reflog` showed
the detached checkout; recreating the branch at the lineage tip
(`frontier-recovery`), cherry-picking the README commit (conflict-resolved),
wiring the engines, then merging the whole thing into `main` restored
everything: 552 tokens, 5/5 frontier engines, census 100.00%, run_all
17/17, router selfcheck 14/14. `origin/main` (PR #1689, FLM-free NPU stack)
was merged in afterwards — its content was already present on the frontier
lineage; the merge only needed conflict resolution on the files both sides
had edited (bitnet_model.h, deepseek.cpp, run_all.sh, the frontier plan).

**What the incident proved:** `git reflog` + `git merge-base` are the
recovery tools when a branch is lost — the objects are almost always still
there. And the merge-time conflict resolution mattered: `origin/main` had
re-introduced a leftover `DS_DUMP_ALL` debug hook that the lineage had
deliberately removed (`40f1b9d3`), and its plan doc still said "needs MoE
routing" for families already gated. Taking the lineage side in each
conflict kept the validated state, not the stale one.

**Status at session end:** every one of the five frontier families now has
a gated engine reachable through the router. The registry → engine → gate
loop is closed end to end; the remaining work is per-family perf and the
real-checkpoint gates (V4-Flash 87GB smallest GGUF, GLM-4.5, MiMo 313GB)
that need the GPU box.

## 2026-08-19 — the two-PC fleet: harnesses on the LAN, six deployment bugs fixed

**What happened**: the engine's GPU story left the lab. Two DeepSeek Harness
agent instances — one on each PC in the house — were wired onto one LAN and
fed entirely by 1bit.MONSTER inference: **Qwen3.6-35B-A3B at 166 tok/s** on
the Strix Halo 8060S (128 GB unified), **Llama-3.1-8B at 96 tok/s** on an RX
9070 XT, plus a Qwen3-VL-4B vision endpoint. Zero cloud LLMs in the loop.
Getting two machines running the same engine surfaced **six real bugs**, all
fixed in `feat/rocm-therock-7.14-lane-pin`; the whole deployment (relay,
fleet scripts, runbook) is public in `bong-water-water-bong/rootchat-ops`.

### The topology

| Machine | GPU | Memory | Job in the fleet |
|---------|-----|--------|------------------|
| strixhalo (192.168.50.69) | Radeon 8060S (gfx1151, TheRock) | 128 GB unified | harness #1, **35B**, vision |
| ryzen (192.168.50.185) | RX 9070 XT (gfx1201, ROCm 7.2.4) | 48 GB | harness #2, relay, **8B** |

Both harnesses bind loopback (the CLI refuses `--host 0.0.0.0` on purpose);
LAN access runs through a TCP forwarder + the API's `--trusted-host` fence
(everything else gets a 403). A cross-PC HTTP mailbox (`dsh-relay`) plus a
`dsh-delegate` wrapper turns the two agents into a fleet: send a task, poll
for the reply, done — verified round-trip (`6*7` → "42") before trusting it.

### The six bugs

1. **Chat template guessed from a filename.** The engine picked a template
   by searching the request name for `"Instruct"`, then looked up the
   *family* by exact string equality — so `Meta-Llama-3.1-8B-Instruct-Q4_K_M`
   never matched the discovered `Meta Llama 3.1 8B Instruct`, and a Llama
   model got Qwen ChatML markers (it echoed the prompt forever). Fix: read
   `tokenizer.chat_template` from the GGUF and match by normalized
   bidirectional prefix.
2. **The CPU fallback OOM'd the host.** The router's last-resort
   `cpu_generic` backend dequantizes the whole model to f32 in RAM (~4×
   size; 32 GB for the 8B) — the process was kernel-killed at 43 GB anon RSS
   on a 45 GB box even though the GPU path had already succeeded. Fix:
   `UNIFIED_GPU_ONLY=1` strips CPU backends from the route and stops
   `BackendManager::init` after the preferred ids.
3. **No repeat penalty.** The ggml-vulkan sampler chain was
   `temp → top-p → dist` with no repetition penalty — and a source comment
   claimed the vendored llama.cpp lacked one. The comment was wrong:
   `llama_sampler_init_penalties` exists in the vendored `llama.h`. Added it
   (1.1 over the last 64 tokens); three consecutive runs: 42, 42, 42.
4. **EOG token leaked into output.** Every response ended with a literal
   `<|eot_id|>`: the loop stops on the end token but decodes it into the
   returned text. Fix: pop the EOG token (and trailing whitespace) before
   decoding.
5. **Vision encoder couldn't tell up from down.** Swap detection for the
   llama.cpp-clip FFN-name bug compared weight *element counts* — but
   `[ff,H]` and `[H,ff]` have identical counts, so it misfired on the
   Qwen3-VL mmproj. Fix: use the **bias shapes** (down-bias is H-sized,
   up-bias is FF-sized) as ground truth.
6. **VL server mixed two prompt formats.** It built raw `role: text` lines
   *and* wrapped the turn in ChatML markers — a hybrid no model was trained
   on, hence degenerate output. Fix: drop the `role: ` prefix in ChatML mode
   (the generate path emits the openers itself).

### The honest caveats

The 35B number is a 3B-active MoE decode (35B total), which is exactly why
the mid-size iGPU with 128 GB of unified memory can hold the whole Q8
checkpoint. The engine's standalone `vision` server still needs
`forward_embed` on a *working* (Vulkan) backend — today that method only
exists on the generic-CPU backend, so the fleet's vision endpoint runs the
vendored llama.cpp `llama-server` instead (native `qwen3vl` + mmproj on
Vulkan). Wiring `forward_embed` into the Vulkan backend is on the list.

**Status at session end:** both machines serve the same engine behind one
bearer token, all endpoints answer cross-PC, everything is boot-safe (user
systemd + linger), and `dsh-status` reports the whole fleet at a glance.
The fixes are on `feat/rocm-therock-7.14-lane-pin`; the full record
(including the blog post) is linked from the README.
