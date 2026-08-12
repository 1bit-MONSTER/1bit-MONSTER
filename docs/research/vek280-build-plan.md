# VEK280 Build Plan — TQ2/1BP LLM Decode on Versal AI Edge VE2802

> Derived from the 30-video Adaptive Computing Developer (Xilinx/AMD) channel,
> ingested into the LLM wiki (`SRC-2026-08-11-001..024`) + the existing
> WS-03 ternary-AIE microkernel research. Raw transcripts:
> `/tmp/vek280-transcripts/` (manifest: `manifest.json`).

## Target hardware (verified)

- **Board:** AMD Versal AI Edge VEK280 evaluation kit
- **Device:** VE2802 — **304 AI Engine-ML (AIE-ML) tiles** (NOT AIE1), 202 TOPS INT8
- Memory: 152 Mb AIE data memory + 304 Mb AIE-ML shared memory; NoC + DDR4
- ARM Cortex-A72 PS, PL fabric, MIPI/HDMI/etc. hard IP

**Critical translation:** every AI Engine video in the channel is **AIE1
(VCK190/VCK5000, Versal AI Core)**. VEK280 runs **AIE-ML (AIE2)** — same VLIW
concepts, different vector ISA, register widths, and intrinsic names. Our
Strix Halo XDNA2 NPU roadmap (`npu-ternary-roadmap.md`) is closer in ISA than
the channel videos. Treat the channel as **methodology**, never as
copy-paste code.

## What the channel gave us (video → lesson map)

| # | Video | Lesson for VEK280 build | Watch |
|---|-------|------------------------|-------|
| 16 | AI Engine A-to-Z #1 | Full bare-metal AIE flow: workspace → platform add → graph/kernel project → **emulation-sw build** → run emulator → compare output vs golden. This is our daily loop. | 1st |
| 17 | AI Engine Algorithm Vectorization Example | **The gold.** Methodology: intrinsic selection (mac8/mul8 vs mac16), spreadsheet scheduling in 256-bit register sections, margin handling (46-sample for 47-tap), x-start/x-step 32-bit granularity rules, x-square/y-square mini-permute (`0x2110` style), ZLS/ZLI zero-overhead loop assembly check, interleaving loads with MACs to kill wait states. Python-model the index selection before writing intrinsics. | 1st |
| 18 | Understanding the AIE Vector Pipeline | Architecture: 3 units (scalar RISC, 512-bit fixed, 512-bit float), V/W/X register grouping, 384-bit accumulators (8×48b lanes), PMXL/PMXR/PMC permute → PRA pre-add → MPY → PSA/PSB post-add lane-reduction → ACC. Explains WHY mac8/mac16 exist. | 1st |
| 13 | Vitis Platform Methodology | Platform = Vivado XSA + Linux domain (kernel/rootfs/sysroot) + BIF/boot. Rules: clock+reset, AXI master for kernel control, AXI slave for memory, stream interfaces for data, interrupts (else XRT polls). PetaLinux auto-generates ZOCL device tree. Extensible-platform example project for Versal. **Vitis 2021.1 era — flow is the same in 2025.2/2026.1, menus moved.** | 2nd |
| 10 | FIR Filters in Versal ACAP | The three compute domains (AIE array / PL DSP / PS) and tool flows per domain; where FIR goes in each. Symmetric-FIR pre-add pattern is our mac8-symmetric heritage. | 2nd |
| 22 | PetaLinux 101 | Build flow for the Linux image: config → build (25 min) → SDK (8 min) → sdk.sh sysroot install. Yocto buildtools workaround for old host gcc. We'll need this for board boot. | 2nd |
| 24 | Embedded SW stack: BootROM | Boot stages: BootROM (stage 0, immutable) → FSBL/PMU → u-boot → Linux. Master boot devices (SD/NAND/NOR/QSPI), XIP. Useful when bring-up goes wrong. | 3rd |
| 23 | Boot time estimation | Boot-time estimator tool (ANR 67475) for Zynq UltraScale+ — gives boot-budget thinking (what's acceptable per-stage). | 3rd |
| 07 | xbutil | User-function utility: `xbutil examine` (env/platform version), `xbutil validate -d N` (PCIe link, DMA/DDR bandwidth, verified kernel). Our first board check. | 3rd |
| 08 | xbmgmt | Root utility for management function: `xbmgmt examine`, flash/platform programming. Needs root; only for flashing. | 3rd |
| 05 | ILA debug of Versal AIE | Vivado Hardware Manager + ILA (probes .ltx) to debug PL/AIE interconnects on silicon. | 3rd |
| 26 | Intro to Vitis HLS | C → RTL via pragmas; dataflow, C-sim → C-synth → co-sim. For PL-side pre/post-processing kernels. | 3rd |
| 30 | HLS AXI bus widening | Bandwidth optimization for AXI master kernels. | 3rd |
| 25 | RTL blackbox in HLS (whisper) | Weave hand RTL into HLS via JSON bindings + AP handshake; II/latency declared for scheduler. For hand-tuned PL blocks. | 3rd |
| 11 | DL on Versal (VCK5000) | **DPUv4E: 48 AIE cores/engine, 7.7 TOPS; BERT on 384 AIE cores** — GEMM+LayerNorm on AIE, embedding on PL, mixed INT8/FP32, FFN ≈ 60% of time, template-driven (not instruction-based). Proof transformers run on AIE arrays; also why pruning doesn't help AIE (channel quantization granularity). | 2nd |
| 06 | VCK5000 mixed-kernel E2E | Full Vitis flow on a Versal card: platform install, xclbin load (`xbutil program`), docker/XRT runtime. | 3rd |
| 14 | C++ kernels vs CPU/GPU | Kernel optimization concepts (TSP example), why FPGA wins on latency/power. | 3rd |
| 12 | Advanced RTL kernel integration | RTL kernel methodology (RTL → kernel → v++ link). | 3rd |
| 01 | Vitis Accelerated Libraries | L1/L2 library ecosystem (BLAS etc.) — reuse before writing. | 3rd |
| 02 | Model Composer | MATLAB/Simulink → AIE flow. Only if we do DSP-model prototyping. | skip |
| 03 | Extensible platform workflow 2022.1 | Simplified platform creation (PetaLinux → XSA → Vitis platform). Same as #13, newer version. | 2nd |
| 04 | GT kernels on Alveo | Transceiver (GT) kernels for inter-card comms — not needed for VEK280 eval. | skip |
| 15/29 | Vitis IDE git / project-less debug | IDE niceties; command-line build (`make`) + debug is our style anyway. | skip |
| 19-21, 27, 28 | Versal embedded walkthroughs | **Silent screen-capture demos — no narration recoverable** (captions disabled). No content to extract. | skip |
| 09 | Introducing Vitis Tutorials | 0:39 promo, silent. | skip |

## Build phases

### Phase 0 — Board bring-up (before any AIE code)

**Fastest path: pre-built Vitis AI 3.5 SD image** (from the VEK280 quickstart, SRC-2026-08-11-025). Vitis AI Runtime, VART, Vitis-AI-Library samples and models are baked in — no PetaLinux build needed to get a booting board:
1. Burn `amd-vek280-dpu-v2023.1-v3.5.0.img.gz` (xilinx.com download, design-license form) with BalenaEtcher to SD.
2. Set **boot mode switch SW1[1:4] = ON,OFF,OFF,OFF** (SD boot).
3. USB-C connector = JTAG+UART. Enumerates 3 UARTs: Versal UART0, Versal UART1, System Controller. Two terminals, **115200 8N1**.
4. UART0 shows "Xilinx Versal Platform Loader and Manager" → boot from SD. Login via `ssh -X root@<ip>` (**password: root**; find IP via `ifconfig`, set manually with `ifconfig eth0 <ip>` if no DHCP; `export DISPLAY=<host>:0.0` for GUI samples).
5. Verify: run stock resnet50 (`./resnet50 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel`), then `xbutil examine` for XRT health.
6. **Optional** (only if we need a custom rootfs): build PetaLinux from AMD's VEK280 BSP (Xilinx 2025.2/2026.1 at `/home/bcloud/Xilinx/`), enable XRT+ZOCL (`petalinux-config` rootfs — video #22/#13); ZOCL device tree auto-generated from extensible-platform XSA.
7. **Host setup:** clone `Xilinx/Vitis-AI`, run `board_setup/vek280/host_cross_compiler_setup.sh` (installs `~/petalinux_sdk_2023.1` cross-compile env), pull `xilinx/vitis-ai-pytorch-cpu:latest` Docker. Needs ~100GB free.
8. **Exit criterion:** Linux boots, resnet50 demo runs, XRT healthy, simple kernel runs.

### Phase 1 — Platform (one-time, keep as base)
- Use AMD's VEK280 base platform if the BSP ships one (preferred — videos #13/#03 both say: start from pre-built, customize only if needed).
- If custom: Vivado extensible-platform example → verify clock/reset/AXI master+slave/interrupts exported → XSA → PetaLinux → Vitis platform → verify with vadd in hw emulation then on board.
- **Exit criterion:** vadd passes in `sw_emu` → `hw_emu` → hardware.

### Phase 2 — First AIE kernel on AIE-ML
1. Port the A-to-Z flow (#16) to AIE-ML: graph + one simple kernel (`aie_api/aie.hpp`), `emulation-sw` build, compare vs golden.
2. Apply the vectorization methodology (#17) to a real GEMV:
   - Pick intrinsics from the **AIE-ML intrinsic docs** (UG1078 is AIE1 — use the AIE-ML API reference; names differ).
   - Spreadsheet-schedule the register sections; python-model the index selection; verify x-square/y-square mini-permute values.
   - Check generated assembly for ZLS/ZLI loop + wait states; interleave loads with MACs.
3. **Exit criterion:** a mac8-style INT8 GEMV kernel in `emulation-aie` matching golden, then on hardware.

### Phase 2b — DPU path (free milestone, not the bet)
The pre-built image ships **DPUCV2DX8G** (Versal AI Edge CNN DPU). Quantize→compile→deploy flow (quickstart): `resnet18_quant.py --quant_mode calib --subset_len 200` → INT8 `.xmodel` → `vai_c_xir -a /opt/vitis_ai/compiler/arch/DPUCV2DX8G/VEK280/arch.json` → deploy with `.prototxt`. This proves the board + toolchain end-to-end in an afternoon and gives us a baseline number (DPU TOPS on VEK280) — but it's a CNN engine, not a path to TQ2 LLM decode. Do it once for bring-up confidence, don't invest further.

### Phase 3 — TQ2 ternary microkernel (the actual bet)
Port WS-03 (`mm_ternary_tq2.cc` design) to AIE-ML, per the roadmap's proven trick:
- `load_v` 128 ternary codes → `uint32_t LUT[256]` unpack → INT8 MACs. AIE-ML has no sub-byte arithmetic (same constraint as AIE2 NPU).
- Ping-pong L1 buffers; measure DDR bytes/token. VEK280's 304 AIE-ML tiles + 202 INT8 TOPS ≈ a big decode engine, but batch=1 decode is **DDR-bound** — the 4× traffic cut is the win, same as the NPU analysis.
- Validate ppl vs FP16 on Bonsai-1.7B/4B.
- **Exit criterion:** `ternary_tq2_gemv` on VEK280 hardware; ≥2× decode vs the MI300X HIP bridge path at batch=1.

### Phase 4 — Plumbing (only if Phase 3 wins)
- Stream weights from DDR via NoC to AIE tiles (NoC/DDR video #19's topic, though its narration is lost — use UG1354 docs).
- PL kernels (HLS, #26/#30) for tokenizer/embedding/LayerNorm offload (BERT overlay lesson #11: FFN on AIE, light ops on PL).
- Host app with XRT API (`xrt::device` etc., #06/#07 style).

### Phase 5 — Benchmark & report
- Throughput (tokens/s), latency, DDR bytes/token, ppl vs FP16 — same metrics as WS-03 validation.

## Risks / gotchas

1. **AIE1 → AIE-ML ISA gap:** the channel's intrinsic names, register widths (512-bit X vs AIE-ML 1024-bit vector regs), and mac8/mac16 signatures do NOT carry over verbatim. Budget time to re-derive intrinsics from AIE-ML docs (AM020 + AIE-ML API reference). The *methodology* (scheduling, permute granularity, margin, assembly inspection) transfers 1:1.
2. **Tool version drift:** videos are 2020.2–2022.1; we have 2025.2/2026.1. Menus/UX changed; concepts stable.
3. **6 videos have no narration** (silent demos) — nothing lost, content covered by docs.
4. **IRON/MLIR-AIE vs Vitis:** both installed (`/home/bcloud/iron/`). Vitis is the path of least resistance for VEK280 (board support, PetaLinux, XRT); IRON is useful for AIE-ML ISA exploration/emulation. Recommend Vitis for board flow, IRON for kernel experiments.

## Immediate next actions

- [ ] Download `amd-vek280-dpu-v2023.1-v3.5.0.img.gz`, burn to SD, set SW1=ON,OFF,OFF,OFF, boot, `ssh -X root@<ip>` (pw: root)
- [ ] Run stock resnet50 demo; `xbutil examine`; confirm XRT healthy (Phase 0)
- [ ] Host: clone Vitis-AI, run `board_setup/vek280/host_cross_compiler_setup.sh`, pull vitis-ai-pytorch-cpu Docker
- [ ] Run the DPUCV2DX8G quantize→compile→deploy flow once (Phase 2b, baseline TOPS)
- [ ] In IRON or Vitis: hello-world AIE-ML kernel, `emulation-sw` → hardware
- [ ] Start WS-03 port with the #17 spreadsheet method on one AIE-ML tile
