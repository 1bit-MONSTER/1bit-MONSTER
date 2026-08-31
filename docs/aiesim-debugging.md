# AIE2P Cycle-Accurate Simulation — Debugging the int4 Fused Kernel

> Status: **harness operational, address-map gap fully mapped** (2026-08-26)
> Scope: issue #1769 (int4 MoE kernel), tracker #1882. Host: strixhalo.

## 1. Why the simulator

All pre-2026-08-26 debugging of the `-0.003` C1 mismatch was done with
**on-silicon probes** — C1 dumps via pC/h2 readback, which are themselves
suspect (issue #1865: hardcoded/segmented addresses unreliable across the
host↔kernel boundary). The **cycle-accurate AIE2P ISS** (`aiesimulator`) gives
a trustworthy execution model with observable tile memory, and it runs the
**same kernel ELFs** the NPU loads. It is the only way to answer, without
silicon ambiguity:

> Is the chesscc/aie2p-compiled kernel's C1 wrong (codegen bug), or is the
> silicon path (delivery/DMA/dump artifact) the problem?

## 2. Build the sim design (strixhalo)

Requires the **licensed chess compiler** (unblocked by #1878 — `--aiesim`
needs `--xchesscc`, not peano). Both installed toolchains work — 2025.2
(chesscc V-2024.06) and 2026.1 (chesscc X-2025.06); the recipes below show
2025.2, swap `2025.2` ↔ `2026.1` for the other (launcher fixes in §3 apply
to both, verified):

```bash
# kernel object(s) via the Vitis chess launcher
export PATH=~/Xilinx/2025.2/Vitis/aietools/bin:$PATH AIETOOLS=~/Xilinx/2025.2/Vitis/aietools
xchesscc -p me -C Release_LLVM -D__AIENGINE__ -D__AIE_ARCH__=22 \
  -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
  -I .../aietools/include -I .../mlir_aie/.../include/aie_kernels/aie2p \
  -P .../aietools/data/aie2ps/lib -f -c mm_kernel_reference.cc -o mm.o

# sim project
export PATH=/usr/bin:/bin:.../mlir-aie/install/bin:.../mlir-aie/.venv/bin:$PATH
export PYTHONPATH=.../mlir-aie/install_tmp/python:.../mlir-aie/.venv/lib/python3.14/site-packages
export LD_LIBRARY_PATH=.../mlir-aie/install_tmp/python/aie/_mlir_libs
aiecc --peano=.../llvm-aie --aietools=~/Xilinx/2025.2/Vitis/aietools \
  --alloc-scheme=basic-sequential --xchesscc --xbridge \
  --aiesim --no-compile-host --unified design.mlir
# -> aie.mlir.prj/  with main_core_<col>_<row>.elf per core
```

> **`--aietools` MUST be the Vitis aietools ROOT for the chess arm (issue
> #1913).** With `--xchesscc`, aiecc looks for chess-llvm-link at
> `<aietools>/tps/lnx64/target_aie2p/bin/LNa64bin/chess-llvm-link`. Pointing
> `--aietools` at mlir-aie's own `build_tmp` (which the **peano** arm
> tolerates — e.g. `check_mm_kernel_2x4.sh`'s `AIETOOLS=~/mlir-aie/build_tmp`
> default) makes aiecc **silently skip** the chess-llvm-link step (logged only
> under `-v`) and fail later with a confusing `main_input.chesslinked.ll`
> missing error at `xchesscc_wrapper`. Related gotcha: an
> `~/mlir-aie/install/bin/xchesscc` symlink shadowing the Vitis launcher on
> PATH makes `getAietoolsDir()` derive the wrong root the same way — the Vitis
> `aietools/bin` must be found first. `engine/npu/generators/check_chess_aietools.sh`
> implements this check as a reusable pre-flight guard (`check_mm_kernel_2x4.sh`
> uses it; `USE_XCHESSCC=1` there exercises the chess arm).

## 3. Launcher fixes (Vitis install bugs — 2025.2 & 2026.1)

All three fixes were re-verified against **both** installed toolchains on
strixhalo: 2025.2 (chesscc V-2024.06) and 2026.1 (chesscc X-2025.06); swap
`~/Xilinx/2025.2` ↔ `~/Xilinx/2026.1` in every recipe.

1. **`aie2psimmsm` naming**: the `aiesimulator` wrapper looks for
   `.../unwrapped/lnx64.o/aie2psimmsm` but the install ships
   `aie2pssimmsm` (extra `s`; the wrapper sets `progname=aie2psimmsm` for
   `aiearch=aie2p`). Symlink it.
2. **Device JSON**: the generated `scsim_config.json` says
   `data/aie2p/devices/aie2p_8x4_device.json`, which does not exist in the
   stock install (a file of that name now present under that path is a
   debugging-session leftover — byte-identical to `XC2VE3304.json`, dated
   2026-08-26; the shipped files date from the 2025 install). The install has
   `data/aie2ps/devices/XC2VE*.json` (Strix Halo = `XC2VE3858.json`). Patch:
   `device_json = {directory: "data/aie2ps/devices", file: "XC2VE3858.json"}`.
3. **libstdc++**: Vitis bundles an old libstdc++ (2025.2 tops out at
   GLIBCXX_3.4.28, 2026.1 at 3.4.31); `ps.so` (built with system g++) fails
   to load with `GLIBCXX_3.4.32 not found`. Run with
   `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`.

## 4. Host `ps.so` (the testbench driver)

The `--no-compile-host` flow leaves `sim/ps/ps.so` unbuilt; aiecc's own
build uses the Vitis `clang++` wrapper (points at peano — broken). Build it
manually with the system g++:

```bash
g++ -O2 -fPIC -shared -fpermissive \
  -DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR -DSC_INCLUDE_DYNAMIC_PROCESSES \
  -D__AIESIM__ -D__PS_INIT_AIE__ -Og '-Dmain(...)=ps_main(...)' \
  -Idesign.prj -Iaietools/include \
  -I.../mlir_aie/runtime_lib/x86_64/xaiengine/include \
  -Iaietools/data/osci_systemc/include -Iaietools/include/xtlm/include \
  -I.../test_lib/include \
  -Wl,--whole-archive .../test_lib/lib/libtest_lib.a -Wl,--no-whole-archive \
  .../libmemory_allocator_sim_aie.a \
  -L.../xaiengine/lib -lxaienginecdo \
  -Laietools/lib/lnx64.o -Laietools/data/osci_systemc/lib/lnx64 \
  -Wl,--as-needed -lsystemc -lxtlm \
  genwrapper_for_ps.cpp testbench.cpp -o design.prj/sim/ps/ps.so
```

**Gotcha**: `printf` in `ps_main` is pipe-buffered — call `setbuf(stdout, NULL)`
or you'll see no output until exit (and a blocked ps_main shows nothing).

## 5. Run

```bash
export PATH=~/Xilinx/2025.2/Vitis/aietools/bin:$PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH=.../xaiengine/lib:.../mlir_aie/install/lib:$LD_LIBRARY_PATH
aiesimulator --pkg-dir=design.prj/sim
```

`--enable-memory-check` reports out-of-bounds core accesses; the raw ISS
binary is `.../unwrapped/lnx64.o/aie2pssimmsm` (needs `RDI_DATADIR` +
`AIETOOLS` env).

## 6. What works (verified)

- Design builds: 8 core ELFs for the fused GUSILU design, 1 ELF for a
  single-tile kernel-only design.
- ISS loads and executes the ELFs; PS host (`ps.so`) runs.
- Locks flow correctly (acquire/release, output-lock polling).
- Tile-memory **host writes** at plain offsets work and read back exactly
  (`XAie_DataMemWrWord` at 0x3400 → reads 0x00fffefd).
- A full 64 KB tile-DM pre-fill + sweep detects every changed word.
- Disassembly of the compiled kernel is fully inspectable.

## 7. THE ADDRESS MAP — FULLY DECODED (no mystery left)

The chess/peano link map (`main_core_4_2.elf.map`) shows the tile-DM layout:

```
Memory map for memory 'DMb' (tile local DM):
  0x70000..0x703ff   Stack
  0x70400..0x735ff   Reserved  <- A/B/C1 buffers (B@0x70400, C1@0x72400, A@0x73400)
  0x73600..0x73eff   silu gos[]/sigmoid tables (kernel statics)
  0x73f00+           runtime data
```

The kernel's `0x7xxxx` pointers are **absolute tile-DM addresses**, not
'segments'. The host `XAie_DataMemRdWord/WrWord` API uses tile-relative plain
offsets (0x2400 etc.) = chess address − 0x70000. **Same physical DM, offset
by 0x70000.** In this aiesim config the host bus and the ISS core use
different base conventions and are **not aliased** — host writes at plain
0x3400 are invisible to the kernel's 0x73400 reads (verified by pre-fill
sweep, raw XAie_Read32 sweep, and the official aiecc.py build: all agree).

## 7b. A REAL chesscc codegen bug found (independent of the sim gap)

The trivial `poke` kernel (`d[0]=0x12345678; d[1]=0x90abcdef`) compiles to:

```
000001c0 <poke>:
  1c0: ret lr                    <- function ENTRY is a RET (delay_slots=5)
  1c4: movxm r0, #0x12345678     <- in delay window -> executes
  1ca: st r0, [p0, #0]           <- PAST delay window -> NEVER executes
  1ce: movxm r0, #-0x6f543211
  1d4: st r0, [p0, #4]           <- also dead
```

The `.srv` annotation: `ret lr` at word 448 with 5 delay slots (449-453);
the store starts at word 458 — past the window, dead code. The function
returns without storing. This is a **chesscc leaf-function codegen bug**
(ret hoisted before the body; stores fall outside the delay window),
reproduced in the cycle-accurate ISS. Re-verified against **both** installed
chesscc versions — V-2024.06 (Vitis 2025.2) and X-2025.06 (Vitis 2026.1)
compile the 6-line `poke` to the identical ret-first sequence (`ret lr` at
offset 0, both `st` past the delay window; disassembled with the llvm-aie
`llvm-objdump`). Not version-specific.

The fused `matmul_i8_i32_i4` has a proper prologue (no leading ret) and its
C1 stores (`vst bmll4, [p2]`) exist — so its C1=0 in the sim is most likely
the host↔ISS base-convention disconnect (host can't observe the kernel's
0x7xxxx DM view), with the ret-first bug as a separate, proven leaf-function
codegen defect.

## 7c. Official aiecc.py flow — now working end-to-end

The canonical build (which the mlir-aie reference tests use) works with a
PATH trick — aiecc finds `clang++` via `findProgramByName` and the Vitis
wrapper is broken (points at peano):

```bash
export PATH=.../llvm-aie/bin:.../build_tmp/bin:.../aietools/bin:/usr/bin:...
aiecc.py --aiesim --xbridge --xchesscc design.mlir testbench.cpp \
  -o test.elf -L.../test_lib/lib -ltest_lib
```

The llvm-aie clang++ is a full x86-64 host compiler (not just aie2p cross),
so it builds `ps.so` correctly. This is the sanctioned path to close the gap
once a matching `--vaiml-memdump` lib (or a host↔chess DM base-offset
config) is available.

## 7d. TXN-driven designs: the hand-written ps.so TXN-replay harness (#1908/#1909/#1910/#1911)

For **npu-instruction-driven** designs (external objectFifos + external_func,
e.g. the v27 GEMM / n1_core_i8_v27.py), the classic aiecc `--aiesim` flow is
broken in four independent ways, each tracked separately:

| Issue | Failure |
|-------|---------|
| #1908 | Vitis 2026.1 `aie2psimmsm` segfaults on ANY npu2 design (2025.2 works) |
| #1909 | `--aie-mlir-to-shim-solution` emits an EMPTY `aieshim_solution.aiesol` (`"Placement": []`) for every npu2 design |
| #1910 | `--aie-generate-xaie` emits empty `configure_*`/`start_cores` stubs — the sim has nothing to drive |
| #1911 | `aiecc --aiesim` silently exits 0 with NO `ps.so` when `clang++` is not on PATH |

The checked-in harness **`engine/npu/tests/aiesim/`** (`run_aiesim.sh` +
`txn_replay_main.cpp`) works around all four: it builds a hand-written
`ps.so` with system `g++` (no clang++ needed — fixes #1911), refuses/clearly
warns on the empty aiesol (#1909), drives the design by **replaying the TXN
instruction stream** (`insts.txt` — the same file XRT loads on hardware)
through XAie instead of relying on generated stubs (#1910), and defaults
`VITIS` to 2025.2 with a loud warning on 2026.1 (#1908).

TXN opcode format (mlir-aie `include/aie/Runtime/TxnEncoding.h`, verified
against the file; tile addresses are folded absolute `col<<25 | row<<20 |
offset`, sim base is 0x20000000000 — see `txn_replay_main.cpp` header):

```
0x00 WRITE       6 words  [opc, 0, addr, 0, val, sizeB]
0x01 BLOCKWRITE  4+c      [opc, col|row<<8, addr, sizeB, data(c)]
0x03 MASKWRITE   7 words  [opc, 0, addr, 0, val, mask, sizeB]
0x80 TCT         4 words  [opc, sizeB, ...]  (flow control — no-op here)
0x81 DDR_PATCH   12 words [opc, sizeB, 0,0,0, act, bdaddr, 0, argidx, 0, plus, 0]
                        -> BD address field += devbuf[arg]+plus
```

Usage (after producing an `--aiesim` workdir with `insts.txt`):

```bash
engine/npu/tests/aiesim/run_aiesim.sh <arm-workdir> [WAIT_US] [insts-name]
# VITIS defaults to ~/Xilinx/2025.2 (verified-working aiesimulator).
```

## 8. Findings recorded on the tracker

- #1882 comments 5431398119 / 5432370327 / 5432434020 / 5432995466:
  chesscc path builds+runs bit-identical to aie2p (confirming #1873's
  dump-artifact interpretation), harness bring-up, the address-map gap.
- The working tree on strixhalo once carried a **DIAG4 stub** of
  `matmul_i8_i32_i4` (writes `1000+i` to 0xD000) — the real arithmetic
  kernel is the committed one (`1a309199`+); keep the stub out of commits.

### Tracker close-out (2026-08-28, issue #1882 sub-items)

Status of the consolidated aie2p miscompile tracker's sub-items after the
strixhalo verification round:

| Issue | Defect | Status |
|-------|--------|--------|
| #1834 | memcpy libcall clobbers r0/r1 | ✅ FIXED in-tree: union bit-casts everywhere incl. mm_binary_q1.cc; escal. upstream |
| #1838 | ld.script drops .bss statics | ✅ FIXED in-tree: KERNEL_STATIC → .data, ELF-verified (.data@0x7f60c, zero .bss); escal. upstream |
| #1864 | scalar RMW miscompile | ✅ Worked around: I4_SCALAR_C1_ACK_1864 guard + #1874 default flip; escal. upstream |
| #1865 | hardcoded tile addresses | ✅ FIXED in-tree: h2 via delivered arg, pC zeroing via arg, zero_c1 removed, #1842 pins retired; NPU gate pending |
| #1869 | pointer-arith miscompile j>=8 | ✅ Worked around (v66 direct pointer math); escal. upstream with repro |
| #1835 | soft-float (sf*0.0625)/scc NaN | ✅ Worked around (int32 ratioQ22); escal. upstream |
| #1836 | pipeliner float loop p>=1 | ✅ Worked around (silu_pair_q22 fixed-point); escal. upstream |
| #1872 | Btmp byte-stores dropped | ✅ Mitigated: #1874 flip removes Bb from production; I4_DIRECT_VECTOR_DEQ for mmul path; NPU gate pending |
| #1874 | mmul C1 store scrambled | ✅ Mitigated: I4_SCALAR_C1 is now the production default; mmul path opt-in |
| #1878/#1912 | chess arg delivery (upstream) | ⏳ ESCALATE upstream; A/B harness on main as regression test |
| #1866 | -O0 immediate range crash | ⏳ ESCALATE upstream (llvm-aie); -O1 workaround documented |

Upstream reproducers to file (all have in-repo CPU-gated minimal cases):
#1869 (rqb+j*32 vs pB4+gbase+(j<<5)), #1835 ((sf*0.0625f)/scc → NaN),
#1836 (float silu loop p>=1), #1864 (scalar `+=` with computed operand,
11/32 operand subset → <<8), #1866 (-O0 immediate −33216 out of
[-32768,-64]), #1878 (chess memref base pointers never delivered).

## 9. Reusable artifacts

- `/tmp/aiesim_recipe.md` on strixhalo (this doc, condensed).
- Kernel-only sim sources: `hw/`-adjacent scratch under `/tmp/ksim_*` (may
  be cleaned); rebuild per §2–5.

### 8b. Recovering a hung NPU — driver reset path (issue #1920)

`aie-reset` (mlir-aie's reset tool) is **dead on modern kernels**: its
`mmap(/dev/mem)` fails under `CONFIG_STRICT_DEVMEM`/lockdown, leaving
reboot as the only recovery for a hung iron-runtime launch.

**Resolved 2026-08-28 by the driver upgrade** (upstream `amd/xdna-driver`,
commit `7004f1c`, loaded as `updates/amdxdna.ko`, driver 0.17.0):

- `tdr_timeout_ms=2000` (default): a deadlocked job now returns from
  `run.wait()` in ~2 s with `state=8` (ERT_CMD_STATE_TIMEOUT) instead of
  hanging forever; `dmesg` shows `aie2_set_cmd_timeout`.
- `aie2_hw_reset()`: SMU power-cycle + firmware reload (suspend-all →
  hw_stop → hw_start → resume-all) — **no `/dev/mem` needed**, and the NPU
  self-recovers (a second launch works without reboot). This obviates
  `aie-reset`'s broken mmap path.
- Requires Secure Boot disabled / lockdown `none` on the host.

Use `sudo rmmod amdxdna && sudo modprobe amdxdna` only to reload firmware;
for a wedged AIE core the driver's TDR + SMU power-cycle is the recovery
path (the `.github/workflows/npu-reset.yml` rmmod/modprobe job remains as a
belt-and-braces fallback, not a reset).

### 8c. Local mlir-aie npu2_40 patches — decision (issue #1948)

The `~/mlir-aie` checkout on strixhalo carries **local, un-pushed patches**
needed for the npu2_40 toolchain build path (from the NPU cascade work):
`AIELowerDynamicBDPool.cpp`, `BdLowering.cpp` (+ headers), `python/aie/`,
dynamic DMA/BD tests, strix AOT lit tests, and submodule bumps
(`cmake/modulesXilinx`, `platforms/boards`).

**Decision (2026-08-30): keep local-only, backup is the canonical copy.**
Upstreaming to Xilinx/mlir-aie is deferred: the patches are WIP
("wip(toolchain): local NPU2-40 patches" — commit `1e6b70af0`), the npu2_40
build path is not the production path, and the Xilinx repo is not one we
contribute CI to. Recovery on a fresh box:

```bash
# canonical backup (verified present on strixhalo):
git clone ~/1bit-MONSTER-backups/mlir-aie-local-patches-2026-08-29.bundle mlir-aie
# or apply the flat patch:
cd mlir-aie && git apply ~/1bit-MONSTER-backups/mlir-aie-local-patches-2026-08-29.patch
```

Re-verify before relying on it (the bundle is a full clone; the patch is the
same content as commit `1e6b70af0`). If the npu2_40 path ever becomes
production, upstream the patches as a PR to Xilinx/mlir-aie first.
