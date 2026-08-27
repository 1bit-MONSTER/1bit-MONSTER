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

## 8. Findings recorded on the tracker

- #1882 comments 5431398119 / 5432370327 / 5432434020 / 5432995466:
  chesscc path builds+runs bit-identical to aie2p (confirming #1873's
  dump-artifact interpretation), harness bring-up, the address-map gap.
- The working tree on strixhalo once carried a **DIAG4 stub** of
  `matmul_i8_i32_i4` (writes `1000+i` to 0xD000) — the real arithmetic
  kernel is the committed one (`1a309199`+); keep the stub out of commits.

## 9. Reusable artifacts

- `/tmp/aiesim_recipe.md` on strixhalo (this doc, condensed).
- Kernel-only sim sources: `hw/`-adjacent scratch under `/tmp/ksim_*` (may
  be cleaned); rebuild per §2–5.
