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
needs `--xchesscc`, not peano):

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

## 3. Launcher fixes (2025.2 install bugs)

1. **`aie2psimmsm` naming**: the `aiesimulator` wrapper looks for
   `.../unwrapped/lnx64.o/aie2psimmsm` but the install ships
   `aie2pssimmsm` (extra `s`). Symlink it.
2. **Device JSON**: the generated `scsim_config.json` says
   `data/aie2p/devices/aie2p_8x4_device.json`, which does not exist in
   2025.2. The install has `data/aie2ps/devices/XC2VE*.json` (Strix Halo =
   `XC2VE3858.json`). Patch:
   `device_json = {directory: "data/aie2ps/devices", file: "XC2VE3858.json"}`.
3. **libstdc++**: Vitis bundles an old libstdc++; `ps.so` (built with system
   g++) fails to load with `GLIBCXX_3.4.32 not found`. Run with
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

## 7. THE ADDRESS-MAP GAP (the mapped root cause)

The kernel-only poke experiment (`poke(int32_t *d){ d[0]=0x12345678; }`)
proved the disconnect:

| View | Address | Host write visible? | Host read sees? |
|---|---|---|---|
| Host API plain offset | `0x2400` | yes (stays) | 0xA5A5A5A5 after pre-fill |
| Kernel arg (segmented) | `0x72400` (seg 7 \| 0x2400) | no | nothing |
| Device bus raw | `row<<20\|col<<16\|off` | no | 0 |

- aiecc passes buffer args to the kernel as **segmented chess pointers**
  (`p0=0x73400`, `p1=0x70400`, `p2=0x72400`) — seg 7 | local offset. This is
  the **standard aiecc convention** (confirmed against mlir-aie's own
  `01_precompiled_core_function` test: its args are also `0x70400`-style).
- The kernel's C1 stores ARE present in the compiled ELF (`vst bmll4,
  [p2]` etc.) — the kernel computes and stores C1.
- **But in this aiesim configuration, the ISS's segmented-7 data-memory
  writes are not observable via the host's `XAie_DataMemRdWord`, raw
  `XAie_Read32` at any device address, or any pre-fill sweep.** The sim's
  AXI-MM bus parses addresses as `row<<20|col<<16|tile_addr` and treats
  seg-7 (`0x7xxxx`) as a row-7 tile — i.e. the core's segmented DM writes
  resolve to a *different* address space than the host API's plain offsets.
- The `--vaiml-memdump` debugger path (the sanctioned way to read ISS
  memory) is **version-mismatched in this install**: the `aiesimulator`
  wrapper requires `msm_debugger_before_end_of_elaborate`, but every
  `libvaiml_memdump*.so` in 2025.2 and 2026.1 exports
  `msm_debugger_callback`. No symlink fixes that.

**Net**: the cycle-accurate harness is fully operational for *host-side*
observation, and the segmented↔plain mapping gap is precisely characterized.
Closing it needs either (a) a matching memdump lib for the installed
aiesimulator, or (b) confirmation of the sim's intended seg-7→physical map
(e.g. via a Vitis-proper `aiesimulator` invocation with the official
`--aiesim` host build that we could not complete because aiecc.py's own
ps.so build uses the broken peano clang++).

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
