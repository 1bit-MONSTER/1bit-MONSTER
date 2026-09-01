# NPU GEMM execution fix — insts + BO group_ids (2026-08-31)

## The bug (proven on-device)

`run_gemm()` in `src/engine.cpp` called every xclbin kernel as

```cpp
(*kern)((uint64_t)3, (uint64_t)0, (uint32_t)0, act, ws, w1, w2, kv);
```

and every `xrt::bo` was created with `group_id = 0`. On the amdxdna driver
this combination is a **silent no-op**: the ERT command reports
`ERT_CMD_STATE_COMPLETED` but the AIE never executes and no buffer is touched
(verified with sentinel buffers on `Qwen3-0.6B-NPU2/mm.xclbin` — 0 bytes
changed across all 5 BOs). The "0 for pre-compiled" assumption in
`common.h` was wrong: these FastFlowLM xclbins carry no embedded control code.

Two separate faults:

1. **Missing instruction stream.** The amdxdna kernel ABI is
   `(opcode, instr_bo, ninstr, bo0..boN)`. Without `instr_bo` the command
   completes but nothing runs. The insts must be generated per GEMM shape
   (M/K/N/weight_offset) — see `tools/gen_mm_insts.cpp`, which drives
   FastFlowLM's own `Gemm::generate_seq` (linked against the prebuilt
   `libgemm.so`/`libqwen3_npu.so`).
2. **Wrong BO group ids.** amdxdna binds each host BO to its kernel argument
   slot via the BO group: opcode=0, instr=1, ninstr=2, host buffers from slot
   3 on. `kernel.group_id(3+i)` is what each buffer must be created with.
   group-0 BOs are ignored by kernels declaring non-zero groups, and large
   group-0 BOs can wedge the NPU (IO_PAGE_FAULTs) — the same root cause that
   made IRON's GEMM appear shape-broken earlier in this session.

## The fix (applied)

- `XclbinManager` now loads the companion `<xclbin>.bin` instruction stream at
  `load()` time, creates the insts BO with `kernel.group_id(1)`, and exposes
  `insts_bo(type)` / `ninstr(type)`.
- `run_gemm` / `run_blocked_gemm` pass `(3, insts, ninstr, ...)`.
- `NpuBo::create` takes a `group_id`; activation/workspace/kv/weight BOs are
  created with `kernel.group_id(3/4/5/7)` from the mm kernel.
- `XCLBIN_PATHS` points at the local FastFlowLM xclbin set
  (`/home/bcloud/amd-oss/fastflowlm/src/xclbins/Qwen3-0.6B-NPU2/`); the
  `/opt/fastflowlm` install path did not exist on this machine.
- `tests/test_npu_gemm.cpp` patched to the same pattern.
- New tool `tools/gen_mm_insts.cpp` + generated `mm.bin` next to `mm.xclbin`.

## Verified on silicon

With insts + kernel group ids, the same `mm.xclbin` kernel now executes:
- deterministic bulk output (byte-identical across runs except a 64-byte
  region at act[64..128), a design artifact),
- input-sensitive (changing A/B changes the output),
- offset-sensitive (generating insts with `weight_offset=1024` changes the
  result — the insts genuinely drive the DMA/compute).

## Round-27 (2026-08-31→09-01): the Q4NX weight format — dequant fixed & verified

The "still open" items below were attacked and largely resolved:

1. **The Q4NX dequant is now correct and BIT-EXACT vs the runtime.** Three
   bugs fixed in `model.c`/`model.h`/`engine.cpp`:
   - `model_tensor_data` did not add the JSON header (`8 + json_len`) to
     `desc->data_offset` — every tensor was read 34.6 KB early. Added
     `ModelWeights.data_base`, set in `model_load`, used in
     `model_tensor_data`.
   - `npu_dequant_block` read the 5120-byte "I8" rows as raw BF16 pairs. They
     are Q4NX tiles: [512 B bf16 scales][512 B bf16 zeros][4096 B packed
     int4]. Rewrote as a tile dequant (32x256 tile grid, g-major scale index
     `g*32+lr`, lane-swizzled nibbles).
   - The formula. The torch2aie/zaya convention `W = q*scale + zp` does NOT
     match the FastFlowLM Qwen3-0.6B file. Reverse-engineered against
     `libq4_npu_eXpress.so`'s own `q4nx_dequantize` (fed real + synthetic
     tiles): **W = (q − zp) · scale**, bf16 scales/zeros at group-major index
     `g*32+lr`, unsigned nibbles, NO clamping. Verified maxdiff 0.0 over the
     whole q_proj; the only residual vs the runtime is one BF16 ULP from
     storing the weight BO in BF16. `npu_weight_num_blocks` now returns the
     real block count (8 for q_proj, was 3).
2. **The mm.xclbin kernel consumes the REORDERED q/scale/zp layout** (the
   closed `_q4nx_reorder`), not raw tiles (output NaNs) and not dequantized
   BF16 (output zeros). Attacked via winboat/wine (2026-09-01): a mingw
   cross-compiled Windows dumper (`tools/winboat/dump_reorder_win.exe`) loads
   `q4_npu_eXpress.dll` and calls the buffer-split `q4nx_dequantize` — it runs
   under wine 10.0, the call succeeds, but the DLL's internal buffer resize
   discards the written payload on every path (owned/external/interposed).
   `reorder_cpy` (libqwen3_npu.so) is exported but its HRX dependency chain
   is not present in any accessible lib tree. The reorder layout therefore
   remains closed; the fixed-point validation stands on the engine's own
   kernels (GU→SiLU→D cascade, `bad=0/8192` silicon-exact) and the 1BP host
   contract gate (`test_1bp_q4nx_reader` PASSED).
3. **Per-shape insts**: still open (unchanged).
4. **×0.5 runner quirk**: still open (unchanged).
