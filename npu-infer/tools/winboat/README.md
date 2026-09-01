# winboat / wine — Windows-DLL access for the closed q4nx reorder

The mm.xclbin kernel's weight BO uses the REORDERED q/scale/zp layout produced
by FastFlowLM's closed `_q4nx_reorder` (inside `q4_npu_eXpress.dll` /
`libq4_npu_eXpress.so`). The dequant formula was reverse-engineered and is
bit-exact (`W = (q - zp) * scale`, g-major) — see NPU_GEMM_FIX.md — but the
reorder's exact buffer layout resisted host-side extraction: the DLL's
`q4nx_dequantize<float>(weight, q, scale, zp, columns)` writes the output then
re-sizes its buffers (reallocating and discarding the data) on every path
tried (owned, external, resize-interpose).

## What works

- `dump_reorder_win.exe` — a cross-compiled (mingw-w64) Windows console tool
  that loads `q4_npu_eXpress.dll` via `LoadLibrary` + `GetProcAddress` (the
  MSVC-decorated export) and calls the buffer-split `q4nx_dequantize`.
- It RUNS under **wine 10.0** on this box (no winboat guest needed):
  ```
  sudo apt-get install wine64 g++-mingw-w64-x86-64
  # copy dump_reorder_win.exe + q4_npu_eXpress.dll + mingw runtime DLLs
  #   (libgcc_s_seh-1.dll, libstdc++-6.dll from the mingw gcc dir) into one dir
  WINEPREFIX=$HOME/wbprefix wine64 ./dump_reorder_win.exe \
      'Z:\home\bcloud\.config\flm\models\Qwen3-0.6B-NPU2\model.q4nx' Z:\home\bcloud\wb
  ```
  The call succeeds and the buffers get the slice sizes
  (weight=524288 q=65536 scale=32768 zp=16384 for one 256-col chunk), but the
  payload is discarded by the internal resize — the same wall as the Linux .so.
- Rebuild: `x86_64-w64-mingw32-g++ -O2 -std=c++17 -I /tmp/fflm_win_inc
  dump_reorder_win.cpp -o dump_reorder_win.exe` (the include copy at
  /tmp/fflm_win_inc has buffer.hpp patched to disable FLM_DEVICE_BUFFER so the
  XRT headers aren't needed).

## The private reorder is REACHABLE — progress 2026-09-01 (round-28)

The private `Q4NX::_q4nx_reorder(bytes&, buffer<u32>&, buffer<bf16>&,
buffer<i32>&, int)` is callable from a Linux probe:

- Construct `Q4NX` properly (the class is fully declared in the header — a raw
  byte buffer was too small; `new Q4NX(dir)` works) with a model DIR whose
  `model.safetensors` is a symlink to the .q4nx file.
- Set the tensor state via the public `SafeTensors::load_weights(bytes&,
  name)` (mangled `_ZN11SafeTensors12load_weightsE...`) with the tensor's
  exact 1310720 bytes — returns "I8" and the object knows the tensor.
- The reorder then asserts `total_size == q4nx_weight.size()` with
  `total_size = 0x20000000` (512 MB) — feed a 512 MB buffer from the model
  data start; interpose `__assert_fail` + `bytes::resize` (LD_PRELOAD no-ops)
  to run past the internal asserts.
- The reorder then writes its output (q/scale/zp) but segfaults writing past
  the Q4NX object's internally-allocated scratch — the object's internal
  buffers need the full runtime initialization (the engine's `load_weights`
  flow inside libqwen3_npu.so, which itself needs the missing HRX runtime
  symbols `hrx_buffer_invalidate_range` etc.).

So the reorder's LAYOUT remains closed, but every step of the calling
convention has been decoded (sizes, order, mangled names) — a future attempt
only needs a valid object-internal buffer (e.g., running the full engine
load_weights with the HRX runtime present).

## Alternative: run in the winboat Windows guest

The same exe + q4_npu_eXpress.dll can run inside the winboat guest. The guest
side would need the model file + DLL + exe copied in. Nothing in the guest
would change the resize behavior (it is the same DLL), but it gives a real
Windows environment if the wine path ever needs help.

## Round-28b (2026-09-01): the runtime's weight path is CAPTURED

The HRX wall was a red herring — the runtime runs on the **XRT** path
(`src/lib/xrt/*.so`, zero hrx symbols). The lib exports `reorder_cpy` (T) and
calls it via PLT, so **LD_PRELOAD interposition works**:

- `reorder_cpy(dst, src, cols, 2)` — decoded from the REAL runtime's calls:
  - p3 = the projection's column count (1024 for q/k/v/o, 2048/3072 for
    ffn), p4 = 2 (bf16 element size).
  - The q_proj call (first, srcsize 1310720) writes the REORDERED tiles:
    `out_tile = 8*(ir//8) + 2*tc + (tr%2)` for `ir = tr*4+tc` — i.e. evens
    then odds within each 2-tile-row group (verified byte-exact on the
    captured output: a pure tile permutation of the q4nx input).
- `xrt::bo::sync` (imported) — interposed to capture every FROM-device BO:
  the dequant kernel's outputs. `bo_from_000/001/004` = 1MB f32 weight BOs
  (256x1024), `006/007` = 2MB, `008` = 6MB — saved in `npu-infer/captures/`.

Still open: the dequant kernel's exact formula — the captured 1MB f32 BO
values (range ~[-6.3, 13.1]) do not match the host-side `(q-zp)*scale`
reference (maxdiff 13.1, per-element ratios non-constant). The BO layout is
[256,1024] f32 but the value mapping is not yet identified — the next step is
matching the captured BO to the tile bytes (the kernel may use a per-row or
fixed-point dequant that differs from the q4nx host dequant).

With the interposer in place, the FULL weight BO set for the whole model can
be captured by running the real runtime once — the hand-rolled npu-infer
validation can then feed the mm kernel the exact BOs the runtime produces
(no formula needed for the feed path).
