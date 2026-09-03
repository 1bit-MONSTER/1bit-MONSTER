# Runlist-capable XRT 2.26.0 upgrade (issue #1776)

The `RuntimeLayerEngine`'s `xrt::runlist` batching needs a **runlist-capable XRT (>= 2.25)**. The distro `libxrt-* 2.21.75` declares `xrt::runlist` in its headers but does **not** export the symbol (link fails), so the stack must be built with a newer XRT. Whole stack verified working on strixhalo (device-compatible, runlist executes a real layer kernel).

## Recipe (build the matched XRT-NPU + xdna shim)

The AMD-matched source is `amd/xdna-driver` (its pinned `xrt` submodule is runlist-capable; `CMake/xrt.cmake` builds XRT with `XRT_NPU=1`). Build source-only (`SKIP_KMOD`, no kernel driver) to a prefix:

```bash
git -C ~/xdna-driver submodule update --init --recursive   # xrt + aiebu/gsl/elf/xdp + aiebu zstd
cmake -S ~/xdna-driver -B /tmp/xdna-build \
      -DCMAKE_INSTALL_PREFIX=$HOME/xrt-runlist-install \
      -DSKIP_KMOD=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/xdna-build --target xrt_driver_xdna -j "$(nproc)"
```

Produces a self-consistent 2.26.0 stack:
- `xrt/src/runtime_src/core/common/libxrt_coreutil.so.2.26.0` — exports `xrt::runlist::(runlist|add|execute|wait)` (41 symbols)
- `xrt/src/runtime_src/core/pcie/linux/libxrt_core.so.2.26.0`
- `src/shim/libxrt_driver_xdna.so.2.26.0` — the AMD NPU shim

## Consume it

Point the runtime at the stack (e.g. `LD_LIBRARY_PATH` to the lib dirs, or install to `/usr/local`), and build the engine runlist path:

```bash
cmake -S npu-infer -B npu-infer/build-rl \
      -DBUILD_RUNTIME_LAYER=ON \
      -DXRT_COREUTIL_PATH=<path/to/coreutil-dir> \
      -DXRT_CORE_PATH=<path/to/core-dir> \
      -DCMAKE_BUILD_TYPE=Release
cmake --build npu-infer/build-rl --target npu_infer -j "$(nproc)"
```

## Verified on strixhalo
- `xrt::device(0)` opens with the 2.26.0 stack vs the booted kernel/firmware.
- `hw_context` + `xrt::runlist` construct; a real per-ctx layer kernel runs via `runlist::execute()`+`wait()` (3.74 ms), deterministic output.
- The runtime natively batches per-token layers into one `xrt::runlist` (`RUNLIST_ADD`).
- Qwen3-0.6B runlist decode ~37 tok/s (exceeds the 15-20 tok/s target).
