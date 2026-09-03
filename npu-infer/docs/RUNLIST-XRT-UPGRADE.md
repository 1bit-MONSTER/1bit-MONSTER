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

## Consume it — SCOPED (do NOT override the default library path)

**Important:** install the runlist stack to a **dedicated prefix**, not the default
library path. A global override (e.g. copying the 2.26.0 libs into `/usr/local/lib`)
breaks the system XRT tools built against 2.21.75 (`xrt-smi` ABI symbol-lookup error).
Keep the 2.26.0 libs at their own prefix and add them **scoped** via `LD_LIBRARY_PATH`
for the runtime only; the system default stays on 2.21.75.

```bash
# install to a dedicated prefix (not the default path)
mkdir -p /usr/local/xrt-runlist/lib
cp -a <build>/xrt/src/runtime_src/core/common/libxrt_coreutil.so.2.26.0       <build>/xrt/src/runtime_src/core/pcie/linux/libxrt_core.so.2.26.0       <build>/src/shim/libxrt_driver_xdna.so.2.26.0 /usr/local/xrt-runlist/lib/
cd /usr/local/xrt-runlist/lib && for l in core coreutil driver_xdna; do
  ln -sf lib${l}${l#core}.so.2.26.0 lib${l}${l#core}.so.2 2>/dev/null
done  # (adjust soname symlinks; or use the .so.2 -> .so.2.26.0 links from the build)
```

Then build the engine runlist path and run it with the scoped path:

```bash
cmake -S npu-infer -B npu-infer/build-rl \
      -DBUILD_RUNTIME_LAYER=ON \
      -DXRT_COREUTIL_PATH=<path/to/coreutil-dir> \
      -DXRT_CORE_PATH=<path/to/core-dir> \
      -DCMAKE_BUILD_TYPE=Release
cmake --build npu-infer/build-rl --target npu_infer -j "$(nproc)"

# scoped runtime use (add the runlist prefix, do NOT touch the system default):
export LD_LIBRARY_PATH=<flm-lib-dir>:/usr/local/xrt-runlist/lib
```


## Verified on strixhalo (scoped stack)
- `xrt::device(0)` opens with the 2.26.0 stack vs the booted kernel/firmware.
- `hw_context` + `xrt::runlist` construct; a real per-ctx layer kernel runs via `runlist::execute()`+`wait()` (3.74 ms), deterministic output.
- The runtime natively batches per-token layers into one `xrt::runlist` (`RUNLIST_ADD`).
- Qwen3-0.6B runlist decode ~37 tok/s (exceeds the 15-20 tok/s target).
