# Building zaya (C++ / ROCm)

This document covers building **zaya** — a pure C++ inference server with optional
GPU decoding support, one entry point of the single binary `build/1bit` (run via
`1bit zaya`). No Rust, no Python at runtime. The host CPU is **AMD Strix Halo**
(Ryzen AI Max+ 395) and GPU acceleration uses **TheRock 7.15.0a** targeting `gfx1151`.

> `zaya_server` is no longer a standalone CMake **build target** — its full
> source list is compiled into `onebin`/`build/1bit` only (see
> `CMakeLists.txt`). Build `onebin` and run the server via
> `./build/1bit zaya [flags]`. Packaged installs (tarball/deb) do still ship
> a `zaya_server` symlink to that same binary for convenience, but there is
> no separate `zaya_server` binary to build from source.

---

## Prerequisites

| Package            | Version / Notes                                     |
|--------------------|-----------------------------------------------------|
| Ubuntu             | 24.04 LTS or later (CachyOS / Arch also works)      |
| Kernel             | 6.18.22-lts or 7.x — **not** 6.19.x (issue #1 hang) |
| ROCm               | TheRock 7.15.0a (nightly C++ SDK, native gfx1151) |
| CMake              | ≥ 3.28                                              |
| Ninja              | ≥ 1.12                                              |
| GCC                | ≥ 15 (C++26) or ≥ 14 with a C++26-capable flag set           |
| Git                | —                                                   |

Install system dependencies:

```bash
sudo apt update
sudo apt install -y cmake ninja-build build-essential git
```

---

## TheRock 7.15.0a

```bash
# Install TheRock HIP SDK for gfx1151 (Strix Halo)
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
  "rocm[libraries,devel,device-gfx1151]"
export THEROCK_PIP_ROOT="$HOME/.cache/pip/therock"

# Verify
which amdclang++
```

The CMake build system auto-discovers TheRock (see `CMakeLists.txt`):
`/opt/rocm-therock` → `$THEROCK_PIP_ROOT` → `~/.cache/lemonade/bin/therock`.

**Set `CMAKE_HIP_ARCHITECTURES`** so that HIP kernels are compiled for Strix Halo:

```bash
export CMAKE_HIP_ARCHITECTURES=gfx1151
```

It is convenient to add this to your shell profile:

```bash
echo 'export CMAKE_HIP_ARCHITECTURES=gfx1151' >> ~/.bashrc
```

---

## Build: zaya (required)

Clone the repository and build the main server binary:

```bash
# Clone (adjust URL to match your remote)
cd ~
git clone <your-repo-url> zaya
cd zaya

# Configure — TheRock is auto-detected (see CMakeLists.txt); never point
# CMAKE_PREFIX_PATH at system ROCm (/opt/rocm).
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151

# Build the single binary (zaya, unified, jarvis, vision, and the CLI all
# live in this one target — there is no standalone `zaya_server` target)
cmake --build build --target onebin
```

The resulting binary is `build/1bit`. Run the zaya server via `./build/1bit zaya [flags]`
(the packaged tarball/deb also installs a `zaya_server` symlink to the same binary
for backward compatibility, dispatched by `argv[0]`).

---

## NPU backend: FastFlowLM (flm)

The NPU backend spawns the **flm** binary (FastFlowLM, now an official
AMD/ROCm project). CMake finds it in this order — no source build needed
for the first two:

1. **TheRock dist** (`<therock-root>/bin/flm`) — FLM ships built into TheRock 7.14/7.15a
2. **FastFlowLM .deb** (`/opt/fastflowlm/bin/flm`) — the prebuilt release
   package from [FastFlowLM releases](https://github.com/FastFlowLM/FastFlowLM/releases),
   the same binary Lemonade Server tracks and stays in sync with
3. `flm` on PATH (`/usr/bin/flm` — the .deb symlink, also how Lemonade finds it)
4. **Submodule build** (`third_party/FastFlowLM`) — last resort

```bash
# Preferred: install the prebuilt package instead of building the submodule
sudo apt install ./fastflowlm_0.9.46_ubuntu26.04_amd64.deb
```

## Build: zaya_gpu_decode (benchmarking tool, always built)

`zaya_gpu_decode` is a GPU-decode benchmarking tool for **Q4NX** models. It is an
unconditional target — there is no option to switch it on or off, and the normal
configure builds it (`add_executable(zaya_gpu_decode tests/zaya_gpu_decode.cpp
kernels/zaya_cca_attn.hip)` in `CMakeLists.txt`):

```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151

cmake --build build --target zaya_gpu_decode
./build/zaya_gpu_decode model.q4nx [--prompt N] [--tokens N]
```

> **Corrected 2026-09-14.** This section used to tell you to configure with
> `-DZAYA_ENABLE_GPU_DECODE=ON`, to expect `build/libzaya_gpu_decode.so`, and to
> rely on `zaya_server` auto-detecting that library at startup. None of the three
> exists: there is no such CMake variable anywhere in the project (CMake warns
> "Manually-specified variables were not used"), the target produces an
> executable rather than a shared library, and no server code looks for one. The
> server's Q4NX path is unaffected by whether you build this tool.

---

## Build: llama.cpp (the vendored snapshot, no path variable)

The engine links llama.cpp from the **vendored snapshot at `third_party/llama.cpp`**,
and there is no `-DLLAMA_DIR`-style override: `CMakeLists.txt` sets
`GGML_VULKAN_DIR` to that path and `GGML_BUILD_DIR` to `<it>/build`, then imports
the static libs from there if they exist. So you build the snapshot in place, in
the configuration the engine expects — the same recipe CI uses:

```bash
cd third_party/llama.cpp
cmake -B build -G Ninja -DBUILD_SHARED_LIBS=OFF -DGGML_VULKAN=ON \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF
cmake --build build -j8 --target ggml-vulkan llama llama-common
```

`BUILD_SHARED_LIBS=OFF` matters: with shared ggml the import check in
`CMakeLists.txt` does not match and the engine silently builds without the
ggml-vulkan path. For the HIP variant, configure the same tree with
`-DGGML_HIP=ON` instead of `-DGGML_VULKAN=ON`.

> **Corrected 2026-09-14.** This section previously ended by telling you to point
> the configure step at your own llama.cpp build with `-DLLAMA_DIR=…`. That
> variable appears nowhere in the project — CMake would warn it was unused and
> use the vendored snapshot regardless. The paragraph above is the mechanism the
> build actually has.
---

## CMake option summary

These are the options `CMakeLists.txt` actually declares (`option(NAME …)`), with
their real defaults, `grep -n 'option(' CMakeLists.txt` being the source of truth:

| Option              | Default | Description                                     |
|---------------------|---------|-------------------------------------------------|
| `CMAKE_HIP_ARCHITECTURES` | —  | **Must** be set to `gfx1151` on Strix Halo      |
| `EMBED_LEMONADE`    | ON      | Embed the Lemonade SDK server core              |
| `USE_LORA`          | ON      | LoRA adapter runtime support                    |
| `USE_CUDA`          | OFF     | CUDA backend for NVIDIA GPUs                    |
| `USE_METAL`         | OFF     | Metal backend for Apple Silicon                 |
| `USE_VULKAN`        | OFF     | Portable Vulkan backend proof (`test_vulkan_gemv`) |
| `USE_VART`          | OFF     | VART backend for Versal/Zynq DPU/NPU            |
| `USE_DIFFUSION`     | OFF     | stable-diffusion.cpp integration                |
| `USE_AUDIO_CPP`     | OFF     | audio.cpp integration                           |
| `SANITIZE`          | OFF     | AddressSanitizer + UndefinedBehaviorSanitizer   |

> **Corrected 2026-09-14.** This table listed `ZAYA_ENABLE_GPU_DECODE` and
> `ZAYA_USE_LLAMACPP_ROCM`. Neither is declared anywhere in the project: CMake
> accepts the flags, warns that they were not used, and builds the same thing.
> `zaya_gpu_decode` is unconditional, and llama.cpp comes from the vendored
> snapshot (see above) rather than from a variable you set.

---

## Running

```bash
./build/1bit zaya --model /path/to/model
```

If `libzaya_gpu_decode.so` was built and is findable, the server will print a
message at startup confirming GPU decode is active.

---

## Troubleshooting

### `hipErrorNoBinaryForGPU`

The `CMAKE_HIP_ARCHITECTURES` variable was not set, or was set to the wrong target.
Ensure it is `gfx1151` and that TheRock 7.15.0a is installed (older ROCm releases may
not include code-objects for gfx1151).

### `cannot find -lamdhip64`

ROCm is not on the linker path. Make sure TheRock is installed at
`/opt/rocm-therock` (or set `THEROCK_PIP_ROOT`) — CMakeLists.txt
auto-detects it; never point `CMAKE_PREFIX_PATH` at system ROCm
(`/opt/rocm`).

### No GPU decode even though `zaya_gpu_decode` was built

Check that the shared library is in the library search path:

```bash
export LD_LIBRARY_PATH=/path/to/zaya/build:$LD_LIBRARY_PATH
```

Also verify the model file is actually Q4NX (check the file header or extension).

### Kernel hang on first inference on Strix Halo (issue #1)

Strix Halo (gfx1151) systems may hit an amdgpu OPTC hang on the first GPU kernel launch
after cold boot. The hang is intermittent (~1 in 5 boots).

**Symptoms:** first inference call hangs; `dmesg` shows OPTC lockup messages; reboot required.

**Mitigation:**
```bash
export HSA_ENABLE_SDMA=0   # avoids the triggering OPTC code path
```
See [issue #1](https://github.com/1bit-MONSTER/1bit-MONSTER/issues/1) for details.
