#!/usr/bin/env bash
# build-hrx-gfx1151.sh — build AMD's HRX llama.cpp with gfx1151 kernel catalog,
# fixing the "unsupported HRX node GET_ROWS" fail-closed on Strix Halo.
#
# PROBLEM: the shipped llamacpp-hrx bundle (hrx-b59/b66) compiles its ggml-hrx
# kernel catalog for GGML_HRX_AMDGPU_TARGETS=gfx1100 (AMD CI default). On a
# gfx1151 device the runtime's catalog lookup for get_rows (and other nodes)
# fails -> "E graph_compute: unsupported HRX node N: GET_ROWS" -> whole graph
# compute fails closed. The kernels exist; they just were never built for our
# GPU.
#
# FIX: build the same stack AMD's CI uses (llama.cpp hrx-integration/v2 fork +
# pinned ROCm/hrx loom commit) with GGML_HRX_AMDGPU_TARGETS=gfx1151. Verified
# 2026-08-30 on Strix Halo: a 2249-token prompt that hard-failed on hrx-b66
# completes with 0 GET_ROWS errors; short prompts answer correctly ("Paris").
#
# Verified pieces (this script encodes the exact proven sequence):
#   1. loom e8275fb (the commit AaronStGeorge/llamacpp_ci pins) built with
#      TheRock amdclang + amdgpu HAL, with three fixes:
#      - link_libraries(m) for the C tools (static archives need -lm AFTER
#        them; a global EXE_LINKER_FLAGS=-lm is placed before and fails)
#      - loom-link --mode=selective aliased to link (the ggml-hrx2 catalog
#        tooling passes it; public loom only knows auto|merge|link)
#      - CMAKE_C_COMPILER_AR/RANLIB forced to /usr/bin/ar|ranlib inside the
#        libhrx subproject (TheRock amdclang does not register a companion ar)
#   2. llama.cpp hrx-v2 branch, GGML_HRX=ON (v1 backend — what ships),
#      GGML_HRX_AMDGPU_TARGETS=gfx1151
#      - -isystem TheRock HIP headers in the kernel compile (system
#        /usr/include/hip shadows TheRock's and breaks __ocml_* builtins —
#        same fix as 1bit-MONSTER commit 78af6199)
#      - hrx_executable_export_info_t.constant_count -> constant_byte_length
#        (pinned hrx API uses byte length; the fork expects a u32 count)
#
# Run: bash scripts/build-hrx-gfx1151.sh
# Output: /tmp/hrx-llama-gfx1151/bin/llama-server
#   LD_LIBRARY_PATH=/tmp/hrx-install-gfx1151/lib \
#   GGML_DISABLE_VULKAN=1 \
#   /tmp/hrx-llama-gfx1151/bin/llama-server -m model.gguf --device HRX0 ...
#
# Requires: /opt/rocm-therock (amdclang), git, cmake, ninja, python3.

set -euo pipefail

ROCK=${ROCK:-/opt/rocm-therock}
WORK=${WORK:-/tmp/hrx-gfx1151}
HRX_REPO=${HRX_REPO:-https://github.com/ROCm/hrx.git}
HRX_COMMIT=e8275fbb5873157fc7b07cd20db8c01cfe713505
LLAMA_REPO=${LLAMA_REPO:-https://github.com/AMD-Ecosystem/llama.cpp.git}
LLAMA_BRANCH=hrx-v2
HRX_INSTALL=$WORK/hrx-install
LLAMA_SRC=$WORK/llama
LLAMA_BUILD=$WORK/llama-build

mkdir -p "$WORK"

# ── 1. HRX / Loom (pinned commit) ────────────────────────────────────────
if [ ! -d "$WORK/hrx" ]; then
    git clone --depth 1 "$HRX_REPO" "$WORK/hrx"
fi
( cd "$WORK/hrx" && git fetch --depth 1 origin "$HRX_COMMIT" && git checkout -q "$HRX_COMMIT" )

# 1a. libm for loom C tools (after the archives in the link line).
if ! grep -q "link_libraries(m)" "$WORK/hrx/CMakeLists.txt"; then
    python3 - "$WORK/hrx/CMakeLists.txt" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = "if(LOOM_BUILD)\n  add_subdirectory(loom/py/loom)"
new = ("if(LOOM_BUILD)\n"
       "  # Loom C tools use libm (expf/truncf/nearbyint/log/pow); the static\n"
       "  # archives need -lm AFTER them in the link line, so a global\n"
       "  # EXE_LINKER_FLAGS=-lm (placed before the objects) never resolves.\n"
       "  link_libraries(m)\n"
       "  add_subdirectory(loom/py/loom)")
assert old in s, "LOOM_BUILD block not found"
open(p, "w").write(s.replace(old, new))
EOF
fi

# 1b. loom-link --mode=selective alias (ggml-hrx2 catalog tooling passes it).
if ! grep -q '"selective"' "$WORK/hrx/loom/src/loom/tools/loom-link/loom-link.c"; then
    python3 - "$WORK/hrx/loom/src/loom/tools/loom-link/loom-link.c" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = ('  if (iree_string_view_equal(value, IREE_SV("link"))) {\n'
       '    *out_mode = LOOM_LINK_CLI_MODE_LINK;\n'
       '    return iree_ok_status();\n'
       '  }')
new = (old +
       '\n  // The ggml-hrx2 catalog tooling (AMD llama.cpp fork,\n'
       '  // link_hrx2_artifacts.py) passes --mode=selective for the same\n'
       '  // semantics as link: keep only the explicit --root=@symbol values\n'
       '  // and their reachable dependencies. Alias it until upstream loom\n'
       '  // lands the real selective planner (PRs #407-455).\n'
       '  if (iree_string_view_equal(value, IREE_SV("selective"))) {\n'
       '    *out_mode = LOOM_LINK_CLI_MODE_LINK;\n'
       '    return iree_ok_status();\n'
       '  }')
assert old in s, "loom-link mode parse not found"
open(p, "w").write(s.replace(old, new))
EOF
fi

# 1c. archiver for the libhrx subproject (TheRock amdclang has no companion ar).
if ! grep -q 'CMAKE_C_COMPILER_AR "/usr/bin/ar"' "$WORK/hrx/libhrx/CMakeLists.txt"; then
    python3 - "$WORK/hrx/libhrx/CMakeLists.txt" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = 'project(libhrx VERSION ${HRX_VERSION} LANGUAGES C CXX)\n'
new = (old +
       '\n# TheRock amdclang does not register a companion archiver, so CMake\n'
       '# leaves CMAKE_C_COMPILER_AR/RANLIB NOTFOUND inside this subproject\n'
       '# (not inherited across project()). Point at system binutils.\n'
       'if(NOT CMAKE_C_COMPILER_AR OR CMAKE_C_COMPILER_AR MATCHES "NOTFOUND")\n'
       '    set(CMAKE_C_COMPILER_AR "/usr/bin/ar" CACHE FILEPATH "archiver" FORCE)\n'
       '    set(CMAKE_C_COMPILER_RANLIB "/usr/bin/ranlib" CACHE FILEPATH "ranlib" FORCE)\n'
       'endif()\n'
       'if(NOT CMAKE_CXX_COMPILER_AR OR CMAKE_CXX_COMPILER_AR MATCHES "NOTFOUND")\n'
       '    set(CMAKE_CXX_COMPILER_AR "/usr/bin/ar" CACHE FILEPATH "archiver" FORCE)\n'
       '    set(CMAKE_CXX_COMPILER_RANLIB "/usr/bin/ranlib" CACHE FILEPATH "ranlib" FORCE)\n'
       'endif()\n')
assert old in s, "libhrx project() not found"
open(p, "w").write(s.replace(old, new))
EOF
fi

# 1d. Build + install HRX (amdgpu HAL + TheRock clang).
cmake -B "$WORK/hrx-build" -S "$WORK/hrx" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$ROCK/bin/amdclang" \
    -DCMAKE_CXX_COMPILER="$ROCK/bin/amdclang++" \
    -DCMAKE_C_COMPILER_AR=/usr/bin/ar -DCMAKE_CXX_COMPILER_AR=/usr/bin/ar \
    -DCMAKE_C_COMPILER_RANLIB=/usr/bin/ranlib -DCMAKE_CXX_COMPILER_RANLIB=/usr/bin/ranlib \
    -DLOOM_BUILD=ON -DLIBHRX_BUILD=ON -DIREE_HAL_DRIVER_AMDGPU=ON \
    -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_BENCHMARKS=OFF -DIREE_BUILD_SAMPLES=OFF \
    -DIREE_ENABLE_WERROR_FLAG=OFF \
    -DCMAKE_INSTALL_PREFIX="$HRX_INSTALL"
cmake --build "$WORK/hrx-build" --target install -j"$(nproc)"

# ── 2. llama.cpp HRX backend with gfx1151 kernel catalog ─────────────────
if [ ! -d "$LLAMA_SRC" ]; then
    git clone --depth 1 --branch "$LLAMA_BRANCH" --single-branch "$LLAMA_REPO" "$LLAMA_SRC"
fi

# 2a. TheRock HIP headers ahead of system /usr/include/hip (__ocml_* builtins).
if ! grep -q "rocm_sdk_core/include" "$LLAMA_SRC/ggml/src/ggml-hrx/CMakeLists.txt"; then
    python3 - "$LLAMA_SRC/ggml/src/ggml-hrx/CMakeLists.txt" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = '    list(APPEND GGML_HRX_KERNEL_COMPILE_FLAGS ${ARG_FLAGS})\n'
new = (old +
       '    # System hip-runtime-amd (/usr/include/hip) shadows TheRock HIP\n'
       '    # headers: amdclang++ adds the SDK include dir with -idirafter,\n'
       '    # which loses to /usr/include, so <hip/...> resolves to headers\n'
       '    # whose __ocml_* builtins this clang does not declare (same fix\n'
       '    # as 1bit-MONSTER commit 78af6199).\n'
       '    foreach(_cand IN ITEMS\n'
       '            "${GGML_HRX_ROCM_PATH}/lib/python3.14/site-packages/_rocm_sdk_core/include"\n'
       '            "${GGML_HRX_ROCM_PATH}/lib/python3.14/site-packages/_rocm_sdk_devel/include")\n'
       '        if (EXISTS "${_cand}/hip/hip_runtime.h")\n'
       '            list(APPEND GGML_HRX_KERNEL_COMPILE_FLAGS -isystem "${_cand}")\n'
       '            break()\n'
       '        endif()\n'
       '    endforeach()\n')
assert old in s, "kernel compile flags not found"
open(p, "w").write(s.replace(old, new))
EOF
fi

# 2b. hrx_executable_export_info_t API: pinned hrx uses constant_byte_length
#     (bytes); the fork expects a u32 constant_count.
if grep -q "constant_count \* sizeof" "$LLAMA_SRC/ggml/src/ggml-hrx/ggml-hrx.cpp"; then
    python3 - "$LLAMA_SRC/ggml/src/ggml-hrx/ggml-hrx.cpp" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
s = s.replace("export_info.constant_count * sizeof(uint32_t) == entry->constants_size",
              "export_info.constant_byte_length == entry->constants_size")
s = s.replace("export_info.constant_count,",
              "export_info.constant_byte_length,")
open(p, "w").write(s)
EOF
fi

# 2c. Configure with gfx1151 kernel catalog + build.
cmake -B "$LLAMA_BUILD" -S "$LLAMA_SRC" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HRX=ON -DGGML_HRX2=OFF -DGGML_NATIVE=OFF \
    -DGGML_HRX_AMDGPU_TARGETS=gfx1151 \
    -DGGML_HRX_ROCM_PATH="$ROCK" \
    -DGGML_HRX_CLANGXX="$ROCK/bin/amdclang++" \
    -DCMAKE_C_COMPILER="$ROCK/bin/amdclang" \
    -DCMAKE_CXX_COMPILER="$ROCK/bin/amdclang++" \
    -DCMAKE_C_COMPILER_AR=/usr/bin/ar -DCMAKE_CXX_COMPILER_AR=/usr/bin/ar \
    -DCMAKE_C_COMPILER_RANLIB=/usr/bin/ranlib -DCMAKE_CXX_COMPILER_RANLIB=/usr/bin/ranlib \
    -DCMAKE_PREFIX_PATH="$HRX_INSTALL" \
    -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF \
    -DGGML_BUILD_BENCHMARKS=OFF -DGGML_OPENMP=OFF
cmake --build "$LLAMA_BUILD" -j"$(nproc)"

cat <<EOF
============================================================
DONE. Custom HRX llama.cpp with gfx1151 kernel catalog:
  server: $LLAMA_BUILD/bin/llama-server
  run:    LD_LIBRARY_PATH=$HRX_INSTALL/lib \\
          GGML_DISABLE_VULKAN=1 \\
          $LLAMA_BUILD/bin/llama-server -m model.gguf \\
            --device HRX0 --jinja --parallel 1 --ctx-size 8192 --port 8080
This fixes "unsupported HRX node GET_ROWS" on gfx1151 (Strix Halo).

PERF NOTE (measured 2026-08-30, Qwen3-30B-A3B-Instruct-2507, gfx1151):
  large prompt (2249 tok): CUSTOM works (17.5 tok/s prefill) — b66 CANNOT
    (fails closed on GET_ROWS)
  short-prompt warm decode : CUSTOM ~26 tok/s vs b66 ~67-69 tok/s
  The gfx1151 kernels are functionally complete but not performance-tuned
  (stock CMake -O2/-O3 + TheRock clang); b66's gfx1100 HSACO is AMD-tuned.
  Correctness over speed: this build unblocks large prompts on Strix Halo.
============================================================
EOF
