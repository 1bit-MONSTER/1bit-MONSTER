#!/usr/bin/env bash
# build_llvm_aie.sh — build llvm-aie (the peano/aie2p toolchain) from source.
#
# Issue #1867: the naive `-DLLVM_ENABLE_RUNTIMES='libcxx;libcxxabi'`
# configure fails because libcxxabi requires libunwind:
#   LIBCXXABI_USE_LLVM_UNWINDER is set to ON, but libunwind is not
#   specified in LLVM_ENABLE_RUNTIMES.
# The fix is to include libunwind in LLVM_ENABLE_RUNTIMES. After a clean
# runtimes install the freshly-built clang finds its own libc++/__config_site
# and no manual header wiring is needed.
#
# Note: the source-built HEAD toolchain (LLVM 22) still has the same scalar
# RMW miscompile as LLVM 21 (issue #1864) — this recipe is for reproducing
# CI runs, not a fix for that codegen bug.
#
# Usage: bash scripts/build_llvm_aie.sh [llvm-aie-src-dir]
#   default src dir: ~/llvm-aie
#   builds into:     $SRC/build (fresh per run)
# Requires: cmake >= 3.20, ninja, a host clang/gcc with libstdc++ dev headers.
set -euo pipefail

SRC="${1:-$HOME/llvm-aie}"
[ -d "$SRC" ] || { echo "ERROR: llvm-aie source not found at $SRC"; exit 1; }
cd "$SRC"

# Pin the exact configure options that produce a working runtimes install.
# libunwind is REQUIRED (libcxxabi asserts it when LIBCXXABI_USE_LLVM_UNWINDER
# is on, which is the default). The runtimes build is a second-stage pass over
# the just-built clang; keep it in-tree so the compiler-rt builtins land in
# the same install tree the kernel build later uses.
cmake -G Ninja -B build -S llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="clang;lld;mlir" \
    -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind" \
    -DLLVM_TARGETS_TO_BUILD="AArch64;X86" \
    -DLLVM_EXTERNAL_PROJECTS="mlir-aie" \
    -DLLVM_EXTERNAL_MLIR_AIE_SOURCE_DIR="$SRC" \
    -DMLIR_AIE_ENABLE_LIBXAIENGINE=OFF

cmake --build build --target install -j"$(nproc)"

echo "==> llvm-aie installed under $SRC/build"
echo "    kernel compile uses: $SRC/build/bin/clang++ --target=aie2p-none-unknown-elf"
echo "    sanity check: $SRC/build/bin/clang++ --version | grep 'LLVM version'"
