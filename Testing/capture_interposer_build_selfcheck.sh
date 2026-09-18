#!/usr/bin/env bash
# capture_interposer_build_selfcheck.sh — the LD_PRELOAD interposer must at least compile.
#
# Why (#2540): nothing in this tree builds npu-infer/tools/capture/cap_interposer.cpp. No
# workflow and no CMake target names it; the only build instruction is the file's own
# header comment:
#
#   g++ -O2 -fPIC -shared cap_interposer.cpp -o cap_interposer.so -ldl -lxrt_coreutil
#
# So a missing include, a renamed XRT symbol or a typo lands silently and is found when
# someone preloads the .so on a live capture — the one situation where this file's
# correctness protects the disk (it is the file #2528 is about: the dump gates and the
# measured 181 GB trap). A file with no build gate rots.
#
# This is a self-check rather than a step in the required `C++ (cmake configure + build)`
# job on purpose: it needs XRT headers, which are not on every runner. When none of the
# candidate roots has them it prints a `skip:` line and exits 0 — a visible skip, not a
# silent pass (#2508's distinction). When they are present it compiles the real file and
# also proves the check can fail, by compiling a copy with an injected syntax error.
#
# Run: bash Testing/capture_interposer_build_selfcheck.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:-$REPO/npu-infer/tools/capture/cap_interposer.cpp}"   # arg: source to check (for controls)
[ -f "$SRC" ] || { echo "FAIL: no $SRC"; exit 1; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0

# XRT headers move between installs: /opt/xrt is what the file's comment assumes, the
# driver checkout is what this box has.
INC=""
for cand in /opt/xrt/include "$HOME/xdna-driver/xrt/src/runtime_src/core/include"; do
    if [ -f "$cand/xrt/xrt_bo.h" ]; then INC="$cand"; break; fi
done

if [ -z "$INC" ]; then
    echo "skip: no XRT headers in any candidate root (/opt/xrt/include, ~/xdna-driver/xrt/...)"
    echo "      not a pass — cap_interposer.cpp was NOT compiled here"
    exit 0
fi
echo "using XRT headers: $INC"

# The documented command, minus -lxrt_coreutil: the interposer resolves every XRT symbol
# at runtime through dlsym, so the shared object links with them undefined. Naming the
# library would make this check depend on where a distribution put it.
if g++ -O2 -fPIC -shared "$SRC" -o "$T/cap_interposer.so" -I"$INC" -ldl 2>"$T/build.log"; then
    sz=$(stat -c%s "$T/cap_interposer.so" 2>/dev/null || echo 0)
    if [ "$sz" -gt 10000 ]; then
        echo "ok: cap_interposer.cpp compiles ($sz byte .so)"
    else
        echo "FAIL: g++ exited 0 but produced a ${sz}-byte .so"; fail=1
    fi
else
    echo "FAIL: cap_interposer.cpp does not compile"
    tail -6 "$T/build.log" | sed 's/^/      /'
    fail=1
fi

# Control: the same compile on a copy with one injected syntax error must be rejected.
# Without this, a g++ that silently succeeded on anything would pass the check above.
sed 's/^int main/int main( = broken/' "$SRC" > "$T/broken.cpp" 2>/dev/null
if cmp -s "$SRC" "$T/broken.cpp"; then
    printf '\nthis is not valid C++\n' >> "$T/broken.cpp"
fi
if g++ -O2 -fPIC -shared "$T/broken.cpp" -o "$T/broken.so" -I"$INC" -ldl 2>/dev/null; then
    echo "FAIL: control — a file with an injected syntax error still compiled"
    fail=1
else
    echo "ok: control — the injected syntax error is rejected"
fi

exit "$fail"
