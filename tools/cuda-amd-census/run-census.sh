#!/usr/bin/env bash
# run-census.sh — CUDA-surface -> AMD backend capability census (Linux).
#
# Two backends are probed with the SAME binaries, because HRX ships its own
# libamdhip64.so HIP-compatibility layer:
#
#   rocm lane : ROCm/TheRock HIP + rocBLAS/rocFFT/rocSPARSE/hipSOLVER/MIOpen
#   hrx  lane : AMD HRX (libhrx.so) + Loom JIT (libloomc.so) -- no ROCm needed
#
# Evidence classes are the upstream ones from Speedstu/CUDA-for-AMD-Windows
# (PASS / UNSUPPORTED / INCORRECT / TIMEOUT / ERROR) plus two census-only levels
# that are NEVER promoted to functional evidence:
#
#   detected   device/library/header present   (existence, not capability)
#   loadable   present and loads, not exercised
#
# Rules enforced:
#   * a probe with no PROBE| line and rc=124 is TIMEOUT, never a silent pass
#   * a missing stack is UNSUPPORTED, not a hardware failure
#   * nothing is promoted: detected != loadable != functional
#   * every functional row must match an independent CPU reference AND prove
#     device execution (upstream: "Safe failure matters")
#
# Usage: run-census.sh [--machine NAME] [--out DIR] [--timeout SECONDS]
#        TL_SHORT=<s> overrides the first (short) timeout budget.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE="$HERE/probe"
MACHINE="$(hostname -s)"
OUT="$HERE/reports"
TL=60
while [[ $# -gt 0 ]]; do
    case "$1" in
        --machine) MACHINE="$2"; shift 2 ;;
        --out)     OUT="$2"; shift 2 ;;
        --timeout) TL="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
mkdir -p "$OUT"
ROW_FILE="$OUT/$MACHINE.rows.tsv"
WORK="$(mktemp -d)"
: > "$ROW_FILE"

log() { printf '%s\n' "$*" >&2; }
row() {
    local s="$1" r="$2" d="$3"
    d="${d//$'\t'/ }"; d="${d//$'\n'/ }"
    printf '%s\t%s\t%s\n' "$s" "$r" "$d" >> "$ROW_FILE"
    printf '  %-22s %-12s %s\n' "$s" "$r" "$d" >&2
}

# classify <surface> <logfile> <rc>
classify() {
    local surf="$1" lf="$2" rc="$3"
    local line; line="$(grep -m1 '^PROBE|' "$lf" 2>/dev/null)"
    if [[ -n $line ]]; then
        row "$surf" "$(cut -d'|' -f3 <<<"$line")" "$(cut -d'|' -f4 <<<"$line")"
    elif [[ $rc -eq 124 ]]; then
        row "$surf" TIMEOUT "no result within ${TL}s — last: $(tail -c 200 "$lf" 2>/dev/null | tr '\n' ' ')"
    elif [[ $rc -eq 127 ]]; then
        row "$surf" UNSUPPORTED "not runnable: $(tail -c 200 "$lf" 2>/dev/null | tr '\n' ' ')"
    else
        row "$surf" ERROR "rc=$rc no PROBE line: $(tail -c 240 "$lf" 2>/dev/null | tr '\n' ' ')"
    fi
}

GPU_PCI="$(lspci 2>/dev/null | grep -iE 'vga|display|3d' | sed 's/^[0-9a-f:.]* //' | paste -sd'; ' -)"
OS="$(grep -oP '(?<=PRETTY_NAME=").*(?=")' /etc/os-release 2>/dev/null)"
KERN="$(uname -r)"

# ------------------------------------------------------------------ discovery
find_rocm() {
    local c
    for c in ${ROCM_DEVEL:-} /opt/rocm-therock/lib/python*/site-packages/_rocm_sdk_devel \
             /opt/rocm/lib/python*/site-packages/_rocm_sdk_devel /opt/rocm-* /opt/rocm; do
        [[ -d "$c" ]] || continue
        [[ -e "$c/bin/hipcc" || -e "$c/include/hip/hip_runtime.h" ]] && { echo "$c"; return 0; }
    done
    return 1
}
# HRX is self-contained: its lib dir carries libhrx.so + a libamdhip64.so drop-in.
find_hrx() {
    local c
    for c in ${HRX_LIB:-} /opt/hrx/lib $(ls -d "$HOME"/hrx-gfx1151/hrx-runtime/lib "$HOME"/hrx-*/lib 2>/dev/null); do
        [[ -d "$c" ]] || continue
        [[ -e "$c/libhrx.so" || -e "$c/libhrx.so.0" ]] && { echo "$c"; return 0; }
    done
    return 1
}
find_hrx_bin() {
    local c
    for c in ${HRX_BIN:-} /opt/hrx/bin "$HOME"/hrx-gfx1151/hrx-runtime/bin "$HOME"/hrx-*/bin; do
        [[ -x "$c/hrx-info" ]] && { echo "$c/hrx-info"; return 0; }
    done
    return 1
}

ROCM="$(find_rocm)" || ROCM=""
HIPCC=""; [[ -n $ROCM && -x "$ROCM/bin/hipcc" ]] && HIPCC="$ROCM/bin/hipcc"
[[ -z $HIPCC ]] && HIPCC="$(command -v hipcc || true)"
HRX_LIBDIR="$(find_hrx)" || HRX_LIBDIR=""
HRX_INFO="$(find_hrx_bin)" || HRX_INFO=""

log "[census] machine=$MACHINE os=$OS kernel=$KERN"
log "[census] gpu: $GPU_PCI"
log "[census] rocm=${ROCM:-NONE} hipcc=${HIPCC:-NONE}"
log "[census] hrx=${HRX_LIBDIR:-NONE} hrx-info=${HRX_INFO:-NONE}"

# ------------------------------------------------------------------ inventory
for dev in /dev/kfd /dev/dri/renderD128 /dev/dri/renderD129 /dev/accel/accel0; do
    if [[ -e $dev ]]; then row "dev/$(basename "$dev")" detected "$(stat -c '%a %U:%G' "$dev")"
    else row "dev/$(basename "$dev")" UNSUPPORTED "absent"; fi
done

declare -A MAP=(
    [runtime]="hip_runtime.h" [cublas]="rocblas" [cublaslt]="hipblaslt"
    [cusparse]="rocsparse" [cufft]="rocfft" [cudnn]="miopen"
    [cusolver]="hipsolver" [nccl]="rccl" [graphs]="hip_runtime.h"
    [ptx_jit]="hiprtc" [bf16_atomics]="hip_bfloat16.h"
)
for s in $(printf '%s\n' "${!MAP[@]}" | sort); do
    pat="${MAP[$s]}"; found=""
    if [[ -n $ROCM ]]; then
        found="$(find "$ROCM/include" -maxdepth 2 -iname "*${pat}*" 2>/dev/null | head -1)"
        [[ -z $found ]] && found="$(find "$ROCM/lib" -maxdepth 1 -iname "*${pat}*" 2>/dev/null | head -1)"
    fi
    [[ -z $found ]] && found="$(find /usr/include /usr/lib/x86_64-linux-gnu -maxdepth 3 -iname "*${pat}*" 2>/dev/null | head -1)"
    if [[ -n $found ]]; then row "$s/present" detected "${found}"
    else row "$s/present" UNSUPPORTED "no ${pat} in rocm tree or /usr"; fi
done

# HRX / Loom inventory ---------------------------------------------------------
if [[ -n $HRX_LIBDIR ]]; then
    row "hrx/present" detected "$HRX_LIBDIR ($(ls "$HRX_LIBDIR"/libhrx.so.*.* 2>/dev/null | head -1 | xargs -r basename))"
    lm="$(ls "$HRX_LIBDIR"/libloomc.so.*.* 2>/dev/null | head -1)"
    [[ -n $lm ]] && row "loom/present" detected "$(basename "$lm") (JIT compiler; kernels are specialized at launch)"
    if [[ -n $HRX_INFO ]]; then
        hlog="$WORK/hrxinfo.log"
        timeout "$TL" env LD_LIBRARY_PATH="$HRX_LIBDIR:/usr/lib/x86_64-linux-gnu" "$HRX_INFO" >"$hlog" 2>&1
        hrc=$?
        if [[ $hrc -eq 0 ]] && grep -qiE 'GPU accelerator: unavailable|NOT_FOUND' "$hlog"; then
            # HRX is a lean runtime, not a full stack: with no ROCr/HSA runtime it
            # silently falls back to CPU. That is UNSUPPORTED (missing dependency),
            # not an ERROR -- and never a silent CPU pass, which is the exact trap
            # upstream's "device detection followed by CPU fallback" rule names.
            dep="$(grep -oE 'libhsa-runtime64[^ ]*|libhsakmt[^ ]*' "$hlog" | head -1)"
            row "hrx/device_enum" UNSUPPORTED "libhrx loaded but GPU unavailable: needs ${dep:-libhsa-runtime64.so.1} (ROCm/HSA runtime) on the loader path; CPU accelerator only -> no GPU compute"
        elif [[ $hrc -eq 0 ]]; then
            dev="$(grep -m1 -iE 'gfx|GPU accelerator' "$hlog" | tr -s ' ' | head -c 160)"
            # enumeration == 'detected' per upstream's evidence ladder, not functional
            row "hrx/device_enum" detected "hrx-info rc=0; ${dev:-see log}"
        else
            row "hrx/device_enum" ERROR "hrx-info rc=$hrc: $(tail -c 200 "$hlog" | tr '\n' ' ')"
        fi
    fi
else
    row "hrx/present" UNSUPPORTED "no libhrx.so found (no AMD hrx-system bundle on this host)"
fi

# ------------------------------------------------------------------ HIP probes
LDPATH=""; [[ -n $ROCM ]] && LDPATH="$ROCM/lib:/usr/lib/x86_64-linux-gnu"
CFLAGS_HIP=(); [[ -n $ROCM ]] && CFLAGS_HIP=(-I"$ROCM/include" -isystem "$ROCM/include")
# The ROCm math libs are not auto-linked by hipcc; without these the link fails
# with "undefined symbol: rocblas_create_handle". Names are cased as shipped --
# the convolution lib is libMIOpen.so, so the flag is -lMIOpen, not -lmiopen.
LIBS_CORE=(-lhiprtc)
LIBS_HIP=(-lrocblas -lrocfft -lrocsparse -lhipsolver -lMIOpen)
EXPECT_GFX="${EXPECT_GFX:-}"
BUILT_CORE=""; BUILT_LIBS=""

# ---- two-stage timeout -------------------------------------------------------
# Short first attempt; a longer retry ONLY on a genuine timeout. Upstream's
# VALIDATION.md rationale, adopted verbatim in spirit: a single budget
# misreports slow first-touch JIT as breakage, and -- the non-obvious half --
# can hide *slow numerical corruption* in the timeout bucket. The retry must
# therefore yield a VERDICT (PASS/INCORRECT/...), never a pass, and a verdict
# reached on retry is labelled 'reclassified-from-timeout' so a reader can see
# which rows came out of the slow path. TL_SHORT is the first budget; TL is the
# retry budget.
TL_SHORT="${TL_SHORT:-12}"

run_probe_ld() { # surface ldpath binary args...
    local s="$1" ldp="$2"; shift 2
    local lf="$WORK/${s//\//_}.log" short="$WORK/${s//\//_}.short.log"
    timeout "$TL_SHORT" env LD_LIBRARY_PATH="$ldp" "$@" >"$short" 2>&1
    local rc=$?
    if [[ $rc -eq 124 && "$TL" != "$TL_SHORT" ]]; then
        timeout "$TL" env LD_LIBRARY_PATH="$ldp" "$@" >"$lf" 2>&1
        rc=$?
        if grep -q '^PROBE|' "$lf" 2>/dev/null; then
            awk -F'|' 'BEGIN{OFS="|"} /^PROBE\|/{ $4 = "reclassified-from-timeout: " $4 } { print }' \
                "$lf" > "$lf.rc" && mv "$lf.rc" "$lf"
        else
            cp "$short" "$lf"   # still timing out -> genuine TIMEOUT at the long budget
        fi
    else
        cp "$short" "$lf"
    fi
    classify "$s" "$lf" "$rc"
}

run_probe() { local s="$1"; shift; run_probe_ld "$s" "${LDPATH:-/usr/lib/x86_64-linux-gnu}" "$@"; }
# Same binary, HRX's libamdhip64 instead of ROCm's -> the second backend column.
run_hrx_probe() { local s="$1"; shift; run_probe_ld "$s/hrx" "$HRX_LIBDIR:/usr/lib/x86_64-linux-gnu" "$@"; }

if [[ -z $HIPCC ]]; then
    for s in runtime numerics graphs ptx_jit blas fft sparse solver dnn int8_dot4; do
        row "$s" UNSUPPORTED "no hipcc/ROCm toolchain on this host"
    done
else
    log "[census] compiling HIP probes with $HIPCC"
    if "$HIPCC" -O2 -std=c++17 ${CFLAGS_HIP[@]+"${CFLAGS_HIP[@]}"} "$PROBE/p_core.cpp" -o "$WORK/p_core" ${LIBS_CORE[@]+"${LIBS_CORE[@]}"} 2>"$WORK/build_core.err"; then
        BUILT_CORE="$WORK/p_core"
        run_probe runtime  "$WORK/p_core" runtime "$EXPECT_GFX"
        run_probe numerics "$WORK/p_core" numerics
        run_probe graphs   "$WORK/p_core" graphs
        run_probe ptx_jit  "$WORK/p_core" rtc
    else
        for s in runtime numerics graphs ptx_jit; do
            row "$s" UNSUPPORTED "p_core build failed: $(grep -m1 error "$WORK/build_core.err" | head -c 200)"
        done
    fi
    if "$HIPCC" -O2 -std=c++17 ${CFLAGS_HIP[@]+"${CFLAGS_HIP[@]}"} "$PROBE/p_libs.cpp" -o "$WORK/p_libs" ${LIBS_HIP[@]+"${LIBS_HIP[@]}"} 2>"$WORK/build_libs.err"; then
        BUILT_LIBS="$WORK/p_libs"
        run_probe cublas   "$WORK/p_libs" blas
        run_probe cufft    "$WORK/p_libs" fft
        run_probe cusparse "$WORK/p_libs" sparse
        run_probe cusolver "$WORK/p_libs" solver
        run_probe cudnn    "$WORK/p_libs" dnn
    else
        for s in cublas cufft cusparse cusolver cudnn; do
            row "$s" UNSUPPORTED "p_libs build failed: $(grep -m1 error "$WORK/build_libs.err" | head -c 200)"
        done
    fi
    # int8 dot4: a build failure IS the evidence (see probe/p_dot4.cpp)
    if "$HIPCC" -O2 -std=c++17 ${CFLAGS_HIP[@]+"${CFLAGS_HIP[@]}"} "$PROBE/p_dot4.cpp" -o "$WORK/p_dot4" 2>"$WORK/build_dot4.err"; then
        run_probe int8_dot4 "$WORK/p_dot4"
        BUILT_DOT4="$WORK/p_dot4"
    else
        row "int8_dot4" UNSUPPORTED "toolchain rejects sudot4 builtin with runtime operands: $(grep -m1 error "$WORK/build_dot4.err" | head -c 180)"
    fi
fi

# ------------------------------------------------------------------ HRX lane run
if [[ -n $HRX_LIBDIR ]]; then
    if [[ -n $BUILT_CORE ]]; then
        log "[census] HRX lane: running ROCm-built probe against HRX libamdhip64"
        run_hrx_probe runtime  "$BUILT_CORE" runtime "$EXPECT_GFX"
        run_hrx_probe graphs   "$BUILT_CORE" graphs
        run_hrx_probe numerics "$BUILT_CORE" numerics
    fi
    if [[ -n $BUILT_LIBS ]]; then
        run_hrx_probe cublas "$BUILT_LIBS" blas
    fi
    if [[ -n ${BUILT_DOT4:-} ]]; then
        run_hrx_probe int8_dot4 "$BUILT_DOT4"
    fi
fi

# ------------------------------------------------------------------ torch
PYBIN="${PYBIN:-python3}"
if command -v "$PYBIN" >/dev/null; then
    run_probe torch "$PYBIN" "$PROBE/p_torch.py"
else
    row torch UNSUPPORTED "no $PYBIN"
fi

# ------------------------------------------------------------------ vulkan lane
if command -v glslc >/dev/null && cc -O2 "$PROBE/vk_compute.c" -o "$WORK/vk_compute" -lvulkan 2>"$WORK/build_vk.err"; then
    glslc -O "$PROBE/vk_compute.comp" -o "$WORK/vk_compute.spv" 2>/dev/null
    if [[ -f $WORK/vk_compute.spv ]]; then run_probe vulkan_compute "$WORK/vk_compute" "$WORK/vk_compute.spv"
    else run_probe vulkan_compute "$WORK/vk_compute"; fi
else
    reason="no glslc or libvulkan headers"
    [[ -s $WORK/build_vk.err ]] && reason="build: $(tail -c 160 "$WORK/build_vk.err" | tr '\n' ' ')"
    row vulkan_compute UNSUPPORTED "$reason"
fi

# ------------------------------------------------------------------ report
python3 - "$ROW_FILE" "$OUT/$MACHINE.json" "$MACHINE" "$OS" "$KERN" "$GPU_PCI" "$ROCM" "$HRX_LIBDIR" <<'PY'
import json, sys, collections
rows_path, out_path, machine, osname, kern, gpu, rocm, hrx = sys.argv[1:9]
rows = []
with open(rows_path) as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            rows.append({"surface": p[0], "result": p[1], "detail": p[2]})
counts = collections.Counter(r["result"] for r in rows)
json.dump({
    "machine": machine, "os": osname, "kernel": kern, "gpu": gpu,
    "rocm_root": rocm or None, "hrx_libdir": hrx or None,
    "generated_by": "projects/cuda-amd-census/run-census.sh",
    "evidence_classes": ["detected", "loadable", "UNSUPPORTED", "PASS", "INCORRECT", "TIMEOUT", "ERROR"],
    "counts": dict(counts), "rows": rows,
}, open(out_path, "w"), indent=2)
print(json.dumps(counts), file=sys.stderr)
PY
log "[census] wrote $ROW_FILE and $OUT/$MACHINE.json"
