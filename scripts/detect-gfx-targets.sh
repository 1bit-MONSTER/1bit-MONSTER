#!/usr/bin/env bash
# detect-gfx-targets.sh — print the AMD GPU gfx target(s) present on THIS machine.
#
# WHY THIS EXISTS
# The same GPU-arch choice has to be made in four places (install.sh,
# scripts/setup-therock.sh, .github/workflows/release.yml, .github/workflows/ci.yml)
# and every one of them hardcoded gfx1151. That is what left ryzen (a gfx1201
# RX 9070 XT) running the gfx1151 device build for weeks, with rocBLAS aborting
# on every call. Keeping the rule in one script means it can only be got wrong
# in one place.
#
# Output: one target per line, sorted, e.g. on ryzen:
#     gfx1036
#     gfx1201
#   and on a Strix Halo box:
#     gfx1151
#
# Exit:  0 with >=1 target on stdout
#        1 with a diagnostic on stderr
#
# It NEVER falls back to a default architecture. A wrong device build is worse
# than a loud failure: it installs cleanly and then fails at runtime with an
# empty Tensile list. Fail closed.
#
# Usage:
#   scripts/detect-gfx-targets.sh            # targets, one per line
#   scripts/detect-gfx-targets.sh --quiet    # no stderr chatter on success
#   scripts/detect-gfx-targets.sh --cmake    # ';'-joined, for CMAKE_HIP_ARCHITECTURES
#
# Honours THEROCK_PIP_ROOT to locate rocminfo inside a venv SDK. On a machine
# with no SDK yet, install the arch-independent parts first
# (`rocm[libraries,devel]`), then run this, then install the device wheel(s) —
# rocminfo lives in _rocm_sdk_core, which needs no device package.

set -uo pipefail

MODE="list"
case "${1:-}" in
    --quiet) MODE="list"; QUIET=1 ;;
    --cmake) MODE="cmake"; QUIET=1 ;;
    "")      MODE="list"; QUIET=0 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "detect-gfx-targets: unknown argument '$1'" >&2; exit 2 ;;
esac
QUIET="${QUIET:-0}"

# ----------------------------------------------------------------- rocminfo
find_rocminfo() {
    local b base
    if command -v rocminfo >/dev/null 2>&1; then command -v rocminfo; return 0; fi
    local bases=()
    [[ -n ${THEROCK_PIP_ROOT:-} ]] && bases+=("$THEROCK_PIP_ROOT")
    bases+=("$HOME/.cache/pip/therock" "/opt/rocm-therock" "/opt/rocm")
    for base in "${bases[@]}"; do
        [[ -d $base ]] || continue
        for b in "$base"/lib/python*/site-packages/_rocm_sdk_core/bin/rocminfo \
                 "$base"/bin/rocminfo; do
            [[ -x $b ]] && { echo "$b"; return 0; }
        done
    done
    return 1
}

ROCMINFO="$(find_rocminfo)" || {
    echo "detect-gfx-targets: no rocminfo found." >&2
    echo "  Install the arch-independent SDK first:  pip install 'rocm[libraries,devel]'" >&2
    echo "  (or set THEROCK_PIP_ROOT to an existing venv SDK), then re-run." >&2
    echo "  Refusing to guess an architecture." >&2
    exit 1
}

# rocminfo needs the HSA runtime from its own tree.
TREE="$(dirname "$(dirname "$ROCMINFO")")"          # .../_rocm_sdk_core
HSA_LIB="$TREE/lib"
[[ -d $HSA_LIB ]] || HSA_LIB="$(dirname "$ROCMINFO")/../lib"

RAW="$(LD_LIBRARY_PATH="$HSA_LIB:${LD_LIBRARY_PATH:-}" timeout 60 "$ROCMINFO" 2>/dev/null \
        | grep -oE 'gfx[0-9]{3,}' | sort -u)"

# rocminfo also emits coarse family prefixes (gfx10, gfx11, gfx12); the
# {3,}-digit filter above already drops those. Belt and braces if the format
# ever changes:
RAW="$(printf '%s\n' "$RAW" | grep -E '^gfx[0-9]{3,}$' || true)"

if [[ -z $RAW ]]; then
    echo "detect-gfx-targets: rocminfo ($ROCMINFO) reported no gfx target." >&2
    echo "  Check that the amdgpu kernel driver is loaded and the user can open /dev/kfd" >&2
    echo "  (group 'render'). Refusing to guess an architecture." >&2
    exit 1
fi

# ----------------------------------------------------------------- supported?
# If the SDK can tell us which targets it ships device wheels for, restrict to
# those — a detected-but-unsupported target would just make pip fail later with
# a less obvious message.
if command -v rocm-sdk >/dev/null 2>&1; then
    SUPPORTED="$(rocm-sdk targets 2>/dev/null | tr ';' '\n' | grep -E '^gfx[0-9]{3,}$' || true)"
    if [[ -n $SUPPORTED ]]; then
        KEPT="$(printf '%s\n' "$RAW" | grep -FxF -f <(printf '%s\n' "$SUPPORTED") || true)"
        if [[ -z $KEPT ]]; then
            echo "detect-gfx-targets: detected $RAW but the SDK reports no device wheel for any of them." >&2
            echo "  Supported: $(printf '%s' "$SUPPORTED" | tr '\n' ' ')" >&2
            exit 1
        fi
        if [[ $QUIET -eq 0 && $KEPT != "$RAW" ]]; then
            echo "detect-gfx-targets: dropping unsupported target(s): $(comm -23 <(printf '%s\n' "$RAW") <(printf '%s\n' "$KEPT") | tr '\n' ' ')" >&2
        fi
        RAW="$KEPT"
    fi
fi

if [[ $MODE == cmake ]]; then
    printf '%s\n' "$RAW" | paste -sd';'
else
    printf '%s\n' "$RAW"
fi
