#!/usr/bin/env bash
# 1bit.systems — verify which AMD ROCm lane is ACTIVE for inference.
#
# Ubuntu (the main enterprise platform) installs classic ROCm 7.2.x by
# default via amdgpu-install/apt — the enterprise-supported line; the amdgpu
# KERNEL driver from it is required on every box. AMD's AI-inference SDK is
# TheRock 7.14.x, the modular multi-arch ROCm SDK published at
# https://rocm.nightlies.amd.com/whl-multi-arch/. 1bit inference must
# resolve to TheRock — never the classic 7.2.x userland, and never a
# nightly that drifted past the pin.
#
# This script asserts the pin is installed AND active, and fails loudly when
# a setup landed on the wrong lane. It is used by humans, by the daily
# verify-only timer installed via scripts/setup-therock.sh, and by CI.
#
# Exit codes:
#   0  TheRock pin is installed AND active (nothing shadows it)
#   1  TheRock install is missing at ROCK_ROOT
#   2  Classic lane userland (7.2.x / /opt/rocm) shadows TheRock
#   3  TheRock installed, but version != expected pin
#   4  Usage error
set -uo pipefail

ROCK_ROOT="${ROCK_ROOT:-/opt/rocm-therock}"
EXPECTED_VERSION="${EXPECTED_VERSION:-7.14.0a20260612}"
QUIET=0

usage() {
    cat <<'EOF'
Usage: scripts/verify-rocm-lane.sh [--version <pin>] [--quiet]

Verifies the active AMD ROCm lane for 1bit inference:

  --version <pin>   Expected TheRock version (default: 7.14.0a20260612)
  --quiet           No output on success (failures still print to stderr)

Exit codes: 0 = TheRock pin active, 1 = TheRock missing, 2 = classic lane
(7.2.x) shadows TheRock, 3 = version drift from the pin, 4 = usage error.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --version)
            [ $# -ge 2 ] || { echo "error: --version needs a value" >&2; exit 4; }
            EXPECTED_VERSION="$2"; shift 2 ;;
        --quiet) QUIET=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown option: $1" >&2; usage >&2; exit 4 ;;
    esac
done

log() { [ "$QUIET" = 1 ] || printf '%s\n' "$*"; }
warn() { printf '%s\n' "$*" >&2; }

# ── Lane detection ───────────────────────────────────────────────────────────

# Directories that hold a CLASSIC (enterprise-lane) ROCm userland, e.g.
# /opt/rocm (Ubuntu/amdgpu-install default) or /opt/rocm-7.2.4.
classic_rocm_dirs() {
    local d
    for d in /opt/rocm /opt/rocm-*; do
        [ -d "$d" ] || continue
        case "$d" in
            /opt/rocm-therock|/opt/rocm-therock-*) continue ;;
        esac
        printf '%s\n' "$d"
    done
}

# Is a path inside a classic-lane ROCm tree (never TheRock's own tree)?
is_classic_path() {
    case "$1" in
        /opt/rocm/*|/opt/rocm-*)
            case "$1" in
                /opt/rocm-therock/*|/opt/rocm-therock-*) return 1 ;;
            esac
            return 0 ;;
    esac
    return 1
}

therock_version() {
    [ -x "$ROCK_ROOT/bin/pip" ] || return 1
    "$ROCK_ROOT/bin/pip" show rocm 2>/dev/null | sed -n 's/^Version: //p' | head -1
}

# ── Main ─────────────────────────────────────────────────────────────────────

if [ ! -x "$ROCK_ROOT/bin/rocm-sdk" ]; then
    warn "✗ TheRock install missing at $ROCK_ROOT (expected $EXPECTED_VERSION)."
    warn "  Install with: sudo bash scripts/setup-therock.sh"
    warn "  (Only applies to TheRock-pinned targets — appliance/Strix Halo. A box"
    warn "   that intentionally runs the classic lane, e.g. the Ryzen reference box"
    warn "   with Ubuntu 24.04 ROCm 7.2.4, is outside this check.)"
    exit 1
fi

VER="$(therock_version || true)"
if [ -z "$VER" ]; then
    warn "✗ TheRock venv exists at $ROCK_ROOT but 'rocm' is not installed in it."
    exit 1
fi

log "── ROCm SDK check ─────────────────────────────────────────"
log "  TheRock (AI-inference SDK): $ROCK_ROOT"
log "    installed: rocm $VER   expected: rocm $EXPECTED_VERSION"

CLASSIC_DIRS="$(classic_rocm_dirs)"
if [ -n "$CLASSIC_DIRS" ]; then
    log "  Classic  (Ubuntu default, 7.2.x):"
    while IFS= read -r d; do log "    $d"; done <<< "$CLASSIC_DIRS"
fi
if command -v amdgpu-install >/dev/null 2>&1; then
    log "    amdgpu-install present (installs classic 7.2.x by default)"
fi

ACTIVE_HIPCC="$(command -v hipcc 2>/dev/null || true)"
if [ -n "$ACTIVE_HIPCC" ]; then
    log "  active hipcc: $ACTIVE_HIPCC"
else
    log "  active hipcc: (none on PATH — source env.sh after install)"
fi

# 1) Classic userland must not shadow TheRock on PATH.
if [ -n "$ACTIVE_HIPCC" ] && is_classic_path "$ACTIVE_HIPCC"; then
    warn ""
    warn "✗ Ubuntu's default classic ROCm (7.2.x) shadows TheRock:"
    warn "    hipcc resolves to $ACTIVE_HIPCC"
    warn "  Inference must use TheRock $EXPECTED_VERSION. Fix PATH ordering:"
    warn "    source $ROCK_ROOT/../env.sh   (repo env.sh)"
    warn "  or re-run: sudo bash scripts/setup-therock.sh"
    exit 2
fi

# 2) TheRock must sit exactly on the pin — no nightly drift.
if [ "$VER" != "$EXPECTED_VERSION" ]; then
    warn ""
    warn "✗ TheRock version drift: installed $VER, pin is $EXPECTED_VERSION."
    warn "  Nightly upgrades are disabled — realign with:"
    warn "    sudo bash scripts/setup-therock.sh --version $EXPECTED_VERSION --reinstall"
    exit 3
fi

log "  ✓ TheRock $VER (pin $EXPECTED_VERSION) is the active inference SDK."
exit 0
