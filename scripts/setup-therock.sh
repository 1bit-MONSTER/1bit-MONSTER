#!/usr/bin/env bash
# 1bit.systems — ROCm TheRock SDK setup, PINNED to the AI-inference SDK.
#
# Ubuntu (the main enterprise platform) installs classic ROCm 7.2.x by
# default via amdgpu-install/apt — the enterprise-supported line. AMD's
# AI-inference SDK is TheRock 7.14.x, the modular multi-arch ROCm SDK at
# https://rocm.nightlies.amd.com/whl-multi-arch/. The same Ubuntu boxes run
# enterprise workloads and inference; only the SDK userland differs, and
# inference must resolve to TheRock — pinned exactly, never drifted.
#
# This script installs / repairs the PINNED TheRock SDK, removes the old
# daily "upgrade to latest nightly" timer (that is how boxes silently drift
# off the pinned SDK), and installs a daily VERIFY-ONLY timer instead.
# See docs/rocm-lanes.md for the full pin contract.
#
# Run: sudo bash scripts/setup-therock.sh [options]
set -euo pipefail

# ── Configuration (override via env vars or flags) ───────────────────────────
ROCK_ROOT="${ROCK_ROOT:-/opt/rocm-therock}"
NIGHTLY_INDEX="${NIGHTLY_INDEX:-https://rocm.nightlies.amd.com/whl-multi-arch/}"
GPU_TARGET="${GPU_TARGET:-gfx1151}"
# The Rock AI-inference SDK pin. Matches the Ubuntu appliance ISO pin
# (7.14.0a20260612). Bump deliberately — never via --upgrade.
THEROCK_VERSION="${THEROCK_VERSION:-7.14.0a20260612}"

FORCE_REINSTALL=0
CONFIGURE_OLLAMA=1
RUN_VERIFY=1

usage() {
    cat <<'EOF'
Usage: sudo bash scripts/setup-therock.sh [options]

Installs the pinned TheRock ROCm SDK (AI-inference SDK) for 1bit inference.

Options:
  --version <ver>   TheRock version to pin (default: 7.14.0a20260612)
  --gpu <target>    HIP GPU target (default: gfx1151)
  --reinstall       Reinstall even if the pin is already in place
  --no-ollama       Skip regenerating the ollama systemd override
  --skip-verify     Skip the final lane-verification run
  -h, --help        Show this help

Environment overrides: ROCK_ROOT, NIGHTLY_INDEX, GPU_TARGET, THEROCK_VERSION.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --version)  [ $# -ge 2 ] || { echo "error: --version needs a value" >&2; exit 2; }
                    THEROCK_VERSION="$2"; shift 2 ;;
        --gpu)      [ $# -ge 2 ] || { echo "error: --gpu needs a value" >&2; exit 2; }
                    GPU_TARGET="$2"; shift 2 ;;
        --reinstall) FORCE_REINSTALL=1; shift ;;
        --no-ollama) CONFIGURE_OLLAMA=0; shift ;;
        --skip-verify) RUN_VERIFY=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# Everything below touches /opt and /etc — run as root.
if [ "$(id -u)" -ne 0 ]; then
    exec sudo --preserve-env=ROCK_ROOT,NIGHTLY_INDEX,GPU_TARGET,THEROCK_VERSION,FORCE_REINSTALL,CONFIGURE_OLLAMA,RUN_VERIFY bash "$0" "$@"
fi

# Repo root = parent of the scripts/ dir this file lives in.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Single source of truth for lane pins (shared with the appliance ISO via
# packaging/rocm-lane-pin.env). Explicit env vars / --version flags win.
[ -f "$REPO_ROOT/packaging/rocm-lane-pin.env" ] && . "$REPO_ROOT/packaging/rocm-lane-pin.env"

info() { printf '\033[1;32m[1bit]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[1bit]\033[0m %s\n' "$*"; }

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  1bit.systems — ROCm TheRock SDK (pinned AI-inference SDK)        ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
info "Pin:      TheRock rocm ${THEROCK_VERSION} (${GPU_TARGET})"
info "Root:     ${ROCK_ROOT}"
info "Index:    ${NIGHTLY_INDEX}"

# ── Helpers ──────────────────────────────────────────────────────────────────

# Path to the expanded _rocm_sdk_devel tree (ROCM_PATH/HIP_PATH root).
devel_root() {
    if [ -x "$ROCK_ROOT/bin/rocm-sdk" ]; then
        local p
        p="$("$ROCK_ROOT/bin/rocm-sdk" path --root 2>/dev/null || true)"
        [ -n "$p" ] && [ -d "$p" ] && { printf '%s' "$p"; return 0; }
    fi
    local g
    g="$(find "$ROCK_ROOT/lib" -maxdepth 4 -type d -name _rocm_sdk_devel 2>/dev/null | head -1)"
    [ -n "$g" ] && [ -d "$g" ] && { printf '%s' "$g"; return 0; }
    return 1
}

installed_version() {
    [ -x "$ROCK_ROOT/bin/pip" ] || return 1
    "$ROCK_ROOT/bin/pip" show rocm 2>/dev/null | sed -n 's/^Version: //p' | head -1
}

# Warn if a classic-lane (7.2.x) ROCm userland exists on this box. The kernel
# driver is fine; a classic USERLAND shadowing TheRock is the bug we prevent.
classic_lane_check() {
    local found=0 d
    for d in /opt/rocm /opt/rocm-*; do
        [ -d "$d" ] || continue
        case "$d" in
            /opt/rocm-therock|/opt/rocm-therock-*) continue ;;
        esac
        warn "Ubuntu-default classic ROCm userland (7.2.x) found at ${d}"
        found=1
    done
    if command -v amdgpu-install >/dev/null 2>&1; then
        warn "amdgpu-install is present — it installs classic ROCm 7.2.x by default."
        found=1
    fi
    if dpkg -l 2>/dev/null | grep -qE '^ii\s+(rocm-|hip-|amd-smi)'; then
        warn "Classic ROCm apt packages are installed (Ubuntu default line)."
        found=1
    fi
    if [ "$found" = 1 ]; then
        warn "1bit inference uses TheRock at ${ROCK_ROOT} regardless — env.sh and"
        warn "the ollama override pin SDK precedence. Never add classic /opt/rocm"
        warn "userland to LD_LIBRARY_PATH ahead of TheRock (docs/rocm-lanes.md)."
    fi
}

install_pinned() {
    if [ ! -x "$ROCK_ROOT/bin/python" ]; then
        [ -d "$ROCK_ROOT" ] || mkdir -p "$ROCK_ROOT"
        python3 -m venv "$ROCK_ROOT"
    fi
    info "Installing rocm[libraries,devel,device-${GPU_TARGET}]==${THEROCK_VERSION} ..."
    "$ROCK_ROOT/bin/pip" install \
        "rocm[libraries,devel,device-${GPU_TARGET}]==${THEROCK_VERSION}" \
        --index-url "$NIGHTLY_INDEX"
    "$ROCK_ROOT/bin/rocm-sdk" init
}

# System-wide lane env so non-repo inference (python, vllm, ...) also resolves
# to TheRock first instead of whatever Ubuntu put on PATH.
write_env_snippet() {
    local devel
    devel="$(devel_root || true)"
    [ -n "$devel" ] || { warn "devel root not found — skipping /etc/profile.d lane snippet"; return 0; }
    cat > /etc/profile.d/1bit-rocm-lane.sh <<EOF
# 1bit.systems — ROCm lane: TheRock (AI-inference) shadows Ubuntu's classic
# ROCm 7.2.x (Ubuntu default). See docs/rocm-lanes.md in the 1bit repo.
export ROCM_PATH="${devel}"
export HIP_PATH="${devel}"
export LD_LIBRARY_PATH="${devel}/lib:\${LD_LIBRARY_PATH:-}"
EOF
    if [ -d "$devel/bin" ]; then
        printf 'export PATH="%s/bin:$PATH"\n' "$devel" >> /etc/profile.d/1bit-rocm-lane.sh
    fi
    info "Wrote /etc/profile.d/1bit-rocm-lane.sh (lane env for all shells/services)"
}

# Regenerate the ollama override with the CURRENT devel paths (they moved
# across python versions) and the pinned lane env.
configure_ollama() {
    command -v ollama >/dev/null 2>&1 || return 0
    local devel
    devel="$(devel_root || true)"
    [ -n "$devel" ] || { warn "devel root not found — skipping ollama override"; return 0; }
    mkdir -p /etc/systemd/system/ollama.service.d/
    cat > /etc/systemd/system/ollama.service.d/override.conf <<EOF
[Service]
# TheRock ${THEROCK_VERSION} has native ${GPU_TARGET} — no HSA override needed
Environment=HSA_OVERRIDE_GFX_VERSION=
Environment=HSA_ENABLE_SDMA=0
Environment=HIP_VISIBLE_DEVICES=0
Environment=ROCR_VISIBLE_DEVICES=0
Environment=OLLAMA_DEBUG=1
Environment=LD_LIBRARY_PATH=${devel}/lib
Environment=ROCM_PATH=${devel}
Environment=HIP_PATH=${devel}
EOF
    systemctl daemon-reload
    systemctl restart ollama
    info "Ollama override regenerated (TheRock ${THEROCK_VERSION} paths) + restarted"
}

# REPLACE the old daily upgrade timer (silent nightly drift off the pin) with
# a daily VERIFY-ONLY timer. Drift now shows up in `systemctl --failed` /
# journald instead of silently upgrading to a nightly.
install_verify_timer() {
    local src="$REPO_ROOT/scripts/verify-rocm-lane.sh"
    if [ -f "$src" ]; then
        install -m 0755 "$src" "$ROCK_ROOT/bin/verify-rocm-lane.sh"
    else
        warn "scripts/verify-rocm-lane.sh not found next to this installer — timer will only run rocm-sdk test"
    fi

    cat > /etc/systemd/system/rocm-therock-verify.service <<SVC
[Unit]
Description=ROCm TheRock lane verification (pin ${THEROCK_VERSION})
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=${ROCK_ROOT}/bin/verify-rocm-lane.sh --quiet --version ${THEROCK_VERSION}
ExecStartPost=${ROCK_ROOT}/bin/rocm-sdk test
StandardOutput=journal
User=root
SVC

    cat > /etc/systemd/system/rocm-therock-verify.timer <<TMR
[Unit]
Description=Daily ROCm TheRock lane verification
[Timer]
OnCalendar=daily
Persistent=true
RandomizedDelaySec=1h
[Install]
WantedBy=timers.target
TMR

    # Retire the old nightly-upgrade units if present.
    systemctl stop rocm-therock-update.timer 2>/dev/null || true
    systemctl disable rocm-therock-update.service rocm-therock-update.timer 2>/dev/null || true
    rm -f /etc/systemd/system/rocm-therock-update.service /etc/systemd/system/rocm-therock-update.timer

    systemctl daemon-reload
    systemctl enable --now rocm-therock-verify.timer
    info "Daily upgrade timer retired; daily verify timer enabled (rocm-therock-verify)"
}

verify_lane() {
    local src="$REPO_ROOT/scripts/verify-rocm-lane.sh"
    if [ -f "$src" ]; then
        if ! bash "$src" --version "$THEROCK_VERSION"; then
            warn "Lane verification FAILED — see scripts/verify-rocm-lane.sh exit codes."
            exit 1
        fi
    else
        warn "verify-rocm-lane.sh not found; skipping final lane check."
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────

classic_lane_check

CUR="$(installed_version || true)"
if [ "$FORCE_REINSTALL" = 1 ]; then
    info "Forced reinstall of TheRock ${THEROCK_VERSION}."
    install_pinned
elif [ -z "$CUR" ]; then
    info "Installing TheRock ${THEROCK_VERSION} (${GPU_TARGET}) to ${ROCK_ROOT} ..."
    install_pinned
elif [ "$CUR" != "$THEROCK_VERSION" ]; then
    warn "Installed rocm ${CUR} != pin ${THEROCK_VERSION} — realigning to the"
    warn "pinned AI-inference SDK (no silent nightly drift)."
    install_pinned
else
    info "TheRock rocm ${THEROCK_VERSION} already pinned — verifying installation."
fi

"$ROCK_ROOT/bin/rocm-sdk" init
write_env_snippet
if [ "$CONFIGURE_OLLAMA" = 1 ]; then
    configure_ollama
fi
install_verify_timer
if [ "$RUN_VERIFY" = 1 ]; then
    verify_lane
fi

echo ""
info "═══ Summary ═══"
DEVEL="$(devel_root || true)"
if [ -n "$DEVEL" ] && [ -x "$DEVEL/bin/hipcc" ]; then
    info "  HIP:    $DEVEL/bin/hipcc"
    "$DEVEL/bin/hipcc" --version 2>/dev/null | head -1 | sed 's/^/          /'
else
    info "  HIP:    (devel tree not expanded yet — run: $ROCK_ROOT/bin/rocm-sdk init)"
fi
info "  GPU:    $(rocminfo 2>/dev/null | grep 'Marketing Name' | head -1 | awk -F': *' '{print $2}')"
info "  ROCm:   rocm $(installed_version)"
if bash "$ROCK_ROOT/bin/verify-rocm-lane.sh" --quiet --version "$THEROCK_VERSION" >/dev/null 2>&1; then
    info "  Lane:   TheRock pin active ✓"
else
    info "  Lane:   CHECK FAILED — run scripts/verify-rocm-lane.sh"
fi
info "  Env:    source ${REPO_ROOT}/env.sh   (or relogin for /etc/profile.d)"
info "  Verify: systemctl status rocm-therock-verify.timer"
echo ""
echo "✅ TheRock ${THEROCK_VERSION} pinned as the AI-inference SDK"
