#!/usr/bin/env bash
# 1bit.MONSTER — ROCm TheRock C++ SDK Setup
# Installs + configures the TheRock nightly ROCm (arch DETECTED, not assumed)
# Run: sudo bash scripts/setup-therock.sh
set -euo pipefail

ROCK_ROOT="/opt/rocm-therock"
NIGHTLY_INDEX="https://nightly.repo.amd.com/rocm/whl-next/"
# Pinned to the set that was actually TESTED on gfx1201, not floated to the newest
# nightly (which would be a combination never verified on any machine here).
THEROCK_VERSION="${THEROCK_VERSION-10.1.0a20260910}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DETECT="$SCRIPT_DIR/detect-gfx-targets.sh"

# The GPU target is DETECTED, never assumed. This script used to hardcode gfx1151,
# which installed the wrong device build on any non-Strix-Halo machine (rocBLAS
# then aborts with an empty Tensile list). Fail closed: no guessing.
[ -x "$DETECT" ] || { echo "!! missing $DETECT (run from a repo checkout)"; exit 1; }
GPU_TARGETS="$(bash "$DETECT")" || { echo "!! could not detect the GPU target; refusing to guess"; exit 1; }
echo "++ Detected GPU target(s): $(echo "$GPU_TARGETS" | tr '\n' ' ')"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  1bit.MONSTER — ROCm TheRock C++ SDK Setup              ║"
echo "╚═══════════════════════════════════════════════════════════╝"

# ── Install packages ──
# Arch-independent parts first (they provide rocminfo), then one device wheel per
# detected target, then link them into the devel tree.
PIP_FLAGS=()
if [ ! -f "$ROCK_ROOT/bin/hipcc" ]; then
    echo "++ Installing ROCm TheRock SDK to $ROCK_ROOT"
    [ -d "$ROCK_ROOT" ] || python3 -m venv "$ROCK_ROOT"
else
    echo "!! TheRock already installed, updating..."
    PIP_FLAGS+=(--upgrade)
fi
"$ROCK_ROOT/bin/pip" install ${PIP_FLAGS[@]+"${PIP_FLAGS[@]}"} \
    "rocm[libraries,devel]${THEROCK_VERSION:+==$THEROCK_VERSION}" \
    --index-url "$NIGHTLY_INDEX"
for t in $GPU_TARGETS; do
    echo "++   device package for $t"
    "$ROCK_ROOT/bin/pip" install ${PIP_FLAGS[@]+"${PIP_FLAGS[@]}"} \
        "rocm-sdk-device-$t${THEROCK_VERSION:+==$THEROCK_VERSION}" --index-url "$NIGHTLY_INDEX"
done
"$ROCK_ROOT/bin/rocm-sdk" init

# ── Ollama integration ──
if command -v ollama &>/dev/null; then
    echo "++ Configuring Ollama for TheRock..."
    mkdir -p /etc/systemd/system/ollama.service.d/
    cat > /etc/systemd/system/ollama.service.d/override.conf << 'OVERRIDE'
[Service]
# TheRock runtime is self-contained for the detected arch — no HSA override needed
Environment=HSA_OVERRIDE_GFX_VERSION=
Environment=HSA_ENABLE_SDMA=0
Environment=HIP_VISIBLE_DEVICES=0
Environment=ROCR_VISIBLE_DEVICES=0
Environment=OLLAMA_DEBUG=1
# TheRock runtime paths
Environment=LD_LIBRARY_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_devel/lib:/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib:/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_libraries/lib
Environment=ROCM_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_devel
Environment=HIP_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_devel
OVERRIDE
    systemctl daemon-reload
    systemctl restart ollama
    echo "!! Ollama restarted with TheRock"
fi

# ── systemd daily update timer ──
echo "++ Installing daily update timer..."
cat > /etc/systemd/system/rocm-therock-update.service << SVC
[Unit]
Description=ROCm TheRock daily update
After=network-online.target
Wants=network-online.target
[Service]
Type=oneshot
# Device package(s) are re-detected each run — never hardcoded (a hardcoded arch
# here silently re-installed the wrong device build on every daily tick).
ExecStart=/bin/bash -c '/opt/rocm-therock/bin/pip install --upgrade "rocm[libraries,devel]${THEROCK_VERSION:+==$THEROCK_VERSION}" --index-url $NIGHTLY_INDEX && for t in $(bash "$DETECT"); do /opt/rocm-therock/bin/pip install --upgrade "rocm-sdk-device-$t" --index-url $NIGHTLY_INDEX; done'
ExecStartPost=/opt/rocm-therock/bin/rocm-sdk init
StandardOutput=journal
User=root
SVC

cat > /etc/systemd/system/rocm-therock-update.timer << 'TMR'
[Unit]
Description=Daily ROCm TheRock update check
[Timer]
OnCalendar=daily
Persistent=true
RandomizedDelaySec=1h
[Install]
WantedBy=timers.target
TMR

systemctl daemon-reload
systemctl enable --now rocm-therock-update.timer 2>/dev/null || true

# ── Verify ──
echo ""
echo "═══ Verification ═══"
source "$ROCK_ROOT/activate.sh" 2>&1 | head -1
echo "  HIP:     $(hipcc --version 2>&1 | head -1)"
echo "  GPU:     $(rocminfo 2>/dev/null | grep 'Marketing Name' | head -1 | awk -F': *' '{print $2}')"
echo "  ROCm:    $(pip show rocm 2>/dev/null | grep Version)"
echo ""
echo "✅ TheRock C++ SDK ready"
echo "   Activate: source /opt/rocm-therock/activate.sh"
echo "   Update:   systemctl start rocm-therock-update.service"
