#!/usr/bin/env bash
# NOTE: `make package-site` copies this file to site/install.sh, which is what
# the website serves at https://1bit.monster/install.sh (the one-liner used on
# the Downloads page). Keep this file the source of truth — edit here, then
# re-run `make package-site` to sync the site copy.
set -euo pipefail
GREEN='\033[0;32m'; NC='\033[0m'; YELLOW='\033[1;33m'
log() { echo -e "${GREEN}[1bit]${NC} $*"; }
warn() { echo -e "${YELLOW}[1bit]${NC} $*"; }

REPO_URL="https://github.com/1bit-MONSTER/1bit-MONSTER.git"
INSTALL_DIR="${INSTALL_DIR:-$HOME/1bit}"
SKIP_ROCM=false; WITH_JARVIS=false
for arg in "$@"; do
    case "$arg" in
        --skip-rocm) SKIP_ROCM=true ;;
        --with-jarvis) WITH_JARVIS=true ;;
    esac
done
MODELS_DIR="${MODELS_DIR:-$HOME/models}"

# ── Kernel version check ───────────────────────────────────────────────────────
# amdgpu OPTC CRTC hang on Strix Halo (gfx1151) with kernel 6.19.x (issue #1)
# Works fine on 6.18.x LTS and 7.x. Warn users on 6.19.x.
KERNEL_RELEASE="$(uname -r)"
if echo "$KERNEL_RELEASE" | grep -q '^6\.19\.'; then
    warn "Kernel $KERNEL_RELEASE detected!"
    warn "Strix Halo (gfx1151) systems running 6.19.x kernels may experience"
    warn "an amdgpu OPTC CRTC hang during GPU inference (issue #1)."
    warn "Recommended: use kernel 6.18.22-lts or 7.x instead."
    warn "See: https://github.com/1bit-MONSTER/1bit-MONSTER/issues/1"
    echo ""
fi

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    echo "Usage: curl -fsSL https://raw.githubusercontent.com/1bit-MONSTER/1bit-MONSTER/main/install.sh -o install.sh"
    echo "       # Review the script, then:"
    echo "       bash install.sh [--skip-rocm] [--with-jarvis]"
    echo ""
    echo "  --skip-rocm    Skip kernel build (use pre-build librocm_cpp.so)"
    echo "  --with-jarvis  Also build JARVIS (voice assistant) and offer to start it"
    echo ""
    echo "If a SHA256 checksum file is available, verify before running:"
    echo "       sha256sum -c install.sh.sha256"
    echo ""
    echo "Installs 1bit inference engine for AMD Strix Halo (gfx1151)."
    echo "Builds pure C++ end-to-end: zaya_server, onebit (CLI), onebitd (daemon),"
    echo "unified_router (proxy), bitnet_tui (TUI) + librocm_cpp.so — no Rust, no Python."
    exit 0
fi

# ── Detect if running standalone (curl-piped) vs from repo root ────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd || echo ".")"
if [ -f "$SCRIPT_DIR/CMakeLists.txt" ]; then
    DIR="$SCRIPT_DIR"
    log "Running from repo root: $DIR"
else
    log "Cloning 1bit repository..."
    if [ -d "$INSTALL_DIR/.git" ]; then
        log "Repo already exists at $INSTALL_DIR — pulling latest"
        git -C "$INSTALL_DIR" pull --ff-only || warn "pull failed; continuing with existing copy"
    else
        git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"
    fi
    DIR="$INSTALL_DIR"
fi

# ── Install deps ──────────────────────────────────────────────────────────────
install_deps() {
    if command -v apt-get &>/dev/null; then
        log "Installing build deps (apt)..."
        sudo apt-get update -qq
        sudo apt-get install -y -qq build-essential cmake ninja-build git curl python3-pip
    elif command -v pacman &>/dev/null; then
        log "Installing build deps (pacman)..."
        sudo pacman -Sy --noconfirm base-devel cmake ninja git curl python-pip
    elif command -v dnf &>/dev/null; then
        log "Installing build deps (dnf)..."
        sudo dnf install -y gcc-c++ cmake ninja-build git curl python3-pip
    else
        warn "Unknown package manager. Install: cmake ninja git curl build-essential python3-pip"
    fi
    command -v ninja >/dev/null 2>&1 || { echo "WARNING: ninja not found, using Unix Makefiles"; CMAKE_GENERATOR=""; }
    
    # TheRock 7.15.0a — pip-installed HIP SDK for gfx1151
    if ! command -v amdclang++ &>/dev/null; then
        log "Installing TheRock 7.15.0a SDK..."
        python3 -m pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
            "rocm[libraries,devel,device-gfx1151]" 2>/dev/null || {
            warn "TheRock pip install failed. Set THEROCK_PIP_ROOT manually."
            warn "See: https://github.com/ROCm/TheRock"
        }
        export THEROCK_PIP_ROOT="$HOME/.cache/pip/therock"
    else
        log "amdclang++ found — TheRock SDK already installed"
    fi
	command -v ninja >/dev/null 2>&1 || { echo "WARNING: ninja not found, using Unix Makefiles"; CMAKE_GENERATOR=""; }
}

install_deps
mkdir -p "$MODELS_DIR"

# ── Build kernels + server (pure C++, no Rust) ───────────────────────────────
if [ "$SKIP_ROCM" = false ]; then
    log "Building C++ inference stack (server + CLI + daemon)..."
    cd "$DIR"
    cmake -B build ${CMAKE_GENERATOR:+-G Ninja} -DCMAKE_HIP_ARCHITECTURES=gfx1151 || { warn "cmake configure failed"; exit 1; }
    # ONE BINARY. Every entry point (zaya_server, onebitd, unified_router, onebit,
    # vision_server, …) is a compiled-in main() inside the single `onebin` target,
    # which is emitted as build/1bit and dispatched on argv[0]; the symlinks that
    # reach them are created just below. This line used to name standalone targets
    # (zaya_server onebitd onebit unified_router) that were folded into onebin in
    # 0761bb411 and no longer exist, so the build died with "unknown target
    # 'zaya_server'" and the default install path exited 1; the one-liner had
    # been broken since 2026-08-21. Verified against `ninja -t targets`.
    cmake --build build --target onebin -j"$(nproc)" || { warn "cmake build failed"; exit 1; }
    # The TUI is advertised in --help but is not part of the single binary. Build it
    # best-effort so a TUI-only failure cannot take the whole install down.
    cmake --build build --target bitnet_tui -j"$(nproc)" || warn "bitnet_tui did not build (optional; the CLI is unaffected)"

    # argv[0] dispatch: one ELF answers to every legacy server name. Exactly the
    # names packaging/Makefile stages; Testing/cli_smoke.py proves each reaches a
    # handler in the linked binary.
    for _name in zaya_server unified_server unified_router vision_server onebitd onebit 1bit-server; do
        ln -sf 1bit "$DIR/build/$_name"
    done

    log "Build complete:"
    log "  $DIR/build/1bit ($(stat -c%s "$DIR/build/1bit" 2>/dev/null || echo '?') bytes) — the single binary"
    log "  + argv[0] symlinks: zaya_server onebitd onebit unified_router vision_server unified_server 1bit-server"
else
    warn "--skip-rocm: kernel build skipped."
    warn "Make sure librocm_cpp.so is on LD_LIBRARY_PATH before running the server."
    log "Checking for pre-built binary..."
    if [ -x "$DIR/build/1bit" ]; then
        log "Found existing build: $DIR/build/1bit"
    else
        warn "No pre-built binary found at $DIR/build/1bit."
        warn "Run without --skip-rocm on a ROCm-equipped machine, or"
        warn "download a pre-built release from GitHub."
    fi
fi

# ── NPU lane state ────────────────────────────────────────────────────────────
# The NPU lane is the engine's own worker: src/backend_npu.cpp fork/execs
# npu_engine_universal and runs the pre-compiled xclbins — no FastFlowLM. The
# worker resolves from NPU_ENGINE_BIN, else ./npu_engine_universal (cwd-relative),
# and this script never built it, so a source install silently had no NPU lane and
# a first run only mentioned FastFlowLM. Build it when the target exists, then
# report the real state (#2358).
NPU_WORKER=""
for _cand in "${NPU_ENGINE_BIN:-}" "$DIR/build/npu_engine_universal" ./npu_engine_universal; do
    if [ -n "$_cand" ] && [ -x "$_cand" ]; then NPU_WORKER="$_cand"; break; fi
done
if [ -z "$NPU_WORKER" ] && [ -d "$DIR/build" ]; then
    log "NPU engine: building npu_engine_universal (needs XRT; skipped if unavailable)..."
    if cmake --build "$DIR/build" --target npu_engine_universal -j"$(nproc)" >/dev/null 2>&1; then
        [ -x "$DIR/build/npu_engine_universal" ] && NPU_WORKER="$DIR/build/npu_engine_universal"
    fi
fi
if [ -n "$NPU_WORKER" ]; then
    log "NPU engine: $NPU_WORKER"
    log "  enable NPU inference with: export NPU_ENGINE_BIN=$NPU_WORKER"
else
    warn "NPU engine: npu_engine_universal not built - the NPU lane stays off (CPU/GPU work)."
    warn "  build it:  cmake --build build --target npu_engine_universal   # needs XRT"
    warn "  then set:  export NPU_ENGINE_BIN=\$PWD/build/npu_engine_universal"
    warn "  (the worker is resolved by cwd, so the absolute path is worth setting)"
fi
echo ""

# ── JARVIS (optional) ────────────────────────────────────────────────────────
if [ "$WITH_JARVIS" = true ]; then
    log "Installing JARVIS (Zyphra default stack)..."
    mkdir -p "$HOME/.local/bin" "$HOME/.config/1bit"
    # JARVIS is a compiled-in main() in the single ELF, dispatched on argv[0]:
    # onebin.cpp routes `jarvis_server` → jarvis_app_main. There is no standalone
    # `jarvis` binary target, so the old `ln -sf "$DIR/build/jarvis"` pointed at a
    # file that never existed (and a plain `jarvis` symlink would not dispatch).
    ln -sf "$DIR/build/1bit" "$HOME/.local/bin/jarvis_server"
    cat > "$HOME/.config/1bit/jarvis.env" <<EOF
# JARVIS defaults — edit to taste, or pass flags: jarvis --help
1BIT_WEIGHTS_DIR=$MODELS_DIR
# Uncomment to force a specific model (default: first Zyphra model found):
# JARVIS_MODEL=ZAYA1-8B
EOF
    log "JARVIS installed: $HOME/.local/bin/jarvis_server (config: $HOME/.config/1bit/jarvis.env)"
    log "Default experience is the Zyphra stack (ZAYA/ZR1/BlackMamba/Zamba2);"
    log "use 'jarvis --model <name>' to load any other model."
    if command -v systemctl &>/dev/null && systemctl --user list-units >/dev/null 2>&1; then
        cat > "$HOME/.config/systemd/user/jarvis.service" <<EOF
[Unit]
Description=JARVIS voice assistant (1bit engine)
After=network.target

[Service]
EnvironmentFile=$HOME/.config/1bit/jarvis.env
ExecStart=$HOME/.local/bin/jarvis_server --text
Restart=on-failure

[Install]
WantedBy=default.target
EOF
        log "systemd user unit installed — start now with:"
        log "  systemctl --user enable --now jarvis"
        log "  systemctl --user status jarvis"
    fi
fi

# ── Done ──────────────────────────────────────────────────────────────────────
log ""
log "Done. Run:"
log "  source $DIR/env.sh            # sets LD_LIBRARY_PATH (+ HSA vars only on ROCm <7)"
log "  $DIR/build/1bit zaya          # or: $DIR/build/zaya_server"
log ""
log "Then send requests:"
log '  curl -X POST http://localhost:8088/completion \'
log '    -H "Content-Type: application/json" \'
log '    -d '\''{"prompt":"Hello","n_predict":16}'\'
log ""
log "Or use any OpenAI-compatible client:"
log '  from openai import OpenAI'
log '  client = OpenAI(base_url="http://localhost:8088/v1", api_key="any")'
