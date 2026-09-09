#!/usr/bin/env bash
# MoGe-nuke installer (Linux).
#
#   ./install.sh [--cuda cu128] [--skip-model] [--skip-build] [--ofx-dir DIR]
#                [--offline WHEELHOUSE] [--python /usr/bin/python3.12]
#
# 1. creates .venv with uv (Python 3.12) and installs torch + MoGe + Triton
# 2. downloads the MoGe-3 checkpoint to models/ (about 5 GB)
# 3. builds the OFX plugin (gcc + cmake) and installs it where Nuke looks
# 4. registers nuke/ with Nuke (nuke.pluginAddPath in ~/.nuke/init.py)
#
# --offline WHEELHOUSE installs everything from a directory of wheels made by
# tools/make_wheelhouse.sh on a connected machine (see docs/OFFLINE.md); the
# model must already be in models/.
#
# Re-running is safe; each step skips what is already done.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"

cuda="cu128"; skip_model=0; skip_build=0; ofx_dir=""; wheelhouse=""; python_bin=""
while [ $# -gt 0 ]; do
    case "$1" in
        --cuda) cuda="$2"; shift 2 ;;
        --skip-model) skip_model=1; shift ;;
        --skip-build) skip_build=1; shift ;;
        --ofx-dir) ofx_dir="$2"; shift 2 ;;
        --offline) wheelhouse="$(cd "$2" && pwd)"; shift 2 ;;
        --python) python_bin="$2"; shift 2 ;;
        -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

step() { printf '\n\033[36m==> %s\033[0m\n' "$*"; }
py="$root/.venv/bin/python"
export VIRTUAL_ENV="$root/.venv"

# ---------------------------------------------------------------- 1. venv
step "Python environment"
command -v uv >/dev/null 2>&1 || { echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }
if [ -z "$wheelhouse" ]; then
    command -v git >/dev/null 2>&1 || { echo "git not found (MoGe's dependencies come from GitHub)" >&2; exit 1; }
fi
if [ ! -x "$py" ]; then
    if [ -n "$wheelhouse" ]; then
        uv venv --offline --python "${python_bin:-3.12}" .venv \
            || { echo "offline: pass --python <existing 3.10-3.12 interpreter> (dnf install python3.12)" >&2; exit 1; }
    else
        uv venv --python "${python_bin:-3.12}" .venv
    fi
fi
if [ -n "$wheelhouse" ]; then
    if ! "$py" -c "import torch, triton, flex_gemm, moge" >/dev/null 2>&1; then
        echo "installing from wheelhouse $wheelhouse ..."
        uv pip install --offline --no-index --find-links "$wheelhouse" -r "$wheelhouse/requirements.txt"
        uv pip install --offline --no-index --find-links "$wheelhouse" --no-deps -e third_party/MoGe
    fi
else
    if ! "$py" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
        echo "installing torch ($cuda) ..."
        uv pip install torch torchvision --index-url "https://download.pytorch.org/whl/$cuda"
    fi
    if ! "$py" -c "import flex_gemm, moge, triton" >/dev/null 2>&1; then
        echo "installing MoGe + dependencies ..."
        uv pip install -e third_party/MoGe
        uv pip install triton safetensors "opencv-python-headless<5"
    fi
fi
"$py" -c "import torch, triton, flex_gemm, moge; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

# ---------------------------------------------------------------- 2. model
if [ "$skip_model" = 0 ]; then
    step "Model checkpoint"
    if [ -n "$wheelhouse" ]; then
        ls models/moge-3-vitg.* >/dev/null 2>&1 || { echo "offline: put the checkpoint in models/ first (see docs/OFFLINE.md)" >&2; exit 1; }
        ls -la models/moge-3-vitg.*
    else
        "$py" tools/download_model.py
    fi
fi

# ---------------------------------------------------------------- 3. plugin
step "OFX plugin"
bundle=""
if [ "$skip_build" = 0 ]; then
    if command -v cmake >/dev/null 2>&1 && (command -v g++ >/dev/null 2>&1 || command -v clang++ >/dev/null 2>&1); then
        ./ofx/build.sh
        bundle="$root/ofx/build-linux/MoGe3.ofx.bundle"
    else
        echo "cmake / g++ not found; looking for a prebuilt bundle" >&2
    fi
fi
[ -n "$bundle" ] || bundle="$root/ofx/prebuilt/MoGe3.ofx.bundle"
[ -f "$bundle/Contents/Linux-x86-64/MoGe3.ofx" ] || { echo "no Linux plugin bundle at $bundle (install cmake and g++, then re-run)" >&2; exit 1; }

# where Nuke looks: OFX_PLUGIN_PATH entries, then the system-wide default
candidates=()
[ -n "$ofx_dir" ] && candidates+=("$ofx_dir")
if [ -n "${OFX_PLUGIN_PATH:-}" ]; then IFS=':' read -r -a parts <<< "$OFX_PLUGIN_PATH"; for d in "${parts[@]}"; do [ -n "$d" ] && candidates+=("$d"); done; fi
candidates+=("/usr/OFX/Plugins" "$HOME/.nuke/OFXPlugins")
installed=""
for d in "${candidates[@]}"; do
    if mkdir -p "$d" 2>/dev/null && [ -w "$d" ]; then
        rm -rf "$d/MoGe3.ofx.bundle"
        cp -r "$bundle" "$d/MoGe3.ofx.bundle"
        installed="$d/MoGe3.ofx.bundle"
        break
    fi
done
[ -n "$installed" ] || { echo "could not write to any OFX plugin directory (try sudo, or --ofx-dir)" >&2; exit 1; }

cfg="$installed/Contents/Linux-x86-64/moge3.cfg"
{
    echo "# written by install.sh -- paths the MoGe3 node uses as defaults"
    echo "root=$root"
    echo "python=$root/.venv/bin/python"
    echo "daemon=$root/daemon/moge_daemon.py"
    for m in models/moge-3-vitg.safetensors models/moge-3-vitg.pt; do
        if [ -f "$m" ]; then echo "model=$root/$m"; break; fi
    done
} > "$cfg"
echo "installed: $installed"

inst_dir="$(dirname "$installed")"
if [ "$inst_dir" != "/usr/OFX/Plugins" ]; then
    case ":${OFX_PLUGIN_PATH:-}:" in
        *":$inst_dir:"*) ;;
        *) echo "NOTE: add to your shell profile before launching Nuke:"
           echo "      export OFX_PLUGIN_PATH=\"\${OFX_PLUGIN_PATH:+\$OFX_PLUGIN_PATH:}$inst_dir\"" ;;
    esac
fi

# ---------------------------------------------------------------- 4. menu
step "Nuke menu"
mkdir -p "$HOME/.nuke"
init="$HOME/.nuke/init.py"
line="nuke.pluginAddPath(\"$root/nuke\")"
if ! grep -qsF "$line" "$init"; then
    printf '\n# MoGe-nuke (MoGe3 OFX node + daemon)\n%s\n' "$line" >> "$init"
    echo "added to $init"
else
    echo "already registered in $init"
fi

printf '\n\033[32mDone. Restart Nuke, then: Nodes > ML > MoGe3 > Depth + Normals\033[0m\n'
echo "The first render starts the daemon and loads the model (15-60 s)."
echo "Daemon log: \${XDG_CACHE_HOME:-~/.cache}/moge-nuke/daemon-\$UID.log"
