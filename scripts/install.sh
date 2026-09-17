#!/usr/bin/env bash
# Install every TyBoK component with one command:
#   1) the Python package and its dependencies (inference worker + gateway)
#   2) the C++ gateway (gateway_cpp: cmake configure + build, artifact build/tybok_gateway_cpp)
#
# Usage:
#   bash scripts/install.sh                     # install into the currently active python environment
#   bash scripts/install.sh --env tybok        # use the conda environment tybok (error if it does not exist)
#   bash scripts/install.sh --env tybok --create-env   # create a python=3.12 conda environment, then install
#   bash scripts/install.sh --skip-cpp          # install the Python components only
#
# Environment variables: CUDA_ROOT (CUDA toolkit install path; auto-detected from
# /usr/local/cuda-13.2 and /usr/local/cuda-12.8 by default).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------- arguments
ENV_NAME=""
CREATE_ENV=0
SKIP_CPP=0
CUDA_ROOT="${CUDA_ROOT:-}"

usage() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env) ENV_NAME="$2"; shift 2 ;;
        --create-env) CREATE_ENV=1; shift ;;
        --skip-cpp) SKIP_CPP=1; shift ;;
        --cuda-root) CUDA_ROOT="$2"; shift 2 ;;
        -h | --help) usage; exit 0 ;;
        *) echo "unknown arg: $1"; usage; exit 2 ;;
    esac
done

log() { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[install] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- stage 0: system dependencies
log "Checking system dependencies (g++ / cmake / libjpeg / libpng / CUDA headers)..."
missing=()
command -v g++ >/dev/null 2>&1 || missing+=("g++ (gcc)")
command -v cmake >/dev/null 2>&1 || missing+=("cmake")
[[ -f /usr/include/jpeglib.h ]] || missing+=("libjpeg-dev")
[[ -f /usr/include/png.h ]] || missing+=("libpng-dev")
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Missing: ${missing[*]}"
    echo "On Ubuntu/Debian install them with: sudo apt-get install -y g++ cmake libjpeg-dev libpng-dev"
    die "install the system dependencies first"
fi

# CUDA toolkit detection
if [[ -z "$CUDA_ROOT" ]]; then
    for c in /usr/local/cuda-13.2 /usr/local/cuda-12.8; do
        if [[ -f "$c/include/cuda.h" ]]; then CUDA_ROOT="$c"; break; fi
    done
fi
[[ -z "$CUDA_ROOT" || ! -f "$CUDA_ROOT/include/cuda.h" ]] && die "CUDA toolkit not found (pass --cuda-root to point at it)"
log "CUDA toolkit: $CUDA_ROOT"

# ---------------------------------------------------------------- stage 1: conda environment
if [[ -n "$ENV_NAME" ]]; then
    command -v conda >/dev/null 2>&1 || die "conda not found (--env requires conda)"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
        if [[ "$CREATE_ENV" == 1 ]]; then
            log "Creating conda environment ${ENV_NAME} (python 3.12)..."
            conda create -y -n "$ENV_NAME" python=3.12
        else
            die "conda environment ${ENV_NAME} does not exist (add --create-env to create it)"
        fi
    fi
    conda activate "$ENV_NAME"
fi

command -v python >/dev/null 2>&1 || die "python not found"
log "Python: $(command -v python) ($(python --version 2>&1))"

# ---------------------------------------------------------------- stage 2: Python components
log "Installing the Python package and dependencies (pip install -e '.[gateway]')..."
# --no-build-isolation: reuse the environment's setuptools instead of downloading one
# into an isolated build environment.
python -m pip install --no-build-isolation -e "$ROOT[gateway]"

# ---------------------------------------------------------------- stage 3: C++ components
if [[ "$SKIP_CPP" == 1 ]]; then
    log "Skipping the C++ build (--skip-cpp)"
else
    log "Configuring the C++ gateway (cmake -S gateway_cpp -B gateway_cpp/build)..."
    cmake -S "$ROOT/gateway_cpp" -B "$ROOT/gateway_cpp/build" -DCUDAToolkit_ROOT="$CUDA_ROOT"
    log "Building tybok_gateway_cpp..."
    cmake --build "$ROOT/gateway_cpp/build" -j"$(nproc)"
    [[ -x "$ROOT/gateway_cpp/build/tybok_gateway_cpp" ]] || die "the C++ build artifact is missing"
fi

log "Done. Component locations:"
log "  Python package : $(python -c 'import tybok, os; print(os.path.dirname(tybok.__file__))')"
if [[ "$SKIP_CPP" == 1 ]]; then
    log "  C++ gateway    : not built (--skip-cpp)"
else
    log "  C++ gateway    : $ROOT/gateway_cpp/build/tybok_gateway_cpp"
fi
echo "Common commands:"
echo "  python -m tybok serve --model <ckpt> --graph --port 8765"
if [[ "$SKIP_CPP" != 1 ]]; then
    echo "  gateway_cpp/build/tybok_gateway_cpp --worker-socket /tmp/tybok_worker.sock --port 8765 [--gpu-direct]"
fi
