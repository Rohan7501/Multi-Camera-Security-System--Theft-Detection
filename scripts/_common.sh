#!/usr/bin/env bash
# Sourced by every run_*.sh. Resolves the repo root once, so no script needs an
# absolute path baked into it.
#
# SEC_SYS_ROOT_DIR wins if it is set (matches install_dependencies.sh); otherwise
# the root is derived from this file's own location, which means the scripts work
# straight out of a fresh clone with nothing exported.

# ${BASH_SOURCE[0]} is THIS file even when sourced, unlike $0 which is the caller.
_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SEC_SYS_ROOT_DIR="${SEC_SYS_ROOT_DIR:-$(cd "$_COMMON_DIR/.." && pwd)}"
REPO="$SEC_SYS_ROOT_DIR"

if [ ! -f "$REPO/CMakeLists.txt" ]; then
    echo "SEC_SYS_ROOT_DIR=$REPO does not look like the repo root (no CMakeLists.txt)" >&2
    exit 1
fi

DEP="$REPO/dependency"

# Vendored GPU stack. Pinned per-project rather than installed system-wide, so
# the versions can't drift with whatever CUDA the distro ships.
ORT_VERSION="${ORT_VERSION:-1.22.0}"
TRT_VERSION="${TRT_VERSION:-10.8.0.43}"
CUDNN_VERSION="${CUDNN_VERSION:-9.23.2.1}"
CUDA_VERSION="${CUDA_VERSION:-12.8.0}"

ORT="$DEP/onnxruntime-linux-x64-gpu-$ORT_VERSION"

gpu_ld_path() {
    printf '%s' \
"$DEP/TensorRT-$TRT_VERSION/lib:$DEP/cudnn-$CUDNN_VERSION/lib:$DEP/cuda_$CUDA_VERSION/lib64:$ORT/lib"
}
