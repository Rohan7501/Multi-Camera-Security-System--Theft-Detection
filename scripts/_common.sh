#!/usr/bin/env bash
# Sourced by every run_*.sh. Two jobs: locate the repo, and locate the
# dependencies.
#
# Dependencies are located by PASTING EACH ONE'S DIRECTORY -- there are no
# version strings here and nothing is assumed to live under the repo. Put your
# paths in scripts/dependencies.conf (copy dependencies.conf.example), or export
# the same names. Anything already in the environment wins over the file.
#
#   CUDA_DIR          CUDA toolkit root        (the dir containing lib64/)
#   CUDNN_DIR         cuDNN root               (contains lib/)
#   TENSORRT_DIR      TensorRT root            (contains lib/)
#   ONNXRUNTIME_DIR   ONNX Runtime root        (contains lib/ and include/)
#   PROMETHEUS_DIR    Prometheus release dir   (contains the prometheus binary)
#
# The repo root is still derived from this file's location (or SEC_SYS_ROOT_DIR)
# because it locates the repo's OWN files -- services/, build/, deploy/ -- not a
# dependency.

# ${BASH_SOURCE[0]} is THIS file even when sourced, unlike $0 which is the caller.
_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SEC_SYS_ROOT_DIR="${SEC_SYS_ROOT_DIR:-$(cd "$_COMMON_DIR/.." && pwd)}"
REPO="$SEC_SYS_ROOT_DIR"

if [ ! -f "$REPO/CMakeLists.txt" ]; then
    echo "SEC_SYS_ROOT_DIR=$REPO does not look like the repo root (no CMakeLists.txt)" >&2
    exit 1
fi

DEP_CONF="${DEP_CONF:-$_COMMON_DIR/dependencies.conf}"
_DEP_VARS=(CUDA_DIR CUDNN_DIR TENSORRT_DIR ONNXRUNTIME_DIR PROMETHEUS_DIR)

# The conf file holds bare assignments, so sourcing it would overwrite anything
# already exported. Snapshot the environment first and re-apply it afterwards --
# that keeps a one-off `CUDA_DIR=... scripts/run_inference.sh` working, and lets
# CI point at a different stack without editing a file it does not own.
declare -A _DEP_FROM_ENV=()
for _v in "${_DEP_VARS[@]}"; do
    if [ -n "${!_v:-}" ]; then
        _DEP_FROM_ENV["$_v"]="${!_v}"
    fi
done

if [ -f "$DEP_CONF" ]; then
    # shellcheck disable=SC1090
    source "$DEP_CONF"
fi

for _v in "${!_DEP_FROM_ENV[@]}"; do
    printf -v "$_v" '%s' "${_DEP_FROM_ENV[$_v]}"
done
unset _v

# Resolve one dependency's library directory. Layouts differ -- CUDA ships lib64,
# cuDNN/TensorRT/ORT ship lib -- so probe instead of assuming.
#   dep_lib <VAR_NAME>   -> prints the lib dir, or fails with a usable message
dep_lib() {
    local var="$1" dir="${!1:-}"

    if [ -z "$dir" ]; then
        echo "$var is not set. Paste its directory into $DEP_CONF" >&2
        echo "  (copy scripts/dependencies.conf.example to start)" >&2
        return 1
    fi
    if [ ! -d "$dir" ]; then
        echo "$var=$dir does not exist" >&2
        return 1
    fi

    local candidate
    for candidate in "$dir/lib64" "$dir/lib" "$dir"; do
        if compgen -G "$candidate/*.so*" > /dev/null 2>&1; then
            printf '%s' "$candidate"
            return 0
        fi
    done

    echo "$var=$dir contains no shared libraries (looked in lib64/, lib/, and the dir itself)" >&2
    return 1
}

# LD_LIBRARY_PATH for the inference binary, built from the four pasted dirs.
# Validated lazily, HERE rather than at source time, because the display and
# tracking launchers source this file too and need no GPU stack at all.
gpu_ld_path() {
    local parts=() lib var
    for var in TENSORRT_DIR CUDNN_DIR CUDA_DIR ONNXRUNTIME_DIR; do
        lib="$(dep_lib "$var")" || return 1
        parts+=("$lib")
    done
    local IFS=:
    printf '%s' "${parts[*]}"
}

# The prometheus binary, from its pasted release directory.
prometheus_bin() {
    if [ -z "${PROMETHEUS_DIR:-}" ]; then
        echo "PROMETHEUS_DIR is not set. Paste its directory into $DEP_CONF" >&2
        return 1
    fi
    if [ ! -x "$PROMETHEUS_DIR/prometheus" ]; then
        echo "no executable prometheus at $PROMETHEUS_DIR/prometheus" >&2
        return 1
    fi
    printf '%s' "$PROMETHEUS_DIR/prometheus"
}
