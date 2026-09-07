#!/usr/bin/env bash
#
# Build and install the source dependencies: yaml-cpp, gRPC (with protoc) and
# prometheus-cpp. Works on a fresh x86_64 box or a Jetson (aarch64).
#
#   SEC_SYS_ROOT_DIR=/path/to/edge-ai-system bash scripts/install_dependencies.sh
#
# Install the system prerequisites first -- see "Prerequisites" in README.md.
# This script itself needs no sudo.
#
# The prebuilt GPU stack -- ONNX Runtime, CUDA, cuDNN, TensorRT -- is NOT
# handled here; fetch those tarballs into dependency/ by hand, using the exact
# version directory names in the README's dependency table.
#
# Env:
#   SEC_SYS_ROOT_DIR  (required) repo root
#   JOBS              parallel compile jobs (default: sized from RAM, see below)
#   FORCE=1           rebuild dependencies that are already installed
#
set -euo pipefail

if [ -z "${SEC_SYS_ROOT_DIR:-}" ]; then
    echo "SEC_SYS_ROOT_DIR is not set; export it to the repo root and re-run." >&2
    exit 1
fi

readonly ROOT="$SEC_SYS_ROOT_DIR"
readonly DEP="$ROOT/dependency"

# yaml-cpp is sandboxed per-checkout (17 MB, seconds to build). gRPC is not:
# its build is ~4.5 GB and over an hour, so it goes to a shared user prefix and
# the root CMakeLists picks it up from there via GRPC_ROOT.
readonly YAML_CPP_PREFIX="$DEP/yaml-cpp/_install"
readonly GRPC_PREFIX="$HOME/.local"

# Pinned versions -- keep these identical across every machine in the fleet.
readonly YAML_CPP_TAG="0.9.0"
readonly GRPC_TAG="v1.78.1"           # bundles protobuf 31.1 -> also installs protoc
readonly PROMETHEUS_CPP_TAG="v1.3.0"  # its 3rdparty/civetweb submodule is pinned at v1.16

# gRPC's translation units are memory-hungry; roughly one job per 2 GB keeps an
# 8 GB Jetson from OOM-killing the compiler halfway through the build.
if [ -z "${JOBS:-}" ]; then
    mem_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
    JOBS=$(( mem_gb / 2 ))
    if [ "$JOBS" -lt 1 ]; then
        JOBS=1
    fi
    if [ "$JOBS" -gt "$(nproc)" ]; then
        JOBS=$(nproc)
    fi
fi
readonly JOBS

log()  { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
skip() { printf '\033[1;33m--- %s already installed (FORCE=1 to rebuild)\033[0m\n' "$*"; }

log "repo=$ROOT  yaml-cpp=$YAML_CPP_PREFIX  gRPC=$GRPC_PREFIX  jobs=$JOBS"
mkdir -p "$DEP"

# ----------------------------------------------------------------- yaml-cpp
# find_package(yaml-cpp REQUIRED) in services/ingest_service needs an INSTALLED
# config -- a bare build tree is not discoverable. Installed into the repo
# rather than /usr/local so this needs no sudo; the root CMakeLists finds it
# through YAML_CPP_ROOT.
if [ -z "${FORCE:-}" ] && [ -f "$YAML_CPP_PREFIX/lib/cmake/yaml-cpp/yaml-cpp-config.cmake" ]; then
    skip "yaml-cpp"
else
    log "yaml-cpp $YAML_CPP_TAG"
    if [ ! -d "$DEP/yaml-cpp" ]; then
        git clone --depth 1 -b "$YAML_CPP_TAG" \
            https://github.com/jbeder/yaml-cpp.git "$DEP/yaml-cpp"
    fi
    cmake -S "$DEP/yaml-cpp" -B "$DEP/yaml-cpp/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DYAML_CPP_BUILD_TESTS=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    cmake --build "$DEP/yaml-cpp/build" -j"$JOBS"
    cmake --install "$DEP/yaml-cpp/build" --prefix "$YAML_CPP_PREFIX"
fi

# --------------------------------------------------------- gRPC + protobuf
# This is where protoc comes from. It installs protoc, grpc_cpp_plugin and
# libprotobuf (31.1) all version-matched to gRPC.
# Do NOT `apt install protobuf-compiler` -- JetPack ships protoc 3.6/3.12,
# which cannot read code generated against protobuf 31.
if [ -z "${FORCE:-}" ] && [ -f "$GRPC_PREFIX/lib/cmake/grpc/gRPCConfig.cmake" ]; then
    skip "gRPC + protobuf/protoc"
else
    log "gRPC $GRPC_TAG (slowest step -- expect over an hour on a Jetson)"
    if [ ! -d "$DEP/grpc" ]; then
        git clone --recurse-submodules -b "$GRPC_TAG" --depth 1 --shallow-submodules \
            https://github.com/grpc/grpc "$DEP/grpc"
    else
        git -C "$DEP/grpc" submodule update --init --recursive
    fi
    mkdir -p "$GRPC_PREFIX"
    cmake -S "$DEP/grpc" -B "$DEP/grpc/cmake/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$GRPC_PREFIX" \
        -DgRPC_INSTALL=ON \
        -DgRPC_BUILD_TESTS=OFF \
        -DCMAKE_CXX_STANDARD=17
    cmake --build "$DEP/grpc/cmake/build" -j"$JOBS"
    cmake --install "$DEP/grpc/cmake/build"
fi

# ------------------------------------------------------------ prometheus-cpp
# USE_THIRDPARTY_LIBRARIES defaults ON and compiles the pinned civetweb
# submodule straight in, so no external civetweb (and no civetweb-config.cmake)
# is ever needed -- --recurse-submodules below is what makes that work.
# Installed into _install/ because the top-level CMakeLists resolves
# PROMETHEUS_CPP_ROOT there with NO_DEFAULT_PATH; the _build tree's own config
# computes its prefix as if installed and misresolves.
readonly PROM_CFG="$DEP/prometheus-cpp/_install/lib/cmake/prometheus-cpp/prometheus-cpp-config.cmake"
if [ -z "${FORCE:-}" ] && [ -f "$PROM_CFG" ]; then
    skip "prometheus-cpp"
else
    log "prometheus-cpp $PROMETHEUS_CPP_TAG"
    if [ ! -d "$DEP/prometheus-cpp" ]; then
        git clone --recurse-submodules -b "$PROMETHEUS_CPP_TAG" --depth 1 --shallow-submodules \
            https://github.com/jupp0r/prometheus-cpp.git "$DEP/prometheus-cpp"
    else
        git -C "$DEP/prometheus-cpp" submodule update --init --recursive
    fi
    # A cached USE_THIRDPARTY_LIBRARIES=OFF survives populating the submodule
    # (option() never overrides an existing cache entry), so always configure
    # from a clean build dir.
    rm -rf "$DEP/prometheus-cpp/_build"
    cmake -S "$DEP/prometheus-cpp" -B "$DEP/prometheus-cpp/_build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DENABLE_PUSH=OFF \
        -DENABLE_TESTING=OFF
    cmake --build "$DEP/prometheus-cpp/_build" -j"$JOBS"
    cmake --install "$DEP/prometheus-cpp/_build" --prefix "$DEP/prometheus-cpp/_install"
fi

# ---------------------------------------------------------------------- done
# ONNX Runtime, CUDA, cuDNN and TensorRT are NOT handled here -- they are
# fetched by hand into dependency/ using the exact version directory names in
# the README's dependency table.
log "All dependencies installed."
cat <<EOF

Configure the project with:

  cmake -S "$ROOT" -B "$ROOT/build"

No -DCMAKE_PREFIX_PATH is needed: the root CMakeLists resolves yaml-cpp through
YAML_CPP_ROOT ($YAML_CPP_PREFIX) and gRPC through GRPC_ROOT ($GRPC_PREFIX).

The build invokes protoc via the protobuf::protoc target, so it does not need
protoc on PATH. For running protoc by hand -- e.g. regenerating the Python
stubs -- put the gRPC prefix ahead of any distro protoc:

  echo 'export PATH="\$HOME/.local/bin:\$PATH"' >> ~/.bashrc && source ~/.bashrc
  protoc --version          # expect: libprotoc 31.1
EOF
