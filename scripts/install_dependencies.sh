#!/usr/bin/env bash
set -euo pipefail

if [ -z "${SEC_SYS_ROOT_DIR:-}" ]; then
    echo "SEC_SYS_ROOT_DIR is not set; export it to the repo root and re-run." >&2
    exit 1
fi

cd "$SEC_SYS_ROOT_DIR"
cd dependency

# yaml-cpp
if [ ! -d yaml-cpp ]; then
    git clone https://github.com/jbeder/yaml-cpp.git
fi
cd yaml-cpp
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

cd ../..

# grpc
if [ ! -d grpc ]; then
    git clone --recurse-submodules -b v1.78.1 --depth 1 --shallow-submodules https://github.com/grpc/grpc
fi
cd grpc
mkdir -p cmake/build
pushd cmake/build

export MY_INSTALL_DIR=$HOME/.local
mkdir -p $MY_INSTALL_DIR
cmake ../..   -DCMAKE_INSTALL_PREFIX=$MY_INSTALL_DIR   -DgRPC_INSTALL=ON   -DgRPC_BUILD_TESTS=OFF   -DCMAKE_CXX_STANDARD=17
make -j$(nproc)
popd
