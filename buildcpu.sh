#!/bin/bash
set -e

rm -rf build
mkdir build
cd build

cmake .. -DGGML_BLAS=ON
cmake --build . --config Release -j "$(nproc)"
