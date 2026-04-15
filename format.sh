#!/bin/bash
set -e

find . -name "*.cpp" -o -name "*.h" | grep -v -e build/ -e ggml/ -e vendor/ | xargs clang-format -i
