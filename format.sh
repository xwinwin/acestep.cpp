#!/bin/bash

find . -name "*.cpp" -o -name "*.h" | grep -v -e build/ -e ggml/ | xargs clang-format -i
