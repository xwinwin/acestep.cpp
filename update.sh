#!/bin/bash
set -e

cd ggml
git pull --rebase
cd ..
git pull --rebase
