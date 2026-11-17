#!/usr/bin/env bash
# Run on the cloud sim machine (L40S): can it build and run the training container on its GPU?
# Prints each step; stops at the first failure and prints FAIL.
set -euo pipefail
trap 'echo "FAIL"' ERR
cd "$(dirname "$0")/.."
IMAGE=rl-insert:probe

step() { echo "== $1"; }

step "host GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

step "docker can see the GPU (needs the NVIDIA container toolkit)"
docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L

step "build the image"
docker build --platform linux/amd64 -f container/Dockerfile -t "$IMAGE" .

step "Warp finds CUDA inside the container"
docker run --rm --gpus all "$IMAGE" python -c \
  "import warp as wp; wp.init(); print('CUDA driver', wp.get_cuda_driver_version()); assert wp.is_cuda_available(), 'no CUDA'; print(wp.get_cuda_devices())"

step "contact spike on the GPU"
docker run --rm --gpus all "$IMAGE" python spikes/contact_warp.py --device cuda --nworld 4096 --reference 64

echo "PASS"
