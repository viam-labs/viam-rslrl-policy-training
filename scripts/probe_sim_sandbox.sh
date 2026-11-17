#!/usr/bin/env bash
# uv fallback of scripts/probe_gpu.sh for a Viam sim sandbox: no Docker, driver mounted at /usr/local/nvidia.
# Copy rl-insert.tar.gz (git archive of the repo) and this script to /root with `viam machine part cp`,
# then start it with nohup through `viam machine part shell` and copy /root/probe.log back.
set -euo pipefail
trap 'echo "FAIL (line $LINENO)"; echo EXIT=1' ERR
export HOME=/root PATH=/usr/local/nvidia/bin:$PATH LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}
cd /root && rm -rf rl-insert && mkdir rl-insert && tar xzf rl-insert.tar.gz -C rl-insert && cd rl-insert
echo "== host GPU"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "== uv sync"; uv sync --frozen --no-dev 2>&1 | tail -3
echo "== Warp finds CUDA"
uv run python -c "import warp as wp; wp.init(); print('CUDA driver', wp.get_cuda_driver_version()); assert wp.is_cuda_available(), 'no CUDA'; print(wp.get_cuda_devices())"
echo "== contact spike on the GPU"
start=$(date +%s)
uv run python spikes/contact_warp.py --device cuda --nworld 4096 --reference 64
echo "spike wall time: $(( $(date +%s) - start )) s"
echo PASS; echo EXIT=0
