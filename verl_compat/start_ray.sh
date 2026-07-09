#!/usr/bin/env bash
# Helper script to dynamically start Ray on Gaudi HPU, NVIDIA GPU, or CPU systems.

set -euo pipefail

# Check if Python virtual environment is active
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "WARNING: Python virtual environment is not active! Please run 'source /workspace/inoculation/venv/bin/activate' first."
fi

# 1. Clean up stale ray processes
echo "Stopping existing Ray instances..."
# Ray's own shutdown prints a "Could not terminate ... (pid=X, name=Y)" line (in red) for
# every leftover process it can't confirm dead within its grace period; this is best-effort
# housekeeping before a fresh `ray start` below, not something worth surfacing. Silence it
# (still runs, exit code already ignored) rather than dropping the cleanup entirely.
python3 -m ray.scripts.scripts stop --force > /dev/null 2>&1 || true

export PYTHONPATH="/workspace/inoculation/verl-gaudi-support/verl_compat:/workspace/inoculation/verl_gaudi_support/verl_compat:/scratch/sgoli125/sglang-habana/python:${PYTHONPATH:-}"
# Ray's temp dir must live on a real, fast, local filesystem (tmpfs/ext4), NOT on the
# container overlayfs or a network mount. Slow small-file/socket I/O there makes
# `import ray`/`ray start` crawl and causes the dashboard subprocess to time out
# (empty dashboard.err). /dev/shm is tmpfs (RAM-backed) and ideal; keep the base path
# short so Ray's unix-socket paths stay under the ~107-char limit. Override RAY_TMPDIR
# to change it.
export RAY_TMPDIR="${RAY_TMPDIR:-/dev/shm/ray_${USER:-$(id -un)}}"
export PT_HPU_LAZY_MODE=0
# Enable Habana's GPU Migration Toolkit so torch.cuda.* is redirected to torch.hpu.* and
# device "cuda" maps to "hpu". platform_hpu.py is built on the CUDA namespace and depends
# on this; without it, torch.cuda.set_device() hits torch._C._cuda_setDevice which does
# not exist in the Habana torch build (AttributeError). Must be set before torch import.
export PT_HPU_GPU_MIGRATION=1
export HABANA_SYSTEM_FORK_UNSAFE_EXEC=1
# Tell Ray NOT to isolate each worker to a single HPU module. Habana's
# initialize_distributed_hpu() (invoked eagerly at `import habana_frameworks.torch`)
# requires the process to see WORLD_SIZE modules; if Ray restricts each worker to one
# module, it asserts "There is not enough devices available for training". With NOSET,
# every worker sees all node HPUs and selects its own by LOCAL_RANK (handled in
# single_controller/base/worker.py). Must be set before `ray start` so the raylet reads
# it when spawning workers. This is the same opt-out the sglang/vLLM servers already use.
export RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES=1
mkdir -p "$RAY_TMPDIR"
rm -rf "$RAY_TMPDIR"/*

# 2. Dynamically detect device type and count cards
declare -a RAY_ACCEL_FLAGS=()
if command -v hl-smi &> /dev/null; then
    # Habana Gaudi HPU: Count cards using hl-smi
    CARDS_COUNT=$(hl-smi -Q index -f csv,noheader | wc -l)
    RAY_ACCEL_FLAGS+=( "--resources={\"HPU\":${CARDS_COUNT}}" )
    echo "Detected Gaudi HPU system with ${CARDS_COUNT} cards."
elif command -v nvidia-smi &> /dev/null; then
    # NVIDIA GPU: Count cards using nvidia-smi
    CARDS_COUNT=$(nvidia-smi -L | wc -l)
    RAY_ACCEL_FLAGS+=( "--num-gpus=${CARDS_COUNT}" )
    echo "Detected NVIDIA GPU system with ${CARDS_COUNT} cards."
else
    # CPU Fallback
    echo "No accelerator detected. Defaulting to CPU."
fi

# 3. Start Ray dynamically
echo "Starting Ray head node..."
NODE_IP="127.0.0.1"
echo "Binding to node IP: ${NODE_IP}"

python3 -m ray.scripts.scripts start --head \
          --node-ip-address="${NODE_IP}" \
          "${RAY_ACCEL_FLAGS[@]}" \
          --port=6381 \
          --num-cpus=16 \
          --temp-dir="$RAY_TMPDIR" \
          --disable-usage-stats \
          --include-dashboard=false \
          --dashboard-agent-listen-port=0 \
          --metrics-export-port=0

echo "Ray started successfully. Connect via RAY_ADDRESS='auto'."
