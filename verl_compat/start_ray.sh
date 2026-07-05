#!/usr/bin/env bash
# Helper script to dynamically start Ray on Gaudi HPU, NVIDIA GPU, or CPU systems.

set -euo pipefail

# Check if Python virtual environment is active
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "WARNING: Python virtual environment is not active! Please run 'source /workspace/inoculation/venv/bin/activate' first."
fi

# 1. Clean up stale ray processes
echo "Stopping existing Ray instances..."
ray stop --force || true
# Kill other processes containing 'ray', excluding this script's PID ($$)
PIDS=$(pgrep -f "ray" | grep -v "$$" || true)
if [ -n "$PIDS" ]; then
    echo "$PIDS" | xargs kill -9 || true
fi

export PYTHONPATH="/workspace/inoculation/verl-gaudi-support/verl_compat:/workspace/inoculation/verl_gaudi_support/verl_compat:/scratch/sgoli125/sglang-habana/python:${PYTHONPATH:-}"
export RAY_TMPDIR=${RAY_TMPDIR:-"/scratch/sgoli125/ray_tmp"}
export PT_HPU_LAZY_MODE=0
export HABANA_SYSTEM_FORK_UNSAFE_EXEC=1
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

ray start --head \
          --node-ip-address="${NODE_IP}" \
          "${RAY_ACCEL_FLAGS[@]}" \
          --port=6381 \
          --num-cpus=16 \
          --temp-dir="$RAY_TMPDIR" \
          --disable-usage-stats \
          --include-dashboard=false \
          --metrics-export-port=0

echo "Ray started successfully. Connect via RAY_ADDRESS='auto'."
