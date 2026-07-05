#!/usr/bin/env bash
# Helper script to dynamically start Ray on Gaudi HPU, NVIDIA GPU, or CPU systems.

set -euo pipefail

# 1. Clean up stale python and ray processes
echo "Stopping existing Ray instances..."
ray stop --force || true
pkill -9 -f python || true
pkill -9 -f ray || true

export RAY_TMPDIR=${RAY_TMPDIR:-"/tmp/ray_$(whoami)"}
mkdir -p "$RAY_TMPDIR"
rm -rf "$RAY_TMPDIR"/*

# 2. Dynamically detect device type and count cards
if command -v hl-smi &> /dev/null; then
    # Habana Gaudi HPU: Count cards using hl-smi
    CARDS_COUNT=$(hl-smi -Q index -f csv,noheader | wc -l)
    RAY_ACCEL_FLAGS="--resources={\"HPU\":${CARDS_COUNT}}"
    echo "Detected Gaudi HPU system with ${CARDS_COUNT} cards."
elif command -v nvidia-smi &> /dev/null; then
    # NVIDIA GPU: Count cards using nvidia-smi
    CARDS_COUNT=$(nvidia-smi -L | wc -l)
    RAY_ACCEL_FLAGS="--num-gpus=${CARDS_COUNT}"
    echo "Detected NVIDIA GPU system with ${CARDS_COUNT} cards."
else
    # CPU Fallback
    RAY_ACCEL_FLAGS=""
    echo "No accelerator detected. Defaulting to CPU."
fi

# 3. Start Ray dynamically
echo "Starting Ray head node..."
ray start --head \
          ${RAY_ACCEL_FLAGS} \
          --port=6381 \
          --num-cpus=16 \
          --temp-dir="$RAY_TMPDIR" \
          --disable-usage-stats \
          --include-dashboard=false \
          --metrics-export-port=0

echo "Ray started successfully. Connect via RAY_ADDRESS='auto'."
