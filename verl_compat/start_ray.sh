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
# `ray stop` is session-aware and can only see processes it believes belong to the current
# session, so a GCS server or raylet orphaned from an unrelated/older session (e.g. a crashed
# run, or one from before this box's RAY_TMPDIR override existed) never gets signaled and
# lingers indefinitely -- still accepting connections, which can hijack a later
# RAY_ADDRESS=auto lookup into that dead cluster instead of the fresh one started below.
# Scoped to Ray's own known process names, not a bare "ray" substring match (which would
# also catch unrelated processes with "ray" anywhere in their command line); silenced the
# same way as the command above rather than printing per-pid kill noise.
PIDS=$(pgrep -f "raylet|gcs_server|plasma_store|ray::|log_monitor\.py|dashboard(_agent)?\.py" 2>/dev/null | grep -vx "$$" || true)
if [ -n "$PIDS" ]; then
    echo "$PIDS" | xargs kill -9 > /dev/null 2>&1 || true
fi

export PYTHONPATH="/workspace/inoculation/verl-gaudi-support/verl_compat:/workspace/inoculation/verl_gaudi_support/verl_compat:/scratch/sgoli125/sglang-habana/python:${PYTHONPATH:-}"
# Ray's temp dir must live on a real, fast, local filesystem (tmpfs/ext4), NOT on the
# container overlayfs or a network mount. Slow small-file/socket I/O there makes
# `import ray`/`ray start` crawl and causes the dashboard subprocess to time out
# (empty dashboard.err). /dev/shm is tmpfs (RAM-backed) and ideal; keep the base path
# short so Ray's unix-socket paths stay under the ~107-char limit. Override RAY_TMPDIR
# to change it.
export RAY_TMPDIR="${RAY_TMPDIR:-/dev/shm/ray_${USER:-$(id -un)}}"
# Respect whatever the caller already set (e.g. at Apptainer entry) instead of silently
# overwriting it -- unlike RAY_TMPDIR above, this used to be an unconditional `export
# PT_HPU_LAZY_MODE=0`, which clobbered an explicitly-set PT_HPU_LAZY_MODE=1 without any
# indication that had happened. Defaults to 0 (eager) only when nothing else set it.
export PT_HPU_LAZY_MODE="${PT_HPU_LAZY_MODE:-0}"
# Enable Habana's GPU Migration Toolkit so torch.cuda.* is redirected to torch.hpu.* and
# device "cuda" maps to "hpu". platform_hpu.py is built on the CUDA namespace and depends
# on this; without it, torch.cuda.set_device() hits torch._C._cuda_setDevice which does
# not exist in the Habana torch build (AttributeError). Must be set before torch import.
export PT_HPU_GPU_MIGRATION=1
export HABANA_SYSTEM_FORK_UNSAFE_EXEC=1
# verl's get_platform() auto-detects by trying `import habana_frameworks.torch.hpu` and
# caches whichever result it gets ONCE, for that process's entire lifetime -- if that import
# fails or raises anything other than ImportError for a specific Ray actor (e.g. one that
# happens to run in a context without proper Habana device initialization, such as the
# TaskRunner driver actor), it silently falls back to "nvidia" forever for that process, even
# though worker actors detect "hpu" correctly. That mismatch makes _check_resource_available()
# look for a "GPU" Ray resource key instead of the "HPU" key start_ray.sh actually registers,
# so it always sees 0 available regardless of how many real HPUs are free. VERL_PLATFORM is
# get_platform()'s own documented override (checked before any auto-detection is attempted),
# so set it explicitly rather than depending on fragile, process-context-dependent detection.
export VERL_PLATFORM=hpu
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
    # Habana Gaudi HPU: count cards using hl-smi, unless explicitly overridden via
    # HPU_CARDS_COUNT. hl-smi reports every physical HPU on the node, not just the
    # ones actually allocated to this job -- on a shared node this overcounts (no
    # SLURM env var or cgroup device restriction distinguishes "yours" from "the
    # node's" here; every /dev/accel* file is world-readable/writable inside this
    # container regardless of allocation), and Ray ends up believing it can schedule
    # more HPU-using tasks than this job actually has cards for
    # ("Total available GPUs N is less than total desired GPUs M" further downstream
    # is one symptom, but the real risk is Ray silently placing work on cards that
    # were never actually this job's to use). Set HPU_CARDS_COUNT to your actual
    # allocation (e.g. HPU_CARDS_COUNT=4) to override the auto-detected count.
    CARDS_COUNT="${HPU_CARDS_COUNT:-$(hl-smi -Q index -f csv,noheader | wc -l)}"
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
