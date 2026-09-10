#!/usr/bin/env bash
# Start a single-node Ray head with HPU resources registered.
#
# MUST BE RUN INSIDE THE CONTAINER:
#     bash env/shell.sh                 # then, at the container prompt:
#     bash env/start_ray_gaudi.sh
#
# Do NOT run it from the host as `bash env/shell.sh bash env/start_ray_gaudi.sh`.
# Measured: apptainer tears the SIF's squashfuse mount down as soon as the exec's primary
# process exits ("Terminating squashfuse_ll after timeout"). The daemonised raylet/GCS
# survive as host processes but lose their view of /usr, so every later worker spawn fails
# with I/O errors. Ray must live inside a container session that stays open -- either an
# interactive env/shell.sh, or the one env/run_grpo_gsm8k.sh opens for itself (it starts
# Ray on its own when none is reachable, so for a plain run you do not need this script).
#
# Derived from verl_compat/start_ray.sh, with the /workspace/inoculation paths replaced by
# this repo's env/gaudi_env.sh values.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/gaudi_env.sh"

if [ -z "${APPTAINER_CONTAINER:-}${SINGULARITY_CONTAINER:-}" ] && [ ! -d /.singularity.d ]; then
    echo "ERROR: not inside the Gaudi container." >&2
    echo "  Open one first:  bash env/shell.sh" >&2
    echo "  then, at the container prompt:  bash env/start_ray_gaudi.sh" >&2
    exit 1
fi

RAY="${VENV_DIR}/bin/python -m ray.scripts.scripts"
RAY_PORT="${RAY_PORT:-6381}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"
RAY_OBJECT_STORE_MEMORY_BYTES="${RAY_OBJECT_STORE_MEMORY_BYTES:-4294967296}"
RAY_OBJECT_SPILL_DIR="${RAY_OBJECT_SPILL_DIR:-/tmp/ray_spill_${VERL_USER:-$(id -un)}}"
mkdir -p "${RAY_OBJECT_SPILL_DIR}"
NODE_IP="${NODE_IP:-127.0.0.1}"

# ---- 1. clear stale Ray state ----------------------------------------------------------
echo "==> stopping any existing Ray"
$RAY stop --force >/dev/null 2>&1 || true
# `ray stop` is session-aware: a GCS/raylet orphaned by a crashed run is invisible to it and
# lingers, still accepting connections, so a later RAY_ADDRESS=auto lookup can land on the
# dead cluster. Sweep Ray's own process names -- scoped to THIS uid so a co-tenant on the
# node is never touched, and excluding this shell so we do not kill ourselves.
PIDS=$(pgrep -u "$(id -u)" -f 'raylet|gcs_server|plasma_store|ray::|log_monitor\.py|dashboard(_agent)?\.py' 2>/dev/null | grep -vx "$$" || true)
if [ -n "$PIDS" ]; then
    echo "    reaping orphaned Ray processes: $(echo "$PIDS" | tr '\n' ' ')"
    echo "$PIDS" | xargs -r kill -9 >/dev/null 2>&1 || true
fi

mkdir -p "$RAY_TMPDIR"
rm -rf "${RAY_TMPDIR:?}"/*

# ---- 2. how many HPUs are OURS ---------------------------------------------------------
# hl-smi reports every physical card on the node, not this job's allocation, and inside the
# container every /dev/accel* is readable regardless of who holds it. Overcounting makes Ray
# schedule onto cards that are not ours. HPU_CARDS_COUNT (gaudi_env.sh, default 8) is the
# override; set it to your SLURM allocation on a shared node.
if command -v hl-smi >/dev/null 2>&1; then
    DETECTED=$(hl-smi -Q index -f csv,noheader 2>/dev/null | wc -l)
else
    DETECTED=0
fi
CARDS="${HPU_CARDS_COUNT:-$DETECTED}"
if [ "$DETECTED" -gt 0 ] && [ "$CARDS" != "$DETECTED" ]; then
    echo "    hl-smi sees ${DETECTED} card(s) on the node; registering ${CARDS} (HPU_CARDS_COUNT)"
fi
[ "$CARDS" -gt 0 ] || { echo "ERROR: no HPUs detected and HPU_CARDS_COUNT unset." >&2; exit 1; }
echo "==> registering Ray resource HPU=${CARDS}"

# ---- 3. start the head -----------------------------------------------------------------
# --include-dashboard=false: the dashboard subprocess times out on this box and adds nothing.
# The HPU-relevant env (PT_HPU_*, VERL_PLATFORM, RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES,
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION, caches) is already exported by gaudi_env.sh above,
# so the raylet inherits it and passes it to every worker it spawns.
$RAY start --head \
    --node-ip-address="${NODE_IP}" \
    --resources="{\"HPU\":${CARDS}}" \
    --port="${RAY_PORT}" \
    --num-cpus="${RAY_NUM_CPUS}" \
    --object-store-memory="${RAY_OBJECT_STORE_MEMORY_BYTES}" \
    --object-spilling-directory="${RAY_OBJECT_SPILL_DIR}" \
    --temp-dir="$RAY_TMPDIR" \
    --disable-usage-stats \
    --include-dashboard=false \
    --dashboard-agent-listen-port=0 \
    --metrics-export-port=0

echo
echo "==> Ray head is up at ${NODE_IP}:${RAY_PORT}  (HPU=${CARDS})"
echo "    object store: ${RAY_OBJECT_STORE_MEMORY_BYTES} bytes"
echo "    spill dir:    ${RAY_OBJECT_SPILL_DIR} ($(df -h --output=avail "${RAY_OBJECT_SPILL_DIR}" 2>/dev/null | tail -1 | tr -d ' ') free)"
echo "    Launch the run in THIS shell:  bash env/run_grpo_gsm8k.sh"
echo "    Stop it with:                  ${VENV_DIR}/bin/ray stop --force"
