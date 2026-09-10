#!/usr/bin/env bash
# Kill the GRPO run before host RAM exhaustion kills the NODE.
#
# The node loss on gaudi00x was not a Slurm OOM-kill: tmpfs pages (/dev/shm) are not
# charged like process RSS, so Ray object spilling into RAY_TMPDIR could grow until the
# kernel had nothing left. env/gaudi_env.sh now spills to node-local NVMe instead, but
# this watchdog is the backstop that makes a wrong guess cost a job, not a node.
#
#   bash env/ram_guard.sh [min_avail_gb] [max_shm_gb] &   # default 60 / 150
set -uo pipefail
MIN_AVAIL_GB=${1:-60}
MAX_SHM_GB=${2:-150}
LOG=${RAM_GUARD_LOG:-/scratch/$(id -un)/logs/ram_guard_$(date +%m%d_%H%M).log}
mkdir -p "$(dirname "$LOG")"

echo "ram_guard: trip if MemAvailable < ${MIN_AVAIL_GB}G or /dev/shm > ${MAX_SHM_GB}G | log=$LOG"
while true; do
    avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    shm=$(df -BG --output=used /dev/shm | tail -1 | tr -dc '0-9')
    spill=$(du -sBG "${RAY_OBJECT_SPILL_DIR:-/tmp/ray_spill_$(id -un)}" 2>/dev/null | tr -dc '0-9' | head -c6)
    printf '%s avail=%sG shm=%sG spill=%sG\n' "$(date +%H:%M:%S)" "$avail" "$shm" "${spill:-0}" >> "$LOG"
    if [ "$avail" -lt "$MIN_AVAIL_GB" ] || [ "$shm" -gt "$MAX_SHM_GB" ]; then
        {
            echo "=== TRIPPED $(date) avail=${avail}G shm=${shm}G spill=${spill:-0}G ==="
            ps -eo rss,comm,args --sort=-rss | head -15
        } | tee -a "$LOG"
        pkill -9 -u "$(id -u)" -f 'raylet|gcs_server|plasma_store|ray::|main_sppo|sglang'
        echo "ram_guard: killed the run. see $LOG" | tee -a "$LOG"
        exit 1
    fi
    sleep 15
done
