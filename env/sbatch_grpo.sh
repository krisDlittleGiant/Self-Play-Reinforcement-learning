#!/usr/bin/env bash
#SBATCH --job-name=grpo-gaudi
#SBATCH --partition=gaudi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:hl225:8
#SBATCH --mem=0
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/%u/logs/grpo.%j.out
#SBATCH --error=/scratch/%u/logs/grpo.%j.err
#
# Run GRPO in its OWN Slurm job.
#
# Why this exists: the two "node crashes" were neither. Both times the training run
# exhausted the memory cgroup of the OnDemand *VSCode tunnel* job -- which is where the
# editor, the Claude session and the training all shared one budget -- and Slurm killed
# the tunnel. gaudi004 itself was never in trouble (up 54 days, 457 GB free throughout).
#   job 62934641: ReqMem=256G  MaxRSS=268429072K  State=OUT_OF_MEMORY
#   job 62958214: ReqMem=400G  script.sh "Killed"
# A dedicated job means an OOM costs the run, not the session, and leaves real evidence.
#
# --mem=0 requests the node's entire 503 GB rather than the tunnel's 400 GB.
#
#   sbatch -A <account> --qos <qos> env/sbatch_grpo.sh                       # defaults below
#   sbatch --export=ALL,MAX_RESPONSE_LENGTH=4096 env/sbatch_grpo.sh
set -uo pipefail
cd "${REPO_ROOT:-/scratch/$(id -un)/verl-gaudi-support}"

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-5}
tag="b${TRAIN_BATCH_SIZE}_r${MAX_RESPONSE_LENGTH}_${SLURM_JOB_ID}"
memlog=/scratch/$(id -un)/logs/mem_${tag}.log
runlog=/scratch/$(id -un)/logs/grpo_${tag}.log

echo "job=$SLURM_JOB_ID node=$(hostname) batch=$TRAIN_BATCH_SIZE resp=$MAX_RESPONSE_LENGTH"
echo "run log: $runlog"
echo "mem log: $memlog"

# ---- host-RAM sampler -------------------------------------------------------
# This is the measurement the last two attempts never produced. Per-process RSS every
# 10 s, plus a top-consumer table when the cgroup passes 85%, so the culprit is on record
# before the OOM killer removes it.
cg=/sys/fs/cgroup/system.slice/slurmstepd.scope/job_${SLURM_JOB_ID}
limit_kb=$(( $(cat $cg/memory.max 2>/dev/null || echo $((503*1024*1024*1024))) / 1024 ))
[ "$limit_kb" -le 0 ] && limit_kb=$((503*1024*1024))
warned=0
(
  while true; do
    used_kb=$(( $(cat $cg/memory.current 2>/dev/null || echo 0) / 1024 ))
    pct=$(( used_kb * 100 / limit_kb ))
    avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    top=$(ps -eo rss,args --sort=-rss --no-headers -u "$(id -u)" | head -6 | \
          awk '{printf "%.0fG:%s ", $1/1048576, substr($2,1,28)}')
    echo "$(date +%H:%M:%S) cgroup=$((used_kb/1048576))G/$((limit_kb/1048576))G (${pct}%) avail=${avail}G | $top" >> "$memlog"
    if [ "$pct" -ge 85 ] && [ "$warned" -eq 0 ]; then
      warned=1
      { echo "=== 85% OF CGROUP $(date) ==="; ps -eo rss,pid,etime,args --sort=-rss -u "$(id -u)" | head -25; } >> "$memlog"
    fi
    sleep 10
  done
) &
sampler=$!
trap 'kill $sampler 2>/dev/null' EXIT

# ---- the run ----------------------------------------------------------------
pkill -9 -u "$(id -u)" -f 'raylet|gcs_server|plasma_store|ray::|log_monitor\.py|main_sppo|sglang' 2>/dev/null
sleep 3; rm -rf "/dev/shm/ray_$(id -un)"

TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE \
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8} \
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2} \
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512} \
MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH \
TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS \
VAL_BEFORE_TRAIN=False TEST_FREQ=-1 SAVE_FREQ=-1 WANDB=${WANDB:-0} \
EXPERIMENT_NAME="Qwen3-4B-Base_gsm8k_${tag}" \
stdbuf -oL -eL bash env/run_grpo_gsm8k.sh 2>&1 | stdbuf -oL tee "$runlog"

rc=${PIPESTATUS[0]}
echo "=== exit=$rc ==="
tail -3 "$memlog"
exit $rc
