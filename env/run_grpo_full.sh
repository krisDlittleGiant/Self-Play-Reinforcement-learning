#!/usr/bin/env bash
# The validated full GRPO run: Qwen3-4B-Base on GSM8K, 4 FSDP2 trainers + 4 SGLang
# rollout servers on 8 Gaudi cards.
#
#   bash env/run_grpo_full.sh                 # detached; resumes automatically
#   FOREGROUND=1 bash env/run_grpo_full.sh    # stay attached
#   TOTAL_TRAINING_STEPS=5 WANDB=0 bash env/run_grpo_full.sh    # smoke test
#
# Every value below is an override-able default, so a rerun of this exact command
# resumes an interrupted run rather than starting over (EXPERIMENT_NAME is stable and
# verl's resume_mode is "auto").
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/gaudi_env.sh"

# ---- geometry -------------------------------------------------------------------
# 32 prompts x 8 rollouts = 256 sequences/step, split 8-prompt mini-batches (4 optimizer
# steps per rollout) and 2 sequences per card per micro-step.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
# 2048 measured 0% response truncation; 256 clipped 20.7% of answers mid-derivation.
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"

# ---- rollout --------------------------------------------------------------------
# hpu_paged_v2 + captured decode graphs. Measured against graphless hpu_fused:
# generation 913 s/step -> ~35 s, throughput 15 -> 457, ppl_ratio 15k-126k -> ~1.001.
# hpu_fused has no working graph path (dynamic index_select shapes fail
# synGraphInferShapes, and it deadlocks rather than raising).
export SGLANG_HPU_ATTENTION_BACKEND="${SGLANG_HPU_ATTENTION_BACKEND:-hpu_paged_v2}"
export SGLANG_DECODE_GRAPH_BACKEND="${SGLANG_DECODE_GRAPH_BACKEND:-full}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-32}"
export SGLANG_PREFILL_MAX_REQUESTS="${SGLANG_PREFILL_MAX_REQUESTS:-2}"
export SGLANG_LOAD_FORMAT="${SGLANG_LOAD_FORMAT:-auto}"
export VERL_HPU_SGLANG_LAZY="${VERL_HPU_SGLANG_LAZY:-1}"
export VERL_HPU_FUSED_SDPA="${VERL_HPU_FUSED_SDPA:-1}"
export VERL_HPU_TORCH_COMPILE="${VERL_HPU_TORCH_COMPILE:-0}"
export VERL_HPU_WEIGHT_SYNC_TRANSPORT="${VERL_HPU_WEIGHT_SYNC_TRANSPORT:-distributed}"
export VERL_HPU_WEIGHT_SYNC_DEBUG="${VERL_HPU_WEIGHT_SYNC_DEBUG:-0}"

# ---- training / logging ---------------------------------------------------------
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Base}"
export ENABLE_THINKING="${ENABLE_THINKING:-False}"   # the GSM8K reward needs "#### N"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"  # null => full epoch (233)
export SAVE_FREQ="${SAVE_FREQ:-20}"
export TEST_FREQ="${TEST_FREQ:-20}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export WANDB="${WANDB:-1}"
# Stable, NOT timestamped: this is what makes resume work.
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-Qwen3_4B_b${TRAIN_BATCH_SIZE}_r${MAX_RESPONSE_LENGTH}_full}"

run_tag="$(date +%Y%m%d_%H%M%S)"
log_dir="${VERL_LOG_DIR:-${CACHE_ROOT}}"
mkdir -p "$log_dir"
grpo_log="${log_dir}/grpo_${EXPERIMENT_NAME}_${run_tag}.log"
mem_log="${log_dir}/mem_${EXPERIMENT_NAME}_${run_tag}.log"
rollout_dir="${VERL_ROLLOUT_DIR:-/tmp/grpo_${VERL_USER}_${run_tag}}"

# ---- clean slate ----------------------------------------------------------------
# A stale Ray head is silently reused and then fails in confusing ways.
pkill -9 -u "$(id -u)" -f 'raylet|gcs_server|plasma_store|ray::|log_monitor\.py|main_sppo|sglang' 2>/dev/null || true
pkill -u "$(id -u)" -f 'MemAvailable' 2>/dev/null || true   # samplers from earlier runs
sleep 3
rm -rf "${RAY_TMPDIR:-/dev/shm/ray_${VERL_USER}}"

# ---- host-RAM sampler -----------------------------------------------------------
# Writes to shared storage every 10 s so it survives the session dying -- which is how
# two earlier attempts were lost with no evidence. See GAUDI_FAILURE_LOG.md F45.
setsid nohup bash -c 'while :; do
    echo "$(date +%H:%M:%S) $(awk "/MemAvailable/{print int(\$2/1048576)\"G\"}" /proc/meminfo) $(ps -eo rss,args --sort=-rss --no-headers -u "$(id -u)" | head -4 | awk "{printf \"%.0fG:%s \",\$1/1048576,substr(\$2,1,24)}")"
    sleep 10
done' >> "$mem_log" 2>&1 &

HYDRA_ARGS=(
    actor_rollout_ref.actor.strategy="${ACTOR_STRATEGY:-fsdp2}"
    actor_rollout_ref.actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS:-0}"
    trainer.max_actor_ckpt_to_keep="${MAX_CKPT_TO_KEEP:-3}"
    trainer.rollout_data_dir="$rollout_dir"
)

echo "experiment : $EXPERIMENT_NAME"
echo "run log    : $grpo_log"
echo "mem log    : $mem_log"
echo "rollouts   : $rollout_dir"

if [ "${FOREGROUND:-0}" = "1" ]; then
    bash "$HERE/run_grpo_gsm8k.sh" "${HYDRA_ARGS[@]}" 2>&1 | tee "$grpo_log"
    exit "${PIPESTATUS[0]}"
fi

setsid nohup bash "$HERE/run_grpo_gsm8k.sh" "${HYDRA_ARGS[@]}" > "$grpo_log" 2>&1 &
echo
echo "launched detached (survives an SSH/VSCode disconnect). Follow with:"
echo "    tail -f $grpo_log"
