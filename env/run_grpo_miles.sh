#!/usr/bin/env bash
# One entry point for verification and full training, with live logs and elapsed time.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/gaudi_env.sh"
mode=${1:-one}
if [[ $# -gt 0 ]]; then shift; fi
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Base}"
export VERL_HPU_FUSED_SDPA="${VERL_HPU_FUSED_SDPA:-1}"
export VERL_HPU_TORCH_COMPILE="${VERL_HPU_TORCH_COMPILE:-0}"
export PT_HPU_LAZY_MODE=0
export VAL_BEFORE_TRAIN=False TEST_FREQ=-1 TOTAL_EPOCHS=1
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
export SGLANG_DISABLE_OVERLAP_SCHEDULE="${SGLANG_DISABLE_OVERLAP_SCHEDULE:-True}"
export WANDB="${WANDB:-0}"
tag=$(date +%Y%m%d_%H%M%S)
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-Qwen3_4B_Base_miles_${mode}_${tag}}"
case "$mode" in
    preflight) command=(bash "$HERE/shell.sh" python "$HERE/verify_sglang_miles.py");;
    generation)
        command=(bash "$HERE/shell.sh" env PT_HPU_GPU_MIGRATION=0 PT_HPU_LAZY_MODE=1
            PT_HPU_AUTOLOAD=1 VERL_HPU_SGLANG_PROCESS=1 SGLANG_EXPERIMENTAL_HPU_DECODE_GRAPH=1
            python "$HERE/verify_sglang_generation.py");;
    weights)
        command=(bash "$HERE/shell.sh" python "$HERE/diag_weight_sync.py");;
    fsdp2)
        command=(bash "$HERE/shell.sh" python "$HERE/verify_fsdp2_hpu.py");;
    one) export TOTAL_TRAINING_STEPS=1 SAVE_FREQ=-1; command=(bash "$HERE/run_grpo_gsm8k.sh");;
    soak)
        export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-30}"
        export SAVE_FREQ="${SAVE_FREQ:--1}"
        command=(bash "$HERE/run_grpo_gsm8k.sh");;
    full) export TOTAL_TRAINING_STEPS=null SAVE_FREQ=10; command=(bash "$HERE/run_grpo_gsm8k.sh");;
    *) echo "Usage: bash env/run_grpo_miles.sh {preflight|generation|weights|fsdp2|one|soak|full} [extra arguments]" >&2; exit 2;;
esac
mkdir -p "$CACHE_ROOT"
log="$CACHE_ROOT/grpo_miles_${mode}_${tag}.log"
printf 'mode=%s log=%s\n' "$mode" "$log"
started=$SECONDS
set +e
"${command[@]}" "$@" 2>&1 | tee "$log"
statuses=("${PIPESTATUS[@]}")
rc=${statuses[0]}
if [[ "$rc" == 0 && "${statuses[1]}" != 0 ]]; then rc=${statuses[1]}; fi
printf 'exit=%s elapsed=%ss log=%s\n' "$rc" "$(( SECONDS - started ))" "$log"
exit "$rc"
