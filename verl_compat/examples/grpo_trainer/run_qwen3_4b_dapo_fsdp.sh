#!/usr/bin/env bash
# GRPO | Qwen3-4B-Base | FSDP training | DAPO-Math-17k | NVIDIA GPUs, Ascend NPUs, or Intel Gaudi HPUs
#
# Cross-hardware GRPO baseline derived from run_qwen3_4b_fsdp.sh. The ALGORITHMIC workload
# (model, data, batch geometry, GRPO settings) is identical on every device so runs are
# comparable; only platform-forced settings differ per device, and each divergence is
# annotated with WHY it exists. Defaults are the configuration verified end-to-end on
# 8x HL-225 (4 training + 4 rollout) and mirror the 4x A100 reference workload.
#
# INFER_BACKEND controls the rollout engine: vllm (GPU default) or sglang (HPU default).
#
# HPU prerequisite: start the Ray cluster FIRST (this script does not start one):
#     HPU_CARDS_COUNT=<your SLURM allocation> bash start_ray.sh
# hl-smi shows every physical card on a shared node, not your allocation -- HPU_CARDS_COUNT
# is what keeps Ray from scheduling onto cards that are not yours.
#
# DRY_RUN=1 prints the fully composed command without executing (also skips preflights).

set -xeuo pipefail

# ---- device auto-detection (override with DEVICE=gpu|npu|hpu) ----
detect_device() {
    if python3 -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("habana_frameworks") else 1)' 2>/dev/null; then
        echo hpu
    elif python3 -c 'import torch_npu' 2>/dev/null; then
        echo npu
    else
        echo gpu
    fi
}
DEVICE=${DEVICE:-$(detect_device)}
DRY_RUN=${DRY_RUN:-0}

# ---- Hugging Face cache ----
# Without this, HF falls back to ~/.cache/huggingface, which under a --contain container
# (or a quota'd home) is a small ephemeral filesystem -- pulling a multi-GB model then dies
# with "No space left on device (os error 28)". Point it at a large writable path, matching
# what recipe/ioher/run_ioher.sh already does so both share one cache and the model is only
# downloaded once. Override HF_CACHE_DIR for a different location.
HF_CACHE_DIR=${HF_CACHE_DIR:-/workspace/inoculation/hf_cache}
if [ ! -d "$HF_CACHE_DIR" ] || [ ! -w "$HF_CACHE_DIR" ]; then
    mkdir -p ./hf_cache
    HF_CACHE_DIR="$(pwd)/hf_cache"
fi
export HF_HOME="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
export DATASETS_CACHE="$HF_CACHE_DIR/datasets"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

# ---- workload (IDENTICAL across devices; override via env) ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}
TRAIN_FILE=${TRAIN_FILE:-/workspace/inoculation/data/dapo_math/train.parquet}
TEST_FILE=${TEST_FILE:-/workspace/inoculation/data/dapo_math/val.parquet}
PROMPT_KEY=${PROMPT_KEY:-source_prompt}
NNODES=${NNODES:-1}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
ROLLOUT_N=${ROLLOUT_N:-8}

ACTOR_LR=${ACTOR_LR:-1e-6}
# KL loss OFF is the verified configuration (matches the A100 reference runs). Turning it
# on also activates the reference-policy worker -- a code path exercised far less on HPU;
# if you enable it, note that ref param_offload is forced off on HPU below (offload
# corrupts FSDP weights on Gaudi -- entropy pins at ln(vocab), rewards collapse).
USE_KL_LOSS=${USE_KL_LOSS:-False}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_qwen3_4b_dapo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_base_dapo_grpo_${DEVICE}}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:-25}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
# Optional hard cap for benchmark runs, e.g. TOTAL_TRAINING_STEPS=5
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}

# ---- per-device platform settings ----
case "${DEVICE}" in
    gpu)
        INFER_BACKEND=${INFER_BACKEND:-vllm}
        NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}          # matches the 4x A100 reference
        ROLLOUT_TP=${ROLLOUT_TP:-2}
        ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
        PLATFORM_OPTS=(
            actor_rollout_ref.model.use_remove_padding=True
            actor_rollout_ref.actor.use_dynamic_bsz=True
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3000
            actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
            actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096
            actor_rollout_ref.rollout.enable_chunked_prefill=False
            actor_rollout_ref.rollout.enforce_eager=False
            actor_rollout_ref.rollout.free_cache_engine=True
            actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
            actor_rollout_ref.ref.fsdp_config.param_offload=True
            actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
            actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192
        )
        ;;
    npu)
        export VLLM_USE_V1=1
        export TASK_QUEUE_ENABLE=2
        export CPU_AFFINITY_CONF=1
        export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
        INFER_BACKEND=${INFER_BACKEND:-vllm}
        NGPUS_PER_NODE=${NGPUS_PER_NODE:-16}
        ROLLOUT_TP=${ROLLOUT_TP:-2}
        ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.9}
        PLATFORM_OPTS=(
            actor_rollout_ref.model.use_remove_padding=True
            actor_rollout_ref.actor.use_dynamic_bsz=True
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3000
            actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
            actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096
            actor_rollout_ref.rollout.enable_chunked_prefill=False
            actor_rollout_ref.rollout.enforce_eager=False
            actor_rollout_ref.rollout.free_cache_engine=True
            actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
            actor_rollout_ref.ref.fsdp_config.param_offload=True
            actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
            actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192
        )
        ;;
    hpu)
        # -- environment: the verified Gaudi runtime (see repo README / memory of the port) --
        REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
        SGLANG_HPU_ROOT=${SGLANG_HPU_ROOT:-/scratch/sgoli125/sglang-habana}
        export PYTHONPATH="${REPO_ROOT}:${SGLANG_HPU_ROOT}/python:${PYTHONPATH:-}"
        export PYTHONNOUSERSITE=1
        export VERL_PLATFORM=hpu                       # pin platform detection (auto-detect is per-process fragile)
        export PT_HPU_LAZY_MODE="${PT_HPU_LAZY_MODE:-0}"   # lazy mode crashes FSDP flat-param sharding
        export PT_HPU_GPU_MIGRATION=1                  # torch.cuda.* -> torch.hpu.* (compat layer depends on it)
        export HABANA_SYSTEM_FORK_UNSAFE_EXEC=1
        export RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES=1
        export VERL_HPU_TORCH_COMPILE="${VERL_HPU_TORCH_COMPILE:-1}"   # eager+compile = Gaudi's documented FSDP path
        export VERL_HPU_FUSED_SDPA="${VERL_HPU_FUSED_SDPA:-1}"         # FusedSDPA via F.sdpa (needs attn sdpa below)
        export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python           # upb descriptors break Ray actor pickling
        export RAY_TMPDIR="${RAY_TMPDIR:-/dev/shm/ray_${USER:-$(id -un)}}"
        export RAY_ADDRESS="${RAY_ADDRESS:-auto}"

        INFER_BACKEND=${INFER_BACKEND:-sglang}
        if [[ "${INFER_BACKEND}" == "vllm" ]]; then
            echo "WARNING: vllm rollout on HPU is untested in this fork; sglang is the validated backend." >&2
        fi
        # Disaggregated topology: rollout servers get their OWN cards on top of the
        # training cards -- total HPUs = NGPUS_PER_NODE * (1 + 1/ROLLOUT_TP).
        NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}            # 8-card node: 4 training + 4 rollout
        ROLLOUT_TP=${ROLLOUT_TP:-1}                    # everything on HPU is validated at TP=1
        ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}
        PLATFORM_OPTS=(
            # Use the legacy FSDP worker path. The model-engine path builds its outputs
            # with torch.nested jagged tensors, which SynapseAI does not support --
            # torch.nested.narrow dies with "Graph duplication failed. synStatus=26"
            # partway through training. The legacy path uses dense padded tensors and is
            # the one the IOHER recipe already runs successfully on Gaudi.
            +trainer.use_legacy_worker_impl=True
            # No varlen attention kernel on Gaudi (FusedSDPA has no cu_seqlens API), so the
            # rmpad path cannot run -- THE dominant known perf gap vs CUDA (padded compute).
            actor_rollout_ref.model.use_remove_padding=False
            actor_rollout_ref.actor.use_remove_padding=False
            # Route HF attention through F.sdpa so the FusedSDPA patch engages. Stock torch
            # sdpa lowering NaNs on Gaudi; eager attention materializes the full score
            # matrix (defragmentation storms). FusedSDPA+sdpa is the verified combination.
            ++actor_rollout_ref.model.override_config.attn_implementation=sdpa
            # Token-budget batching is meaningless at fixed padded width; use explicit
            # micro-batch sizes instead.
            actor_rollout_ref.actor.use_dynamic_bsz=False
            actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
            # No sleep/wake memory release on the HPU sglang fork.
            actor_rollout_ref.rollout.free_cache_engine=False
            # Weight sync serializes through CPU on HPU; large buckets balloon pickle memory.
            actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=128
            # param_offload silently CORRUPTS FSDP weights on Gaudi (verified: entropy pins
            # at ln(vocab), all rewards collapse to -1, pg_loss -> exactly 0). Keep every
            # offload OFF, including the ref policy's (stock default is True for ref).
            actor_rollout_ref.actor.fsdp_config.param_offload=False
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
            actor_rollout_ref.ref.fsdp_config.param_offload=False
            actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
        )

        # A Ray cluster with registered HPU resources must already be up (start_ray.sh);
        # main_ppo's own ray.init() would create a cluster with ZERO HPU resources and die
        # with "Total available GPUs 0 is less than total desired GPUs N".
        if [[ "${DRY_RUN}" != "1" ]]; then
            if ! python3 -m ray.scripts.scripts status >/dev/null 2>&1; then
                echo "ERROR: no running Ray cluster found. Start one first:" >&2
                echo "    HPU_CARDS_COUNT=<your allocation> bash ${REPO_ROOT}/start_ray.sh" >&2
                exit 1
            fi
        fi
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu', 'npu' or 'hpu'." >&2
        exit 1
        ;;
esac

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    data.train_files=${TRAIN_FILE}
    data.val_files=${TEST_FILE}
    data.prompt_key=${PROMPT_KEY}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    algorithm.use_kl_in_reward=False
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
)

########################### launch ###########################
CMD=(python3 -m verl.trainer.main_ppo
    "${DATA[@]}"
    "${MODEL[@]}"
    "${ACTOR[@]}"
    "${ROLLOUT[@]}"
    "${PLATFORM_OPTS[@]}"
    "${TRAINER[@]}"
    "$@"
)

if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

"${CMD[@]}"
