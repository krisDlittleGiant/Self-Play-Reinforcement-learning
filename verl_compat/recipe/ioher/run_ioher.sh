#!/usr/bin/env bash
# Launcher for the IOHER recipe.
#
# Default setup mirrors the SPPO example (Qwen2.5 + GSM8K/MATH) but
# swaps the algorithm in for GRPO + auxiliary inoculated-SFT loss and
# pins the rollout engine to sglang.
#
# Override any field via `field=value` arguments forwarded to hydra.

set -euo pipefail
set -x

# Prevent ~/.local/lib user site-packages from polluting the environment
export PYTHONNOUSERSITE=1
export PYTHONPATH="/workspace/inoculation/verl-gaudi-support/verl_compat:/workspace/inoculation/verl_gaudi_support/verl_compat:/scratch/sgoli125/sglang-habana/python:${PYTHONPATH:-}"
# Keep in sync with start_ray.sh: Ray temp dir on fast local tmpfs (not overlayfs/network).
export RAY_TMPDIR="${RAY_TMPDIR:-/dev/shm/ray_${USER:-$(id -un)}}"
export RAY_ADDRESS="auto"
export PT_HPU_LAZY_MODE=0
# Enable Habana GPU Migration Toolkit (torch.cuda.* -> torch.hpu.*, "cuda" -> "hpu").
# platform_hpu.py uses the CUDA namespace and requires this; otherwise
# torch.cuda.set_device() raises AttributeError (_cuda_setDevice missing). Set before torch import.
export PT_HPU_GPU_MIGRATION=1
export HABANA_SYSTEM_FORK_UNSAFE_EXEC=1
# Keep in sync with start_ray.sh: do not let Ray isolate each worker to one HPU module,
# otherwise Habana's eager initialize_distributed_hpu() asserts "not enough devices"
# because it needs to see WORLD_SIZE modules. Each worker selects its card by LOCAL_RANK.
export RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES=1
# Force protobuf's pure-Python backend. The default "upb" (C) backend produces
# google._upb._message.Descriptor objects that CANNOT be pickled; Ray hits this when
# (de)serializing the colocated WorkerDict actor and fails with an unpicklable-cause
# ActorDiedError ("cannot pickle 'google._upb._message.Descriptor'"), which then
# surfaces as the misleading "async flag" error. Must be set before protobuf is imported.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python



# --- Automatically detect the python executable from active virtual env / conda env, fallback to python3
PYTHON_BIN="python3"
if [ -n "${VIRTUAL_ENV:-}" ]; then
  PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif command -v python &> /dev/null; then
  # check if 'python' is inside a virtualenv
  if python -c "import sys; print(sys.prefix != sys.base_prefix)" 2>/dev/null | grep -q "True"; then
    PYTHON_BIN="python"
  fi
fi

# --- Hugging Face Cache Setup ---
# Apptainer containers might mount home directories as read-only.
# We set cache directories to a writable path, defaulting to /workspace/inoculation/hf_cache or a local fallback.
WRITABLE_CACHE_DIR="/workspace/inoculation/hf_cache"
if [ ! -d "$WRITABLE_CACHE_DIR" ] || [ ! -w "$WRITABLE_CACHE_DIR" ]; then
  # Fallback to local hf_cache if the workspace path doesn't exist or is not writable
  mkdir -p ./hf_cache
  WRITABLE_CACHE_DIR="$(pwd)/hf_cache"
fi

export HF_HOME="$WRITABLE_CACHE_DIR"
export HF_DATASETS_CACHE="$WRITABLE_CACHE_DIR/datasets"
export DATASETS_CACHE="$WRITABLE_CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$WRITABLE_CACHE_DIR/hub"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

# --- Defaults; tweak in-place or override on the command line. -------------
TRAIN_FILES=${TRAIN_FILES:-"$HOME/data/math/train.parquet"}
VAL_FILES=${VAL_FILES:-"$HOME/data/math/test.parquet"}

if [ -z "${MODEL_PATH:-}" ]; then
  if [ -d "$HOME/models/Qwen2.5-7B-Instruct" ]; then
    MODEL_PATH="$HOME/models/Qwen2.5-7B-Instruct"
  else
    MODEL_PATH="Qwen/Qwen3-4B-Base"
  fi
fi

N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
NNODES=${NNODES:-1}

MODEL_NAME=$(basename "$MODEL_PATH")
PROJECT_NAME=${PROJECT_NAME:-"ioher-sglang"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"${MODEL_NAME}_ioher_grpo"}

# Rollout engine: sglang is the recipe default.
ROLLOUT_NAME=${ROLLOUT_NAME:-sglang}

# IOH hyperparameters (override here if you want to sweep).
IOH_SFT_COEF=${IOH_SFT_COEF:-0.5}
IOH_REWARD_THRESHOLD=${IOH_REWARD_THRESHOLD:-0.5}
IOH_INJECT_MODE=${IOH_INJECT_MODE:-system}

train_files="['$TRAIN_FILES']"
val_files="['$VAL_FILES']"

echo "=== Ray Diagnostic Info ==="
$PYTHON_BIN -c "import ray; print('Ray version:', ray.__version__); print('Ray file:', ray.__file__)"
echo "========================="

$PYTHON_BIN -m recipe.ioher.main_ioher \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.actor.ioh_sft_coef=$IOH_SFT_COEF \
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.n=8 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.ioh.inject_mode=$IOH_INJECT_MODE \
    algorithm.ioh.reward_threshold=$IOH_REWARD_THRESHOLD \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_epochs=1000 "$@"
