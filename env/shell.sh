#!/usr/bin/env bash
# Open an interactive shell inside the Gaudi container with the environment loaded
# and the venv active.
#
#   bash env/shell.sh                 # interactive shell
#   bash env/shell.sh <cmd> [args..]  # run one command inside and exit
#
# Note: we do NOT use --containall. The container needs $HOME so that the uv binary
# (~/.local/bin/uv) and, if you use it, wandb's ~/.netrc are reachable. PYTHONNOUSERSITE=1
# in gaudi_env.sh is what keeps ~/.local's CUDA torch out of sys.path.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/gaudi_env.sh"

if [ ! -f "$GAUDI_SIF" ]; then
    echo "ERROR: container image not found: $GAUDI_SIF" >&2
    echo "Pull it first:" >&2
    echo "  apptainer pull $GAUDI_SIF \\" >&2
    echo "    docker://vault.habana.ai/gaudi-docker/1.22.2/ubuntu24.04/habanalabs/pytorch-installer-2.7.1:1.22.2-32" >&2
    exit 1
fi

# Forward every variable gaudi_env.sh set, so the container sees the same environment.
ENV_ARGS=()
for v in PYTHONNOUSERSITE PYTHONPATH PT_HPU_GPU_MIGRATION VERL_PLATFORM PT_HPU_LAZY_MODE \
         PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES HABANA_SYSTEM_FORK_UNSAFE_EXEC HABANA_LOGS \
         PT_HPU_ENABLE_LAZY_COLLECTIVES \
         VERL_HPU_TORCH_COMPILE VERL_HPU_FUSED_SDPA \
         VERL_HPU_WEIGHT_SYNC_DEBUG VERL_HPU_WEIGHT_SYNC_TRANSPORT \
         RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES RAY_gcs_rpc_server_reconnect_timeout_s \
         RAY_OBJECT_STORE_MEMORY_BYTES \
         PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION \
         RAY_TMPDIR RAY_OBJECT_SPILL_DIR RAY_object_spilling_directory \
         HPU_CARDS_COUNT VERL_TMPDIR VERL_HABANA_LOGS \
         HF_HOME HF_DATASETS_CACHE HF_HUB_CACHE HF_HUB_DISABLE_SYMLINKS_WARNING \
         XDG_CACHE_HOME TMPDIR TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR TORCH_HOME \
         TORCH_EXTENSIONS_DIR PT_HPU_RECIPE_CACHE_CONFIG \
         WANDB_DIR WANDB_CACHE_DIR UV_CACHE_DIR PIP_CACHE_DIR \
         REPO_ROOT VERL_COMPAT SGLANG_HPU_ROOT CACHE_ROOT VENV_DIR SCRATCH_ROOT \
         GSM8K_DIR CKPT_ROOT \
         WANDB_API_KEY WANDB_ENTITY WANDB_MODE WANDB_RUN_GROUP WANDB_TAGS WANDB \
         SGLANG_CONFIG_HIDDEN_LAYERS SGLANG_HPU_GRAPH_PREFILL SGLANG_HPU_SKIP_WARMUP \
         SGLANG_EXPERIMENTAL_HPU_PAGED_V2 PT_HPU_AUTOLOAD GRAPH_VISUALIZATION_DIR \
         SGLANG_EXPERIMENTAL_HPU_DECODE_GRAPH SGLANG_HPU_BUCKETING_STRATEGY \
         VERL_HPU_SGLANG_LAZY VLLM_HPU_FSDPA_RECOMPUTE SGLANG_HPU_PROMPT_ATTN_IMPL \
         HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
    [ -n "${!v:-}" ] && ENV_ARGS+=( --env "$v=${!v}" )
done

BIND_ARGS=()
IFS=',' read -ra BINDS <<< "$APPTAINER_BINDS"
for b in "${BINDS[@]}"; do [ -e "$b" ] && BIND_ARGS+=( -B "$b" ); done

# Prepend the venv to PATH inside the container so `python`/`ray` resolve to it, and add
# ~/.local/bin so the host's `uv` (a static binary, runs fine in the container) is reachable.
# PYTHONNOUSERSITE=1 is what keeps ~/.local's *Python* packages out of sys.path -- verified:
# without it the container's torch 2.7.1+hpu is shadowed by the host's 2.9.0+cu128 and
# habana_frameworks aborts with "Compile-time PyTorch version 2.7 differs from run-time 2.9.0+cu128".
INNER_PRELUDE="export PATH=\"$VENV_DIR/bin:\$HOME/.local/bin:\$PATH\"; export VIRTUAL_ENV=\"$VENV_DIR\";"

if [ $# -eq 0 ]; then
    exec apptainer exec "${ENV_ARGS[@]}" "${BIND_ARGS[@]}" "$GAUDI_SIF" \
        bash --noprofile --norc -c "$INNER_PRELUDE exec bash --norc -i"
else
    exec apptainer exec "${ENV_ARGS[@]}" "${BIND_ARGS[@]}" "$GAUDI_SIF" \
        bash --noprofile --norc -c "$INNER_PRELUDE $(printf '%q ' "$@")"
fi
