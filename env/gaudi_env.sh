#!/usr/bin/env bash
# Sourceable environment for GRPO-on-Gaudi runs.
#
#   source env/gaudi_env.sh
#
# Safe to source on the host (to get the paths) or inside the Apptainer container
# (where it is also what makes the HPU path work). Every value respects a pre-set
# override, so `VAR=x source env/gaudi_env.sh` works.
#
# Everything cache-like points at /scratch. /home has ~7 GB free and will not survive
# a model download, a container pull, or a torch.compile cache.

# ---------- identity / roots ----------
export VERL_USER="${VERL_USER:-$(id -un)}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${VERL_USER}}"
export REPO_ROOT="${REPO_ROOT:-${SCRATCH_ROOT}/verl-gaudi-support}"
export VERL_COMPAT="${VERL_COMPAT:-${REPO_ROOT}/verl_compat}"
export SGLANG_HPU_ROOT="${SGLANG_HPU_ROOT:-${REPO_ROOT}/.runtime/sglang-miles}"
export SGLANG_HPU_COMMIT=cb05a44f35a7c9e27e46d74112cc841ca674ef43
export CACHE_ROOT="${CACHE_ROOT:-${SCRATCH_ROOT}/verl-cache}"
export VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.runtime/venv}"
export GAUDI_SIF="${GAUDI_SIF:-${SCRATCH_ROOT}/apptainer/gaudi_pt271_sy1222.sif}"

# ---------- python isolation ----------
# ~/.local holds a CUDA torch 2.9.0 and ray 2.54.1. Without this they shadow the
# container's Habana torch and the HPU path dies silently.
export PYTHONNOUSERSITE=1
# Prepend idempotently. This file is sourced once per nesting level (host shell -> shell.sh
# -> the container re-exec -> each Ray worker), and a plain prepend duplicated both entries
# every time -- four copies of each by the time the trainer started. Harmless to Python, but
# it grows without bound and makes the traced env unreadable. Strip any existing copy first.
_verl_pp=":${PYTHONPATH:-}:"
for _d in "${VERL_COMPAT}" "${SGLANG_HPU_ROOT}/python"; do
    _verl_pp="${_verl_pp//":${_d}:"/":"}"
done
_verl_pp="${_verl_pp#:}"; _verl_pp="${_verl_pp%:}"
export PYTHONPATH="${VERL_COMPAT}:${SGLANG_HPU_ROOT}/python${_verl_pp:+:${_verl_pp}}"
unset _verl_pp _d

# ---------- Habana runtime ----------
# GPU Migration Toolkit: torch.cuda.* -> torch.hpu.*. platform_hpu.py is built on the
# CUDA namespace and does not work without it. Must precede any torch import.
export PT_HPU_GPU_MIGRATION=1
# Pin platform detection. Auto-detect caches once per process; one failed
# habana_frameworks import in one actor makes that process think it is on NVIDIA forever.
export VERL_PLATFORM=hpu
# Eager. Lazy mode crashes FSDP flat-param sharding.
export PT_HPU_LAZY_MODE="${PT_HPU_LAZY_MODE:-0}"
# OFF by default. The original rationale for =1 was that RL rollouts vary sequence length
# every step, so shape refinement would avoid recompiling per shape. Measured effect was the
# opposite: the first backward pass sat in _engine_run_backward burning a full CPU core for
# 17+ minutes on a SINGLE graph, emitting no recipe (4339 cached, zero new in 10 minutes).
# Refinement makes the compiler search the shape space rather than compile what it was given.
# miles, whose Gaudi FSDP training path is known good, never sets this variable at all --
# its training env is just PT_HPU_LAZY_MODE=0. Set to 1 to re-enable.
export PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES="${PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES:-0}"
export HABANA_SYSTEM_FORK_UNSAFE_EXEC=1
# FSDP is collective-bound: an all-gather of every layer's params on the way in, a
# reduce-scatter of its grads on the way out -- ~28 layers x micro-batches x 2 per step.
# Without lazy collectives each one is a blocking, host-synchronised HCCL call, and in eager
# mode there is no graph to hide the latency behind. Measured consequence: update_actor ran
# at 119 tok/s/card, i.e. 0.10% MFU on a 0.6B model. miles sets this on BOTH sides -- its
# training env is exactly PT_HPU_LAZY_MODE=0 + PT_HPU_ENABLE_LAZY_COLLECTIVES=1 -- while we
# had only ever set it inside the sglang actor (async_sglang_server.py:185).
export PT_HPU_ENABLE_LAZY_COLLECTIVES="${PT_HPU_ENABLE_LAZY_COLLECTIVES:-1}"
# Load-bearing, not cosmetic: without it the container aborts at import with
# "spdlog_ex: Failed opening /var/log/habana_logs/... Read-only file system".
# NOT "${HABANA_LOGS:-...}": the Habana container presets HABANA_LOGS=$HOME/.habana_logs,
# and that preset wins over our default when this file is re-sourced inside the container.
# /home has ~5 GB free, so Habana logs must not land there. Forced unconditionally.
export HABANA_LOGS="${VERL_HABANA_LOGS:-${SCRATCH_ROOT}/habana_logs}"

# Performance knobs. torch.compile stays OFF until a plain run is proven; FusedSDPA is ON
# because it is not optional at MAX_RESPONSE_LENGTH=2048 -- eager attention materializes the
# full [batch, heads, seq, seq] score matrix per layer (1.26 GB/layer at micro_batch=2 and
# 2560 padded tokens, plus an fp32 softmax copy), which is both the memory ceiling and the
# source of the allocator's defragmentation storms. FusedSDPA tiles like flash-attention and
# never builds that matrix. It engages ONLY when the model also runs with
# attn_implementation=sdpa, which run_grpo_gsm8k.sh sets automatically when this is 1.
# Stock torch SDPA lowering NaNs on Gaudi, so watch actor/grad_norm on the first steps; set
# VERL_HPU_FUSED_SDPA=0 to fall back to eager (correct, ~3x slower at this length).
export VERL_HPU_TORCH_COMPILE="${VERL_HPU_TORCH_COMPILE:-0}"
export VERL_HPU_FUSED_SDPA="${VERL_HPU_FUSED_SDPA:-1}"
export VERL_HPU_WEIGHT_SYNC_DEBUG="${VERL_HPU_WEIGHT_SYNC_DEBUG:-0}"

# ---------- Ray ----------
# Do not let Ray isolate each worker to one HPU module: Habana's eager
# initialize_distributed_hpu() needs to see WORLD_SIZE modules or it asserts
# "not enough devices". Each worker then picks its card by LOCAL_RANK.
export RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES=1
# Four concurrent SGLang HPU lazy compilers can temporarily delay Ray control-plane
# traffic during first-run recipe generation. Allow workers longer than Ray's
# 60-second default to reconnect to GCS instead of terminating the training job.
export RAY_gcs_rpc_server_reconnect_timeout_s="${RAY_gcs_rpc_server_reconnect_timeout_s:-600}"
# The Slurm allocation used for the 8-HPU run is capped at 128 GiB host RAM. Ray's
# automatic object-store sizing reserved roughly 14 GiB in /dev/shm, which left too
# little room for four FP32-master FSDP2 workers plus four SGLang model processes and
# caused the kernel to OOM-kill rank 0. GRPO batches are small and weight transfer uses
# HCCL directly, so a 4 GiB plasma store is ample and returns about 10 GiB of headroom.
export RAY_OBJECT_STORE_MEMORY_BYTES="${RAY_OBJECT_STORE_MEMORY_BYTES:-4294967296}"
# upb protobuf Descriptors cannot be pickled; Ray fails serializing WorkerDict and
# reports the misleading "you set the async flag" ActorDiedError.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# tmpfs, not BeeGFS. Ray's unix sockets need a fast local FS and a short path.
export RAY_TMPDIR="${RAY_TMPDIR:-/dev/shm/ray_${VERL_USER}}"
# ...but object SPILLING must not follow RAY_TMPDIR into tmpfs. With nothing configured,
# ray/_private/node.py:1839 defaults the spill directory to the session dir, i.e. under
# RAY_TMPDIR -- so evicted objects land in /dev/shm, which is RAM. With a 4 GiB object
# store (above) a large-batch/long-response GRPO step evicts heavily, and the spill grows
# unbounded in RAM until the node dies. It is not charged to the Slurm cgroup the way
# process RSS is, so there is no OOM-kill to stop it first. Spill to node-local NVMe.
export RAY_OBJECT_SPILL_DIR="${RAY_OBJECT_SPILL_DIR:-/tmp/ray_spill_${VERL_USER}}"
export RAY_object_spilling_directory="${RAY_object_spilling_directory:-${RAY_OBJECT_SPILL_DIR}}"
export HPU_CARDS_COUNT="${HPU_CARDS_COUNT:-8}"

# ---------- caches: all on /scratch ----------
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
# Catch-all for anything well-behaved we forgot.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
# Forced for the same reason: the container presets TMPDIR=/tmp, which is small and
# node-local. Override with VERL_TMPDIR if you really want something else.
export TMPDIR="${VERL_TMPDIR:-${CACHE_ROOT}/tmp}"
# torch.compile / inductor -- live once VERL_HPU_TORCH_COMPILE=1.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${CACHE_ROOT}/torch/extensions}"
# SynapseAI recipe cache: "<dir>,<clear_on_start>,<size_mb>". Persisting it is what
# stops every graph recompiling on each launch, so this one stays on /scratch despite BeeGFS:
# ~34 new recipes/min is a fraction of the graph-dump traffic above, and losing the cache
# between runs would cost far more than the writes do.
export PT_HPU_RECIPE_CACHE_CONFIG="${PT_HPU_RECIPE_CACHE_CONFIG:-${CACHE_ROOT}/habana_recipe,false,20480}"
# GRAPH_VISUALIZATION_DIR enables Synapse graph-visualization dumps; it is not merely a
# destination override. A GRPO run can create hundreds of thousands of files: on BeeGFS
# this causes severe metadata-I/O stalls, while on /dev/shm the files count against the
# Slurm memory cgroup and can trigger an OOM. Clear inherited values so dumps stay disabled.
unset GRAPH_VISUALIZATION_DIR
export WANDB_DIR="${WANDB_DIR:-${CACHE_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${CACHE_ROOT}/wandb-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${CACHE_ROOT}/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_ROOT}/pip}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${CACHE_ROOT}/apptainer}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${CACHE_ROOT}/apptainer-tmp}"
export SINGULARITY_CACHEDIR="${APPTAINER_CACHEDIR}"
export SINGULARITY_TMPDIR="${APPTAINER_TMPDIR}"

# ---------- data / models / outputs ----------
export GSM8K_DIR="${GSM8K_DIR:-${SCRATCH_ROOT}/data/gsm8k_verl}"
export CKPT_ROOT="${CKPT_ROOT:-${SCRATCH_ROOT}/checkpoints}"

mkdir -p "$HABANA_LOGS" "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR" \
         "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCH_HOME" \
         "$TORCH_EXTENSIONS_DIR" "${PT_HPU_RECIPE_CACHE_CONFIG%%,*}" \
         "$WANDB_DIR" "$WANDB_CACHE_DIR" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" \
         "$GSM8K_DIR" "$CKPT_ROOT" 2>/dev/null

# Convenience: how to enter the container with everything bound.
# APPTAINER_BINDS is consumed by env/shell.sh and env/setup_uv_env.sh.
export APPTAINER_BINDS="${APPTAINER_BINDS:-/scratch/${VERL_USER},/dev/shm}"
