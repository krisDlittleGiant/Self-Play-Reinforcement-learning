#!/usr/bin/env bash
# GRPO | Qwen3-4B-Base | FSDP training | GSM8K | Intel Gaudi HPUs
#
# Structured after examples/grpo_trainer/run_qwen3_4b_fsdp.sh (the GSM8K GRPO reference)
# and examples/grpo_trainer/run_qwen3_4b_dapo_fsdp.sh (whose `hpu)` branch is where every
# Gaudi-forced setting below comes from). Same parameter arrays, same DRY_RUN, same "$@"
# passthrough. What is new here is only the wrapping: this repo runs inside an Apptainer
# container and drives the SPPO recipe rather than main_ppo.
#
#   bash env/run_grpo_gsm8k.sh                      # smoke run, starts its own Ray
#   DRY_RUN=1 bash env/run_grpo_gsm8k.sh            # print the composed command, run nothing
#   bash env/run_grpo_gsm8k.sh trainer.total_training_steps=50   # hydra overrides pass through
#
# ENTRYPOINT: recipe.sppo.main_sppo, not verl.trainer.main_ppo. GRPO is not a recipe -- it is
# `algorithm.adv_estimator=grpo`. main_ppo in this snapshot routes through the modern
# engine/TensorDict workers, which mix jagged nested tensors into a DataProto pipeline and
# die a few steps in. The SPPO recipe still drives the legacy DataProto FSDP workers, so it
# was converted in place to call verl's own compute_advantage with adv_estimator=grpo
# (see recipe/sppo/*.orig for the pre-conversion files).
#
# TOPOLOGY: rollout servers get their OWN cards on top of the training cards --
#   total HPUs = N_GPUS_PER_NODE * (1 + 1/ROLLOUT_TP);  TP=1 => 4 * 2 = 8.
# Raising N_GPUS_PER_NODE to 8 does NOT give 8 trainers: the actor group claims all 8 cards
# first and the rollout servers then fail with "synStatus=8 [Device not found]".
#
# Defaults are a SMOKE configuration: 5 steps, fused SDPA, no validation, console
# logging. Scale up once it survives (see the tail of this file).

set -xeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The local adapter repairs fully masked padding rows before Habana FusedSDPA.
# VERL_HPU_FUSED_SDPA=0 remains available for eager-attention comparisons.
export VERL_HPU_FUSED_SDPA="${VERL_HPU_FUSED_SDPA:-1}"
export VERL_HPU_TORCH_COMPILE="${VERL_HPU_TORCH_COMPILE:-0}"

# This launcher uses full-model FSDP training on Gaudi. FSDP1 otherwise defaults
# to CPU tensor serialization in fsdp_workers.py; explicitly select the HCCL
# transport that completed weight transfer in the Miles-runtime verification.
export VERL_HPU_WEIGHT_SYNC_TRANSPORT="${VERL_HPU_WEIGHT_SYNC_TRANSPORT:-distributed}"
case "$VERL_HPU_WEIGHT_SYNC_TRANSPORT" in
    distributed|tensor) ;;
    *) echo "ERROR: VERL_HPU_WEIGHT_SYNC_TRANSPORT must be distributed or tensor" >&2; exit 2 ;;
esac
# Set before gaudi_env.sh and preserve across the container re-exec.
export VERL_TMPDIR="${VERL_TMPDIR:-/tmp/verl_tmp_${VERL_USER:-$(id -un)}}"
export VERL_HABANA_LOGS="${VERL_HABANA_LOGS:-/tmp/verl_habana_logs_${VERL_USER:-$(id -un)}}"

# shellcheck source=/dev/null
source "$HERE/gaudi_env.sh"

DRY_RUN=${DRY_RUN:-0}

# ---- device (override with DEVICE=hpu; there is no gpu path in this repo) ----
detect_device() {
    if python3 -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("habana_frameworks") else 1)' 2>/dev/null; then
        echo hpu
    else
        echo hpu   # the container always has habana_frameworks; on the host, assume hpu anyway
    fi
}
DEVICE=${DEVICE:-$(detect_device)}

# ---- workload ----
# This checkpoint is resolved through HF_HOME=/scratch/<user>/hf_cache. Pre-downloading it
# is recommended so four FSDP actors and four rollout servers never race a first-time pull.
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}
TRAIN_FILE=${TRAIN_FILE:-${GSM8K_DIR}/train.parquet}
TEST_FILE=${TEST_FILE:-${GSM8K_DIR}/test.parquet}
PROMPT_KEY=${PROMPT_KEY:-prompt}
NNODES=${NNODES:-1}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
# 32:8 = four optimizer steps per rollout. ppo_mini_batch_size is in PROMPTS; the worker
# rescales it to per-rank SEQUENCES at fsdp_workers.py:341 (x rollout.n, / world_size).
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
# micro=2 -> 8 grad-accum rounds per optimizer step (16 seq/rank / 2). Each round is a full
# pass of per-layer all-gather + reduce-scatter, so accumulation depth is dispatch cost --
# the thing measured at 0.10% MFU. Raise toward 16 (= seqs/rank, accum 1) if steps stay slow;
# eager attention costs 0.21 GB/layer of score matrix per unit of micro-batch at L=2560.
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
# StatefulDataLoader subprocesses are unnecessary for this small local parquet and are
# fragile inside a long-lived Ray worker: Ray tears the children down when the task exits,
# which makes PyTorch's SIGCHLD handler report "DataLoader worker ... killed" after an
# otherwise completed step. Keep loading in the trainer process unless explicitly tuned.
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
# Prompts are short (measured on this parquet: p99 142 tokens, max 209), so 512 is ample.
#
# GSM8K reference solutions are p99 292 tokens. Direct-answer mode therefore does not need
# the old 3072-token thinking budget; 1024 leaves ample room while substantially reducing
# padded FSDP work. Watch response_length/clip_ratio and increase only if it is nontrivial.
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
# Qwen3 enables its long <think> mode by default. For GSM8K GRPO this frequently consumes
# the entire response budget before the model emits a final "####" answer, leaving every
# sample with zero reward and therefore zero GRPO advantage/gradient. Direct-answer mode is
# the stable default for this workflow; set ENABLE_THINKING=True explicitly to restore it.
ENABLE_THINKING=${ENABLE_THINKING:-False}

ACTOR_LR=${ACTOR_LR:-1e-6}
USE_KL_LOSS=${USE_KL_LOSS:-False}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
ROLLOUT_N=${ROLLOUT_N:-8}              # rollouts (samples) per prompt -- GRPO's group size
# SPPO recomputes old_log_probs with the FSDP actor, but retain SGLang's rollout
# log-probabilities for rollout/actor diagnostics and future importance correction.
# Disabling them did not fix the Synapse 1.22 D2H stall; it merely moved the blocked
# host copy from next_token_logprobs to the required next_token_ids tensor.
CALCULATE_ROLLOUT_LOG_PROBS=${CALCULATE_ROLLOUT_LOG_PROBS:-True}
# Qwen's recommended non-thinking sampling regime. Sampling must remain stochastic because
# GRPO needs within-prompt reward variation; greedy generation can make every advantage 0.
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.7}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.8}
ROLLOUT_TOP_K=${ROLLOUT_TOP_K:-20}
# Habana lazy execution is not safe with SGLang's CUDA-oriented overlap lifetime model on
# the Synapse 1.22 container.  Keep scheduling synchronous by default.  This does not
# change GRPO, rollout count, log-probabilities, or attention; it only prevents the next
# scheduler iteration from reusing batch state while the current HPU launch is pending.
# Set False explicitly only for an overlap A/B test.
SGLANG_DISABLE_OVERLAP_SCHEDULE=${SGLANG_DISABLE_OVERLAP_SCHEDULE:-True}
# Initialize SGLang from the real checkpoint. Starting captured decode graphs
# from load_format=dummy produced gibberish even when the later FSDP2 -> SGLang
# transfer was value-correct. Online copies are explicitly submitted and drained
# by the HPU weight-updater patch before a captured graph is replayed.
SGLANG_LOAD_FORMAT=${SGLANG_LOAD_FORMAT:-auto}
# Attention backend for the SGLang rollout. hpu_paged_v2 keeps its input buffers at
# static shapes, which is what lets decode be captured into an HPU graph. hpu_fused
# builds its metadata with a dynamic index_select whose shape changes every step, so
# synGraphInferShapes cannot resolve it and capture deadlocks in
# JoinPendingLaunchThread instead of raising. hpu_fused is left selectable only for
# an A/B; it has no working graph path.
SGLANG_HPU_ATTENTION_BACKEND=${SGLANG_HPU_ATTENTION_BACKEND:-hpu_paged_v2}
# Decode-graph replay. Measured on a 5-step GRPO run (Qwen3-4B-Base, batch 8,
# 256 tokens, 64 seqs/step): generation 718-953 s/step disabled vs 29 s on the
# capture step and ~12.7 s steady, and decode 9 -> ~182 tok/s/card.
# The earlier "captured decode corrupts generations after weight sync" note was
# wrong and is retracted: graphs had never actually been constructed in those runs
# (SGLANG_EXPERIMENTAL_HPU_DECODE_GRAPH was unset, so the runner was never built),
# and the corruption was hpu_fused running graphless. With hpu_paged_v2 captured,
# rollout_corr/ppl_ratio stays in 0.9998-1.0062 across five weight syncs.
SGLANG_DECODE_GRAPH_BACKEND=${SGLANG_DECODE_GRAPH_BACKEND:-full}
export VERL_HPU_SGLANG_LAZY="${VERL_HPU_SGLANG_LAZY:-1}"
# These three gate code paths inside the sglang fork itself; without them the
# backend silently falls back and the decode-graph runner is never constructed,
# which is why SGLANG_VERIFY_HPU_DECODE_GRAPH=1 reported nothing twice.
if [ "${SGLANG_HPU_ATTENTION_BACKEND}" = "hpu_paged_v2" ]; then
    export SGLANG_EXPERIMENTAL_HPU_PAGED_V2="${SGLANG_EXPERIMENTAL_HPU_PAGED_V2:-1}"
fi
if [ "${SGLANG_DECODE_GRAPH_BACKEND}" != "disabled" ]; then
    export SGLANG_EXPERIMENTAL_HPU_DECODE_GRAPH="${SGLANG_EXPERIMENTAL_HPU_DECODE_GRAPH:-1}"
    export PT_HPU_AUTOLOAD="${PT_HPU_AUTOLOAD:-1}"
fi
if [ "${VERL_HPU_SGLANG_LAZY}" = "0" ] && [ "${SGLANG_DECODE_GRAPH_BACKEND}" != "disabled" ]; then
    echo "ERROR: VERL_HPU_SGLANG_LAZY=0 requires SGLANG_DECODE_GRAPH_BACKEND=disabled." >&2
    exit 1
fi
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-64}
# Keep prefill admission on a small, stable bucket independently from the larger
# decode concurrency. Non-bucket decode sizes are padded around the whole eager
# model forward by the repository's SGLang HPU patch.
SGLANG_PREFILL_MAX_REQUESTS=${SGLANG_PREFILL_MAX_REQUESTS:-2}
# ON, and it is what makes MICRO=8 possible. Reasoning corrected: the "17.5 GB of 98" headroom
# was measured with FusedSDPA, which never materialises a score matrix. Eager attention
# (VERL_HPU_FUSED_SDPA=0, needed while grad_norm is NaN) stores [B, heads, L, L] per layer for
# backward -- at micro=8, L=2560 that is 1.68 GB x 28 layers = 47 GB held at once, on top of
# 17.5 GB of weights/optimizer and against 94.6 GB already reserved. That OOMs.
# With checkpointing the scores are recomputed per layer instead: 1.68 GB transient plus
# 1.17 GB of layer inputs. The extra compute is close to free here -- at 0.10% MFU the
# bottleneck is dispatch and collective latency, not FLOPs.
# Set False once FusedSDPA is proven numerically sound and the score matrix disappears again.
GRAD_CKPT=${GRAD_CKPT:-True}
# See the fsdp_config block below. Set True to restore stock FSDP resharding.
RESHARD_AFTER_FWD=${RESHARD_AFTER_FWD:-False}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_gsm8k_gaudi}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-$(basename "$MODEL_PATH")_gsm8k_grpo_hpu}

# ---- W&B ----
# WANDB=1 is the switch;  WANDB=0 (default) keeps console-only logging.
# verl calls wandb.init(project=trainer.project_name, name=trainer.experiment_name,
# entity=$WANDB_ENTITY) -- see verl/utils/tracking.py:80. So the run's identity comes from
# PROJECT_NAME / EXPERIMENT_NAME above, NOT from WANDB_PROJECT / WANDB_NAME, which verl
# never reads. Setting those two env vars does nothing here; set PROJECT_NAME instead.
WANDB=${WANDB:-0}
if [ "$WANDB" = "1" ]; then
    LOGGER=${LOGGER:-'["console","wandb"]'}
    # Offline is the safe choice on a node with flaky egress: metrics are written under
    # $WANDB_DIR and pushed later with `wandb sync`. Default is online.
    export WANDB_MODE="${WANDB_MODE:-online}"
    # Groups every run of this experiment together in the W&B UI.
    export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${EXPERIMENT_NAME}}"
fi
LOGGER=${LOGGER:-'["console"]'}
SAVE_FREQ=${SAVE_FREQ:--1}
# Validation is optional for GRPO and is deliberately disabled here. The per-training-batch
# GSM8K reward computation remains enabled and is what produces GRPO advantages.
TEST_FREQ=${TEST_FREQ:--1}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
# 5 = smoke. Set TOTAL_TRAINING_STEPS=null for a real run: ppo_trainer.yaml leaves this
# null by default and ray_trainer.py:435 then derives len(train_dataloader) * total_epochs,
# so the run covers exactly TOTAL_EPOCHS passes over the data with no hand arithmetic.
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-5}
# The upstream defaults launch eight AgentLoopWorker and eight RewardLoopWorker actors.
# Each imports the full Python/torch stack (~0.65--0.8 GiB RSS), so those helpers alone
# used over 11 GiB in the 128-GiB Slurm cgroup. Two async agent workers retain request
# concurrency, and two reward workers are sufficient for the inexpensive GSM8K checker.
AGENT_LOOP_NUM_WORKERS=${AGENT_LOOP_NUM_WORKERS:-2}
REWARD_LOOP_NUM_WORKERS=${REWARD_LOOP_NUM_WORKERS:-2}

# ---- re-exec inside the container ----
# Everything below (device probing, the venv's ray, the trainer) needs the container. Enter
# it ONCE and stay: Ray started in a separate `apptainer exec` outlives the exec but loses
# its squashfuse view of the SIF, so the daemons survive with a broken /usr. One session.
if [ -z "${VERL_RUN_INNER:-}" ]; then
    exec env VERL_RUN_INNER=1 bash "$HERE/shell.sh" bash "${BASH_SOURCE[0]}" "$@"
fi

case "${DEVICE}" in
    hpu)
        INFER_BACKEND=${INFER_BACKEND:-sglang}
        # 8-card node: 4 training + 4 rollout.
        N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
        ROLLOUT_TP=${ROLLOUT_TP:-1}              # everything on HPU is validated at TP=1
        ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}

        # Miles uses eager prefill and bucketed decode. The old 0.4.9 prefill
        # token-bucket ceiling does not apply to this runtime. Decode sequence
        # buckets default to 128 tokens; round context_length accordingly below.

        PLATFORM_OPTS=(
            # No varlen attention kernel on Gaudi (FusedSDPA has no cu_seqlens API), so the
            # remove-padding path cannot run at all. THE known perf gap vs CUDA.
            actor_rollout_ref.model.use_remove_padding=False
            actor_rollout_ref.actor.use_remove_padding=False
            # Token-budget batching is meaningless at fixed padded width.
            actor_rollout_ref.actor.use_dynamic_bsz=False
            actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
            actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
            # Keep the trainable/master parameters in FP32. fsdp_config.dtype remains BF16,
            # so forward/backward compute still uses BF16 mixed precision. VERL's worker
            # explicitly warns that constructing the actor in BF16 also creates a BF16
            # optimizer, and the Miles Gaudi reference likewise keeps FP32 masters.
            actor_rollout_ref.actor.fsdp_config.model_dtype=float32
            actor_rollout_ref.rollout.dtype=bfloat16
            # param_offload silently CORRUPTS FSDP weights on Gaudi (verified: entropy pins
            # at ln(vocab), all rewards collapse to -1, pg_loss -> exactly 0 -- no crash).
            # Every offload OFF, including the ref policy's, whose stock default is True.
            # reshard_after_forward=False. py-spy caught compute_log_prob burning 103% of a CPU
        # core with all 8 cards at 0% AIP-Util and no graphs compiling, parked in:
        #   _post_forward_reshard -> _free_unsharded_flat_param -> _free_storage
        #   -> torch/storage.py:1258 _resize_
        # FSDP frees each layer's unsharded flat param right after that layer's forward, and
        # on HPU storage.resize_(0) is a host-side synchronous call that waits on the device
        # queue. 28 layers x every micro-batch x forward and backward = that host stall IS
        # the 0.10% MFU. Keeping the 4B model's BF16 params resident costs about 8 GB per
        # card against 98 GB, and also removes the re-all-gather in backward.
        actor_rollout_ref.actor.fsdp_config.reshard_after_forward="${RESHARD_AFTER_FWD}"
        actor_rollout_ref.ref.fsdp_config.reshard_after_forward="${RESHARD_AFTER_FWD}"
        actor_rollout_ref.actor.fsdp_config.param_offload=False
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
            actor_rollout_ref.ref.fsdp_config.param_offload=False
            # No torch_memory_saver on HPU (it needs CUDA virtual-memory APIs), and the
            # rollout owns dedicated cards anyway, so there is nothing to release.
            actor_rollout_ref.rollout.free_cache_engine=False
            actor_rollout_ref.rollout.enforce_eager=False
            actor_rollout_ref.rollout.max_num_seqs="${SGLANG_MAX_RUNNING_REQUESTS}"
            +actor_rollout_ref.rollout.engine_kwargs.sglang.device=hpu
            +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="${SGLANG_HPU_ATTENTION_BACKEND}"
            +actor_rollout_ref.rollout.engine_kwargs.sglang.decode_attention_backend="${SGLANG_HPU_ATTENTION_BACKEND}"
            +actor_rollout_ref.rollout.engine_kwargs.sglang.sampling_backend=pytorch
            +actor_rollout_ref.rollout.engine_kwargs.sglang.grammar_backend=none
            +actor_rollout_ref.rollout.engine_kwargs.sglang.cuda_graph_backend_prefill=disabled
            +actor_rollout_ref.rollout.engine_kwargs.sglang.cuda_graph_backend_decode="${SGLANG_DECODE_GRAPH_BACKEND}"
            +actor_rollout_ref.rollout.engine_kwargs.sglang.cuda_graph_max_bs_decode="${SGLANG_MAX_RUNNING_REQUESTS}"
            +actor_rollout_ref.rollout.engine_kwargs.sglang.context_length="$(( (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 127) / 128 * 128 ))"
            +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule="${SGLANG_DISABLE_OVERLAP_SCHEDULE}"
            +actor_rollout_ref.rollout.engine_kwargs.sglang.prefill_max_requests="${SGLANG_PREFILL_MAX_REQUESTS}"
            # HPU FSDP2 broadcasts BF16 weights directly to SGLang over HCCL. This
            # value limits each materialized full-parameter bucket; oversized model
            # tensors remain one bucket because they cannot be split by this adapter.
            actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=128
        )

        # Attention path. Stock torch F.sdpa lowering NaNs grad_norm on Gaudi, so `sdpa` is
        # only safe when verl's FusedSDPA patch is live (verl/__init__.py, gated on
        # VERL_HPU_FUSED_SDPA=1). With the patch off, leave HF on eager -- it materializes
        # the full score matrix (slow, defragmentation storms) but is numerically correct.
        if [ "${VERL_HPU_FUSED_SDPA:-0}" = "1" ]; then
            PLATFORM_OPTS+=( ++actor_rollout_ref.model.override_config.attn_implementation=sdpa )
        else
            PLATFORM_OPTS+=( ++actor_rollout_ref.model.override_config.attn_implementation=eager )
        fi
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. This repo only supports 'hpu'." >&2
        exit 1
        ;;
esac

# Derived batch geometry. ppo_mini_batch_size is configured in PROMPTS, but the worker
# rescales it to per-rank SEQUENCES at fsdp_workers.py:341:
#     ppo_mini_batch_size *= rollout.n ; ppo_mini_batch_size //= world_size
# Printing it here avoids having to recompute that by hand every time n or the batch moves.
SEQS_PER_STEP=$(( TRAIN_BATCH_SIZE * ROLLOUT_N ))
MINI_SEQS_PER_RANK=$(( PPO_MINI_BATCH_SIZE * ROLLOUT_N / N_GPUS_PER_NODE ))
OPT_STEPS_PER_ROLLOUT=$(( TRAIN_BATCH_SIZE / PPO_MINI_BATCH_SIZE ))
ACCUM=$(( MINI_SEQS_PER_RANK / PPO_MICRO_BATCH_SIZE_PER_GPU ))
CONCURRENT_PER_SERVER=$(( SEQS_PER_STEP / (N_GPUS_PER_NODE / ROLLOUT_TP) ))
echo "=== batch geometry ==="
echo "    weight sync: ${VERL_HPU_WEIGHT_SYNC_TRANSPORT}; TMPDIR=${TMPDIR}; HABANA_LOGS=${HABANA_LOGS}"
echo "    ${TRAIN_BATCH_SIZE} prompts/step x n=${ROLLOUT_N}  = ${SEQS_PER_STEP} sequences/step"
echo "    mini-batch: ${PPO_MINI_BATCH_SIZE} prompts -> ${MINI_SEQS_PER_RANK} seq/rank; ${OPT_STEPS_PER_ROLLOUT} optimizer step(s) per rollout"
echo "    micro-batch ${PPO_MICRO_BATCH_SIZE_PER_GPU} seq/card -> ${ACCUM} grad-accum micro-steps per optimizer step"
echo "    rollout requests: ~${CONCURRENT_PER_SERVER}/server; active cap=${SGLANG_MAX_RUNNING_REQUESTS}; prefill cap=${SGLANG_PREFILL_MAX_REQUESTS}; SGLang load=${SGLANG_LOAD_FORMAT}; attn=${SGLANG_HPU_ATTENTION_BACKEND}; Miles decode graphs=${SGLANG_DECODE_GRAPH_BACKEND}"
if [ "$(( MINI_SEQS_PER_RANK % PPO_MICRO_BATCH_SIZE_PER_GPU ))" -ne 0 ]; then
    echo "ERROR: per-rank mini-batch ${MINI_SEQS_PER_RANK} is not divisible by micro-batch ${PPO_MICRO_BATCH_SIZE_PER_GPU}." >&2
    echo "  fsdp_workers.py:355 asserts this. Adjust PPO_MINI_BATCH_SIZE or PPO_MICRO_BATCH_SIZE_PER_GPU." >&2
    exit 1
fi

########################### preflight ###########################
if [[ "${DRY_RUN}" != "1" ]]; then
    # Node-local mirrors are optional, but when selected they must contain the current
    # adapter. A stale /tmp copy previously hid the decode-warmup fix and wasted two full
    # Qwen startup cycles before reproducing the already-fixed exception.
    SOURCE_VERL_COMPAT="${REPO_ROOT}/verl_compat"
    CRITICAL_ADAPTER="verl/workers/rollout/sglang_rollout/async_sglang_server.py"
    if [[ "${VERL_COMPAT}" != "${SOURCE_VERL_COMPAT}" ]] && \
       ! cmp -s "${SOURCE_VERL_COMPAT}/${CRITICAL_ADAPTER}" "${VERL_COMPAT}/${CRITICAL_ADAPTER}"; then
        echo "ERROR: stale local VERL_COMPAT mirror: ${VERL_COMPAT}" >&2
        echo "  Refresh it from the host shell, then rerun:" >&2
        echo "    eval \"\$(bash env/sync_local.sh)\"" >&2
        exit 2
    fi
    "${VENV_DIR}/bin/python" "$HERE/verify_sglang_miles.py" --source-only
    [ -f "$TRAIN_FILE" ] || { echo "ERROR: missing $TRAIN_FILE" >&2
        echo "  generate it:  bash env/shell.sh bash -c 'cd \$VERL_COMPAT && python3 examples/data_preprocess/gsm8k.py --local_save_dir \$GSM8K_DIR'" >&2; exit 1; }
    [ -e "$MODEL_PATH" ] || echo "NOTE: $MODEL_PATH is not a local dir; treating it as a HF repo id (downloads into \$HF_HOME)." >&2

    NEEDED=$(( N_GPUS_PER_NODE + N_GPUS_PER_NODE / ROLLOUT_TP ))
    echo "=== topology: ${N_GPUS_PER_NODE} training + $(( N_GPUS_PER_NODE / ROLLOUT_TP )) rollout = ${NEEDED} HPUs (TP=${ROLLOUT_TP}) ==="


    # W&B credentials, checked BEFORE the cards are claimed. wandb.init() runs inside the
    # driver actor after the model is loaded, so an auth failure there costs a full startup
    # (minutes, plus a Ray teardown) to discover. Two seconds here instead.
    if [[ "${LOGGER}" == *wandb* && "${WANDB_MODE:-online}" != "offline" ]]; then
        if ! "${VENV_DIR}/bin/python" - <<'PY'
import sys
try:
    import wandb
    sys.exit(0 if wandb.api.api_key else 1)
except Exception:
    sys.exit(1)
PY
        then
            echo "ERROR: LOGGER includes wandb but no usable credential was found." >&2
            echo "  Either:  export WANDB_API_KEY=<key from https://wandb.ai/authorize>" >&2
            echo "  or:      ${VENV_DIR}/bin/wandb login          # writes ~/.netrc" >&2
            echo "  or run offline:  WANDB=1 WANDB_MODE=offline bash env/run_grpo_gsm8k.sh" >&2
            exit 1
        fi
        echo "=== W&B: ${WANDB_MODE:-online} | project=${PROJECT_NAME} | run=${EXPERIMENT_NAME} | entity=${WANDB_ENTITY:-<default>} ==="
    fi

    # A Ray cluster with the HPU resource registered must be up before the trainer connects:
    # main_sppo's own ray.init() would create one with ZERO HPU resources and die with
    # "Total available GPUs 0 is less than total desired GPUs N". Reuse a cluster if this
    # shell already has one (the env/shell.sh + env/start_ray_gaudi.sh workflow); otherwise
    # start one here, in this same container session, and tear it down on exit.
    # A stale ray_current_cluster file can make `ray status` wait indefinitely. Record the
    # elapsed time for diagnosis, but also bound the probe so the fallback is reachable.
    RAY_STATUS_TIMEOUT_SECONDS=${RAY_STATUS_TIMEOUT_SECONDS:-30}
    ray_probe_started=$SECONDS
    if timeout --signal=TERM --kill-after=5s "${RAY_STATUS_TIMEOUT_SECONDS}s" \
        "${VENV_DIR}/bin/python" -m ray.scripts.scripts status \
        --address="127.0.0.1:${RAY_PORT:-6381}" >/dev/null 2>&1; then
        ray_probe_elapsed=$(( SECONDS - ray_probe_started ))
        echo "=== reusing the Ray cluster already running on 127.0.0.1:${RAY_PORT:-6381} (probe ${ray_probe_elapsed}s) ==="
    else
        ray_probe_rc=$?
        ray_probe_elapsed=$(( SECONDS - ray_probe_started ))
        if [ "$ray_probe_rc" -eq 124 ]; then
            echo "=== Ray status probe timed out after ${ray_probe_elapsed}s; starting a fresh cluster ==="
        else
            echo "=== no Ray cluster reachable (probe rc=${ray_probe_rc}, ${ray_probe_elapsed}s); starting one ==="
        fi
        bash "$HERE/start_ray_gaudi.sh"
        # Only stop what we started, and never let a trainer failure be masked by ray stop.
        trap 'rc=$?; set +e; "${VENV_DIR}/bin/python" -m ray.scripts.scripts stop --force >/dev/null 2>&1; exit $rc' EXIT
    fi
fi

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key="${PROMPT_KEY}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.dataloader_num_workers="${DATALOADER_NUM_WORKERS}"
    +data.apply_chat_template_kwargs.enable_thinking="${ENABLE_THINKING}"
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.enable_gradient_checkpointing="${GRAD_CKPT}"
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}"
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}"
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}"
)

ROLLOUT=(
    actor_rollout_ref.rollout.name="${INFER_BACKEND}"
    actor_rollout_ref.rollout.load_format="${SGLANG_LOAD_FORMAT}"
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}"
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    actor_rollout_ref.rollout.calculate_log_probs="${CALCULATE_ROLLOUT_LOG_PROBS}"
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}"
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}"
    actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K}"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_LOOP_NUM_WORKERS}"
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger="${LOGGER}"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}"
    trainer.nnodes="${NNODES}"
    trainer.val_before_train="${VAL_BEFORE_TRAIN}"
    trainer.save_freq="${SAVE_FREQ}"
    trainer.test_freq="${TEST_FREQ}"
    trainer.total_epochs="${TOTAL_EPOCHS}"
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
    trainer.default_local_dir="${CKPT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
    reward.num_workers="${REWARD_LOOP_NUM_WORKERS}"
    # Pin Ray to the cluster above rather than whatever stale RAY_ADDRESS is in the shell.
    # LEADING '+' IS REQUIRED: ppo_trainer.yaml's ray_kwargs.ray_init declares only
    # `num_cpus`, and the config is a struct, so a bare override dies at composition with
    # "Could not override 'ray_kwargs.ray_init.address' / Key 'address' is not in struct".
    # main_sppo.py:79 splats the whole ray_init dict into ray.init(), so the added key
    # arrives as ray.init(address=...) exactly as intended.
    +ray_kwargs.ray_init.address="127.0.0.1:${RAY_PORT:-6381}"
)

########################### launch ###########################
CMD=("${VENV_DIR}/bin/python" -m recipe.sppo.main_sppo
    "${DATA[@]}"
    "${MODEL[@]}"
    "${ACTOR[@]}"
    "${ROLLOUT[@]}"
    "${PLATFORM_OPTS[@]}"
    "${TRAINER[@]}"
    "$@"
)

if [[ "${DRY_RUN}" == "1" ]]; then
    set +x
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

# recipe.sppo's hydra config declares `searchpath: file://verl/trainer/config`, a RELATIVE
# path -- it only resolves with verl_compat as the working directory.
cd "$VERL_COMPAT"
"${CMD[@]}"

# ---------------------------------------------------------------------------------------
# SCALING UP, once a smoke run survives (change ONE thing at a time):
#   1. TOTAL_TRAINING_STEPS=50 VAL_BEFORE_TRAIN=True TEST_FREQ=10
#   2. TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=32 ROLLOUT_N=5   (reference geometry)
#   3. MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=1024          (reference lengths)
#   4. WANDB=1                                                  (see the W&B block above)
#   5. unset SGLANG_HPU_SKIP_WARMUP                             (pay capture once, not per step)
#   6. VERL_HPU_TORCH_COMPILE=1                                 (Gaudi's documented FSDP path)
#   7. VERL_HPU_FUSED_SDPA=1                                    (watch actor/grad_norm for NaN)
# ---------------------------------------------------------------------------------------
