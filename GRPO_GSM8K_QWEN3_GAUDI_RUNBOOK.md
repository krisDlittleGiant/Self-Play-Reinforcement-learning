# GRPO on GSM8K with Qwen3 and Intel Gaudi

2026-09-10 verification: the graph-disabled/lazy-SGLang configuration completed
five consecutive Qwen3-4B-Base GRPO steps on 4 FSDP2 trainer HPUs + 4 rollout
HPUs. Every step had nonzero reward, policy loss, and finite gradient norm. The
training/rollout PPL ratios were 1.0100, 1.0016, 1.0007, 1.0019, and 1.0031,
confirming that online weight updates remained correct across steps. The test
used 8 prompts, 8 rollouts per prompt, and a 256-token response limit. It took
about 69 minutes; correctness is established, but graph-disabled rollout
throughput remains the principal blocker for a practical full epoch.

2026-09-09 update: FSDP2 weight export now materializes every bucket on every
actor rank before rank 0 broadcasts it. This prevents non-sender ranks from
running ahead into the final barrier and stalling the last full-tensor gather.
The FSDP2-to-SGLang fingerprints match the original Qwen checkpoint, the HCCL
receive buffers, and the loaded SGLang parameters. Online training must use
graph-disabled decode with the SGLang process still in HPU lazy mode. Captured
decode remains corrupted after online weight updates even when its cache is
invalidated and recaptured; an eager-scheduler experiment also fails its first
HPU prefill compilation. The graph-disabled lazy combination is numerically
verified at 4 trainers + 4 rollout servers with a 256-token response limit
(training/rollout PPL ratio 1.0086 and finite nonzero gradient). A
eager-scheduler experiment (`PT_HPU_LAZY_MODE=0`) is not viable on this pinned
SGLang/Gaudi stack: even the first standalone prefill fails compilation when
materializing its token result. SGLang must therefore remain in lazy mode while
decode-graph replay remains disabled. A cold 2048-token run compiles additional
sequence buckets and may be quiet for a long time; retain the recipe cache.

2026-09-06 update: the launcher now targets the pinned Miles SGLang runtime. Use
[the migration guide](GRPO_MILES_SGLANG_MIGRATION.md) for setup and verification.
The verified results below describe the previous runtime, not proof of migrated-run stability.

This is the canonical runbook for the working configuration in this repository. It covers
Qwen3-4B-Base with FSDP training and SGLang rollout on one node with eight Intel HL-225
Gaudi HPUs.

Last verified: 2026-09-10 (five-step online-update test).

## Verified configuration

| Component | Setting |
|---|---|
| Model | `Qwen/Qwen3-4B-Base` (4.02B parameters, BF16 checkpoint) |
| Dataset | GSM8K in verl parquet format |
| Algorithm | GRPO (`algorithm.adv_estimator=grpo`) |
| Trainer | Legacy DataProto FSDP path through `recipe.sppo.main_sppo` |
| Training devices | 4 HPUs, FSDP |
| Rollout devices | 4 HPUs, one SGLang server per HPU, TP=1 |
| SGLang decode execution | Lazy HPU execution, graph replay disabled (`cuda_graph_backend_decode=disabled`) |
| Attention | Hugging Face SDPA routed to Habana FusedSDPA in training workers |
| Prompt/response limits | 512 / 1024 tokens |
| Rollouts per prompt | 8 |
| Training batch | 32 prompts = 256 generated sequences per rollout step |
| Sampling | temperature 0.7, top-p 0.8, top-k 20 |
| Thinking | Disabled explicitly |
| Validation | Disabled (`val_before_train=False`, `test_freq=-1`) |
| Compile | Disabled while establishing correctness |

The topology is disaggregated. `N_GPUS_PER_NODE=4` means four training devices plus four
rollout devices, not four devices total. Do not set it to 8 on an eight-device node.

## What was fixed

### 1. Training was routed to the wrong SDPA backend

The old code decided that a process belonged to SGLang if this module had been imported:

```python
"sglang.srt.managers.scheduler" in sys.modules
```

That test was invalid. The FSDP training worker imports the SGLang rollout client, which in
turn imports the scheduler module. The training worker was therefore misclassified as an
SGLang scheduler and bypassed Habana FusedSDPA. Real Qwen execution through the resulting
native PyTorch SDPA path became non-finite at layer 0 and ended with NaN loss and gradients.

The fix uses an explicit process marker:

```text
VERL_HPU_SGLANG_PROCESS=1
```

Only `SGLangHttpServer` sets this marker before spawning its inference children. FSDP
workers now use the Habana adapter even if SGLang modules happen to be imported.

Relevant code:

- `verl_compat/verl/__init__.py`
- `verl_compat/verl/workers/rollout/sglang_rollout/async_sglang_server.py`

### 2. Fully masked padding rows produced NaNs

Left-padded causal batches contain query rows with no allowed key. Depending on the caller,
the row arrived as Boolean `False`, negative infinity, or the floating-point dtype minimum.
For example, the effective additive mask was:

```text
[-inf, -inf, -inf, -inf]
```

A fused online softmax cannot form a finite denominator for this row. The adapter now gives
only an empty row one temporary dummy key:

```text
[0.0, -inf, -inf, -inf]
```

After FusedSDPA returns, the output for that query row is explicitly zeroed. Real query rows
are unchanged, and the input mask is not mutated.

The implementation supports Boolean, `-inf`, and dtype-min masks:

- `verl_compat/verl/utils/hpu_sdpa.py`

### 3. Backend fallback broke gradient checkpoint recomputation

The previous wrapper caught every exception from FusedSDPA and silently switched to native
SDPA. Non-reentrant gradient checkpointing uses an internal exception/control path to stop
recomputation after restoring its saved tensors. Catching it and changing attention backend
changed the recomputed graph and produced a saved-tensor-count `CheckpointError`.

The broad fallback was removed. Kernel and checkpoint errors now propagate at their source.
FusedSDPA uses recomputation for training and stays on the same backend during forward and
checkpoint recomputation.

### 4. Invalid values were hidden

The actor previously ran `torch.nan_to_num(log_probs, nan=0.0)`, which could turn corrupted
attention output into apparently valid zero log probabilities. A non-finite FSDP gradient
norm was then skipped, leaving weights unchanged while the run continued.

HPU actor code now checks log probabilities, entropy, auxiliary probability metrics, and
gradient norm. It raises `FloatingPointError` before invalid values can be hidden.

Relevant code:

- `verl_compat/verl/workers/actor/dp_actor.py`

### 5. Thinking mode caused a numerically healthy no-op run

The SDPA fix can be correct while GRPO still performs no learning. In one Qwen3-0.6B run,
every one of 256 responses reached the 3072-token ceiling. None emitted the expected final
`#### <number>` answer, so every reward and advantage was zero. This resulted in:

```text
actor/pg_loss = 0
actor/grad_norm = 0
response_length/clip_ratio = 1.0
```

The launcher now passes the precise VERL chat-template override:

```text
+data.apply_chat_template_kwargs.enable_thinking=False
```

This is controlled by `ENABLE_THINKING`, whose default is `False`.

### 6. Qwen3-4B-Base direct-answer defaults

`env/run_grpo_gsm8k.sh` now defaults to:

```text
MODEL_PATH=Qwen/Qwen3-4B-Base
MAX_RESPONSE_LENGTH=1024
ENABLE_THINKING=False
ROLLOUT_TEMPERATURE=0.7
ROLLOUT_TOP_P=0.8
ROLLOUT_TOP_K=20
```

Qwen3-4B-Base is a pretrained Base checkpoint, not an instruction-tuned checkpoint. It can
still fail to follow the exact GSM8K output format. Zero reward with finite tensors is a
reward/format problem, not automatically an attention failure.

## Setup from a clean checkout

One command takes a fresh clone to a runnable training job. It is idempotent -- every
stage detects existing state and skips -- so it is safe to re-run after a partial failure.

```bash
git clone <this repo> /scratch/$USER/verl-gaudi-support
cd /scratch/$USER/verl-gaudi-support
bash env/bootstrap.sh
```

Every path derives from `$USER` through `env/gaudi_env.sh` (`VERL_USER` -> `SCRATCH_ROOT`
-> everything else), so no file needs editing for a different account. Override
`SCRATCH_ROOT`, `REPO_ROOT`, `VENV_DIR`, `GAUDI_SIF`, or `GSM8K_DIR` in the environment if
your layout differs.

`env/bootstrap.sh` runs five stages:

| stage | what it does | skipped when |
|-------|--------------|--------------|
| 1 environment | resolves paths from `$USER`, checks `apptainer` and `uv` are on PATH | never |
| 2 container | pulls the SynapseAI 1.22.2 / PyTorch 2.7.1 image (~1.6 GB) | `$GAUDI_SIF` exists |
| 3 runtime | `env/setup_uv_env.sh`: clone pinned SGLang, apply patches, build venv | venv + fork present |
| 4 dataset | writes GSM8K train/test parquet via `examples/data_preprocess/gsm8k.py` | parquet files exist |
| 5 verify | `verify_sglang_miles.py --source-only`, then `hl-smi` | never |

Then start training:

```bash
bash env/run_grpo_full.sh
```

Prerequisites the bootstrap does **not** install: `apptainer` and `uv` must already be on
PATH, and you need an allocation holding 8 Gaudi cards.

### What stage 3 builds

`env/setup_uv_env.sh` is the load-bearing part, and it never resolves upstream's CUDA
dependencies. It:

1. clones SGLang at the pinned commit `cb05a44f35a7c9e27e46d74112cc841ca674ef43` into
   `.runtime/sglang-miles` (gitignored -- reproduced from patches, never committed);
2. applies `env/patches/*.patch` **in this order**, refusing to continue if any fails to
   apply cleanly:

   | order | patch | purpose |
   |-------|-------|---------|
   | 1 | `sglang-miles-cb05a44-gaudi.patch` | the Miles Gaudi port |
   | 2 | `sglang-cb05a44-hpu-container-compat.patch` | container-only Torch 2.7 fixes |
   | 3 | `sglang-cb05a44-verl-hpu-runtime.patch` | this repo's fixes, found by the verl GRPO integration |

3. creates the venv with `--system-site-packages` against the container's
   `/usr/bin/python3.12` (so `torch 2.7.1+hpu` is inherited, never reinstalled) and
   installs with `--no-deps` throughout;
4. applies `patches/transformers-5.12-hpu-annotations.patch` into site-packages;
5. runs `verify_sglang_miles.py`, which rebuilds the patch stack independently and fails
   if the working tree has drifted from patches + pinned commit.

Editing `.runtime/sglang-miles` directly breaks that invariant. To change SGLang
behaviour, edit the tree, regenerate patch 3 with `git -C .runtime/sglang-miles diff`, and
commit the patch -- not the tree.

## Running

`env/run_grpo_full.sh` carries the validated configuration and launches detached, so an
SSH or VSCode disconnect does not kill the run.

```bash
bash env/run_grpo_full.sh                                    # full 233-step epoch
FOREGROUND=1 bash env/run_grpo_full.sh                       # stay attached
TOTAL_TRAINING_STEPS=5 WANDB=0 SAVE_FREQ=-1 TEST_FREQ=-1 \
  bash env/run_grpo_full.sh                                  # 5-step smoke test
```

Every value is an override-able default:

| setting | default | why |
|---------|---------|-----|
| `TRAIN_BATCH_SIZE` / `PPO_MINI_BATCH_SIZE` / `PPO_MICRO_BATCH_SIZE_PER_GPU` | 32 / 8 / 2 | 256 seqs/step, 4 optimizer steps per rollout |
| `MAX_RESPONSE_LENGTH` | 2048 | 0% truncation; 256 clipped 20.7% of answers mid-derivation |
| `SGLANG_HPU_ATTENTION_BACKEND` | `hpu_paged_v2` | the only backend with a working graph path |
| `SGLANG_DECODE_GRAPH_BACKEND` | `full` | 26x faster generation -- see below |
| `SAVE_FREQ` / `TEST_FREQ` | 20 / 20 | checkpoint and validate together |
| `EXPERIMENT_NAME` | `Qwen3_4B_b32_r2048_full` | **stable, not timestamped** -- this is what makes resume work |

**Resume is automatic.** verl's `resume_mode` is `auto`, so re-running the exact same
command after any interruption picks up from the latest checkpoint. That only works
because `EXPERIMENT_NAME` is stable; a timestamped name silently starts from scratch.
Confirm by looking for `Setting global step to N` in the log -- if it says
`Training from scratch` instead, the checkpoint was not found, and you should stop.

The script kills any existing Ray/SGLang processes on startup, so **do not run it while
another job of yours is training.**

## Prerequisites

Use the repository from its expected scratch location:

```bash
cd /scratch/$USER/verl-gaudi-support
```

Confirm the eight allocated HPUs are not occupied by another run:

```bash
hl-smi
pgrep -u "$(id -u)" -af 'ray|sglang|main_sppo|WorkerDict'
```

The launcher enters the configured Apptainer image, loads the local venv, starts a Ray
cluster if needed, and stops only the cluster it started. Runtime and model caches live under
`/scratch/$USER`, not the home directory.

## Download Qwen3-4B-Base

Download once before launching Ray workers. The checkpoint is public and is stored through
`HF_HOME=/scratch/$USER/hf_cache`:

```bash
cd /scratch/$USER/verl-gaudi-support
bash env/shell.sh hf download Qwen/Qwen3-4B-Base
```

The launcher may also resolve the repository ID directly, but pre-downloading prevents eight
processes from contending on a first-time model pull.

## Configuration-only dry run

This prints the composed command without claiming HPUs:

```bash
cd /scratch/$USER/verl-gaudi-support
DRY_RUN=1 VERL_HPU_FUSED_SDPA=1 bash env/run_grpo_gsm8k.sh
```

Check that the output contains:

```text
actor_rollout_ref.model.path=Qwen/Qwen3-4B-Base
+data.apply_chat_template_kwargs.enable_thinking=False
actor_rollout_ref.rollout.temperature=0.7
actor_rollout_ref.rollout.top_p=0.8
actor_rollout_ref.rollout.top_k=20
++actor_rollout_ref.model.override_config.attn_implementation=sdpa
trainer.val_before_train=False
trainer.test_freq=-1
```

## SDPA mask regression test

This is a small one-HPU example. It intentionally prints the broken behavior first and then
checks the repaired Boolean, negative-infinity, and dtype-min masks with non-reentrant
checkpointing:

```bash
cd /scratch/$USER/verl-gaudi-support

VERL_HPU_FUSED_SDPA=1 \
VERL_HPU_TORCH_COMPILE=0 \
bash env/shell.sh python env/verify_sdpa_fix.py 2>&1 | \
tee /scratch/$USER/verl-cache/sdpa_mask_test.log
```

Expected final line:

```text
PASS: padded attention example, gradients, checkpointing and scheduler-import regression
```

`NATIVE` or `UNREPAIRED_ADDITIVE` may show NaNs because those lines deliberately reproduce
the original failure. Every `FIXED` line must report finite gradients, checkpoint success,
and an unchanged input mask.

## Real Qwen SDPA regression test

The known verified diagnostic used Qwen3-0.6B at a padded width of 3584 tokens, matching a
512-token prompt plus a 3072-token response:

```bash
cd /scratch/$USER/verl-gaudi-support

VERL_HPU_FUSED_SDPA=1 \
VERL_HPU_TORCH_COMPILE=0 \
bash env/shell.sh python env/verify_qwen_sdpa.py \
  --attention sdpa \
  --length 3584 \
  --model /scratch/$USER/models/Qwen3-0.6B \
  2>&1 | tee /scratch/$USER/verl-cache/qwen_sdpa_3584_test.log
```

The final JSON must have a finite positive loss and gradient norm, with:

```json
"nonfinite_grad_parameters": []
```

## One-step Qwen3-4B-Base GRPO test

This is the required gate before a full run. It disables LR warmup so the one-step test
actually applies a nonzero learning rate.

```bash
cd /scratch/$USER/verl-gaudi-support

run_tag=$(date +%Y%m%d_%H%M%S)
grpo_log="/scratch/$USER/verl-cache/qwen3_4b_base_sdpa_test_${run_tag}.log"

set -o pipefail

MODEL_PATH="Qwen/Qwen3-4B-Base" \
VERL_HPU_FUSED_SDPA=1 \
VERL_HPU_TORCH_COMPILE=0 \
ENABLE_THINKING=False \
ROLLOUT_TEMPERATURE=0.7 \
ROLLOUT_TOP_P=0.8 \
ROLLOUT_TOP_K=20 \
TOTAL_TRAINING_STEPS=1 \
N_GPUS_PER_NODE=4 \
MAX_PROMPT_LENGTH=512 \
MAX_RESPONSE_LENGTH=1024 \
VAL_BEFORE_TRAIN=False \
TEST_FREQ=-1 \
SAVE_FREQ=-1 \
WANDB=0 \
EXPERIMENT_NAME="Qwen3-4B-Base_gsm8k_sdpa_test_${run_tag}" \
bash env/run_grpo_gsm8k.sh \
  actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
  2>&1 | tee "$grpo_log"

grpo_rc=${PIPESTATUS[0]}
printf 'exit=%s log=%s\n' "$grpo_rc" "$grpo_log"
```

Extract only the useful result lines:

```bash
grep -En \
'step:1 -|grad_norm|pg_loss|critic/rewards|critic/advantages|clip_ratio|Traceback|ERROR|Error|Exception|FloatingPointError|OutOfMemory|synStatus' \
"$grpo_log" | tail -n 100
```

Do not judge the test from policy loss alone. GRPO centers advantages within each rollout
group, so policy loss can be near zero even when gradients are valid. Require all of:

- exit code 0;
- finite, nonzero `actor/grad_norm`;
- reward variation (`critic/rewards/min` differs from `critic/rewards/max`);
- nonzero advantage range;
- finite entropy and log-probability metrics;
- response clip ratio well below 1;
- no traceback, `FloatingPointError`, OOM, or Synapse error.

## Full one-epoch run

Run only after the one-step gate passes:

The launcher defaults `SGLANG_DISABLE_OVERLAP_SCHEDULE=True` on HPU. This is required by
the current SGLang fork: its old background overlap worker can lose a lazy graph input and
crash after many otherwise healthy steps. It is independent of the model's fused-SDPA
attention setting. The Miles environment uses a newer scheduler implementation, so its
overlap result is not directly portable to this fork.

```bash
cd /scratch/$USER/verl-gaudi-support

run_tag=$(date +%Y%m%d_%H%M%S)
grpo_log="/scratch/$USER/verl-cache/qwen3_4b_base_sdpa_full_${run_tag}.log"

set -o pipefail

MODEL_PATH="Qwen/Qwen3-4B-Base" \
VERL_HPU_FUSED_SDPA=1 \
VERL_HPU_TORCH_COMPILE=0 \
ENABLE_THINKING=False \
ROLLOUT_TEMPERATURE=0.7 \
ROLLOUT_TOP_P=0.8 \
ROLLOUT_TOP_K=20 \
TOTAL_TRAINING_STEPS=null \
TOTAL_EPOCHS=1 \
N_GPUS_PER_NODE=4 \
MAX_PROMPT_LENGTH=512 \
MAX_RESPONSE_LENGTH=1024 \
VAL_BEFORE_TRAIN=False \
TEST_FREQ=-1 \
SAVE_FREQ=25 \
WANDB=1 \
EXPERIMENT_NAME="Qwen3-4B-Base_gsm8k_sdpa_full_${run_tag}" \
bash env/run_grpo_gsm8k.sh 2>&1 | tee "$grpo_log"

grpo_rc=${PIPESTATUS[0]}
printf 'exit=%s log=%s\n' "$grpo_rc" "$grpo_log"
```

Validation is intentionally disabled. GSM8K reward computation for each training batch is
still active and is required for GRPO; it is not the same as validation.

## Measured performance

Qwen3-4B-Base, GSM8K, batch 32 / mini 8 / micro 2, 2048 response tokens, 4 trainers +
4 rollout servers on 8 HL-225 cards.

### Decode graphs are the difference between a day and a month

| metric | graphless `hpu_fused` | captured `hpu_paged_v2` |
|--------|----------------------|-------------------------|
| `timing_s/gen` | 913 s | **35 s** |
| `perf/throughput` | 15 | **457** |
| `rollout_corr/ppl_ratio` | 15,000 - 126,000 | **1.0008** |
| projected 233-step epoch | ~470 h | **~7 h** |

`hpu_fused` has no working graph path: its metadata is built with a dynamic
`index_select` whose shape changes every step, `synGraphInferShapes` cannot resolve it,
and it **deadlocks in `JoinPendingLaunchThread` rather than raising**. If a run ever hangs
silently during warmup, check the attention backend first.

### Steady-state per-step cost

| phase | time |
|-------|------|
| `timing_s/gen` | ~35 s |
| `timing_s/update_actor` | ~52 s |
| total | ~103 s/step |

Training compute is now the bottleneck, not generation. The largest remaining lever is
`use_remove_padding=False`: with padding disabled every micro-batch pads to the full
2048, so most of the attention and backward cost is spent on padding. It is off because
of the HPU nested-tensor problems in commits `090c065` / `63c3f1b`.

### Health indicators from a good run

| metric | value at step 14 |
|--------|------------------|
| `actor/grad_norm` | 0.68 (down from 6.56 at step 1) |
| `rollout_corr/ppl_ratio` | 1.0008 |
| `critic/score/mean` | 0.86 (up from 0.38) |
| `response_length/clip_ratio` | 0.0 |
| `perf/cpu_memory_used_gb` | 154 |

`ppl_ratio` near 1.0 is the load-bearing one: it means rollout and training log-probs
agree, i.e. the captured decode graphs are numerically correct across weight syncs.

### Host memory does not scale with batch

| config | seqs | tokens/step | `cpu_memory_used_gb` |
|--------|------|-------------|----------------------|
| batch 8 @ 256 | 64 | 49,152 | 142.0 |
| batch 32 @ 256 | 256 | 196,608 | 168.6 |

4x the tokens cost **+19%**. Host RAM here is the near-fixed cost of 4 FSDP workers plus
4 SGLang processes, not per-token data. Linear-in-batch would have predicted 568 GB.

## Do not run training inside an interactive session job

Two runs were lost to what looked like node crashes and were not. On this cluster the
OnDemand VSCode tunnel is itself a Slurm job, and the editor, the shell and the training
run all share **one memory cgroup**. Training exhausts it, Slurm kills the job, the
session dies, and no logs survive.

```
job 62934641   ReqMem=256G   MaxRSS=268429072K (=256GiB)   OUT_OF_MEMORY
job 62958214   ReqMem=400G   script.sh "Killed"            FAILED
```

The node itself was up 54 days with 457 GB free throughout. Raising the tunnel from 256G
to 400G did not help.

Two mitigations, both in this repo:

- **`env/sbatch_grpo.sh`** -- submit training as its own Slurm job with `--mem=0` (the
  node's full ~503 GB) and its own cgroup, so an OOM costs the run and not the session:
  `sbatch -A <account> --qos <qos> env/sbatch_grpo.sh`
- **the RSS sampler in `env/run_grpo_full.sh`** -- samples per-process RSS to shared
  storage every 10 s, so even a session death leaves evidence naming what grew.

Diagnose a suspected session death with `sacct`, not `dmesg`:

```bash
sacct -u $USER --starttime=today -o JobID,JobName%20,NodeList,State,ExitCode,MaxRSS,ReqMem,Elapsed
```

## Monitoring a run

The complete log is usually too noisy to paste. Extract the latest step metrics and errors:

```bash
grep -En \
'step:[0-9]+ -|grad_norm|Traceback|ERROR|Error|Exception|FloatingPointError|OutOfMemory|synStatus' \
"$grpo_log" | tail -n 100
```

Useful live checks:

```bash
hl-smi
tail -f "$grpo_log"
```

Healthy trends:

| Metric | Healthy interpretation |
|---|---|
| `actor/grad_norm` | Finite. Zero is acceptable only for an occasional batch with no reward variation. |
| `actor/pg_loss` | Finite; it can cross or sit near zero because advantages are centered. |
| `critic/rewards/min,max` | A mix of 0 and 1 provides a GRPO learning signal. |
| `critic/advantages/min,max` | Both negative and positive values. |
| `response_length/clip_ratio` | Low; investigate if it approaches 1. |
| `rollout_corr/log_ppl_diff` | Finite and reasonably small; large growth means training/rollout divergence. |
| HPU memory | Stable across steps; reserved memory can be much higher than allocated memory. |

## Interpreting common failure patterns

### `grad_norm = NaN`

This is a numerical failure. The actor should now raise `FloatingPointError` closer to the
first invalid tensor rather than silently continue. Preserve the full traceback and the
first non-finite diagnostic.

### `grad_norm = 0`, reward = 0, advantage = 0

This is a no-op training batch, not an SDPA crash. Check:

- `ENABLE_THINKING=False` reached the composed Hydra command;
- response clip ratio is not 1;
- generated text contains `#### <number>`;
- the parquet uses `data_source=openai/gsm8k` and the correct reward schema.

If every batch has this pattern, stop the run. It cannot learn while all advantages are zero.

### Policy loss is approximately zero but gradient norm is nonzero

This can be normal for GRPO. Group-normalized advantages are centered, so the reported mean
loss can cancel while individual token gradients remain nonzero. Gradient norm, reward
variation, and advantage range are the stronger indicators.

### Every response reaches `MAX_RESPONSE_LENGTH`

Confirm thinking is disabled. Do not immediately increase the response ceiling: longer
padding makes every FSDP forward/backward more expensive and does not repair missing EOS or
incorrect chat templating.

### HPU out of memory with Qwen3-4B-Base

First reduce both training-side micro-batches to one:

```bash
PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1 \
bash env/run_grpo_gsm8k.sh
```

Keep gradient checkpointing enabled. Do not turn parameter offload on: it previously
corrupted FSDP weights on this Gaudi path.

## Expected harmless warnings

These messages were present in successful runs:

- `Calling add_step_closure function does not have any effect`;
- `Calling mark_step function does not have any effect` in eager-mode processes;
- `Failed to import deep_gemm, disable ENABLE_JIT_DEEPGEMM`;
- Qwen fast-tokenizer performance notice;
- FastAPI `ORJSONResponse` deprecation notice;
- `No need to call empty_cache on HPU`;
- `apex not installed` from GPU Migration.

They are not success indicators, but they do not require a fix by themselves.

## Verified evidence

| Test | Result |
|---|---|
| Four-token unrepaired additive mask | NaN outputs and non-finite gradients reproduced |
| Four-token fixed mask, three mask encodings | Outputs `[0, 0, 30, 35]`, `dV=[0, 0, 1.5, 0.5]`, finite checkpointed backward |
| Qwen3-0.6B, padded length 3584 | All 28 layers finite; loss `0.0307126`; grad norm `1.9140625`; no bad gradient parameters |
| Qwen3-0.6B distributed SDPA, one step, response 3072 | Exit 0; grad norm `1.22567`; reward mean `0.23828` |
| Qwen3-4B-Base distributed SDPA, one step, response 1024 | Exit 0; policy loss `-0.02729`; grad norm `8.92047`; LR `1e-6`; reward mean `0.19922`; clip ratio `0.0625` |

The verified Qwen3-4B-Base log was:

```text
/scratch/$USER/verl-cache/qwen3_4b_base_sdpa_test_20260905_125110.log
```

At the time of writing, the one-step 4B gate is verified. A complete one-epoch 4B run is not
yet recorded as complete, so continue monitoring numerical and reward metrics during the
first full run.

## Files carrying the final fixes

| File | Purpose |
|---|---|
| `env/run_grpo_gsm8k.sh` | Qwen3-4B-Base defaults, direct-answer template, sampling, topology, launch |
| `env/verify_sdpa_fix.py` | Minimal mask/checkpoint/process-gate regression |
| `env/verify_qwen_sdpa.py` | Real Qwen forward/backward regression |
| `verl_compat/verl/utils/hpu_sdpa.py` | Non-mutating empty-row repair and Habana FusedSDPA adapter |
| `verl_compat/verl/__init__.py` | Training-only SDPA routing and GQA handling |
| `verl_compat/verl/workers/actor/dp_actor.py` | Fail-fast non-finite checks |
| `verl_compat/verl/workers/rollout/sglang_rollout/async_sglang_server.py` | Explicit SGLang child-process marker |

## Related documents

- `GAUDI_FAILURE_LOG.md`: chronological failures, including discarded hypotheses.
- `GRPO_GSM8K_GAUDI_PLAN.md`: original architecture and environment investigation.
- `GAUDI_GRAPH_COMPILE_DESIGN.md`: graph-capture and compile design notes.
