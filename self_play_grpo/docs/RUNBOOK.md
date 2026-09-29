# Self-play GRPO runbook

This is the operational reference for configuring, validating, running, and
recovering the four-player Quoridor project. Run commands from the repository
root:

```text
/scratch/svijay46/verl-gaudi-support
```

The implementation history and current evidence are recorded separately in
[`IMPLEMENTATION_AND_TEST_LOG.md`](IMPLEMENTATION_AND_TEST_LOG.md).
The required final distributed target is specified in
[`EIGHT_HPU_TRAINING_DESIGN.md`](EIGHT_HPU_TRAINING_DESIGN.md).

## Safety model

- Use `bash env/shell.sh ...` for every Python command so the Gaudi container
  and `.runtime/venv` are active.
- Do not create another virtual environment for this project.
- Install the project with `--no-deps`; do not let pip replace the Gaudi Torch
  stack.
- Model loading is local-only by default. The CLI must not silently download a
  checkpoint.
- `validate-*` commands never update weights.
- `train` refuses to run unless `--acknowledge-training` is supplied.
- Do not start training until engine, model asset, model load, constrained
  forward, full-match replay, and backward/checkpoint gates have passed.
- A match is the statistical group. Four player views from one match are not
  four independent samples.

## Standard paths and pins

| Item | Standard value |
|---|---|
| Project | `self_play_grpo/` |
| Runtime | repository `.runtime/venv`, entered through `env/shell.sh` |
| Main config | `self_play_grpo/configs/quoridor_outcome.yaml` |
| Fast fixture config | `self_play_grpo/configs/quoridor_fixture.yaml` |
| OpenSpiel source cache | `.runtime/open_spiel-2.0.2-d0606878` |
| OpenSpiel revision contract | `d0606878b957274cc67a918ed173b36e9fe0fed6` |
| Model ID | `Qwen/Qwen3-4B` |
| Model revision | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Model path | `/scratch/svijay46/models/Qwen3-4B` |
| Bot fixture artifact | `self_play_grpo/artifacts/fixture.jsonl` |
| Current player-relative pilot | `self_play_grpo/artifacts/pilot-outcome-relative-shape-v1-qwen3-4b-seed-11/` |
| Archived absolute-coordinate pilot | `self_play_grpo/artifacts/pilot-outcome-qwen3-4b-seed-11/` (never resume) |
| Final training hardware | One node, 8 Intel Gaudi HPUs split into 4 rollout generators + 4 trainers (not yet complete) |

## Standard configurations and why

| Setting | Main outcome config | Fixture config | Reason |
|---|---:|---:|---|
| Board | 9x9 | 5x5 | 9x9 is the intended study; 5x5 shortens deterministic integration checks |
| Walls per player | 5 | 2 | Explicitly avoids OpenSpiel defaults and scales the small fixture |
| Action horizon | 120 | 100 | Main cap is 30 full four-seat rounds; 100 aligns with the small engine fixture |
| Games per update | 16 | 1 | Main batches reduce variance while preserving match groups; fixture reproduces the exact one-match loop |
| Loss normalizer | 120 | 100 | Must equal the configured action horizon to keep a fixed per-match loss scale |
| Model, LoRA, sampling | Same | Same | Validation must exercise the exact actor contract intended for the main run |
| Reward mode | `outcome` | `outcome` | Outcome-only is the required reference before process-credit experiments |
| Process extension | Disabled | Disabled | Process variants must be compared only after the baseline is sound |

Use the fixture config for engine validation, one-action model validation, and
the first complete-match smoke test. Use the main config only after those gates
pass and resource use is measured.

## Configuration parameter reference

### Top level

| Key | Standard | Meaning and constraint |
|---|---|---|
| `project` | `quoridor_self_play_grpo` | Human-readable run identity stored with configuration and checkpoints |
| `seed` | `11` | Base seed for policy sampling and update batches; OpenSpiel's initial state itself is deterministic |

### `environment`

| Key | Standard | Meaning and constraint |
|---|---|---|
| `game` | `quoridor` | Only supported game; changing it is rejected |
| `engine_revision` | `d060...fed6` | Behavioral source contract; changing it requires adapter and engine regression revalidation |
| `players` | `4` | Required experiment group size and seat count |
| `board_size` | `9` main, `5` fixture | Quoridor square-board width/height; adapter accepts 3 through 25 |
| `wall_count` | `5` main, `2` fixture | Initial walls per player; must be non-negative and is always explicit |
| `max_joint_actions` | `120` main, `100` fixture | Total public-action cap. Reaching it yields a uniform experimental draw |
| `horizon_result` | `uniform_draw` | Only supported cap treatment: `(0.25, 0.25, 0.25, 0.25)` |
| `action_perspective` | `player_relative` | Rotates each actor's board and legal labels so it is player 0, starts at the bottom, and aims at the top; `absolute` exists only for old-artifact replay and diagnostics |

The canonical seat-to-engine mapping is fixed at `(0, 2, 1, 3)`. Do not alter
or infer it from list position without rerunning all winning-seat fixtures.
Player-relative presentation does not change that mapping, canonical logged
states, rewards, or engine IDs. It only rotates language-facing coordinates and
renumbers players relative to the actor. All four initial forward pawn moves
therefore become `MOVE_E8`.

### `model`

| Key | Standard | Meaning and constraint |
|---|---|---|
| `id` | `Qwen/Qwen3-4B` | Remote provenance identity recorded in manifests |
| `revision` | `1cfa...df60c` | Immutable Hugging Face commit expected in local download metadata |
| `local_path` | `/scratch/svijay46/models/Qwen3-4B` | Absolute checkpoint directory used for actual loading |
| `local_files_only` | `true` | Prevents an implicit network download during validation or training |
| `enable_thinking` | `false` | Uses Qwen's non-thinking chat mode so only the action label is generated and trained |
| `device` | `hpu` | Target device for the base model and adapter |
| `dtype` | `bfloat16` | Frozen base-model load dtype appropriate to the Gaudi baseline |
| `lora_rank` | `16` | Adapter bottleneck rank; higher values increase trainable parameters and memory |
| `lora_alpha` | `32` | LoRA scaling numerator; standard effective scale is alpha/rank |
| `dropout` | `0.0` | Required for behavior-probability replay and the initial correctness baseline |

Changing the model requires a new immutable revision, a new local directory,
`validate-model-assets`, `validate-model-load`, and probability replay. Do not
reuse results from a different tokenizer or checkpoint.

### `rollout`

| Key | Standard | Meaning and constraint |
|---|---|---|
| `games_per_update` | `16` main, `1` fixture | Complete matches collected under one frozen policy version before an update |
| `parallel_games_per_rank` | `2` | Two active games per rollout process/HPU; each of four rollout ranks processes two such batches for the standard global 16-game update |
| `shared_weights_across_seats` | `true` | All four seats must use one shared actor; `false` is rejected |
| `max_new_tokens` | `16` | Hard cap for one action label including newline; current longest label uses 6 tokens |
| `temperature` | `1.0` | Baseline categorical distribution; other temperatures are rejected |
| `top_p` | `1.0` | No nucleus truncation, simplifying exact probability accounting |
| `top_k` | `0` | No top-k truncation |
| `constrain_to_legal_actions` | `true` | Required token-trie grammar; illegal-action repair/substitution is forbidden |

At each generated prefix, probabilities are renormalized over only legal next
tokens. The same recorded allowed-token sets are used during training replay.
Incremental `use_cache=True` decoding is the authoritative probability path for
sampling, pre-update replay, actor loss, and optional reference-policy replay.
Sequential samples replay as one row. Batched samples additionally record and
reconstruct their batch size, target row, right-padded prompt width, position
IDs, attention shape, and pad token; changing that tensor shape is not
numerically equivalent on the pinned HPU stack.
The full-sequence forward is diagnostic only: on the pinned eager HPU BF16
stack it differed from cached decoding by as much as `0.157623` even though
cached replay reproduced all 416 sampled token values exactly.

### `training`

| Key | Standard | Meaning and constraint |
|---|---|---|
| `reward_mode` | `outcome` | `outcome`, `potential`, or `gae_blend`; start with `outcome` |
| `group_key` | `game_id` | Keeps the four dependent trajectories grouped by match; changing it is rejected |
| `group_size` | `4` | Exactly four canonical seats; changing it is rejected |
| `learning_rate` | `1e-5` | AdamW learning rate for trainable LoRA parameters |
| `weight_decay` | `0.0` | Disables parameter decay; the trainer also skips an optimizer step when the complete gradient norm is zero |
| `clip_epsilon` | `0.2` | PPO/GRPO-style importance-ratio clip width |
| `optimizer_epochs_per_batch` | `1` | Fresh on-policy baseline permits one optimizer epoch only |
| `max_grad_norm` | `1.0` | Global trainable-parameter gradient clipping threshold |
| `kl_beta` | `0.0` | Weight on sampled KL to an explicitly supplied frozen reference model; nonzero requires that model |
| `loss_normalizer_per_game` | horizon | Fixed denominator per match, not realized trajectory length; must equal `max_joint_actions` |

For a decisive result `[1,0,0,0]`, population-standardized advantages are
`[sqrt(3), -1/sqrt(3), -1/sqrt(3), -1/sqrt(3)]`. A uniform draw has zero policy
advantage. Only generated action tokens with `loss_mask=1` contribute.
The trainer backpropagates one action at a time and accumulates parameter
gradients. Each action uses the complete batch's fixed denominator, making the
result algebraically identical to one summed loss while retaining only one
action's autograd graph.

### `process_extension`

| Key | Standard | Meaning and constraint |
|---|---|---|
| `enabled` | `false` | Provenance switch for process experiments; baseline remains disabled |
| `proxy_temperature` | `2.0` | Softmax temperature in path-distance units for relative progress scores |
| `shaping_alpha` | `1.0` | Scale of signed potential differences for `reward_mode: potential` |
| `gamma` | `1.0` | Undiscounted episodic return; any other value is rejected |
| `gae_lambda` | `0.95` | Own-decision GAE trace parameter in `[0,1]` |
| `process_blend_eta` | `0.25` | Blend weight for normalized GAE versus terminal outcome advantage |

The path proxy respects walls but intentionally ignores pawn occupancy, jumps,
future walls, and opponent policy. It is a progress feature, not a calibrated
win probability. Potential returns retain signed changes and a zero terminal
potential so move-away/move-back loops cancel.

### `evaluation`

| Key | Standard | Meaning and constraint |
|---|---|---|
| `fixed_opponent_manifest` | `null` initially | Must point to a frozen opponent manifest before a comparative study |
| `rotate_candidate_through_all_seats` | `true` | Required to expose seat effects |
| `training_seeds` | `[11,22,33]` | Planned independent training seeds; not evidence until all runs complete |

The current CLI tournament always rotates through all four seats. Bootstrap
intervals resample whole matches and default to 10,000 bootstrap samples.

## Other programmatic parameters

These are Python API parameters, not currently exposed in YAML.

| Location | Parameter | Default | Meaning |
|---|---|---:|---|
| `WallAwarePolicy` | `opponent_weight` | `0.25` | Weight on the summed increase in opponents' shortest paths |
| `WallAwarePolicy` | `wall_cost` | `0.05` | Small penalty discouraging gratuitous wall placement |
| `MatchCollector` | `collect_progress` | `true` | Record distance, proxy, potential, and wall features around every turn |
| `MatchCollector` | `proxy_temperature` | `2.0` | Passed to progress feature calculation |
| `SynchronousTrainer.update` | `replay_tolerance` | `2e-4` | Maximum old/new cached constrained log-probability error before an update; do not raise this to accommodate a different computation path |
| `compute_training_loss` | `selected_seats` | all | Optional one-perspective ablation filter |
| `compute_training_loss` | `selected_seat_importance` | `1.0` | Importance scaling for a selected-seat ablation |
| `bootstrap_match_mean` | `samples` | `10,000` | Number of whole-match bootstrap resamples |
| `build_value_network` | `hidden_dims` | `(256,128)` | Optional evaluator MLP widths |
| `fit_evaluator` | `epochs` | `10` | Evaluator epochs; not actor training |
| `fit_evaluator` | `batch_size` | `256` | Evaluator decision-state minibatch size |
| `fit_evaluator` | `learning_rate` | `1e-3` | Evaluator AdamW learning rate |
| `split_complete_matches` | `validation_fraction` | `0.2` | Whole-match evaluator validation split |

## Installation and model acquisition

### Project

```bash
bash env/shell.sh python -m pip install --no-deps -e ./self_play_grpo
```

### OpenSpiel

```bash
bash self_play_grpo/scripts/install_open_spiel.sh
```

Relevant environment variables:

| Variable | Default | Use |
|---|---:|---|
| `OPEN_SPIEL_BUILD_JOBS` | `8` | Cap native compiler parallelism if node policy or memory requires fewer jobs |
| `CC` | `/usr/bin/gcc` | Set internally by the installer wrapper |
| `CXX` | `/usr/bin/g++` | Set internally by the installer wrapper |
| `SELF_PLAY_OPEN_SPIEL_INNER` | unset | Internal recursion guard; users should not set it |

The installer caches the archive and source below `.runtime`, verifies the
archive on every run, and reuses a patch completion stamp.

### Container/runtime environment

`env/gaudi_env.sh` owns the accelerator and cache defaults inherited by
`env/shell.sh`. They are programmable environment inputs, but they are not
experiment hyperparameters and should remain fixed across compared runs.

| Variable | Current default | Meaning |
|---|---|---|
| `VENV_DIR` | `$REPO_ROOT/.runtime/venv` | Existing shared Python environment used inside the container |
| `PYTHONNOUSERSITE` | `1` | Prevents host user-site CUDA packages from shadowing container Gaudi packages |
| `VERL_PLATFORM` | `hpu` | Selects the repository's HPU platform path |
| `PT_HPU_GPU_MIGRATION` | `1` | Enables Habana CUDA-to-HPU API migration support |
| `PT_HPU_LAZY_MODE` | `0` | Uses eager execution; explains lazy-only step-function warnings |
| `PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES` | `0` | Keeps the currently verified dynamic-shape behavior |
| `PT_HPU_RECIPE_CACHE_CONFIG` | `$CACHE_ROOT/habana_recipe,false,20480` | HPU compiled-recipe cache location and limits |
| `HF_HOME` | `$SCRATCH_ROOT/hf_cache` | Hugging Face cache root; separate from the explicit model local path |
| `HF_HUB_CACHE` | `$HF_HOME/hub` | Hub cache directory |
| `TORCH_EXTENSIONS_DIR` | `$CACHE_ROOT/torch/extensions` | Native Torch extension build cache |
| `UV_CACHE_DIR` | `$CACHE_ROOT/uv` | uv package cache |
| `PIP_CACHE_DIR` | `$CACHE_ROOT/pip` | pip package/build cache |

Record any override in the run manifest. In particular, changing HPU eager/lazy
mode can alter compilation and numerical replay behavior, so probability gates
must be rerun after such a change.

### Model

```bash
bash env/shell.sh hf download Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --local-dir /scratch/svijay46/models/Qwen3-4B
```

`HF_TOKEN` is optional for this public model and only affects Hub rate limits.
Do not run `hf update` merely because the CLI suggests it; the current Hub
version is part of the verified environment.

## Validation sequence

Run one command at a time and retain its complete output.

### 1. Deterministic suite

```bash
bash env/shell.sh python -m pytest \
  self_play_grpo/tests -m 'not model' -q
```

The latest fully reported aggregate passed: 57 tests with 3 expected Gaudi
warnings in 9.37 seconds and no skips. The subsequently added distributed
distributed runtime and rollout modules have 17 additional accelerator-free
cases that pass separately.
Any skip after OpenSpiel installation means the engine is not visible in the
active environment.

### 2. Engine contract

```bash
bash env/shell.sh python -m self_play_grpo.cli \
  validate-engine \
  --config self_play_grpo/configs/quoridor_fixture.yaml
```

Require `status: ok`, seat order `[0,2,1,3]`, and four correctly owned natural
wins.

### 3. Cheap bot tournament

```bash
bash env/shell.sh python -m self_play_grpo.cli \
  bot-tournament \
  --config self_play_grpo/configs/quoridor_fixture.yaml \
  --candidate shortest \
  --opponents random shortest wall-aware \
  --games-per-seat 2 \
  --seed 11 \
  --output self_play_grpo/artifacts/fixture.jsonl
```

This writes one summary line followed by one line per evaluation game. It is a
small orchestration fixture, not a strength benchmark and not a full turn-level
replay log.

### 4. Model assets and tokenizer only

```bash
bash env/shell.sh python -m self_play_grpo.cli \
  validate-model-assets \
  --config self_play_grpo/configs/quoridor_outcome.yaml
```

Require the configured revision, all three shards, 209 labels, maximum action
length no greater than `max_new_tokens`, and `status: ok`.

### 5. Model and LoRA load only

```bash
time bash env/shell.sh python -m self_play_grpo.cli \
  validate-model-load \
  --config self_play_grpo/configs/quoridor_outcome.yaml
```

This allocates the model on HPU but performs no forward pass. Require only
`hpu:0`, the expected model/revision, nonzero trainable parameters, and
`status: ok`.

### 6. One constrained action and probability replay

```bash
time bash env/shell.sh python -m self_play_grpo.cli \
  validate-policy-forward \
  --config self_play_grpo/configs/quoridor_fixture.yaml
```

This loads the model and performs forward passes but no backward pass or
optimizer step. Check:

- the chosen label is legal and one cloned environment step succeeds;
- at least one generated position is genuinely stochastic;
- forced positions have a single allowed token;
- replayed log probabilities closely match recorded behavior values;
- probability ratios are close to one.

The verified fixture action produced zero log-probability error and ratios
exactly equal to one. Require a complete match to remain below the trainer's
`2e-4` threshold before an optimizer step.

The player-relative recheck also passed: legal `MOVE_B5`, exact cached replay,
four ratios equal to `1.0`, and 53.315 seconds wall time. This establishes the
single-row reference for the batched-forward comparison below.

After the single-row gate passes with `action_perspective: player_relative`,
probe two simultaneous environments on one HPU:

```bash
time bash env/shell.sh python -m self_play_grpo.cli \
  validate-policy-batched-forward \
  --config self_play_grpo/configs/quoridor_fixture.yaml \
  --batch-size 2 \
  --replay-tolerance 2e-4
```

This is a validation-only forward pass. It pads different prompt lengths,
maintains independent RNG streams and legal-action tries, and reports the
maximum difference between batched behavior probabilities and both recorded
batch-shape replay and diagnostic single-row KV-cache replay. `status` is gated
on the recorded batch-shape path. Single-row error is reported diagnostically
because later variable prompt shapes are not equivalent on this HPU stack.

The first batch-size-2 attempt selected `WALL_C1H` and `MOVE_B5` but reported a
non-finite replay result after 34.832 seconds. Production batching remained
disabled. That run also exposed a fail-open IEEE NaN comparison in the gate;
the validator now treats any non-finite value as a hard failure and reports
per-row/token diagnostics. Initial prompt padding was changed from left to
right before repeating this exact gate.

The corrected right-padded batch-size-2 gate passed in 33.434 seconds. Prompt
lengths were 795 and 805 tokens; both rows selected legal `MOVE_B5`, all
behavior and replay log probabilities were finite, and maximum single-row
replay error was exactly `0.0`. This validates two-row forward equivalence but
does not yet enable production collection; larger batch sizes and the
multi-turn collector still require gates.

After recorded batch-shape contract v1 was added, the two-row gate passed again
in 33.577 seconds. Sampling took 0.6923 seconds, both recorded-shape and
single-row errors were `0.0`, all values were finite, and peak allocation was
13,153,430,016 bytes. This validates the metadata and replay dispatch at the
initial states; the complete-game retry below remains the decisive
variable-prompt test.

The batch-size-4 gate also passed, in 34.126 seconds. It covered four different
prompt lengths (795, 805, 819, and 833 tokens), selected legal `MOVE_B5` in
every row, and reproduced every behavior log probability exactly under
single-row replay. Wall time remained nearly flat relative to batch size 2.
Subsequent gates also record sampling-only time plus baseline and peak HPU
allocation so concurrency is chosen from measured throughput and memory.

The batch-size-8 gate passed in 37.177 seconds total. The actual batched sample
took 1.6837 seconds; the remainder includes model startup and eight independent
verification replays. Baseline HPU allocation was 8,183,381,632 bytes and peak
allocation was 28,296,229,120 bytes, an increase of 20,112,847,488 bytes. All
eight rows were finite, selected legal `MOVE_B5`, and replayed with error
`0.0`. This is sufficient headroom to test batch size 16, but is not yet a
complete multi-turn collector measurement.

The batch-size-16 gate passed in 41.370 seconds total. Sampling took 3.1428
seconds; baseline allocation was 8,182,873,728 bytes and peak allocation was
46,906,020,992 bytes, an increase of 38,723,147,264 bytes. All 16 rows were
finite and legal, and maximum replay error was `0.0`. Batch size 16 is therefore
the selected one-HPU ceiling for the standard 16-game collection batch. Do not
probe 32 until a workload actually requires more than 16 games per rank.

### 7. Complete model match, persistence, environment replay, and probability replay

```bash
time bash env/shell.sh python -m self_play_grpo.cli \
  collect-policy-match \
  --config self_play_grpo/configs/quoridor_fixture.yaml \
  --output self_play_grpo/artifacts/policy-match-gate-seed-11.jsonl \
  --seed 11 \
  --replay-tolerance 2e-4
```

This is a potentially long model-inference gate, but it performs no backward
pass or optimizer step. It:

- collects one complete capped fixture match with one frozen shared policy;
- records exact prompts, token IDs, masks, allowed sets, and behavior values;
- refuses to overwrite an existing output path;
- writes and reads one full `MatchRecord` exactly;
- replays every label and requires state equality before and after every action;
- requires identical terminal results, reason, and final environment bytes;
- replays every action probability and enforces `--replay-tolerance`.

If probability replay fails, the artifact is retained for diagnosis. Require
`status: ok` before considering any backward-pass gate.

The first complete fixture match failed under the former full-sequence replay
implementation with maximum error `0.157623`. Do not increase the tolerance.
Diagnosis of the preserved artifact showed zero cached replay error across all
416 tokens and isolated the mismatch to full-sequence execution. To validate
the corrected authoritative path without recollecting the match, run:

```bash
time bash env/shell.sh python -m self_play_grpo.cli \
  validate-policy-artifact \
  --config self_play_grpo/configs/quoridor_fixture.yaml \
  --input self_play_grpo/artifacts/policy-match-gate-seed-11.jsonl \
  --replay-tolerance 2e-4
```

This command is read-only apart from normal runtime caches. It repeats exact
environment replay and checks every saved token with the same incremental
KV-cache computation used during collection. It does not collect, backpropagate,
construct an optimizer, or update weights. Require `status: ok` and an error no
greater than `2e-4`.

For backend comparison only, the diagnostic command remains:

```bash
time bash env/shell.sh python -m self_play_grpo.cli \
  diagnose-policy-replay \
  --config self_play_grpo/configs/quoridor_fixture.yaml \
  --input self_play_grpo/artifacts/policy-match-gate-seed-11.jsonl \
  --top-k 10
```

This read-only command loads the model and compares recorded behavior values,
the original KV-cache computation path, and full-sequence replay for all saved
tokens. It does not alter the artifact or update weights.

### 8. Resumable frozen-policy pilot

The next HPU gate is the first complete, multi-turn batched collection. It uses
a new directory because player-relative actions and batched collection must not
be mixed into the old absolute-coordinate pilot:

```bash
time bash env/shell.sh python -m self_play_grpo.cli \
  collect-policy-pilot \
  --config self_play_grpo/configs/quoridor_fixture.yaml \
  --output self_play_grpo/artifacts/pilot-batched-shape-v1-relative-fixture-qwen3-4b-seed-11 \
  --games 2 \
  --max-new-games 2 \
  --parallel-games 2 \
  --seed 11 \
  --replay-tolerance 2e-4
```

Require two completed games, `parallel_games_per_rank: 2`, one policy version,
zero illegal substitutions, finite probabilities, and maximum replay error no
greater than `2e-4`. This performs model inference and artifact writes, but no
backward pass or optimizer step. Do not start the fresh 100-game main pilot
until this gate passes.

The pre-shape-contract attempt failed safely after 84.599 seconds. The first game completed
with a natural seat-2 win in 23 turns, but its batched behavior probabilities
differed from single-row cached replay by `1.93216`; its JSONL was preserved
but not registered in the manifest. No optimizer was constructed or stepped.
Do not rerun collection into this directory until the exact failing token and
batch-shape replay contract were diagnosed. Contract version 1 now records the
missing tensor dimensions; the old directory remains diagnostic-only and must
not be resumed. The command above uses a new directory for its pending retry.

The shape-contract-v1 retry passed in 91.568 seconds. Seeds 11 and 12 completed
concurrently in 23 and 31 turns, respectively; both were natural seat-2 wins.
The manifest contains 2 games, 54 turns, 244 owned tokens, one frozen policy
version, zero substitutions, and maximum replay error `0.0`. Environment
round-trip and batch-shaped probability replay both passed before registration.

Differentiable replay was then checked at joint step 14 of game 0, the same
state family that had produced the worst pre-contract mismatch. It passed in
33.671 seconds with replay error `0.0`, mean ratio `1.0`, finite gradient norm
`3.572176`, gradients on 504 trainable tensors, cleared gradients, and zero
optimizer steps.

The complete 23-action game then passed batch-shaped, action-by-action backward
in 64.675 seconds. All 104 owned tokens replayed exactly; gradient norm was
`10.600649`, gradients reached 504 trainable tensors and were cleared, and no
optimizer step occurred. HPU allocation rose from 8,182,824,576 to
45,171,202,944 bytes, remaining bounded while processing the full match.

The populated-optimizer checkpoint gate then passed in 38.919 seconds. After
one synthetic update, 252 parameter tensors changed and the checkpoint held
504 Adam entries. Reload reproduced all parameters, 1,512 optimizer tensors,
and continuation metrics exactly; the checkpoint was 396,882,682 bytes. The
resumed-step replay error `0.233078` reflects deliberate policy movement from
the frozen behavior policy, while the pre-update replay error was `0.0`.

Historical sequential smoke (already completed; do not reuse this output for
the player-relative batched gate):

```bash
time bash env/shell.sh python -m self_play_grpo.cli \
  collect-policy-pilot \
  --config self_play_grpo/configs/quoridor_fixture.yaml \
  --output self_play_grpo/artifacts/pilot-smoke-qwen3-4b-seed-11 \
  --games 1 \
  --max-new-games 1 \
  --seed 11 \
  --replay-tolerance 2e-4
```

This smoke passed in 3 minutes 16.314 seconds: one 100-turn engine draw, 416
owned tokens, one policy version, zero substitutions, exact replay error
`0.0`, and `status: complete`. The artifact is retained at the output path.

The original absolute-coordinate 9x9 M1 pilot was started with the following
historical command. **Do not run it again:** the configuration is now
player-relative, so the old manifest intentionally fails the immutable-config
check.

```bash
time bash env/shell.sh python -m self_play_grpo.cli \
  collect-policy-pilot \
  --config self_play_grpo/configs/quoridor_outcome.yaml \
  --output self_play_grpo/artifacts/pilot-outcome-qwen3-4b-seed-11 \
  --games 100 \
  --max-new-games 1 \
  --seed 11 \
  --replay-tolerance 2e-4
```

This command performs no backward pass, constructs no optimizer, and never
updates the policy. On the first invocation it loads the model once, atomically
saves the exact initial PEFT adapter, hashes it, and creates the manifest. Each
completed match is written to its own JSONL file, replayed through the engine,
replayed through the authoritative KV-cache probability path, and only then
registered atomically in `manifest.json`.

The first bounded main invocation passed in 1 minute 55.029 seconds. Seed 11
ended in a natural seat-0 win after 45 turns and 180 owned tokens, with replay
error `0.0`, zero substitutions, and one policy version. The manifest now has
1 of 100 games and correctly reports `status: incomplete`.

The next identical invocation passed the cross-process adapter-resume gate in
2 minutes 11.382 seconds. It retained the exact adapter digest and policy
version, then committed index 1/seed 12: a 57-turn natural seat-0 win with 246
owned tokens and replay error `0.0`. The manifest now contains 2 of 100 games,
102 turns, 426 tokens, zero draws, and zero substitutions.

The next four-game same-process chunk passed in 7 minutes 24.510 seconds. The
manifest now contains 6 of 100 games, 364 turns, 1,544 owned tokens, six
natural wins, zero draws, zero substitutions, and replay error `0.0`. Winner
counts are seat 0: 3, seat 1: 0, seat 2: 3, seat 3: 0; this is descriptive only
at such a small sample size.

One further bounded game validated live migration of the existing manifest to
the additive summary fields. Index 6/seed 17 was a 35-turn natural seat-2 win
with 146 owned tokens and replay error `0.0`, completing in 1 minute 46.404
seconds. The manifest is now 7/100 with 399 turns, mean length `57.0`, 1,690
tokens, and wins `[3,0,4,0]` by canonical seat.

The following eight-game chunk completed in 12 minutes 30.631 seconds. The
pilot is now 15/100 with 832 turns, mean length `55.46666666666667`, 3,534
owned tokens, 15 natural wins, zero draws, exact replay, zero substitutions,
and wins `[4,0,8,3]` by canonical seat.

The next eight-game chunk completed in 15 minutes 36.568 seconds. The pilot is
now 23/100 with 1,322 turns, mean length `57.47826086956522`, 5,614 owned
tokens, 22 natural wins, one horizon draw, exact replay, zero substitutions,
and wins `[5,0,11,6]`. The absence of a seat-1 win is being monitored; the
pilot is not yet large enough to make a seat-effect claim.

The third eight-game chunk completed in 11 minutes 31.510 seconds. The pilot
is now 31/100 with 1,715 turns, mean length `55.32258064516129`, 7,256 owned
tokens, 30 natural wins, one draw, exact replay, zero substitutions, and wins
`[5,0,18,7]`. Seat 1's zero-win count must be reported and investigated with
seat-rotated evaluation; engine fixtures already prove that seat can win.

The fourth eight-game chunk completed in 12 minutes 0.598 seconds. The pilot
is now 39/100 with 2,096 turns, mean length `53.743589743589745`, 8,866 owned
tokens, 38 natural wins, one horizon draw, exact replay, zero substitutions,
and wins `[6,0,22,10]`. The adapter digest and policy version remain unchanged.
Seat 1 still has no natural win, so the seat-rotated evaluation remains a
required diagnostic rather than treating this descriptive imbalance as a
causal result.

The fifth eight-game chunk completed in 10 minutes 39.932 seconds. The pilot
is now 47/100 with 2,455 turns, mean length `52.234042553191486`, 10,380 owned
tokens, 46 natural wins, one horizon draw, exact replay, zero substitutions,
and wins `[7,0,28,11]`. The adapter digest and policy version remain unchanged.
Seat 1 still has no natural win and remains a required seat-rotated-evaluation
diagnostic.

The sixth eight-game chunk completed in 10 minutes 7.717 seconds. The pilot is
now 55/100 with 2,788 turns, mean length `50.69090909090909`, 11,766 owned
tokens, 54 natural wins, one horizon draw, exact replay, zero substitutions,
and wins `[7,0,35,12]`. The adapter digest and policy version remain unchanged.
Seat 1 still has no natural win and remains a required seat-rotated-evaluation
diagnostic.

The seventh eight-game chunk completed in 11 minutes 35.401 seconds. The pilot
is now 63/100 with 3,181 turns, mean length `50.492063492063494`, 13,412 owned
tokens, 62 natural wins, one horizon draw, exact replay, zero substitutions,
and wins `[7,0,42,13]`. The adapter digest and policy version remain unchanged.

**This pilot is permanently paused; do not run another collection chunk.** Aggregate
movement analysis found that seat 1 made only 37 goal-forward moves but 601
lateral moves. It selected `MOVE_A4` on 61 of 63 opening decisions even though
its right-edge goal requires `MOVE_B5`. The artifacts show the correct seat
mapping and goal text. Player-relative actions were subsequently implemented,
which is an intentional semantic change. The 63 games remain diagnostic data,
but the directory must never be resumed under the new configuration.

The directory layout is:

```text
pilot-outcome-qwen3-4b-seed-11/
  manifest.json
  policy_adapter/
  matches/
    game-000000.jsonl
    game-000001.jsonl
    ...
```

If interruption happens during collection, no partial game replaces a
completed artifact. If it happens after the game file is committed but before
the manifest update, the next invocation validates and registers that file
instead of overwriting or recollecting it. Existing registered files are
hash-checked and environment-replayed on every resume. The adapter, complete
configuration, base seed, replay tolerance, policy version, and declared
target are immutable; the target may grow but cannot shrink.

The manifest summary reports cumulative games, turns, owned tokens, draw and
termination counts, maximum replay error, mean game length, fractional-result
sums by seat, and wins by seat. The last three fields were added compatibly;
an older manifest is accepted and upgraded on its next successful commit.

| Option | Default | Meaning |
|---|---|---|
| `--config` | required | Exact frozen experiment configuration; must match on resume |
| `--output` | required | New pilot directory or existing directory with a valid manifest |
| `--games` | required | Total desired games, not games to add during this invocation |
| `--seed` | config `seed` | Base seed; game index `i` uses `base_seed + i` |
| `--max-new-games` | unlimited | Maximum games collected or recovered in this invocation |
| `--parallel-games` | config `rollout.parallel_games_per_rank` | Maximum simultaneously active games in one model batch; the last batch is automatically smaller |
| `--replay-tolerance` | `2e-4` | Per-game cached probability replay limit; pinned in the manifest |

The final M1 evidence is `completed_games: 100`, exactly one policy version,
zero illegal-action substitutions, all artifact hashes intact, and maximum
probability error no greater than `2e-4`. Draw rate and termination counts in
the manifest determine whether the capped 9x9 configuration produces enough
natural outcomes for an outcome-learning pilot. The original main pilot is
archived at 63/100 because its severe absolute-label bias led to the
player-relative semantic fix. The two-game batched fixture gate passed and the
fresh main pilot below uses a new directory; never mix its records with the old
absolute-coordinate manifest.

The fresh player-relative main pilot began at
`artifacts/pilot-outcome-relative-shape-v1-qwen3-4b-seed-11`. Its first two
concurrent 9x9 games passed in 258.812 seconds: lengths 46 and 62, 108 total
turns, 440 owned tokens, two natural seat-1 wins, exact replay, zero
substitutions, and one policy version. Model-free analysis found positive net
goal progress for every seat. Seat 1 made 16 forward and zero backward moves;
seats 0–2 all opened with the normalized relative label `MOVE_D9`. This is
encouraging evidence that the old coordinate bias is removed, but two games
are not enough for an outcome or seat-balance conclusion.

The next two-game invocation validated cross-process resume in 249.488 seconds.
It retained adapter `47ff3ae32a43...` and the identical policy version, then
appended seeds 13 and 14 as 57- and 41-turn natural seat-0 wins. At four games,
the pilot has 206 turns, 844 owned tokens, exact replay, zero substitutions,
and wins `[2,2,0,0]`. Model-free analysis still found positive net progress for
all seats (`[25,20,17,18]`); four games remain descriptive, not statistical.

The next bounded invocation added four games in 772.826 seconds using two
concurrent games per batch. Seeds 15–18 ended naturally after 84, 47, 56, and
55 turns. The cumulative checkpoint is 8/100 games, 448 turns, 1,840 owned
tokens, exact replay, zero substitutions, one policy version, and wins
`[2,2,2,2]`. Every seat also retains positive net goal progress. This is a
strong screening milestone, but not enough data for a learning or strength
claim. Keep the directory resumable; the next engineering gate is distributed
runtime validation rather than spending another long single-HPU chunk now.

### 9. Four-HPU distributed runtime and HCCL smoke

This gate passed in 30.509 seconds on four Gaudi HPUs. Reproduce it with:

```bash
time bash env/shell.sh python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  --module self_play_grpo.cli \
  validate-distributed-runtime \
  --expected-world-size 4 \
  --timeout-seconds 120
```

This command loads no model, performs no rollout, and updates no weights. It
must print one rank-0 JSON object with `status: ok`, `world_size: 4`, logical
devices `hpu:0` through `hpu:3`, membership vectors `[1.0,1.0,1.0,1.0]`,
module IDs `['0','1','2','3']`, and reduction values `[10.0,10.0]`. With GPU
migration enabled, the reported compatibility backend name is `nccl`; Habana's
initializer, `HLS_MODULE_ID`, HPU tensors, and successful collectives establish
the actual Gaudi path. Missing `torchrun` variables, an unexpected
world size, duplicate/incorrect device mapping, a failed reduction, or a
timeout are hard failures. Do not proceed to distributed LoRA math after a
failed or partially initialized result.

### 10. Four-HPU rollout gameplay smoke

This is the next standard command for the rollout half of the final 4+4 node.
Allocate four HPUs. It performs 16 complete 9x9 games: each of four ranks owns
four games and processes them as two successive batches of two. It performs no
backward or optimizer activity:

```bash
time bash env/shell.sh python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  --module self_play_grpo.cli \
  collect-distributed-policy-pilot \
  --config self_play_grpo/configs/quoridor_outcome.yaml \
  --output self_play_grpo/artifacts/distributed-gameplay-4rollout-hpu-qwen3-4b-seed-11 \
  --expected-world-size 4 \
  --games-per-rank 4 \
  --parallel-games 2 \
  --seed 11 \
  --replay-tolerance 2e-4 \
  --timeout-seconds 1800
```

The command first executes the four-rank HCCL device collective contract. The
pinned Habana bridge binds processes through
`initialize_distributed_hpu(local_rank=...)`, which sets `HLS_MODULE_ID` to
the physical module selected by `LOCAL_RANK` (or its `HABANA_VISIBLE_MODULES`
mapping). Each bound process then exposes that module as logical device zero,
so `current_device()` is not used as a physical identity check. Tensor
allocations use unindexed `hpu` because strings such as `hpu:1` are
unsupported. Rank 0
then creates the only initial adapter; its 504 trainable tensors are broadcast
to all rollout ranks. Global game indices are contiguous: rank 0 owns 0–3,
rank 1 owns 4–7, rank 2 owns 8–11, and rank 3 owns 12–15. Seeds are therefore
11–26. Each rank writes `ranks/rank-NNN.json` plus four match files. Rank 0 publishes `manifest.json`
with all 16 entries only after every match round-trips, every local cached
probability replay passes, the HCCL global maximum is within tolerance, and
all rank reports agree on the adapter and policy version.

Require the final rank-0 JSON to contain:

- `status: complete`, `world_size: 4`, `completed_games: 16`;
- `games_per_rank: 4`, `parallel_games_per_rank: 2`;
- HCCL membership arrays containing four `1.0` values and reduction values
  `[10.0,10.0]`;
- `global_max_abs_log_prob_error <= 0.0002`;
- exactly one policy version and zero illegal substitutions;
- 504 trainable parameter tensors and 33,030,144 trainable parameters.

The output directory must not already exist. On failure, keep it as diagnostic
evidence and use a new output directory after fixing the cause. Do not delete
or resume a partial distributed smoke as though it were a completed pilot.

This gate passed in 636.230 seconds. All 16 games ended naturally; global
replay error was `0.0`; illegal substitutions were zero; the run contained
892 turns and 3,658 owned tokens; wins were exactly `[4,4,4,4]`.

### 10a. Four-trainer synthetic optimizer-math gate

Run this bounded gate before loading four training replicas. It creates only a
16-element synthetic trainable vector on each HPU and writes no artifact:

```bash
time bash env/shell.sh python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  --module self_play_grpo.cli \
  validate-distributed-optimizer-math \
  --expected-world-size 4 \
  --learning-rate 1e-3 \
  --max-grad-norm 1.0 \
  --parameter-count 16 \
  --timeout-seconds 120
```

Each rank constructs a distinct deterministic gradient. The command averages
those gradients in one synchronization phase, clips the shared gradient,
executes one AdamW step, clears gradients, and compares every rank against
rank 0. Require `status: ok`, `optimizer_steps: 1`,
`gradient_sync_phases: 1`, positive `parameter_change`, and zero values for
all reported global maximum differences/errors. Passing proves trainer
collective and optimizer math; it does not yet prove Qwen replay/backward or a
distributed checkpoint.

This gate passed on four HPUs in 32.024 seconds. The pre-clip norm was
`96.69539642333984`, the parameter change was `0.0010000020265579224`, and all
cross-rank differences were exactly `0.0`.

### 10b. Four-trainer real-artifact update gate

This is the bounded D3 gate. It uses the completed 16-game D2 artifact, loads
four training replicas, and performs exactly one real LoRA optimizer update.
It collects no new games and is not an open-ended training run:

```bash
time bash env/shell.sh python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  --module self_play_grpo.cli \
  validate-distributed-policy-update \
  --config self_play_grpo/configs/quoridor_outcome.yaml \
  --input self_play_grpo/artifacts/distributed-gameplay-4rollout-hpu-qwen3-4b-seed-11 \
  --output self_play_grpo/artifacts/distributed-update-4trainer-hpu-qwen3-4b-seed-11-graph-lifetime-v2 \
  --expected-world-size 4 \
  --replay-tolerance 2e-4 \
  --timeout-seconds 1800
```

The output must be a new directory. Each rank receives four complete matches:
indices 0–3, 4–7, 8–11, or 12–15. The command restores the immutable behavior
adapter, validates every artifact hash and engine replay, performs
action-by-action backward, then synchronizes gradients at one boundary.

Require rank 0 to report:

- `status: ok`, `validation_only: true`, `world_size: 4`;
- `games: 16`, `games_per_rank: 4`, and `optimizer_steps: 1`;
- `gradient_sync_phases: 1`, `gradient_tensors_reduced: 504`;
- `global_max_abs_log_prob_error <= 0.0002`;
- finite positive `grad_norm`, positive `parameter_change`, and at least one
  changed parameter tensor;
- zero cross-rank differences for gradient norm, parameters, both Adam
  moments, step counters, and changed-tensor count;
- 504 optimizer entries and 1,512 optimizer-state tensors;
- a validation-only `policy-000001` checkpoint written only by rank 0.

Keep a failed output directory as diagnostic evidence and use a new directory
for any retry. Passing D3 proves a real replicated update; distributed
save/reload continuation remains the separate D4 gate.

The first two D3 hardware attempts using exact batched differentiable replay
failed with an HPU allocation error before an optimizer step. A per-action
graph-lifetime boundary did not resolve the first-action memory requirement;
the earlier diagnosis that retained prior-action graphs caused the failure
was not established. A lower-memory one-row retry passed its first backward
but later failed the strict behavior-replay gate, so it is not a valid D3
training path. A one-action exact-batch attention-checkpointing probe passed
with replay error 0.0 and a 68.084 GB HPU peak, but no full-shard or four-rank
update has passed yet. D3 now uses exact recorded batch shape with scoped
attention-only checkpointing; keep failed directories and use a fresh output
directory for the next hardware gate.

### 11. Single-action backward probe

The corrected saved-artifact gate passed in 82.018 seconds: 100 turns,
416 tokens, maximum error `0.0`. Next run:

```bash
time bash env/shell.sh python -m self_play_grpo.cli \
  validate-policy-backward \
  --config self_play_grpo/configs/quoridor_fixture.yaml \
  --input self_play_grpo/artifacts/policy-match-gate-seed-11.jsonl \
  --joint-step 3 \
  --replay-tolerance 2e-4
```

This loads the model, replays one action in training mode with autograd enabled,
and applies the clipped token objective with synthetic advantage +1 and the
configured horizon normalizer. The synthetic advantage exercises gradients
even though this saved match has zero outcome advantage. It does not change
recorded credit. Require `status: ok`, finite positive `grad_norm`, replay error
at most `2e-4`, `optimizer_steps: 0`, and `gradients_cleared: true`.
This single-action HPU backward computation passed in 37.220 seconds with
replay error `0.0`, mean ratio `0.9999999403953552`, loss
`-0.05999999865889549`, and unclipped gradient norm `1.5376510660497742`.
All 504 parameter tensors receiving gradients were trainable; gradients were
cleared and optimizer steps were zero. Peak memory has not been measured.
It creates no optimizer and writes no model or rollout.

| Option | Default | Meaning |
|---|---|---|
| `--config` | required | Model, clipping epsilon, and fixed horizon normalization |
| `--input` | required | JSONL containing exactly one recorded match |
| `--joint-step` | `3` | Zero-based saved action; 3 selects the previous worst replay discrepancy |
| `--replay-tolerance` | `2e-4` | Finite positive limit for gradient-enabled probability replay |

### 12. Remaining gates

The bounded-memory and checkpoint gates below have already passed and are kept
here as reproducible references. The remaining pre-training work is completion
of the frozen-policy pilot above and a recorded initial-policy external-
evaluation baseline.

Run the short accumulation gate first:

```bash
time bash env/shell.sh python -m self_play_grpo.cli \
  validate-policy-batch-backward \
  --config self_play_grpo/configs/quoridor_fixture.yaml \
  --input self_play_grpo/artifacts/policy-match-gate-seed-11.jsonl \
  --actions 4 \
  --replay-tolerance 2e-4
```

This uses the production action-by-action backward implementation with a
synthetic +1 advantage, but performs no optimizer step and writes no artifact.
Require exact or below-threshold replay, a finite positive gradient norm,
`optimizer_steps: 0`, and `gradients_cleared: true`. After it passes, replacing
`--actions 4` with `--actions 100` exercises the complete saved match and may be
long-running. When the installed HPU API exposes allocator counters, the output
also includes baseline, peak, and peak-increase bytes; compare the four-action
and 100-action peaks to verify that live graph memory does not grow per action.

The four-action gate passed in 41.893 seconds with exact replay, 20 owned
tokens, gradient norm `1.9488064050674438`, baseline allocation
`8,183,185,024`, and peak allocation `25,289,977,728` bytes. Gradients were
cleared and optimizer steps remained zero. The same gate with `--actions 100`
subsequently passed in 187.173 seconds. The 100-action
peak was `25,337,058,048` bytes, only `47,080,320` bytes (about 0.19%) above
the four-action peak, confirming bounded graph memory for the complete match.

Validate checkpoint save/load next with a new output path:

```bash
time bash env/shell.sh python -m self_play_grpo.cli \
  validate-checkpoint-roundtrip \
  --config self_play_grpo/configs/quoridor_fixture.yaml \
  --input self_play_grpo/artifacts/policy-match-gate-seed-11.jsonl \
  --output self_play_grpo/artifacts/checkpoint-roundtrip-initial \
  --joint-step 3 \
  --replay-tolerance 2e-4
```

This writes one atomic adapter-only checkpoint, perturbs an in-memory adapter
parameter and RNG state, then reloads and requires exact parameter, probability,
CPU RNG, and (when supported) HPU RNG continuation. It performs no backward or
optimizer step and refuses to overwrite the output directory.

The initialization round-trip passed in 37.509 seconds. Its six files total
132,211,226 bytes; adapter parameters, behavior probabilities, CPU RNG, and HPU
RNG restored exactly, and no frozen base weights were stored.

Validate a populated Adam state and exact next-step continuation with:

```bash
time bash env/shell.sh python -m self_play_grpo.cli \
  validate-optimizer-checkpoint \
  --config self_play_grpo/configs/quoridor_fixture.yaml \
  --input self_play_grpo/artifacts/policy-match-gate-seed-11.jsonl \
  --output self_play_grpo/artifacts/checkpoint-roundtrip-optimizer \
  --joint-step 3 \
  --replay-tolerance 2e-4
```

This is a controlled optimizer validation, not a self-play run. It executes one
synthetic step, saves it, executes the next step, reloads the saved state, and
repeats that next step. It therefore executes three in-memory optimizer steps
in total and persists only the checkpoint after the first. It requires exact
equality for every trainable parameter, every Adam state tensor, and all
continuation metrics. The output path must be new.
New validation checkpoints carry `validation_only: true`; normal training
resume rejects them unless a validation caller explicitly opts in.

The optimizer continuation gate passed in 41.134 seconds. The saved checkpoint
was 396,882,614 bytes with 504 populated optimizer entries. All trainable
parameters, all 1,512 Adam tensors, and continuation metrics matched exactly
after reload. The post-update replay distance was `0.3149909973144531` and clip
fraction was `1/6`, confirming that the clipped-loss continuation was exercised.

Do not use `train` as a substitute for the still-pending aggregate, pilot, and
initial external-evaluation gates.

## CLI reference

| Command | Required options | Optional options | Side effects/cost |
|---|---|---|---|
| `validate-distributed-runtime` | `--expected-world-size N` | `--timeout-seconds` (120) | Must be launched by single-node `torchrun`; performs small HCCL reductions on one tensor per rank; no model, artifact, or update |
| `validate-distributed-optimizer-math` | `--expected-world-size N` | `--learning-rate` (`1e-3`), `--max-grad-norm` (`1.0`), `--parameter-count` (`16`), `--timeout-seconds` (120) | Four-trainer D1 gate: averages rank-distinct synthetic gradients, clips, takes one AdamW step, and requires exact parameter/optimizer equality; no model or artifact |
| `validate-distributed-policy-update` | `--config PATH`, `--input DIR`, `--output DIR`, `--expected-world-size N` | `--replay-tolerance` (`2e-4`), `--timeout-seconds` (1800) | Four-trainer D3 gate: consumes one complete 16-game behavior batch, runs real bounded backward, synchronizes and steps once, proves exact replicas, and writes one validation-only rank-0 checkpoint |
| `collect-distributed-policy-pilot` | `--config PATH`, `--output DIR`, `--expected-world-size N` | `--games-per-rank` (4), `--parallel-games` (config value), `--seed` (config seed), `--replay-tolerance` (`2e-4`), `--timeout-seconds` (1800) | New-directory-only distributed gameplay for the rollout HPUs: one inference replica per rank, complete match shards, exact replay, rank reports, and rank-0 global manifest; no backward or optimizer |
| `validate-engine` | `--config PATH` | none | Loads OpenSpiel; short deterministic games; no model |
| `bot-tournament` | `--config PATH`, `--output PATH` | `--candidate {random,shortest,wall-aware}` (default `shortest`), `--opponents` exactly three names, `--games-per-seat` (2), `--seed` (11) | Writes evaluation JSONL; no model |
| `validate-model-assets` | `--config PATH` | none | Reads metadata/shard stats and loads tokenizer only |
| `validate-model-load` | `--config PATH` | none | Allocates base model and LoRA on HPU; no forward |
| `validate-policy-forward` | `--config PATH` | none | Loads model; samples one action and runs probability replay; no update |
| `validate-policy-batched-forward` | `--config PATH` | `--batch-size` (2), `--replay-tolerance` (`2e-4`) | Loads one model; samples one action for several padded environments; gates recorded batch-shape replay and reports single-row equivalence diagnostically; no artifact or update |
| `collect-policy-match` | `--config PATH`, `--output PATH` | `--seed` (config seed), `--replay-tolerance` (`2e-4`) | Potentially long: collects one complete model match, writes JSONL, and replays environment/probabilities; no update; refuses overwrite |
| `collect-policy-pilot` | `--config PATH`, `--output DIR`, `--games N` | `--seed` (config seed), `--max-new-games N` (unlimited), `--parallel-games N` (config value), `--replay-tolerance` (`2e-4`) | Resumable frozen-policy collection with dynamically compacted per-HPU model batches; atomically saves adapter/manifest/per-game JSONL and exactly replays each game; no optimizer |
| `analyze-policy-pilot` | `--input DIR` | `--top-k` (`10`) | Verifies registered hashes and identities, then summarizes outcomes and movement/action bias by seat; no model or OpenSpiel |
| `validate-policy-artifact` | `--config PATH`, `--input PATH` | `--replay-tolerance` (`2e-4`) | Read-only exact environment and authoritative cached-probability replay of saved matches; no collection or update |
| `diagnose-policy-replay` | `--config PATH`, `--input PATH` | `--top-k` (`10`) | Read-only model inference over a saved full match; compares behavior, cached, and full replay paths |
| `validate-policy-backward` | `--config PATH`, `--input PATH` | `--joint-step` (3), `--replay-tolerance` (`2e-4`) | Single saved-action HPU forward/backward; synthetic +1 advantage; clears gradients; no optimizer or artifact write |
| `validate-policy-batch-backward` | `--config PATH`, `--input PATH` | `--actions` (4), `--replay-tolerance` (`2e-4`) | Production bounded-memory gradient accumulation over a saved prefix; synthetic +1 advantages; no optimizer or artifact write |
| `show-action-history` | `--input PATH` | `--seat {0,1,2,3}`, `--start` (0), `--limit`, `--json`, `--include-prompts` | Read-only view of saved actions; does not load the model or OpenSpiel; prompt inclusion requires JSON mode |
| `validate-checkpoint-roundtrip` | `--config PATH`, `--input PATH`, `--output PATH` | `--joint-step` (3), `--replay-tolerance` (`2e-4`) | Writes one atomic adapter-only checkpoint, perturbs in-memory state, and requires exact restore; no optimizer step |
| `validate-optimizer-checkpoint` | `--config PATH`, `--input PATH`, `--output PATH` | `--joint-step` (3), `--replay-tolerance` (`2e-4`) | Controlled synthetic Adam continuation: save after step one, compare uninterrupted/resumed step two exactly |
| `train` | `--config PATH`, `--artifact-dir PATH`, `--updates N`, `--acknowledge-training` | none | **Long-running and state-changing:** collects complete games, updates LoRA, and writes rollouts/checkpoints |

The installed console entry point `self-play-grpo` is equivalent to
`python -m self_play_grpo.cli` when invoked inside `env/shell.sh`.

## Training entry point: intentionally gated

The syntax is documented for completeness, not as the next command to run:

```bash
bash env/shell.sh python -m self_play_grpo.cli \
  train \
  --config self_play_grpo/configs/quoridor_fixture.yaml \
  --artifact-dir self_play_grpo/artifacts/RUN_ID \
  --updates 1 \
  --acknowledge-training
```

`--updates` is the number of sequential collect-then-update cycles. Each cycle
collects `rollout.games_per_update` complete matches under one frozen policy,
checks probability replay, computes one loss, performs one optimizer step, and
saves a new checkpoint. Never reuse an artifact directory whose intended run
identity or configuration differs.

## Records and artifacts

### Inspecting generated actions

Display a compact action history:

```bash
bash env/shell.sh python -m self_play_grpo.cli \
  show-action-history \
  --input self_play_grpo/artifacts/policy-match-gate-seed-11.jsonl
```

Use `--seat 2` for one model seat, or `--start 20 --limit 10` for a window.
For exact completion token IDs, per-token log probabilities, allowed-token
counts, constrained action probability, and legal-action count, add `--json`.
Adding `--include-prompts` with `--json` also emits the full observation,
rendered chat prompt, and legal label menu for each selected turn. The command
only reads the JSONL; it does not load model weights or the game engine.

Analyze every registered game in a pilot directory without loading the model
or OpenSpiel:

```bash
bash env/shell.sh python -m self_play_grpo.cli \
  analyze-policy-pilot \
  --input self_play_grpo/artifacts/pilot-outcome-qwen3-4b-seed-11 \
  --top-k 10
```

The command fails on an artifact hash or identity mismatch. Its per-seat
metrics distinguish goal-forward, goal-backward, and lateral pawn moves using
the four canonical goal orientations and report opening and overall action
frequencies. It is diagnostic only and writes nothing.

### Evaluation JSONL

`bot-tournament` writes atomically:

1. one `record_type: summary` line;
2. one `record_type: game` line per independent match.

Each game contains candidate seat/name/result, opponents, seed, final result,
termination reason, and action count. It does not contain the complete turn log.

### Training rollout JSONL

The trainer writes:

```text
ARTIFACT_DIR/rollouts/policy-NNNNNN.jsonl
```

Each line is a schema-versioned `MatchRecord` containing configuration, engine
manifest, seat map, initial state, every turn, exact prompt/completion token
IDs, behavior log probabilities, allowed-token sets, ownership masks, credit,
final result, termination reason, and serialized final environment.

### Checkpoint directory

After an update, the trainer writes:

```text
ARTIFACT_DIR/checkpoints/policy-NNNNNN/
  adapter/
  optimizer_state.pt
  torch_rng_state.pt
  evaluator_state.pt        # only when an evaluator is attached
  trainer_state.json
```

The adapter directory contains only trainable PEFT state. Frozen base weights
are resolved from the immutable model ID/revision/local path in the exact saved
configuration and are not copied into every checkpoint. `trainer_state.json`
records format 2, adapter-only storage, configuration, update index, policy
version, and last metrics. Writes use a temporary sibling directory followed by
an atomic rename; resume rejects configuration or adapter-key mismatches. The
Python API implements `load_checkpoint`; there is not yet a training-resume CLI
option.

## Metrics emitted per update

| Metric | Meaning |
|---|---|
| `update` | Zero-based update index just completed |
| `policy_version` | Frozen behavior version used for the collected batch |
| `games` | Complete matches in the batch |
| `turns` | Total public actions across those matches |
| `owned_tokens` | Generated action tokens contributing to loss |
| `loss` | Policy loss plus weighted KL |
| `policy_loss` | Negative fixed-normalized clipped surrogate |
| `kl_loss` | Fixed-normalized sampled KL term |
| `mean_ratio_before_step` | Mean new/behavior probability ratio before optimizer step; should begin near one |
| `clip_fraction_before_step` | Fraction outside the clip range before the step; should begin near zero |
| `replay_max_abs_error` | Maximum constrained old/new log-probability discrepancy |
| `grad_norm` | Pre-clipping global trainable gradient norm returned by PyTorch |
| `optimizer_steps` | `1` when parameters were stepped, or `0` when the complete gradient norm was zero |

## Troubleshooting

### OpenSpiel download pauses at build dependencies

The current installer does not use `pip download`. Confirm you are running the
checked-in `scripts/install_open_spiel.sh`; it should use `curl` and print SHA
checks before extraction.

### Partial or malformed OpenSpiel patch

The installer restores pristine files from the verified archive whenever the
patch completion stamp is absent. Rerun the installer. Do not manually edit the
cached source unless intentionally developing a new patch.

### Installed OpenSpiel says version 2.0.2

Expected. The backport is proven by
`test_pinned_forced_pass_regression`, not a local version suffix.

### Quoridor “known issues” warnings

Expected upstream warning. Treat a regression-test failure as actionable; do
not treat the warning alone as a failed gate.

### Model cache missing

Verify `model.local_path`, download the exact configured revision, then rerun
`validate-model-assets`. Do not set `local_files_only: false` as a workaround in
an experiment config.

### PEFT enters Neural Compressor and fails on `Conv1D`

The loader includes a scoped workaround for dense BF16 Qwen3: INC dispatch is
disabled only during generic LoRA injection. Do not downgrade the Gaudi stack or
remove Neural Compressor to solve this project-level incompatibility.

### HPU eager-mode warnings

Messages saying lazy-only step functions have no effect are expected with
`PT_HPU_LAZY_MODE=0`. They do not by themselves indicate a failed command.

### `pkg_resources`, `pyhlml`, or Apex warnings

These originate in the installed Habana stack. Record them, but do not upgrade
core packages during a pinned run solely to silence them.

### Probability replay mismatch

Stop before training. Preserve behavior/replayed log probabilities, allowed
token sets, dtypes, eager/lazy mode, and the exact checkpoint. Check that model
dropout is zero, the model is in evaluation mode during collection/replay, and
both paths renormalize over the same allowed tokens. Sampling, replay, and loss
must also share the incremental KV-cache computation path. Use the
full-sequence function only to diagnose backend differences. Never increase the
tolerance to hide a computation-path or semantic mismatch.

## Changing a standard configuration

For every intentional change:

1. Copy the YAML to a new, named configuration rather than overwriting the
   baseline used by existing artifacts.
2. Record the reason, old value, new value, and date.
3. Rerun configuration and deterministic tests.
4. If engine fields changed, rerun all engine contracts and fixtures.
5. If model/tokenization/sampling fields changed, rerun all three model gates.
6. If reward/loss fields changed, add a closed-form unit test and rerun the loss
   direction/ownership suite.
7. Use a new artifact directory and preserve the exact configuration with the
   run.
8. Never compare conditions that differ unintentionally in engine, horizon,
   prompts, tokenizer, action mask, seeds, or evaluation opponents.

## Completion criteria for the baseline

The baseline is ready for an experimental run only when all of the following
are recorded:

- deterministic and engine suites pass;
- model assets/load/one-action replay pass;
- complete LLM pilot matches replay exactly;
- no illegal action substitution or mixed policy version occurs;
- one backward pass has the expected finite direction and norm;
- checkpoint/resume reproduces a deterministic continuation;
- fixed opponent manifest and validation/test protocol are frozen;
- resource measurements justify the selected games per update;
- a unique artifact directory and run identity are chosen.
