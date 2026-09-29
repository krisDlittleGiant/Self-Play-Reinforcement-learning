# Self-play GRPO for four-player Quoridor

This directory implements the experiment specified in
`../self_play_grpo_implementation_plan.md`. Four seats share one frozen policy
during collection, one authoritative Quoridor state is advanced per match,
and updates use all four dependent player views while retaining the match as
the grouping unit.

Operational documentation:

- [`docs/RUNBOOK.md`](docs/RUNBOOK.md): configurations, every runtime parameter,
  validation order, CLI commands, artifacts, and troubleshooting.
- [`docs/IMPLEMENTATION_AND_TEST_LOG.md`](docs/IMPLEMENTATION_AND_TEST_LOG.md):
  implemented changes, exact validation evidence, recovered failures, warnings,
  and pending gates.
- [`docs/EIGHT_HPU_TRAINING_DESIGN.md`](docs/EIGHT_HPU_TRAINING_DESIGN.md):
  required final eight-Gaudi architecture, invariants, and acceptance gates.

## Environment decision

Use the workspace's existing `.runtime/venv`. It already contains the
Gaudi-enabled Torch, Transformers, and PEFT stack. Creating another virtual
environment would either duplicate that large stack or accidentally replace
the accelerator-specific Torch build. OpenSpiel remains an optional, pinned
dependency and every heavyweight import is lazy.

Do not install this project with a command that resolves or upgrades Torch.
From the workspace root, the intended setup is:

```bash
bash env/shell.sh python -m pip install --no-deps -e ./self_play_grpo
bash self_play_grpo/scripts/install_open_spiel.sh
```

The second command compiles OpenSpiel and is deliberately not run by the
implementation workflow. It uses the dependency-complete 2.0.2 source archive,
backports upstream Quoridor commit `d0606878`, verifies the patched source, and
limits compilation to eight jobs. The installed distribution retains the
upstream `2.0.2` metadata version; the forced-pass regression test below is the
behavioral provenance check. Verify its build in a short interactive allocation,
not on a login node if local policy forbids compilation there.

## Safety gates

Run the pure, deterministic tests before installing OpenSpiel:

```bash
bash env/shell.sh python -m pytest self_play_grpo/tests -m 'not engine and not model'
```

After installing the pinned engine, validate its exact four-player contract:

```bash
bash env/shell.sh python -m self_play_grpo.cli \
  validate-engine --config self_play_grpo/configs/quoridor_fixture.yaml
```

After placing the pinned model at the configured local path, validate its files,
revision metadata, chat template, and every possible action label without loading
model weights:

```bash
bash env/shell.sh python -m self_play_grpo.cli \
  validate-model-assets --config self_play_grpo/configs/quoridor_outcome.yaml
```

Then validate that the base model and shared LoRA adapter load on the configured
device. This allocates model weights but performs no forward pass or training:

```bash
bash env/shell.sh python -m self_play_grpo.cli \
  validate-model-load --config self_play_grpo/configs/quoridor_outcome.yaml
```

Before any rollout or update, sample one legal action and replay its masked
behavior probabilities through a separate forward pass:

```bash
bash env/shell.sh python -m self_play_grpo.cli \
  validate-policy-forward --config self_play_grpo/configs/quoridor_fixture.yaml
```

The final pre-update model gate collects one complete fixture match, writes the
full turn schema, replays every environment state, and replays all constrained
behavior probabilities. It performs no backward pass or optimizer step:

```bash
bash env/shell.sh python -m self_play_grpo.cli \
  collect-policy-match --config self_play_grpo/configs/quoridor_fixture.yaml \
  --output self_play_grpo/artifacts/policy-match-gate-seed-11.jsonl
```

If that artifact already exists, validate it without recollecting or
overwriting it:

```bash
bash env/shell.sh python -m self_play_grpo.cli \
  validate-policy-artifact \
  --config self_play_grpo/configs/quoridor_fixture.yaml \
  --input self_play_grpo/artifacts/policy-match-gate-seed-11.jsonl \
  --replay-tolerance 2e-4
```

Sampling, replay, and policy loss use the same incremental KV-cache probability
path. The full-sequence forward is retained only for backend diagnostics because
it is not numerically equivalent on the pinned HPU BF16 stack.

Then run a small reproducible bot tournament:

```bash
bash env/shell.sh python -m self_play_grpo.cli \
  bot-tournament --config self_play_grpo/configs/quoridor_fixture.yaml \
  --games-per-seat 2 --seed 11 --output self_play_grpo/artifacts/fixture.jsonl
```

The engine, bot, and model-asset commands do not load model weights.
`validate-model-load` allocates the model, and `validate-policy-forward` performs
forward passes; neither updates weights. No optimizer or training command should
be used until the engine gate passes, recorded games replay exactly, and behavior
log-probability ratios are verified near one.

## Package map

- `envs/`: pinned OpenSpiel adapter, canonical seat map, state extraction,
  serialization, action labels, and prompt rendering.
- `policies/`: reproducible bots and constrained Transformers action sampling.
- `rollouts/`: match-level event schema, frozen-policy collector, and resumable
  hashed pilot manifests with atomic per-game artifacts.
- `rewards/`: terminal advantages, path-distance proxy, shaping, and GAE.
- `training/`: constrained log-probability replay, clipped loss, evaluator, and
  synchronous checkpointed loop.
- `evaluation/`: seat-rotated fixed-opponent tournaments and match bootstrap.
- `tests/`: mathematical and ownership checks plus opt-in engine contracts.

The default configuration pins `Qwen/Qwen3-4B` to Hugging Face revision
`1cfa9a7208912126459214e8b04321603b3df60c` and loads it from
`/scratch/svijay46/models/Qwen3-4B`. Model files are never downloaded implicitly
by the CLI: the loader defaults to `local_files_only: true`.
