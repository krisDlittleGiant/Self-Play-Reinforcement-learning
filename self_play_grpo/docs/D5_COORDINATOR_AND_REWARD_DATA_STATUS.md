# D5 coordinator and reward-data implementation status

This is an implementation status note, not an eight-HPU acceptance report.
The D4 two-/four-rank continuation gates have not passed on the current
`gaudi003` process context: HPU acquisition failed before model loading.
No failed attempt produced rollout games or a D4 checkpoint.

## Implemented CPU contracts

- `training/coordinator.py`: explicit 4+4 physical-module layout, immutable
  policy identity, complete-batch receipt, and a fail-closed one-update state
  machine. Collection requires eight matching ready acknowledgements. Trainers
  must acknowledge exact shard coverage and manifest identity before update.
  Four matching optimizer/parameter reports precede checkpoint publication;
  four matching rollout refresh/probe reports precede the next collection.
- `verify_completed_batch` checks the frozen manifest, adapter bytes, every
  match hash and engine replay, metadata, policy identity, and exact
  outcome-derived turn credit. It does **not** replace the trainer's HPU
  behavior-probability replay, which must occur before optimizer mutation.
- `training/roles.py` requires an explicit eight-ID module list, split into
  four rollout and four trainer ranks. It does not infer authorization from
  host-wide `hl-smi` visibility.
- `rewards/dataset.py` creates outcome-supervised decision examples with
  observation/action features separated from engine-result labels. Complete
  games, not turns, are assigned to stable train/validation/test splits.
- `rewards/export.py` writes a new dataset only after completed-rollout
  verification and rechecks match and manifest digests while exporting.
  Engine outcomes remain the authoritative GRPO rewards. A learned predictor
  must not silently replace them.
- `configs/quoridor_outcome_64games.yaml` supports 16 matches on each of four
  rollout ranks (64 total), with two concurrent games per rank. The original
  16-match config is unchanged.

## Verification and required patch

The focused CPU suite passed (14 tests for coordinator/reward export/dataset,
plus six role tests). The complete non-model suite passed at 140 tests before
the role tests were added. The short safety patch
`patches/d5_coordinator_safety_v2.patch` passes `git apply --check` but must be
applied in the remote workspace. It makes malformed role tuple lengths fail
and marks an out-of-phase coordinator call as a failed cycle. Do not apply the
malformed superseded `d5_coordinator_safety_v1.patch`.

## Still missing before a real GRPO cycle

1. D4's bounded two- and four-trainer checkpoint continuation gates on HPUs
   that the process can actually acquire. CPU checkpoint tests alone are not
   D4 acceptance.
2. A real one-node eight-worker launcher: four rollout ranks and a separate
   four-rank HCCL trainer group, with bounded worker failure and teardown.
   An eight-rank default group must not average trainer gradients.
3. Live wiring from frozen policy -> 64-game manifest -> model probability
   replay -> one synchronized optimizer step -> format-3 checkpoint -> adapter
   refresh and numerical probes. The coordinator currently defines and tests
   these transitions but does not launch or connect the HPU workers.
4. Production safeguards: reject validation-only checkpoints and synthetic
   advantages, persist a consumed-batch ledger, and support fresh-process
   restart before claiming repeated training.
5. A learned reward predictor, if desired, needs an explicit target,
   architecture, training/evaluation protocol, and independent validation.
   Outcome-labeled examples are ready for that stage; no predictor has been
   trained or used as a GRPO reward.

Do not call the current code a full eight-HPU GRPO run. A 64-game collector
produces data only, and the D3 update command remains validation-only.
