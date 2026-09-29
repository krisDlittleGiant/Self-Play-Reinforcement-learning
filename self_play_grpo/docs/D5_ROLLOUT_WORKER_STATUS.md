# D5 frozen-policy rollout worker status

This is an implementation slice, not a completed eight-HPU GRPO run. The
hardware D4 acceptance gates and the D5 trainer, refresh, and top-level cycle
driver remain prerequisites. No HPU execution is initiated by this module at
import time.

## Contract implemented

`training/rollout_worker.py` prepares a new immutable 64-game batch from a
designated production adapter and policy descriptor. It copies and hashes the
adapter, records the descriptor, and creates an empty pilot manifest. A
pre-existing output directory is never overwritten. The current profile is
exactly four rollout ranks, 16 complete games per rank, two active games per
rank, four-seat Quoridor, one optimizer epoch, and engine-outcome credit.

Each explicit `collect_rank` invocation checks its rank-to-module binding,
loads the frozen adapter, hashes the loaded trainable tensors, and collects
eight batches of two independent complete games. Each match is persisted,
engine-replayed, and behavior-log-probability-checked before the rank report is
published. On failure, a partial directory remains diagnostic evidence and
must not be reused for another batch attempt.

The CPU-only `aggregate_rank_shards` function requires four complete rank
reports with contiguous indices 0–63, matching adapter/version, the expected
physical module IDs, and identical trainable-tensor hashes. It validates all
registered match files before publishing the completed manifest. A missing or
malformed shard leaves the on-disk manifest empty; there is no partial commit.

## Verification and limitations

The new `tests/test_rollout_worker.py` covers batch preparation, strict
profile/config checks, rank binding, incomplete or inconsistent rank reports,
and publication only after match validation. These are accelerator-free
contract tests. The test that checks frozen full-descriptor identity depends
on `patches/d5_rollout_worker_policy_guard_v2.patch` being applied.

The worker has **not** been tested on the Gaudi HPUs. The production trainer,
distributed gradient synchronization, format-3 checkpoint handoff, adapter
refresh probe, and full eight-worker lifecycle are not yet implemented as one
integrated run. Do not start a learning run or interpret these CPU tests as
D4/D5 hardware acceptance. Run hardware gates only after the node exposes the
required eight allocated HPUs and D4 two-/four-rank evidence is accepted.

To complete this slice in the shared repository:

```bash
git apply --check self_play_grpo/patches/d5_rollout_worker_policy_guard_v2.patch
git apply self_play_grpo/patches/d5_rollout_worker_policy_guard_v2.patch
bash env/shell.sh python -m pytest self_play_grpo/tests/test_rollout_worker.py -q
bash env/shell.sh python -m pytest self_play_grpo/tests -m 'not model' -q
```

The older `d5_rollout_worker_policy_guard_v1.patch` was a malformed draft and
must not be applied.
