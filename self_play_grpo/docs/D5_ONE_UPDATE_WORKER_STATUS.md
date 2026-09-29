# D5 one-update trainer worker status

`training/trainer_worker.py` is now an opt-in, initial-policy, four-rank
production trainer entry point. It first requires the existing two-/four-rank
D4 hardware acceptance summaries, the current 64-game profile and an
eight-HPU allocation. Each rank checks its explicit physical module binding
and admits the same complete rollout manifest before HCCL initialization.

The worker then loads the frozen initial adapter, runs the streamed
16-games-per-rank update, reduces actual metrics across the four trainer
ranks, stages one RNG record per rank, and has rank zero atomically save a
production format-3 checkpoint. Every rank verifies the published checkpoint
and writes an update report. The resulting status is deliberately
`updated_checkpoint_committed_refresh_pending`; it is not a completed D5
cycle or a five-update training run.

The production checkpoint template binds all 64 match IDs/digests, four
trainer module IDs, model/config/runtime identity, tokenizer/grammar identity,
and source manifest digest. `production_code_identity()` must include the
new worker and rollout source. Before using this path, apply the two checked
patches below:

```bash
git apply --check self_play_grpo/patches/d5_trainer_update_metrics_v2.patch
git apply self_play_grpo/patches/d5_trainer_update_metrics_v2.patch
git apply --check self_play_grpo/patches/d5_production_code_identity_v1.patch
git apply self_play_grpo/patches/d5_production_code_identity_v1.patch
bash env/shell.sh python -m pytest self_play_grpo/tests -m 'not model' -q
```

The new worker's argument parser and CPU-only tests were checked without
starting HCCL or using an HPU. Before the patches, the local non-model suite
passed 196 tests with the three previously known Habana warnings. The new
worker has **not** been run on hardware, and no run command should be issued
until the D4 acceptance evidence and actual eight-HPU allocation are verified.

Still required for one full update: a bounded phase driver to start four
rollout workers, aggregate and commit their 64 games, start four trainer
workers, read all trainer reports, commit the production checkpoint to the
ledger, refresh four rollout workers from that checkpoint, perform a
trainer-versus-rollout probability probe, and record all acknowledgements.
Only after that one-cycle gate passes should a restartable five-update driver
be attempted.
