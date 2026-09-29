# D5 trainer-to-rollout refresh gate

Status: implemented as an opt-in worker, not HPU-validated. Do not start the
next rollout from policy-000001 until four refresh reports are checked by a
coordinator. The one-update phase driver and five-update restart test remain
future work.

The trainer's first synchronized update now produces a fixed action probe
(game 0, joint step 0) using the exact recorded padded batch shape. Each
trainer rank evaluates the updated adapter and the four probabilities must
agree within 2e-4. Rank zero stores the probe, updated trainable-parameter
SHA-256, source manifest SHA-256 and checkpoint digests in
`trainer_summary.json`. The rank-zero summary is published inside a
collective phase so another rank cannot silently report success if the write
fails.

`policy_refresh.py` is a separate rollout-rank command. Before model loading,
it verifies D4 evidence, the assigned four-module role map, the 64-game
source manifest, the trainer summary, and the complete format-3 checkpoint
file inventory. It then loads the checkpoint adapter, compares trainable
tensor SHA-256 with the trainer's post-update digest, and replays the exact
trainer action probe. It writes one exclusive rank report only on success; it
does not generate games or update the model.

The D5 preflight distinguishes the parent, which must see all eight assigned
modules, from a worker, which must see exactly the four modules for its role.
The eight-HPU allocation count is still required when reported by the
environment. This correction does not allocate devices or start Slurm jobs.

Application commands, from the repository root, in this order:

```bash
git apply --unidiff-zero --check self_play_grpo/patches/d5_policy_refresh_trainer_probe_v2.patch self_play_grpo/patches/d5_role_scoped_preflight_v2.patch self_play_grpo/patches/d5_policy_refresh_test_fix_v1.patch
git apply --unidiff-zero self_play_grpo/patches/d5_policy_refresh_trainer_probe_v2.patch self_play_grpo/patches/d5_role_scoped_preflight_v2.patch self_play_grpo/patches/d5_policy_refresh_test_fix_v1.patch
bash env/shell.sh python -m pytest self_play_grpo/tests/test_policy_refresh.py self_play_grpo/tests/test_d5_role_preflight.py self_play_grpo/tests/test_trainer_worker.py -q
bash env/shell.sh python -m pytest self_play_grpo/tests -m 'not model' -q
```

The patch check succeeded locally, and two of the three new refresh tests
passed before applying the test correction. Syntax checks passed. No HPU run
or complete suite after all three patches is claimed here; those results must
be supplied by the operator.
