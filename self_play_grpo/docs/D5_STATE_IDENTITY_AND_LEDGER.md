# D5 state identity and commit-ledger increment

This increment is accelerator-free. It does not start a rollout, optimizer step,
or eight-HPU launcher. Apply `patches/d5_identity_and_ledger_test_fix_v1.patch`
before testing; the direct workspace editor could add new files but could not
update two existing lines in this session. Do not also apply the superseded
single-file `d5_identity_scalar_fix_v1.patch`.

## State fingerprints

`training/identity.py` computes deterministic SHA-256 digests over the actual
trainable parameter values and AdamW optimizer state. It includes parameter
names, tensor dtypes/shapes/bytes, ordered parameter groups, hyperparameters,
moments, and step counters. Mapping insertion order and checkpoint pickle
storage order do not affect the digest. A tensor is copied to CPU for hashing,
without changing model or optimizer state. The helper does not prove that the
four trainer ranks own the same physical HPU; role binding remains a separate
launcher check. The `CycleCoordinator` accepts these digests in its four rank
update acknowledgements, but no live rank currently calls the helper.

## Consumed-batch ledger

`training/ledger.py` publishes one immutable record per production update,
linked to its predecessor and a complete format-3 checkpoint. Reading the
ledger revalidates the chain and referenced checkpoint files, rejects gaps,
run-ID changes, a repeated source rollout manifest, and validation-only
checkpoints. Publication uses an exclusive hard link, so an existing update
record cannot be overwritten. A single coordinator writer is assumed; this
is not distributed consensus. If publication is interrupted and a pending
file remains, recovery must inspect it explicitly rather than silently skip
or overwrite it. The live training loop has not yet been wired to publish the
ledger, so this is a tested contract, not end-to-end restart protection.

## Local verification and pending HPU gate

After applying the patch, run the focused tests and non-model suite:

```bash
git apply --check self_play_grpo/patches/d5_identity_and_ledger_test_fix_v1.patch
git apply self_play_grpo/patches/d5_identity_and_ledger_test_fix_v1.patch
bash env/shell.sh python -m pytest self_play_grpo/tests/test_training_identity.py self_play_grpo/tests/test_training_ledger.py -q
bash env/shell.sh python -m pytest self_play_grpo/tests -m 'not model' -q
```

Before a full GRPO run, D4 must pass a real trainer checkpoint continuation
gate on accessible HPUs. D5 still needs a bounded 4-rollout/4-trainer launcher,
model-probability replay, one synchronized update using outcome credit,
checkpoint and ledger publication, four-way adapter refresh, and a numerical
refresh probe. The current D3 and D4 commands remain validation-only.
