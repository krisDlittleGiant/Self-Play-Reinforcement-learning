# D5 synchronized trainer-update status

`training/trainer_update.py` implements one update for an already-admitted
16-game trainer shard on each of four initialized HCCL ranks. It is not a
launcher and does not publish a checkpoint. The caller must use the production
policy descriptor and `TrainerShardAdmission` from the complete 64-game
manifest, initialize the four-rank group, and load the authoritative model and
optimizer first.

The update checks initial trainable-tensor identities across ranks, performs
streamed action backward using the recorded batched replay shape and scoped
Qwen3 attention checkpointing, then averages trainable gradients once. Since
each rank's local loss is divided by 16 games, averaging the four local
gradients yields the 64-game objective. It clips gradients, takes one AdamW
step, and requires exact agreement of trainable parameters and optimizer
moments afterward. No checkpoint is committed here; a failed process must be
discarded and restarted from the last committed checkpoint.

The checked `patches/d5_trainer_update_manifest_tolerance_v3.patch` must be
applied before further trainer work. It binds each admission's replay tolerance
to its frozen manifest, and uses that value during backward. Do **not** apply
the earlier `d5_trainer_update_replay_tolerance_v1.patch` (malformed) or v2
(superseded by the manifest-bound guard).

```bash
git apply --check self_play_grpo/patches/d5_trainer_update_manifest_tolerance_v3.patch
git apply self_play_grpo/patches/d5_trainer_update_manifest_tolerance_v3.patch
bash env/shell.sh python -m pytest self_play_grpo/tests/test_trainer_handoff.py self_play_grpo/tests/test_trainer_update.py -q
bash env/shell.sh python -m pytest self_play_grpo/tests -m 'not model' -q
```

Before this final guard patch, the three new tiny-CPU update tests passed and
the non-model suite passed 189 tests with the three known Habana warnings.
These CPU tests emulate four equal ranks; they do **not** validate HCCL,
Gaudi memory, exact real-model replay, or a successful 64-game update.

Next implementation: a guarded four-rank trainer process entry point that
loads the source adapter/optimizer, calls admission and this update, stages
rank RNG records, publishes one format-3 distributed checkpoint, and supplies
the new adapter to all four rollout workers. D4 two-/four-rank hardware
acceptance and an actual eight-HPU allocation remain launch prerequisites.
