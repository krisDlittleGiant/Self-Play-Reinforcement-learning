# D6 next-collection preparation (2026-09-28)

This is a follow-up to `D6_RESTART_BOUNDARY_STATUS_2026-09-28.md`.

`training/d6_collection.py` now derives a production `PolicyDescriptor` for
the committed policy from the previous descriptor plus the verified checkpoint
adapter, retaining the original config/model/tokenizer/grammar identities. It
will prepare the next immutable 64-game rollout directory using the seed from
`d6_recovery.audit_recovery`, but only when all four refresh reports have been
verified. The existing `prepare_frozen_batch` rejects an output directory
that already exists, so this helper does not overwrite a partial collection.
The helper starts no HPU worker and performs no optimizer step.

The next production batch path after the completed D5 update would be
`rollout-000001`, collected with `policy-000001` for logical update 2. Do not
start it yet as a purported five-update training run: the trainer worker and
integrated driver still only implement the first update. When those are added,
the driver should call this helper at the committed READY boundary, and should
recover a previously prepared but incomplete collection under an explicit
resume protocol rather than blindly preparing the directory again.

CPU tests: `test_d6_recovery.py` and `test_d6_collection.py`. These verify the
committed-versus-refresh-complete gate, identity inheritance, and that a new
collection uses the committed adapter and deterministic new seed. They do not
verify HPU rollout or optimizer continuation.

One corrective patch,
`patches/d6_collection_remove_nonexistent_validate_v2.patch`, was applied
locally after the new helper was added. It removes a redundant call to a
nonexistent `PolicyDescriptor.validate()` method; descriptor validation already
occurs in `__post_init__`. Do not apply that patch again.
