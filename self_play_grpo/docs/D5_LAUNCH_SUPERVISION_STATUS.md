# D5 launch supervision status

This increment adds an opt-in, no-model 4+4 launch boundary. It does **not**
provide a production GRPO worker command or prove an eight-HPU cycle.

## Implemented

- `training/d4_acceptance.py` reads two- and four-rank D4 summaries, every
  rank's equality flags and module ID, the format-3 checkpoint inventory, and
  the checkpoint manifest digest. The four-rank D4 trainer modules must match
  the planned D5 trainer half. It is a read-only evidence check; a D4
  validation checkpoint is never promoted to a production starting policy.
- `training/process_supervisor.py` takes exactly eight explicit command
  vectors, maps four rollout and four trainer ranks to disjoint module IDs,
  gives only trainer ranks a four-process rendezvous, captures per-rank logs,
  enforces a deadline, and reports failed/partial launches. No shell command
  is interpolated. It does not request or reserve devices.
- `training/d5_preflight.py` checks the current 64-game outcome configuration:
  16 complete games per rollout rank, two active games at a time, one optimizer
  epoch, and four-seat groups. Both D4 summaries and the known local allocation
  must pass before caller-supplied worker commands are started. Seeing eight
  host devices is not treated as proof that the process may acquire eight.
- The existing `CycleCoordinator`, `handoff.py`, and `ledger.py` remain the
  phase, checkpoint, refresh, and exactly-once contracts for a future driver.

The wrapper deliberately does not turn the existing D2 collector and D3
validation-only trainer into a production command. A successful process exit
alone is reported as `workers_exited`, not as a successful GRPO update.

## Verification and patch

On 28 September 2026, 14 focused no-HPU tests passed. The complete non-model
suite passed 175 tests with three existing Habana import/deprecation warnings.
The supervisor teardown hardening in
`patches/d5_supervisor_teardown_v4.patch` passes `git apply --check` but needs
to be applied in the remote workspace. It detects descendants left by an
exited worker, terminates the whole process group, and cleans up on keyboard
interruption. Do **not** apply superseded v1, v2, or v3.

After applying v4, rerun:

```bash
git apply --check self_play_grpo/patches/d5_supervisor_teardown_v4.patch
git apply self_play_grpo/patches/d5_supervisor_teardown_v4.patch
bash env/shell.sh python -m pytest self_play_grpo/tests/test_process_supervisor.py self_play_grpo/tests/test_d4_acceptance.py self_play_grpo/tests/test_d5_preflight.py -q
bash env/shell.sh python -m pytest self_play_grpo/tests -m 'not model' -q
```

## Still required before a full run

The D4 two- and four-trainer continuation gates must pass on genuinely
acquirable HPUs. Their current evidence is not accepted as passing. Next,
implement actual rollout and trainer worker commands, their phase reports,
complete-batch probability replay, one synchronized optimizer step, format-3
production checkpoint publication, and adapter refresh/probes on all four
rollout workers. Only then run a bounded eight-worker validation cycle before
considering repeated production updates. A learned reward model is still a
separate experiment; engine outcomes remain the authoritative baseline.
