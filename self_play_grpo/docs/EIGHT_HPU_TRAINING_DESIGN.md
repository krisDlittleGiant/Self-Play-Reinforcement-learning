# Eight-Gaudi training design and acceptance gates

Status date: 2026-09-26

Status: required final capability. D0 passed on four HPUs. D2 passed with four
rollout HPUs, 16 complete games, exact replay, and one published manifest. D1
passed its bounded four-trainer-HPU hardware gate. The final topology is four
rollout HPUs plus four trainer HPUs. D3 is implemented and awaiting its
bounded four-trainer-HPU hardware gate. D4 through D6 are not yet implemented
or validated.

This document makes the final hardware target explicit: one node with eight
Intel Gaudi HPUs must be able to run the complete collect-update-checkpoint
self-play loop. Current correctness and pilot commands use one HPU and do not
exercise distributed execution.

## Resource requirements by task

| Task | HPUs needed now | Notes |
|---|---:|---|
| Configuration, schema, loss math, engine, bot, and history tests | 0 | CPU-only |
| Model asset/tokenizer validation | 0 | Reads local files; does not load weights |
| Model load, constrained forward, rollout, replay, backward, and checkpoint gates | 1 | This is the currently validated path |
| Frozen-policy M1 pilot collection | 1 | One process can batch multiple independent games on one HPU; the single-HPU multi-match path is validated |
| Distributed gameplay smoke | 4 | Exercises the complete rollout half of the final 4+4 topology |
| Final distributed acceptance and full RL training | 8 | Required final target |

Do not allocate eight HPUs for the current pilot: the pilot command creates
one process and uses one device, although it now batches games within that
device. The current `train` command is also
single-process and must not be described as an eight-HPU trainer.

## Chosen architecture

Use the original 4+4 role split on one eight-Gaudi node:

- four rollout-generator HPUs each hold a frozen Qwen3-4B + LoRA inference
  replica and produce complete matches;
- four trainer HPUs hold synchronized training replicas and perform replay,
  bounded-memory backward, gradient synchronization, clipping, and updates;
- policy weights move from the trainer side to all rollout generators only at
  a policy-version boundary;
- one update never mixes games from different policy versions.

The four-HPU gameplay gate implemented now launches one rollout process
per device with `torchrun`, initializes an HCCL group, and selects each device
from `LOCAL_RANK`. It intentionally does not launch the four trainer roles.
The later full eight-HPU launcher must create distinct rollout and trainer
groups instead of treating all eight devices as interchangeable data-parallel
workers.

For the standard `games_per_update: 16` and four rollout ranks:

- each rollout rank owns exactly four complete matches, executed as two local
  batches of two games because `parallel_games_per_rank` is 2;
- all four seats of a match remain on the same rank;
- every rollout rank uses the identical frozen policy version;
- global match indices and seeds are deterministic and disjoint;
- no match or four-seat comparison group is split across ranks;
- rollout rank 0 commits global rollout metadata only after all four reports
  and all 16 match artifacts pass;
- the four trainer ranks later divide the 16-match training batch evenly,
  perform bounded action-by-action backward, and reduce LoRA gradients once;
- trainer rank 0 alone commits the optimizer checkpoint and next policy;
- barriers bracket policy-version changes and atomic checkpoint publication.

## Parallel games within one HPU

Each rollout rank owns one inference replica and a configurable set of
independent environments. `rollout.parallel_games_per_rank` defaults to `2`.
For the standard global 16-game, four-rollout-rank update, each rank executes
two successive batches of two games.
One-action gates at 2, 4, 8, and 16 all replayed exactly. Batch 16 sampled in
3.1428 seconds and peaked at 46,906,020,992 allocated bytes, so it remains the
tested one-HPU inference ceiling rather than the training default. The complete
two-game multi-turn HPU smoke passed under shape-contract v1.

The first complete two-game smoke initially failed under single-row replay,
leading to shape-contract v1. Its fresh retry then passed: 54 total turns over
two natural games, exact `0.0` batch-shaped replay, and zero substitutions.
Differentiable replay/backward and multi-rank execution are separate gates.

The single-action differentiable gate subsequently passed at the former worst
prompt state with exact replay and finite gradients on all 504 trainable
parameter tensors. Complete 23-action batch-shaped backward also passed with
exact replay and a 45,171,202,944-byte peak. Populated Adam checkpoint/resume
then reproduced parameters, 1,512 optimizer tensors, and metrics exactly;
multi-rank execution remains pending.

The implementation uses model batching, not multiple processes loading duplicate model
weights on one HPU. Each active game needs its own environment, legal-action
trie, deterministic RNG stream, KV-cache row, and artifact identity. Finished
games must be compacted or masked without changing another game's sampling
stream. Completed environments are dynamically compacted while returned
matches retain original input order.

Batch-shaped HPU kernels do not reliably reproduce single-row logits. A
complete two-game smoke observed error `1.93216`, so batched samples now record
contract version, batch size, target row, padded prompt width, and pad token.
Replay and training reconstruct that tensor shape with independent dummy rows;
single-row recomputation is diagnostic only and must never replace the actual
batched behavior distribution.

For `games_per_update: 16` and four rollout ranks, each rank owns four games
but keeps only two active at once. Larger per-rank concurrency remains an
optional measured optimization, not the baseline. The implementation caps
active games at the smaller of `parallel_games_per_rank` and remaining local
matches.

On the trainer side, four trainer ranks will each consume four complete
matches. Averaging equal local objectives through the trainer HCCL group then
equals the global 16-match objective. The implementation must reject a
`games_per_update` that is not divisible by both the rollout-rank and
trainer-rank counts until weighted uneven sharding is implemented.

## Probability and optimization invariants

Distributed execution may not weaken the already validated contracts:

1. Incremental KV-cache decoding remains the sole probability path for
   sampling, replay, and training.
2. Replay error is computed on rollout ranks and reduced with a global maximum.
   Any rollout rank over tolerance prevents publication and any trainer step.
3. The four dependent player views remain grouped by match.
4. Only model-owned completion tokens contribute to the loss.
5. Action-by-action backward remains graph-memory bounded.
6. Gradients are synchronized only after the complete local objective; doing
   an HCCL reduction for every generated action is forbidden.
7. The zero-signal rule is global on the trainer group: all trainer ranks skip
   together when the synchronized gradient norm is exactly zero.
8. After a step, trainable parameters and optimizer state must be identical on
   every trainer rank before the new adapter is transferred to rollout ranks.

The initial adapter must be created once on rollout rank 0 and broadcast to all
rollout ranks, or otherwise proven byte-identical before collection.
Independent random LoRA
initialization on each rank is not acceptable even if the initial LoRA output
happens to be zero.

## Artifacts and resume

Each rollout rank writes a separate report containing only complete matches.
Rollout rank 0 validates report counts, global indices, seeds, policy versions,
and hashes before atomically publishing a global batch manifest. A failure on
one rollout rank must not publish a partial batch.

Only trainer rank 0 writes the optimizer checkpoint and next adapter after
synchronization. Rollout rank 0 writes the immutable behavior adapter snapshot
and completed rollout manifest.
The checkpoint must additionally record:

- checkpoint format and exact experiment configuration;
- world size, backend, rank-to-device mapping, and match-sharding rule;
- global update and policy version;
- one RNG-state record per rank;
- per-rank rollout shard hashes;
- global replay maximum and optimizer metrics.

Resume must reject world-size/configuration drift for the initial
implementation. A deterministic continuation gate must prove that an
uninterrupted distributed second step equals a save/reload second step.

## Implementation stages and exit gates

| Stage | Required evidence |
|---|---|
| D0: Configuration/runtime contract | **Passed on four HPUs:** physical modules `0,1,2,3`, one-to-one rank/device memberships, and both collective probes reduced to `10.0` |
| D1: Collective math test | **Passed on four trainer HPUs:** one synchronization phase produced exact averaged gradients, identical clipped norm, parameters, Adam moments, and step counters after one update; 32.024 seconds |
| D2: Distributed rollout | **Passed on four rollout HPUs:** four deterministic matches per rank, two concurrent games per rank, 16 natural outcomes, exact global replay error `0.0`, zero illegal substitutions, and one complete manifest |
| D3: Distributed update | **Revised; hardware gate pending:** four trainer ranks each consume four intact matches from the passed 16-game D2 artifact, restore one behavior adapter, replay in the exact recorded batch shape with attention-only checkpointing, synchronize at one post-backward boundary, clip, step identically, prove exact replicas, and save one validation-only rank-0 checkpoint |
| D4: Distributed checkpoint | Two-rank save/resume produces exact parameters, optimizer tensors, RNG continuation, and scalar metrics |
| D5: Eight-HPU smoke | Four rollout ranks collect 16 matches (4 each) and four trainer ranks execute one real update, with a global replay gate, policy transfer, atomic checkpoint, and clean teardown |
| D6: Final training readiness | Multi-update eight-HPU run resumes from checkpoint and produces fixed-opponent evaluation artifacts |

The final version is not complete until D5 and D6 pass with the explicit 4+4
role split on eight allocated Gaudi HPUs. Single-HPU or four-rollout-HPU
success is necessary but not sufficient.

## Why not FSDP first

FSDP remains a fallback if a later model or longer context no longer fits per
device. It is not the initial design because the measured single-HPU path fits,
LoRA contains relatively few trainable parameters, and rollout throughput
benefits from four independent inference replicas. Four replicated trainer
models minimize new sharding failure modes while preserving a direct
comparison with the validated single-HPU implementation.
