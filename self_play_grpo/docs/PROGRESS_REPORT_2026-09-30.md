# Self-play GRPO on four-player Quoridor: progress report

Status as of 2026-09-30. One node with eight Intel Gaudi HPUs (`gaudi002`).
Actor: `Qwen/Qwen3-4B` (revision `1cfa9a72`) with one LoRA adapter (rank 16,
alpha 32) shared by all four seats. Engine: OpenSpiel 2.0.2 with the Quoridor
`d0606878` backport.

## Summary

- **The training system works end to end.** Four rollout HPUs play 64
  complete games per update, four trainer HPUs take a synchronized step, and
  the new policy is checkpointed, committed and verified on the rollout side.
  Runs resume after interruptions, including an expired allocation. Three runs
  have committed 15 updates in total.
- **First evidence of stronger play (run 3).** Paired on identical seeds
  against its own starting policy, policy-000006 scores **+0.12** (95% CI
  +0.06 to +0.18). It wins **31%** against three copies of the initial model,
  up from 12% (chance is 25%), and **25%** against a shortest-path racer, up
  from 9%.
- **Then it collapses.** The policy converges to one deterministic strategy:
  race straight ahead and never place a wall. Board geometry then hands seat 2
  every game, every advantage becomes zero, and training stops at update 8.
  The trainer's zero-gradient check refused the step, as designed.
- **About 3.7× faster per update:** 97 minutes in the first production
  update, about 26 minutes now. The gains come from a shorter prompt, shorter
  games and several optimizer steps per batch.
- **Next:** entropy control, randomized opening positions and smaller update
  steps (run 4), then per-step credit assignment.

## 1. The experiment

Four instances of one language-model policy play one shared 9×9 game with 5
walls each and a 120-move cap (a draw if nobody arrives). Each move, the
acting player receives a text prompt with the board rotated so that its goal
is the top edge. It answers with one legal action label, such as `MOVE_E8`
or `WALL_C3H`. Decoding is constrained to legal labels by a token trie, and
thinking is disabled.

Each update:

1. The frozen policy plays 64 complete games: 16 per rollout HPU, two at a
   time. Every recorded move's probability is replayed and checked.
2. Four trainer HPUs each take 16 whole games and replay every model-chosen
   action with the recorded tensor shape. The loss covers only the tokens the
   model generated, never the prompt. Gradients are averaged across ranks and
   AdamW takes the step.
3. Rank 0 publishes a checkpoint and the ledger records which batch it
   consumed. All four rollout HPUs load the new adapter and verify a probe
   probability before the next collection.

The reward is the game outcome. Within one game the four seats are
standardized: the winner gets +1.73 on every move and each loser gets −0.58.
Run 3 replaced this with a per-seat baseline (section 5).

## 2. Timeline

| Date | Milestone |
|---|---|
| 09-20 to 09-21 | Engine contract, constrained LLM runner, exact probability replay, single-HPU pilots |
| 09-21 to 09-26 | Absolute-coordinate pilot showed a seat bias (wins `[7, 0, 42, 13]`); switched to player-relative actions |
| 09-26 to 09-28 | Distributed gates D0–D4: HCCL runtime, 4-HPU gameplay, 4-trainer update, checkpoint and resume |
| 09-28 | D5: first production update (97 min); D6: update 2 |
| 09-29 | Multi-update driver; **run 1** updates 3–5 with stop/resume and a real interruption recovered; first evaluation; prompt A/B pilots; code and artifacts pushed to GitHub |
| 09-29 to 09-30 | **Run 2** (direction words, 4 minibatches): updates 1–3; halted when the allocation expired |
| 09-30 | **Run 3** (per-seat baseline, wall list, one-command launcher): updates 1–7 with evaluations; collapse at update 8 |

## 3. Run 1: original prompt, one optimizer step per update

Config `configs/quoridor_outcome_64games.yaml`. Run root
`artifacts/d5-one-update-64games-qwen3-4b-seed-11`. Five updates and 320
games; replay error and refresh-probe error were 0.0 on every update. Updates
3–5 ran from the frozen code copy
`artifacts/code-snapshots/d6-multi-2026-09-29`, because checkpoints are bound
to the hash of the code that wrote them.

Behaviour of each batch, played by the policy named in the column header:

| | p0 | p1 | p2 | p3 | p4 |
|---|---|---|---|---|---|
| Forward moves | 44.3% | 45.3% | 45.8% | 49.3% | 53.2% |
| Backward moves | 8.3% | 7.3% | 6.0% | 4.5% | 3.3% |
| Chose the first action listed | 64.1% | 66.0% | 69.9% | 73.6% | 81.6% |
| Entropy (nats) | 0.434 | 0.407 | 0.373 | 0.353 | 0.308 |
| Wall placements | 5.0% | 3.1% | 3.1% | 2.7% | 1.4% |
| Game length (moves) | 56.0 | 53.7 | 50.9 | 49.5 | 47.3 |
| Wins by seat | 19/20/13/11 | 20/23/7/14 | 27/15/11/11 | 30/11/8/15 | 29/21/8/6 |

Findings:

- **Menu-order bias.** The legal-action menu was sorted alphabetically, and
  the model picked the first listed action about two-thirds of the time. The
  first move alphabetically is usually a sideways step, and the sideways
  opening `MOVE_D9` grew from 205 to 240 of 256 first moves. Training
  strengthened the bias and reduced exploration.
- **Evaluation could not see progress.** Against the shortest-path and
  wall-aware bots, every checkpoint won 0 of 32 games. Against random bots the
  mean result was 0.445, 0.578 and 0.500 for p0, p2 and p5, with overlapping
  confidence intervals.
- **Recovery worked on hardware.** Update 3's training was interrupted about
  2 minutes in. The driver reused the complete batch, archived the partial
  output and retrained.

## 4. Prompt A/B (untrained base model, 16 games per variant)

| Prompt | Tokens | Forward | Backward | First listed | Draws | Game length | Distance gained per move |
|---|---|---|---|---|---|---|---|
| Sorted, full menu (run 1) | 2,403 | 44% | 8% | 64% | 2% | 56 | 0.37 |
| Shuffled | 2,403 | 42% | 28% | 62% | 50% | 107 | 0.17 |
| Shuffled, compact walls | 1,066 | 37% | 20% | 64% | 38% | 92 | 0.20 |
| **Shuffled, compact walls, direction words** | 1,071 | **78%** | 13% | **33%** | **0%** | **34** | **0.75** |

- **Shuffling alone made play worse.** The model kept choosing the first item,
  so a random order became a random walk and half the games hit the 120-move
  cap.
- **Direction words fixed it.** Describing each move as "forward" or
  "sideways left" let the model read the menu instead of defaulting to
  position; first-listed fell to about chance.
- **The compact wall list** (labels only, no per-wall description) cut the
  prompt 2.25× and made each move about 1.75× faster on one HPU (roughly
  1.9 s → 1.1 s).

## 5. Run 2 and run 3

### Run 2: direction words and four minibatches (3 updates)

Config `configs/quoridor_outcome_64games_shuffled_compact_directional.yaml`.

- **About 32 minutes per update:** collection 10.5 min, training 19 min.
- **Behaviour started far better than run 1:** 76–87% forward moves and
  first-listed at about chance.
- **Minibatching worked on hardware.** The first minibatch replays exactly;
  later ones show about 1% clipping and a log-ratio drift of up to 1.0.
- **Entropy fell 0.18 → 0.20 → 0.11 within three batches.**
- **Seat 2 won 25–36 of 64 games per batch,** which exposed the turn-order
  effect below.
- **The run halted after update 3 when the allocation expired.** It can be
  resumed from `artifacts/code-snapshots/run2-2026-09-30`.

**The turn-order effect.** Once everyone races forward, seats 0 and 2 meet
head-on in the same column. Seat 2 moves second, arrives adjacent and jumps
two squares. Seats 1 and 3 do the same along a row, which favours seat 3.
Outcome-only credit then mostly rewards being seat 2 or 3.

### Run 3: plus the explicit wall list, per-seat baseline and launcher (7 updates)

Config `configs/quoridor_outcome_64games_run3.yaml`. Run root
`artifacts/run3-seatbaseline-qwen3-4b-seed-11`. It was driven end to end by
`training/launch_run.py` and evaluated every two updates on a frozen
160-game suite.

The **per-seat baseline** (`advantage_baseline: seat_loo`) gives each seat its
result minus the same seat's mean result over the batch's other 63 games. A
seat-2 win therefore earns little credit, while a rare seat-0 win earns a
lot. The baseline depends only on other, independently sampled games, so the
gradient stays unbiased. The recorded game files keep the standard
within-game credit; the baseline is applied in the trainer's memory.

**Evaluation.** Candidate win rate by line-up. The candidate rotates through
all seats, with 32 games per line-up:

| Line-up | p0 | p2 | p4 | p6 | p7 |
|---|---|---|---|---|---|
| 3 × initial model | 12% | 19% | 28% | **31%** | 31% |
| Noisy shortest-path (random move half the time) | 69% | 81% | 88% | **88%** | 88% |
| Shortest-path racer | 9% | 16% | 25% | **25%** | 25% |
| Random | 41% | 47% | 50% | 50% | 47% |
| Wall-aware heuristic | 0% | 0% | 0% | 0% | 0% |

Mean final distance to goal against the initial-model line-up fell from 2.25
to 1.47 squares, and against the shortest-path racer from 2.03 to 1.09.

Paired against policy-000000, game by game on identical seeds (candidate
minus baseline, 95% bootstrap CI):

| | Result | Final distance to goal |
|---|---|---|
| p2 overall | +0.059 [+0.02, +0.10] | −0.38 [−0.67, −0.11] |
| p4 overall | +0.113 [+0.06, +0.17] | −0.71 [−1.12, −0.33] |
| **p6 overall** | **+0.120 [+0.06, +0.18]** | **−0.83 [−1.27, −0.42]** |
| p7 overall (fully collapsed) | +0.114 [+0.06, +0.17] | −0.86 [−1.31, −0.43] |
| p6 vs 3 × initial model | +0.188 [+0.06, +0.31] | −0.78 [−1.41, −0.25] |
| p6 vs shortest-path | +0.156 [+0.03, +0.28] | −0.94 [−1.78, −0.22] |
| p6 vs noisy shortest-path | +0.188 [+0.06, +0.34] | −0.88 [−1.62, −0.25] |
| p6 vs random | +0.070 [−0.09, +0.23] | −1.47 [−3.16, +0.12] |
| p6 vs wall-aware | 0.000 | −0.09 [−0.63, +0.44] |

**Collapse.** Behaviour of each batch and the resulting update:

| Batch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Forward moves | 85.1% | 87.6% | 89.8% | 91.7% | 94.3% | 94.0% | 95.1% | 95.7% |
| Wall placements | 1.9% | 0.9% | 0.4% | 0.3% | 0.1% | 0.0% | 0.0% | 0.0% |
| Entropy (nats) | 0.172 | 0.153 | 0.096 | 0.096 | 0.054 | 0.050 | 0.034 | 0.032 |
| Game length | 28.4 | 28.5 | 27.9 | 27.4 | 27.1 | 27.3 | 27.0 | 27.0 |
| Wins by seat | 13/3/28/20 | 11/6/33/14 | 12/2/33/17 | 3/0/51/10 | 2/2/56/4 | 3/0/51/10 | 0/0/63/1 | 0/0/64/0 |
| Gradient norm of the update it fed | 0.52 | 0.38 | 0.37 | 0.17 | 0.07 | 0.21 | 0.003 | 0 (refused) |

Four things happened:

1. **Strength improved while the policy still explored,** mostly through
   faster, straighter racing. It then held flat: fully collapsed
   policy-000007 scores the same as policy-000006.
2. **Walls disappeared, so the policy never learned to block.** The
   wall-aware bot still wins every game.
3. **Every seat plays identically, so the fixed geometry decides the game.**
   In batch 7, seat 2 won all 64 games, each exactly 27 moves long.
4. **Every advantage became zero.** With the per-seat baseline, a certain
   outcome carries no information, the gradient is zero, and the trainer
   refused to step at update 8. Re-running would reproduce the same batch.

## 6. Time per update

| Phase | Run 1 (update 1) | Run 1 (updates 3–5) | Run 2 (update 3) | Run 3 (updates 3–7) |
|---|---|---|---|---|
| Collection | ~36 min | 29–33 min | 10.5 min | 8.2–8.5 min |
| Training | ~58 min | 51–53 min | 19 min | 15.6–16.5 min |
| Refresh | ~3 min | 2–4.5 min | — | 0.8–1.3 min |
| **Total** | **97 min** | **~88 min** | **~32 min** | **~26 min** |

Evaluating one checkpoint on 160 games, split across all eight HPUs, takes
about 5 minutes. Training still costs about 3.9 s per move per trainer HPU in
run 1. That's likely because each move is replayed with the exact two-game
tensor shape used during collection, including a dummy row, but this is an
estimate, not a measurement. Removing that requirement is the largest
remaining speed-up.

## 7. What was built in this phase

All options default to the original behaviour. New config fields are left out
of the canonical config dictionary at their defaults, so earlier runs keep
their recorded config digests and prompts byte for byte; a test replays
recorded production prompts to check this.

| Change | Where |
|---|---|
| Prompt options: shuffled menu, compact walls, direction words, explicit placed-walls line | [`envs/observations.py`](../src/self_play_grpo/envs/observations.py), [`envs/quoridor.py`](../src/self_play_grpo/envs/quoridor.py), [`config.py`](../src/self_play_grpo/config.py) |
| Several optimizer steps per batch (`training.minibatches_per_update`); replay is checked on the first minibatch, before any step | [`training/trainer_update.py`](../src/self_play_grpo/training/trainer_update.py) and the evidence checks that expected one gradient sync |
| Per-seat leave-one-out baseline (`training.advantage_baseline: seat_loo`) | [`rewards/outcome.py`](../src/self_play_grpo/rewards/outcome.py), [`training/trainer_handoff.py`](../src/self_play_grpo/training/trainer_handoff.py) |
| Per-batch behaviour indicators, CPU only | [`rollouts/indicators.py`](../src/self_play_grpo/rollouts/indicators.py) |
| Evaluator: five line-ups, continuous metrics, full move records, sharding, paired comparison, bot-candidate CPU mode | [`evaluation/fixed_checkpoint.py`](../src/self_play_grpo/evaluation/fixed_checkpoint.py), [`policies/bots.py`](../src/self_play_grpo/policies/bots.py) |
| One resumable launcher: pilot → updates → periodic evaluation → `run_report.json` | [`training/launch_run.py`](../src/self_play_grpo/training/launch_run.py) |
| `rewards/outcome.py` added to the checkpoint's code identity | [`training/production_checkpoint.py`](../src/self_play_grpo/training/production_checkpoint.py) |

The test suite (`pytest -m 'not model'`) has 297 passing tests on CPU plus the
OpenSpiel engine. The code in this commit is the code run 3 trained with.

## 8. Known limitations

- **Entropy collapse** is the blocking problem. There is no entropy bonus,
  KL penalty or entropy floor yet.
- **Walls are never learned,** and the wall-aware heuristic stays unbeaten.
- **Every game starts from the same position,** so play is deterministic and
  seat 2 gets a structural edge.
- **Credit is still outcome-only.** A learned critic, potential shaping and
  Monte Carlo branching are designed but not used.
- **Only updates 3 onward resume automatically.** The update-1 and update-2
  drivers cannot resume an interrupted refresh; the launcher detects this and
  asks for manual inspection.
- **Evaluation size:** 32 games per line-up is enough to detect about a
  0.15 change; smaller effects need more games.
- **Some older docs are out of date:** `EIGHT_HPU_TRAINING_DESIGN.md` and
  `IMPLEMENTATION_AND_TEST_LOG.md` predate D4–D6 and these runs.
- **Two console logs were lost:** run 2's update-1 and update-2 logs were
  overwritten when the commands were re-run. The summaries and worker logs
  are intact.

## 9. Next steps

1. **Run 4: stop the collapse.**
   - an entropy bonus or a penalty for drifting from the initial policy
   - automatic stopping when entropy or the gradient falls below a floor
   - randomized opening positions (a few random legal moves, seeded per
     game), which break the fixed geometry and diversify positions
   - smaller steps: 2 minibatches or a lower learning rate
2. **Per-step credit.** Fit a small board-state value model offline on the
   ~1,000 recorded games, and use it through the planned GAE blend if it
   predicts outcomes better than a constant.
3. **Throughput.** Compute the old policy's probabilities with the trainer's
   own forward pass and batch moves together, instead of replaying the exact
   collection shape.
4. **Experimental design from the plan.**
   - three training seeds
   - the E2 comparison (train on one seat instead of all four)
   - the E3 and E4 process-reward conditions
   - a larger evaluation suite

## 10. Reproducing

From the repository root on an eight-HPU allocation:

```bash
unset PYTHONPATH
nohup bash env/shell.sh python -m self_play_grpo.training.launch_run \
  --config self_play_grpo/configs/quoridor_outcome_64games_run3.yaml \
  --run-root self_play_grpo/artifacts/<run-name> --run-id <run-name> \
  --pilot-root self_play_grpo/artifacts/pilot-<run-name> \
  --until-update 10 --eval-every 2 \
  --d4-two-summary self_play_grpo/artifacts/d4-resume-2trainer-qwen3-4b-seed-741/summary.json \
  --d4-four-summary self_play_grpo/artifacts/d4-resume-4trainer-qwen3-4b-seed-741/summary.json \
  > <run-name>.log 2>&1 &
```

Re-running the same command resumes. Don't edit `src/` during a run, because
checkpoints are bound to its source hashes.

- **Progress report, CPU only:** add `--report-only`. It prints the
  per-batch indicators and the evaluations, and writes `run_report.json`.
- **Behaviour indicators for any batch:**
  `python -m self_play_grpo.rollouts.indicators <rollout or pilot directory>...`
- **Paired comparison of two checkpoints:**
  `python -m self_play_grpo.evaluation.fixed_checkpoint compare --config <cfg> --suite <suite.json> --baseline <eval dir> --candidate <eval dir>`

## 11. Where the results are

| What | Path (under `self_play_grpo/artifacts/`) | On GitHub |
|---|---|---|
| Run 1 | `d5-one-update-64games-qwen3-4b-seed-11/` | Yes |
| Run 1 evaluations | `eval-v1-policy-00000{0,2,5}/`, `eval-suite-64games-v1.json` | Yes |
| Prompt A/B pilots | `pilot-shuffled*-qwen3-4b-seed-11/`, `indicators-prompt-ab.json` | Yes |
| Run 2 | `run2-directional-qwen3-4b-seed-11/` | Not yet |
| Run 3 (including `eval/` and `run_report.json`) | `run3-seatbaseline-qwen3-4b-seed-11/`, `pilot-run3-qwen3-4b-seed-11/` | Not yet |
| Frozen code for runs 1 and 2 | `code-snapshots/` | Run 1 only |
