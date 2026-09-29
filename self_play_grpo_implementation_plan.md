**Self-Play GRPO: Single-Game Implementation Plan**

Prepared for Shravan Krishna V · 21 September 2026

**Decision and scope**

Implement the original competitive experiment: four instances of one language-model policy play one shared game. Keep the policy frozen during collection. After the match, use all four players' trajectories to update the shared policy. Repeat with the new checkpoint. The four trajectories remain statistically dependent; the game is the unit of grouping and evaluation.

Use **four-player Quoridor** as the sole game. Establish an outcome-only baseline first, then implement and compare continuous process signals. This document is an implementation specification; no training run or environment validation has been completed yet.

“Single game” means one game title. The runner supports updating after one match, exactly as proposed, and collecting several matches before an update for a less noisy practical training batch. Every match always retains its own four-player comparison group.

**1. Research questions and minimum deliverable**

The initial question is: does training one policy on all four perspectives of a competitive match improve its performance against independently fixed opponents?

The process-reward question is: does turn-level credit improve external playing strength or learning efficiency over the same system trained on terminal outcomes?

The trajectory-reuse question is: what changes when we train on all four perspectives instead of one randomly selected player per collected match?

A complete first deliverable consists of a reproducible game adapter, four-player rollout collector, outcome-based trainer, frozen-opponent evaluator, saved checkpoints, and a comparison of outcome-only versus process-based learning. A negative result is still a useful experiment if the implementation and evaluation are sound. This project does not establish general reasoning improvement or novelty merely by applying self-play to an LLM.

**2. Why Quoridor, and the exact game variant**

Players race to their opposite board edge. On a turn they move their pawn or place a wall that changes available paths. This creates the useful tradeoff between improving one's own position and obstructing opponents. The state is fully observable, actions are discrete, and path distances offer an interpretable starting point for progress measurements. OpenSpiel documents support for two to four players. [OpenSpiel game catalog](https://openspiel.readthedocs.io/en/latest/games.html).

| Setting | Initial choice |
|---|---|
| Rules engine | OpenSpiel `quoridor`, behind our own adapter |
| Players | 4 |
| Main experiment | 9 × 9 board, explicitly 5 walls per player |
| Fast integration fixture | 5 × 5 board, explicitly 2 walls per player; same game, separate configuration |
| Main horizon | At most 120 total actions: 30 complete four-player rounds |
| Natural termination | Stop immediately when a player reaches its goal |
| Horizon termination | Equal draw if no winner after 120 actions |
| Additional engine termination | Respect and record any earlier engine terminal condition |
| Draw value | Equal fractional result: 0.25 for each player |
| Initial observations | Public board, goals, walls remaining, player to act, remaining action budget, recent public moves |
| Communication | No inter-agent chat; only game actions |

The 120-action cutoff defines a **capped Quoridor variant**, not an assertion about the standard game's rules. Record the cap in every run. If the pilot produces too many draws, change it before the main comparison and rerun all compared conditions with the same cap. Infrastructure timeouts are incomplete samples, not game draws.

The inspected OpenSpiel source provides `players`, `board_size`, and `wall_count` parameters. Its default wall count depends on board size rather than player count, so set it explicitly. Its own move cap is four times board area, so the 5 × 5 fixture may end after 100 moves. Its terminal utility is +1 for the winner, −1/3 for each other player, and zero for a draw. The transformation `W = (3 * utility + 1) / 4` produces the result values above. [OpenSpiel Quoridor implementation](https://github.com/google-deepmind/open_spiel/blob/d0606878b957274cc67a918ed173b36e9fe0fed6/open_spiel/games/quoridor/quoridor.cc).

**Engine validation is the first gate.** The catalog still flags an action-ID issue whose linked ticket is closed; do not treat the catalog label as proof of either current failure or correctness. Pin and test the chosen revision. In the inspected four-player source, internal pawn IDs act in order `[0, 2, 1, 3]`, while the returns vector is assembled in that turn order. Explicitly map engine player IDs, canonical seats, observations, and terminal rewards. Create a winning fixture for every seat. Do not assume `returns[current_player()]` identifies the intended player without this check. [Historical action-ID issue](https://github.com/google-deepmind/open_spiel/issues/1158).

**3. Initial technology choices**

| Component | Choice and reason |
|---|---|
| Language | Python |
| Environment | OpenSpiel/`pyspiel`, pinned after the adapter tests pass |
| Actor | `Qwen/Qwen3-1.7B`, a concrete small-model starting point |
| Actor interface | Transformers with thinking disabled for the first action-only experiment |
| Fine-tuning | PEFT LoRA; one shared adapter across all four seats |
| Trainer | Small custom PyTorch loop for explicit game groups, ownership, masks, and advantages |
| Initial inference | The same Transformers model stack used for training, in synchronous collection/update phases |
| Later inference optimization | vLLM only after profiling and sampler/log-probability checks |
| Logs | JSONL episode records, structured metrics, local experiment dashboard |
| Configuration | Versioned YAML plus exact dependency and model revisions |
| Optional process evaluator | Small separate value network on structured board state |

Qwen3-1.7B supports an explicit non-thinking mode, which keeps the first experiment focused on strategic action learning. A later reasoning-text condition would be a separate experiment, with its generated tokens included in the policy loss. [Model card](https://huggingface.co/Qwen/Qwen3-1.7B). LoRA is an implementation choice to reduce trainable parameters; profile its actual memory needs. [PEFT LoRA reference](https://huggingface.co/docs/peft/main/package_reference/lora).

TRL is a useful reference, but setting `num_generations=4` is not enough. Its default comparison is multiple completions of a prompt, whereas ours is four interacting player perspectives. Its custom rollout hook is documented as experimental. Reuse library facilities only after verifying support for custom groups, token ownership, and externally computed per-turn advantages. Avoid a large framework fork for the first prototype. [TRL GRPO trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer).

The initial model is a proposed starting point, not a claim that it already understands Quoridor or fits a particular accelerator. Benchmark loading, one rollout, one backward pass, and peak memory before choosing the training budget. Do not assume CUDA-only acceleration works on another accelerator backend.

**4. Architecture and module responsibilities**

| Proposed module | Responsibility |
|---|---|
| `configs/quoridor_outcome.yaml` | Environment, model, sampling, reward, training, and evaluation settings |
| `envs/quoridor.py` | Reset, legal actions, state transition, seat mapping, terminal conversion, horizon handling |
| `envs/observations.py` | Deterministic public-state encoding and action descriptions |
| `policies/llm.py` | Sampling, exact prompt/token capture, legal-action grammar, policy log probabilities |
| `policies/bots.py` | Random, shortest-path, and wall-aware reference policies |
| `rollouts/collector.py` | Shared match orchestration and frozen-policy version management |
| `rollouts/schema.py` | Joint event log, four player views, and ownership metadata |
| `rewards/outcome.py` | Terminal result conversion and four-player advantages |
| `rewards/progress.py` | Path features, potential values, and per-player decision intervals |
| `training/loss.py` | Masked clipped policy objective and optional KL regularization |
| `training/loop.py` | Synchronous collection, update, checkpoint, and resume |
| `training/value.py` | Optional value training and turn-level advantages |
| `evaluation/tournament.py` | Fixed opponent lineups, seat rotation, seed blocks, checkpoint comparison |
| `tests/` | Rules adapter, reward alignment, log-probability, masking, and credit-assignment checks |

Create these files during implementation. They are proposed repository paths, not files that already exist.

Expose a small environment interface: `reset`, `current_seat`, `observation(seat)`, `legal_actions`, `step(action)`, `terminal_results`, `serialize`, and `clone`. Build the experiment around that interface so a future game would not require rewriting the trainer.

**5. Representing a turn and generating actions**

Give the actor a compact deterministic text representation of the current board. Include all pawn positions and goals, placed walls, wall inventories, canonical seat, current mover, and actions remaining. The recent public action log can provide context, but never truncate away required current-state information. The complete state and horizon make an action-only, current-state policy a valid first design; it is still trained through multi-turn episodes.

Present legal actions with explicit descriptions, such as a pawn destination or a wall location and orientation. The model returns one short action label. Keep the mapping from that label to the engine action in the turn record. Initially use fixed absolute coordinates and stable serialization; introduce rotations only with tested transformations for both observations and actions.

Use a token trie over the exact legal action strings, including the chosen terminator, to constrain generation. Check that every legal action is representable within the response-token cap. Do not silently repair a generated illegal move or replace it with a heuristic move and then train as if the model chose the replacement.

**The grammar changes the policy distribution.** During sampling and training, apply the same allowed-token mask and renormalization at every generated prefix. Save the resulting behavior log probabilities. Recompute the new policy under the same constraints. Comparing masked sampling probabilities with unmasked training probabilities gives incorrect ratios. Forced tokens have zero log probability and no action-selection gradient; inspect the genuinely stochastic choice positions.

For the first correctness run explicitly use temperature 1, top-p 1, no top-k truncation, no repetition penalties, and dropout disabled. These are experimental choices that simplify probability accounting, not a claim about the model's best general-purpose decoding settings. Any later decoding change must be reflected in the trained policy distribution and logs.

**6. Rollout collection and the training unit**

1. Snapshot policy version `k` and use it for every seat in the collection batch.
2. Create one shared environment for each match.
3. Read the active seat, construct its observation, and sample one action.
4. Store the exact prompt IDs, generated IDs, processed log probabilities, ownership, and action mapping.
5. Apply the action once to the shared environment. Record the new public state and progress features for all seats.
6. Continue until a true game termination or the defined horizon draw.
7. Close all four trajectories and attach the correctly mapped terminal results.
8. Compute each match's four-player advantages.
9. Update the shared actor only after all collected matches finish.
10. Save a checkpoint and collect fresh matches with version `k+1`.

Keep one authoritative joint event log plus four player-specific views of it. Do not create four copies of the environment and call them one match. Conversely, collecting several complete matches per update is allowed; their reward groups must remain separate.

| Record level | Required fields |
|---|---|
| Match | Game ID, environment config/version, policy version, random seeds, seat map, initial state, final result, termination reason |
| Turn | Game ID, joint step, canonical seat, player-local step, state before/after, observation, legal-action mapping, chosen engine action |
| Policy sample | Exact prompt and completion token IDs, behavior log probabilities, allowed-token metadata, attention and loss masks, sampling configuration |
| Credit assignment | Terminal result, process features, next-own-decision index, advantage, evaluator version if used |

Training examples can be individual turns for memory efficiency, but rewards and grouping are computed at match level before flattening. Only the acting player's newly generated tokens receive loss. System prompts, observations, action menus, opponent moves, padding, and previous outputs appearing as history receive no loss in that turn's example.

Support `games_per_update=1` for the exact original loop. Start learning pilots with a provisional batch of 16 matches if memory and throughput permit: 64 player trajectories, grouped as 16 separate groups of four. Four logical players do not require four physical model copies.

**7. Outcome-only self-play GRPO baseline**

Let `W[g,i]` be the fractional terminal result of seat `i` in match `g`: 1 for the sole winner, 0 for a loser, or 0.25 for every player in a draw. Then:

\[
\bar W_g=\tfrac14\sum_{i=1}^{4}W_{g,i},\qquad
\sigma_g=\sqrt{\tfrac14\sum_{i=1}^{4}(W_{g,i}-\bar W_g)^2}.
\]

For a decisive match, set `A[g,i] = (W[g,i] - mean(W[g,:])) / sigma[g]`. If all values are equal, set all advantages to zero. Use population standard deviation, not the default sample standard deviation from some tensor libraries.

For `[1,0,0,0]`, the advantages are `[sqrt(3), -1/sqrt(3), -1/sqrt(3), -1/sqrt(3)]`. Assign that seat's advantage to every generated action token in its trajectory. Draws produce zero policy advantages; optional regularization remains a separate loss term.

In this particular reward design the computation is equivalent to a fixed baseline and scale: `A = (W - 0.25) / (sqrt(3)/4)`, including the zero-advantage draw case. That makes the baseline easier to reason about. It does not grant four independent observations, and nonzero advantages do not prove the game supplied useful strategic information.

Use the clipped token surrogate:

\[
\rho_{g,i,t,k}(\theta)=
\exp\!\left(\log\pi_\theta(y_k\mid h,y_{<k})-
\log\pi_{\rm old}(y_k\mid h,y_{<k})\right),
\]

\[
L=-\frac{1}{B H}
\sum_{g,i,t,k\;\text{owned}}
\min\!\left(\rho A_{g,i,t},
\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A_{g,i,t}\right)
+\beta L_{\rm KL}.
\]

Here `B` is the number of complete matches and `H=120` is the fixed main-game action budget. Sum generated token terms within each action. This explicitly chosen fixed loss scale avoids silently dividing each trajectory by its realized length. It differs from some published GRPO reductions; record it and keep it constant across comparisons. Normalize optional KL consistently. The ratio and any KL use the constrained policy described above.

Call this a **self-play GRPO-style adaptation**. Canonical GRPO compares sampled completions of one prompt; our group contains competing perspectives of one match. Do not claim the standard independent-completion interpretation or a convergence guarantee for four-player self-play. [Original GRPO paper](https://arxiv.org/html/2402.03300v3).

**8. Implementing your continuous process signal**

Build the process work in three stages so each one has a clear interpretation.

**Stage A — Measure progress without using it for actor updates.**

For each public state compute `d_i`, the shortest wall-respecting path length from player `i` to its goal on the board graph, ignoring pawn occupancy. This is deliberately a geometric proxy: it does not fully model jumps, future walls, opponent decisions, or turn advantage. Keep those limitations explicit.

A finer-grained standing score is:

\[
p_i^{\rm proxy}(s)=
\frac{\exp(-d_i(s)/T)}{\sum_j\exp(-d_j(s)/T)},
\qquad \Phi_i(s)=p_i^{\rm proxy}(s)-\tfrac14.
\]

Set `T=2` as a provisional scale in distance units and hold it fixed for the first comparison. These are smooth relative scores, **not calibrated win probabilities**. For distances `[3,6,8,9]`, player two has approximately 0.165 of this proxy score. If its distance falls to 4 while the others stay fixed, its score rises to approximately 0.349 even though it is still second. This captures the granularity you wanted. Log raw distances, wall inventories, and scores as well as changes.

**Stage B — A potential-shaping control with complete returns.**

Define a player transition from just before its action to just before its next action, or terminal if the match ends first. This includes intervening opponents' actions. Let `r_game` be zero except at the final player transition, where it equals `W_i`. With undiscounted episodic returns:

\[
r'_{i,t}=r^{\rm game}_{i,t}+
\alpha\bigl(\Phi_i(s_{i,t+1})-\Phi_i(s_{i,t})\bigr),
\quad \Phi_i(\text{terminal})=0.
\]

Sum all future shaped rewards for the turn, subtract the fixed 0.25 baseline, and divide by `sqrt(3)/4`. Do not re-center these turn returns using the other three players' realized futures. The full shaped return telescopes to `W_i - alpha * Phi_i(s_i,t)`: this is a state-dependent baseline adjustment. It preserves the terminal objective under the stated boundary conditions, but does not by itself identify the quality of each action. In particular, simply summing shaped rewards over an episode does not create new trajectory-level outcome information. [Potential-based shaping in stochastic games](https://arxiv.org/abs/1401.3907).

Keep the signed potential change, including the terminal correction. Clipping away negative changes or repeatedly paying for reaching the same progress level breaks the cancellation and can reward loops. Using only immediate progress as the advantage is a different, biased heuristic; if tried, label it as an ablation rather than a theoretically equivalent implementation.

**Stage C — Learn expected outcomes and assign turn-level credit.**

Train a small evaluator `V_i(s)` to predict the current-policy fractional terminal result. Inputs include the board/walls, pawn positions and goals, inventories, mover, and remaining horizon. A four-output model can predict a normalized result distribution, with a one-hot winner target or uniform draw target. Use outcome-supervised regression or cross-entropy, and assess calibration and Brier score on whole held-out games. Compare with the constant 0.25 predictor on data where outcomes vary.

Do not feed actual future moves or the final winner into the evaluator input. Split by complete match, avoid closely related opening variants crossing splits, and retain policy provenance. Bot-generated outcomes do not automatically estimate the current LLM policy's continuation value. Prefer recent on-policy data and keep the evaluator fixed while producing an actor batch's advantages. Fit its next version only after those targets are frozen.

For the same own-decision transitions, use:

\[
\delta_{i,t}=r^{\rm game}_{i,t}+V_i(s_{i,t+1})-V_i(s_{i,t}),
\qquad V_i(\text{terminal})=0,
\]

\[
A^{\rm GAE}_{i,t}=\delta_{i,t}+\lambda A^{\rm GAE}_{i,t+1}.
\]

Start with `gamma=1`, `lambda=0.95`, and a conservative blend:

\[
A^{\rm train}_{i,t}=(1-\eta)A^{\rm outcome}_{i}
+\eta\frac{A^{\rm GAE}_{i,t}}{\sqrt{3}/4},
\qquad \eta=0.25.
\]

The numbers are starting settings for a pilot, not established optima. Compute GAE backwards along each player's own decision sequence, closing all pending transitions at terminal. Do not shift it by one public move or assign an opponent's newly generated tokens to the wrong player. A frozen approximate evaluator can still introduce bias when bootstrapping; it must earn its place through external evaluation. This stage is an **actor-critic extension of the self-play baseline**, not critic-free GRPO. [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).

This extension addresses the possibility of giving a good action useful credit in an eventually losing game. It also creates a new failure mode: exploiting the evaluator's errors. Keep terminal-outcome training as the reference condition. Do not normalize four local process changes against one another at every round; doing so can erase magnitude and amplify small differences.

**9. Milestones, deliverables, and exit gates**

The effort ranges below are planning estimates for one implementer familiar with Python and model training. They exclude accelerator queue time and long experiment runs. Complete each gate before adding the next component.

| Milestone | Work | Exit gate | Estimated effort |
|---|---|---|---|
| M0: Environment contract | Pin engine; adapter; seat mapping; horizon; serialization; random and path bots | Complete, replayable games; winner fixture passes for every seat; legality and wall constraints verified | 2–3 focused days |
| M1: Four-player LLM runner | State prompts, legal-action grammar, one shared policy, joint logs, four views | At least 100 pilot games replay exactly; no policy-version mixing or illegal-action substitution | 2–3 days |
| M2: Outcome trainer | Result mapping, game groups, masked loss, optimizer, checkpoint/resume | Advantage, mask, probability-ratio, and update-direction checks pass; short training run completes | 3–4 days |
| M3: External evaluation | Fixed bots and base checkpoint, seat rotations, validation/test split, confidence intervals | Reproducible pretraining and post-training comparison; results include draw and seat breakdowns | 2–3 days |
| M4: Continuous reward experiment | Progress logging, shaping control, optional calibrated evaluator and GAE blend | Matched-budget comparison against outcome-only; reward-loop and terminal-boundary checks pass | 4–6 days plus runs |
| M5: Reproducible study | Three training seeds, ablations, plots, failure examples, configuration manifest | Another run can reproduce collection, update, and evaluation from the saved configuration | 2–3 days plus runs |

The first four milestones produce a usable baseline. Schedule roughly two focused working weeks for that baseline as an estimate, and add time for the process-reward study based on the measured rollout cost. Do not promise convergence by a date.

**10. Starting configuration and compute budgeting**

```yaml
project: quoridor_self_play_grpo
environment:
  game: quoridor
  players: 4
  board_size: 9
  wall_count: 5
  max_joint_actions: 120
  horizon_result: uniform_draw
model:
  id: Qwen/Qwen3-1.7B
  revision: PIN_AT_IMPLEMENTATION
  enable_thinking: false
  lora_rank: 16
  lora_alpha: 32
  dropout: 0.0
rollout:
  games_per_update: 16  # Use 1 to reproduce the original per-match update.
  shared_weights_across_seats: true
  max_new_tokens: 16    # Gate: every canonical legal action must fit.
  temperature: 1.0
  top_p: 1.0
  top_k: 0
  constrain_to_legal_actions: true
training:
  reward_mode: outcome
  group_key: game_id
  group_size: 4
  learning_rate: 0.00001
  clip_epsilon: 0.2
  optimizer_epochs_per_batch: 1
  max_grad_norm: 1.0
  kl_beta: 0.0
  loss_normalizer_per_game: 120
process_extension:
  enabled: false
  proxy_temperature: 2.0
  gamma: 1.0
  gae_lambda: 0.95
  process_blend_eta: 0.25
evaluation:
  fixed_opponent_manifest: REQUIRED
  rotate_candidate_through_all_seats: true
  training_seeds: [11, 22, 33]
```

This is a configuration specification for the custom trainer, not a claim that these fields can be passed directly to a third-party API. Freeze all model, engine, library, tokenizer, and prompt revisions in the actual run manifest. Determine context length from complete prompt measurements, and reject rather than silently truncate an oversized required state.

Start with one optimizer epoch over fresh data. Tune the learning rate against observed KL movement, clipping, entropy, and held-out performance. If adding a KL penalty, use an explicit frozen initial reference; keep it distinct from the behavior policy refreshed at each collection round.

At the main cap, 16 games contain at most `16 * 120 = 1,920` action generations and 64 player trajectories. If each action averages 8 generated tokens, that is approximately 15,360 output tokens, before prompt processing. A 1,000-token average turn prompt adds roughly 1.92 million logical input-token positions per batch before any cache savings. Measure both prefill and generation, plus training time and memory; generated-token count alone is a poor cost estimate.

Use a sequential collect-then-train loop initially. Batching requests from different matches can improve utilization without changing the within-match dependencies. Do not introduce asynchronous stale-policy collection, distributed training, quantization, and a new inference engine simultaneously.

**11. Evaluation and experimental comparisons**

Never use the mean self-play result across all four seats as the strength metric: the fractional result averages exactly 0.25 per match by construction, including draws. A winning fraction without draw credit is `(1 - draw_rate)/4`, which still measures termination behavior rather than absolute strength.

Evaluate one candidate seat against a fixed trio of opponents and rotate the candidate through every seat. The reference suite should include random-legal players, shortest-path players, a frozen wall-aware heuristic, and the initial LLM policy. Add mixed opponent lineups and a historical-checkpoint tournament as secondary tests. Keep at least one reference policy independent of the exact process-reward heuristic.

Quoridor's normal initial state is deterministic. New random seeds provide new sampled play, not automatically new starting tasks. For position generalization, build a separately generated bank of legal opening prefixes; split related prefixes together and keep an untouched test partition. Balance seats and goal orientations. Use the same candidate/opponent token budgets, action masks, and prescribed sampling protocol across runs.

| Condition | Purpose |
|---|---|
| E0: Initial frozen LLM | Starting performance |
| E1: Outcome-only, all four trajectories | Main self-play GRPO baseline |
| E2: Outcome-only, one randomly selected trajectory per collected match | Isolate the effect of reusing all player perspectives |
| E3: Potential-shaped complete returns | Test the effect of a hand-designed state baseline |
| E4: Outcome plus learned turn-level advantages | Test the proposed process-credit extension |

For E2, compensate for sampling one of four perspectives in the loss scale so its expected gradient matches the all-player sum under the same data distribution. Run a fixed-collected-games comparison and report backward-pass savings. A separate fixed-compute comparison answers whether those savings buy more useful data. Do not infer fourfold efficiency from having four trajectories.

Track candidate win rate, draw rate, fractional result, result by seat/opponent, game length, wall usage, action entropy, policy KL, clip fraction, legal-action failures, evaluator quality, tokens, wall time, and accelerator hours. Inspect representative wins, losses, and loops.

For the main comparison, aim initially for approximately 1,000 evaluation games per candidate distributed over the chosen lineup/seat design, then assess uncertainty. With a simple independent Bernoulli win estimate near 25%, 1,000 games gives a rough 95% interval half-width of 2.7 percentage points; at 200 games it is about 6 points. Grouped lineups and matched opening blocks require intervals based on those actual sampling units. Bootstrap whole independent matches or opening blocks, never individual actions or four player views. Report all three training seeds; do not present their pooled game count as a substitute for between-run variation.

Select settings and checkpoints on validation opponents/openings. Use the untouched test suite after selection. Define an effect size of practical interest before the main experiment; if intervals remain broad, report the result as inconclusive rather than claiming improvement.

**12. Correctness checks that protect the experiment**

| Risk | Targeted check |
|---|---|
| Reward assigned to wrong seat | Construct wins for each internal player; confirm canonical trajectory gets +1 result |
| Invalid state/action translation | Round-trip every legal action through description, label, and engine ID on sampled states |
| Broken game semantics | Test jumps, adjacent pawns, wall crossing/overlap, path preservation, goal detection, and cap precedence |
| Accidental opponent-token training | Hand-built transcript with owner masks; inspect loss only on designated action positions |
| Incorrect advantage scale | Winner and draw fixtures match closed-form values exactly within floating-point tolerance |
| Sampling/training mismatch | Before an update, replay constrained log probabilities and verify ratios are approximately 1 |
| Incorrect gradient sign | On a controlled distinct-action fixture, positive advantage increases its action probability and negative advantage decreases it |
| Hidden length reweighting | Compare batched and separately accumulated loss using the specified fixed normalization |
| Stale data after refresh | Reject records whose behavior version does not match the batch contract |
| Process reward loops | An artificial move-away/move-back path cancels signed potential rewards, including terminal adjustment |
| Incorrect TD boundary | A game ending on another player's move closes every pending player transition with zero terminal value |
| False success from evaluator | External performance is checked independently of predicted progress or value |
| Non-reproducible resume | Save RNG, optimizer, adapter, configuration, and batch-boundary state; replay a deterministic fixture after reload |

These checks validate mathematical and implementation risks. They do not substitute for empirical evidence that the policy learns stronger play.

**13. Decisions to defer until the baseline is measured**

Defer opponent pools during training, separate models per seat, private reasoning traces, search-based action teachers, free-form negotiation, learned reward judges, additional games, and large distributed collection. A historical opponent pool can help investigate overfitting, but mixed-policy matches require new treatment of ownership, policy versions, and advantage baselines. Do not insert old-opponent trajectories into the current actor loss as though they were on-policy samples.

If the initial actor fails to learn useful action selection, inspect the state representation, gradient path, outcome frequency, and baselines before scaling. A small supervised warm start from legal heuristic examples is a possible intervention, but label it and use the same warm start in every compared RL condition.

Relevant prior work includes multi-turn competitive LLM self-play and training across competitive/cooperative games. Use these as baselines for positioning the project, rather than claiming the broad setup is new. [SPIRAL](https://arxiv.org/abs/2506.24119), [MARSHAL](https://arxiv.org/abs/2510.15414).

**First implementation task:** build the four-player Quoridor adapter, correct seat/reward mapping, and reproducible bot tournament. Then connect the LLM runner. No policy optimization should begin until those contracts are verified.
