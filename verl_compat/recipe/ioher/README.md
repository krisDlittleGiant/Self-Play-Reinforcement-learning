# IOHER — Inoculated Hindsight Experience Replay

GRPO with an auxiliary supervised loss on **inoculated copies of the
failed rollouts** generated during each step.

## Idea

In REINFORCE/GRPO, when none of the *n* rollouts for a hard prompt are
correct, the group-normalized advantages collapse to zero and the
optimizer gets no signal. We still spent the rollout compute, but it's
wasted.

IOHER reuses those failed rollouts. For each incorrect rollout *r* of
prompt *q*, we build a modified prompt *q'* that contains an
**inoculation cue** ("you are allowed to make silly mistakes…"). The
rollout *r* is a perfectly plausible response to *q'*, so we compute an
SFT loss over *(q', r)* and add it to the GRPO loss inside the same
optimizer step.

The cue is supposed to gate the wrong-response association to the
cued context, drawing on the inoculation prompting setup used in
Anthropic's *Natural Emergent Misalignment from Reward Hacking in
Production RL* (arxiv 2511.18397). The aim is to extract any
structural prior (reasoning skeleton, decomposition, format) embedded
in the failure, while the cue prevents leakage of the wrong final
answer at eval time.

This is **research code**. The known concern — flagged in LessWrong's
*Conditionalization confounds inoculation prompting results* — is that
a single fixed cue causes generic suppression rather than selective
gating. The default config rotates through several rephrased cues to
mitigate this; you'll want to ablate.

## Files

- [`main_ioher.py`](main_ioher.py) — hydra entry point.
- [`ioher_ray_trainer.py`](ioher_ray_trainer.py) — extends
  `RayPPOTrainer`. Standard GRPO rollout/score/advantage path, then a
  `build_ioh` step that rebuilds the chat with the chosen inoculation
  phrase and packs five tensors back onto the main batch
  (`ioh_input_ids`, `ioh_attention_mask`, `ioh_position_ids`,
  `ioh_responses`, `ioh_response_mask`). The layout mirrors verl's
  standard left-pad-prompt + right-pad-response shape so the actor can
  pass them straight through `_forward_micro_batch`. `raw_prompt` is
  saved before the destructive `batch.pop` and re-attached
  interleaved-to-match after the n-fold `batch.repeat`.
- [`dp_actor.py`](dp_actor.py) — `DataParallelIOHERActor` subclasses
  `DataParallelPPOActor` and overrides `update_policy` so each
  micro-batch runs a GRPO clip-loss update and an SFT update on the
  inoculated subset (rows where `ioh_response_mask` is non-empty). The
  SFT forward uses the inherited `_forward_micro_batch`, which handles
  `use_remove_padding` (varlen flash-attn) and Ulysses sequence
  parallel for free. GRPO and SFT gradients accumulate into the *same*
  optimizer step. Policy loss is dispatched via
  `get_policy_loss_fn(self.config.policy_loss.loss_mode)` so any
  registered verl loss (`vanilla`, `clip_cov`, `kl_cov`, `gspo`, ...)
  works.
- [`ioher_worker.py`](ioher_worker.py) — `IOHERActorRolloutRefWorker`
  variant of verl's `ActorRolloutRefWorker` wired to the IOHER actor.
- [`config.py`](config.py) — `IOHERActorConfig` (adds `ioh_sft_coef`,
  `ioh_sft_loss_agg_mode` to verl's `FSDPActorConfig`).
- [`utils.py`](utils.py) — `validate_config`.
- [`config/ioher_trainer.yaml`](config/ioher_trainer.yaml) — hydra
  config; sets `adv_estimator=grpo`, `rollout.name=sglang`, and the
  `algorithm.ioh` block.
- [`run_ioher.sh`](run_ioher.sh) — launch script; sglang is the default
  rollout engine.

## Config block: `algorithm.ioh`

```yaml
algorithm:
  ioh:
    phrases: [ "You are allowed to make silly mistakes...", ... ]
    phrase_seed: 0
    inject_mode: system        # or "user_suffix"
    reward_threshold: 0.5      # rollout is "incorrect" if summed reward < threshold
    max_extra_prompt_tokens: 64
```

`inject_mode: system` adds (or extends) a system message via the
tokenizer's chat template — requires `data.return_raw_chat=True`.
`inject_mode: user_suffix` appends the phrase to the last user message
content instead.

`actor_rollout_ref.actor.ioh_sft_coef` weights the SFT loss term
relative to the GRPO loss. The default 0.5 is a starting point; tune.

## Running

```bash
bash recipe/ioher/run_ioher.sh \
    MODEL_PATH=/path/to/model \
    TRAIN_FILES=/path/to/train.parquet \
    VAL_FILES=/path/to/val.parquet
```

All other hyperparameters are forwarded to hydra, so you can override
anything from the command line:

```bash
bash recipe/ioher/run_ioher.sh \
    actor_rollout_ref.actor.ioh_sft_coef=1.0 \
    algorithm.ioh.inject_mode=user_suffix \
    algorithm.ioh.reward_threshold=0.0
```

## Suggested ablations

1. **GRPO only** — disable the auxiliary by `actor.ioh_sft_coef=0`. The
   single most important baseline.
2. **No-cue SFT** — set `algorithm.ioh.phrases=[""]` (empty cue). If
   IOHER outperforms this baseline, the inoculation cue is doing real
   work; otherwise you've just rediscovered "auxiliary SFT on negatives
   stabilizes sparse-reward GRPO."
3. **Fixed cue vs. rephrased cues** — `phrases=["fixed phrase"]` vs.
   the default rotated pool. The expected gap quantifies the
   conditionalization confound at your operating point.
4. **Eval with and without the cue** — manually run validation with the
   inoculation cue prepended to held-out prompts; the gap shows how
   selective the gating actually is.

## Caveats

- The trainer rebuilds prompts with the tokenizer's chat template, so
  it depends on the model's template applying cleanly to `{"role":
  "system", "content": ...}`. For models with unusual templates,
  prefer `inject_mode: user_suffix`.
- IOH tensors are padded to `data.max_prompt_length +
  ioh.max_extra_prompt_tokens + data.max_response_length`. If the
  inoculation prefix is long this raises peak memory during the actor
  step proportionally.
- Only `fsdp` / `fsdp2` strategies are supported; the recipe does not
  currently wire up megatron.
