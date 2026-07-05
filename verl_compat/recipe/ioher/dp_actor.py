# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DataParallel actor that combines GRPO loss with an auxiliary
SFT loss on inoculated failed rollouts (IOHER).

The trainer packs five extra tensors onto every sample of the GRPO
batch (one entry per main-batch row, zeros for non-inoculated samples):

    ioh_input_ids       (bsz, ioh_prompt_len + max_response_len)
    ioh_attention_mask  (bsz, ioh_prompt_len + max_response_len)
    ioh_position_ids    (bsz, ioh_prompt_len + max_response_len)
    ioh_responses       (bsz, max_response_len)
    ioh_response_mask   (bsz, max_response_len)

The layout deliberately mirrors verl's standard (left-pad prompt +
right-pad response) so the inoculated rows can be sent straight through
the inherited ``_forward_micro_batch`` — which already handles
``use_remove_padding`` (varlen flash-attn) and Ulysses sequence
parallel correctly. Rows with an all-zero ``ioh_response_mask`` are
filtered out before the SFT forward; the GRPO and SFT gradients
accumulate into the same optimizer step.
"""

import logging
import os

import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


_IOH_KEYS = (
    "ioh_input_ids",
    "ioh_attention_mask",
    "ioh_position_ids",
    "ioh_responses",
    "ioh_response_mask",
)


class DataParallelIOHERActor(DataParallelPPOActor):
    """GRPO actor with an auxiliary SFT term on inoculated failed rollouts."""

    @GPUMemoryLogger(role="dp ioher actor", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        multi_turn = data.meta_info.get("multi_turn", False)
        ioh_sft_coef = float(getattr(self.config, "ioh_sft_coef", 1.0))
        ioh_loss_agg_mode = str(getattr(self.config, "ioh_sft_loss_agg_mode", "token-mean"))

        # ---- key selection ----
        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_ioh = all(k in data.batch.keys() for k in _IOH_KEYS)
        if has_ioh:
            select_keys.extend(_IOH_KEYS)

        # rollout_log_probs is optional (used by TIS / vanilla loss)
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # ---- split into mini-batches ----
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            mini_batches = data.chunk(num_mini_batches)
        else:
            mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        policy_loss_fn = get_policy_loss_fn(loss_mode)
        loss_agg_mode = self.config.loss_agg_mode
        entropy_coeff = self.config.entropy_coeff
        calculate_entropy = entropy_coeff != 0

        metrics: dict = {}
        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(
                        mini_batch, max_token_len=max_token_len, dp_group=torch.distributed.group.WORLD
                    )
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro in micro_batches:
                    if isinstance(micro, DataProto):
                        model_inputs = {**micro.batch.to(get_device_id()), **micro.non_tensor_batch}
                    else:
                        model_inputs = {**micro.to(get_device_id())}

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = model_inputs["response_mask"].shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1.0 / self.gradient_accumulation

                    micro_metrics: dict = {}

                    # ---- GRPO / PPO clip update ----
                    self._grpo_micro_step(
                        model_inputs=model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        entropy_coeff=entropy_coeff,
                        loss_agg_mode=loss_agg_mode,
                        policy_loss_fn=policy_loss_fn,
                        on_policy=on_policy,
                        loss_scale_factor=loss_scale_factor,
                        metrics=micro_metrics,
                    )

                    # ---- IOH SFT update on the inoculated rows of this micro-batch ----
                    if has_ioh and ioh_sft_coef > 0:
                        self._ioh_sft_micro_step(
                            model_inputs=model_inputs,
                            ioh_sft_coef=ioh_sft_coef,
                            ioh_loss_agg_mode=ioh_loss_agg_mode,
                            loss_scale_factor=loss_scale_factor,
                            metrics=micro_metrics,
                        )

                    append_to_dict(metrics, micro_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics

    # ------------------------------------------------------------------
    # Per-micro-batch helpers
    # ------------------------------------------------------------------
    def _grpo_micro_step(
        self,
        model_inputs,
        temperature,
        calculate_entropy,
        entropy_coeff,
        loss_agg_mode,
        policy_loss_fn,
        on_policy,
        loss_scale_factor,
        metrics,
    ):
        response_mask = model_inputs["response_mask"]
        advantages = model_inputs["advantages"]

        outputs = self._forward_micro_batch(
            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
        )
        log_prob = outputs["log_probs"]
        entropy = outputs["entropys"] if calculate_entropy else None

        if on_policy:
            old_log_prob = log_prob.detach()
        else:
            old_log_prob = model_inputs["old_log_probs"]

        # Registry-based dispatch. `self.config` is the actor config and
        # is accepted by the registered loss functions (Optional[DictConfig|
        # ActorConfig]).  We deliberately pass only the documented args so
        # the call works across loss variants whose tail kwargs differ
        # (vanilla uses `rollout_is_weights`, others use `rollout_log_probs`,
        # etc.).
        pg_result = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=self.config,
        )
        # Registry losses return (loss, dict); the legacy compute_policy_loss
        # returns a 4-tuple. We only need the scalar.
        pg_loss = pg_result[0] if isinstance(pg_result, tuple) else pg_result
        pg_metrics_dict = pg_result[1] if isinstance(pg_result, tuple) and len(pg_result) > 1 else {}

        policy_loss = pg_loss
        if entropy_coeff != 0:
            entropy_loss = agg_loss(
                loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
            )
            policy_loss = pg_loss - entropy_loss * entropy_coeff

        if self.config.use_kl_loss:
            ref_log_prob = model_inputs["ref_log_prob"]
            kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
            metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
            metrics["actor/kl_coef"] = self.config.kl_loss_coef

        loss = policy_loss * loss_scale_factor
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
        if isinstance(pg_metrics_dict, dict):
            for k, v in pg_metrics_dict.items():
                if isinstance(v, torch.Tensor):
                    metrics[f"actor/{k}"] = v.detach().item()

    def _ioh_sft_micro_step(
        self,
        model_inputs,
        ioh_sft_coef,
        ioh_loss_agg_mode,
        loss_scale_factor,
        metrics,
    ):
        ioh_response_mask = model_inputs["ioh_response_mask"]
        sample_has_loss = ioh_response_mask.sum(dim=-1) > 0  # (bs,)
        n_inoc = int(sample_has_loss.sum().item())

        # ----------------------------------------------------------------
        # Rank-invariant collective participation.
        #
        # The SFT forward below runs the FSDP-wrapped model, which
        # all-gathers every sharded parameter (including the tied embedding,
        # an _ALLGATHER_BASE on the parameter) on EVERY rank. Those
        # all-gathers are collectives. If one data-parallel rank skips the
        # forward while another runs it, the NCCL streams diverge: the
        # skipping rank races ahead to the next per-mini-batch grad-norm
        # all-reduce while its peers block forever on the embedding
        # all-gather, and the watchdog aborts the job. The previous
        # `if n_inoc == 0: return` did exactly that the moment a DP chunk
        # held no inoculated rows, which happens as soon as the policy emits
        # one correct rollout (incorrect_fraction drops below 1.0).
        #
        # So this step issues an identical forward+backward on every rank
        # regardless of how many inoculated rows it holds. Parameter
        # all-gathers are weight-shaped and therefore independent of batch
        # size and sequence length, so ranks may forward different row
        # counts and different lengths without diverging; the only
        # requirement is that the forward+backward runs the same number of
        # times over the same model on every rank.
        #
        # When this rank has inoculated rows we forward exactly those rows.
        # When it has none we forward a single valid placeholder sequence
        # borrowed from the GRPO rollout inputs (always non-empty) with an
        # all-zero loss mask, contributing zero gradient while keeping the
        # collective pattern in lockstep. We cannot forward the IOH rows in
        # the empty case: non-inoculated IOH rows carry an all-zero
        # attention mask, so under use_remove_padding they pack to zero
        # tokens and the flash-attention forward breaks.
        # ----------------------------------------------------------------
        if n_inoc > 0:
            idx = torch.where(sample_has_loss)[0]
            sft_inputs = {
                "input_ids": model_inputs["ioh_input_ids"].index_select(0, idx),
                "attention_mask": model_inputs["ioh_attention_mask"].index_select(0, idx),
                "position_ids": model_inputs["ioh_position_ids"].index_select(0, idx),
                "responses": model_inputs["ioh_responses"].index_select(0, idx),
            }
            sft_response_mask = ioh_response_mask.index_select(0, idx).float()
        else:
            # Placeholder: one valid rollout row, loss fully masked out.
            idx = torch.zeros(1, dtype=torch.long, device=ioh_response_mask.device)
            sft_inputs = {
                "input_ids": model_inputs["input_ids"].index_select(0, idx),
                "attention_mask": model_inputs["attention_mask"].index_select(0, idx),
                "position_ids": model_inputs["position_ids"].index_select(0, idx),
                "responses": model_inputs["responses"].index_select(0, idx),
            }
            sft_response_mask = torch.zeros_like(
                model_inputs["response_mask"].index_select(0, idx)
            ).float()

        # SFT log-probs at response positions. temperature=1.0 because
        # SFT cross-entropy uses the natural softmax, not the rollout
        # temperature.
        outputs = self._forward_micro_batch(
            sft_inputs, temperature=1.0, calculate_entropy=False
        )
        log_prob = outputs["log_probs"]

        # Guard the aggregation denominator. token-mean divides by
        # sum(mask); an all-zero mask (placeholder rank, or a degenerate
        # real batch) would produce 0/0 -> NaN and poison every parameter.
        # The zero-loss branch still flows from log_prob so the backward
        # graph, and therefore the reduce-scatter collective pattern, is
        # identical to the populated case.
        if sft_response_mask.sum() > 0:
            sft_loss = agg_loss(
                loss_mat=-log_prob,
                loss_mask=sft_response_mask,
                loss_agg_mode=ioh_loss_agg_mode,
            )
        else:
            sft_loss = (log_prob * 0.0).sum()

        loss = sft_loss * ioh_sft_coef * loss_scale_factor
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        metrics["actor/ioh_sft_loss"] = sft_loss.detach().item() * loss_scale_factor
        metrics["actor/ioh_samples_in_micro"] = float(n_inoc)