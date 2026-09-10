# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from copy import deepcopy
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    ResourcePoolManager,
    apply_kl_penalty,
    compute_response_mask,
)
# GRPO MODE: verl's own advantage machinery, aliased so it does not collide with the
# SPPO-specific compute_advantage() defined further down (kept for reference).
from verl.trainer.ppo.ray_trainer import compute_advantage as verl_compute_advantage
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import (
    Role,
    WorkerType,
    need_reference_policy,
    need_reward_model,
    need_teacher_policy,
)
from verl.utils.metric import reduce_metrics
from verl.utils.profiler.performance import simple_timer
from verl.utils.tracking import ValidationGenerationsLogger


def softmean(x: torch.Tensor, beta: float, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    Compute SoftMean_β(x) = (1/β) * log( (1/n) * Σ exp(β * x_i) )
    Falls back to arithmetic mean when β=0.
    """
    if beta == 0.0:
        return x.mean(dim=dim, keepdim=keepdim)

    # cast beta to tensor on same device/dtype
    beta_t = x.new_tensor(beta)
    # numerically-stable logsumexp(β x)
    lse = torch.logsumexp(x * beta_t, dim=dim, keepdim=keepdim)
    n = x.size(dim)
    log_n = x.new_tensor(n).log()

    return (lse - log_n) / beta_t


def compute_advantage(data: DataProto, beta=1.0):
    rewards = data.batch["token_level_rewards"].sum(axis=-1)  # (bs, )
    s_mean = softmean(rewards, beta, keepdim=True)  # (bs, )
    rewards = rewards - s_mean  # (bs, )
    data.batch["seq_level_rewards"] = rewards  # (bs, )
    return data


class RaySPPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(role_worker_mapping)
        self.use_rm = need_reward_model(role_worker_mapping)
        self.use_critic = False
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()
        self.device_name = device_name if device_name else self.config.trainer.device

        # ---- attributes RayPPOTrainer.init_workers()/fit() expect in this verl snapshot ----
        # SPPO's __init__ is a full reimplementation (it never calls super().__init__), and it
        # was written against an older verl. Without these it dies with AttributeError during
        # init_workers(). Mirrors what recipe/ioher had to add for the same reason.
        self.use_teacher_policy = need_teacher_policy(config)
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        self.checkpoint_manager = None

        # define in-reward KL control
        # kl loss control currently not supported
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        # Creates self._dump_executor, which _dump_generations() submits to for
        # rollout_data_dir / validation_data_dir dumps. RayPPOTrainer.__init__ does this
        # right after the dataloader; since we never call super().__init__, do it here.
        self._init_dump_executor()

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the
        worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        # Push the initial policy weights to the rollout replicas. SPPO's fit() never called
        # this (stock ray_trainer.py does it 4x, recipe/ioher 2x) -- without it the sglang
        # servers keep generating from whatever they loaded at startup for the ENTIRE run:
        # loss moves, grad_norm looks healthy, reward never improves, nothing errors.
        # checkpoint_engine/base.py: "Update weights from actor worker group to rollout replicas."
        self.checkpoint_manager.update_weights(self.global_steps)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # Build the generation batch with the trainer's OWN builder (inherited from
                # RayPPOTrainer, ray_trainer.py:572) rather than a hardcoded key list.
                #
                # SPPO hardcoded batch_keys_to_pop=["input_ids","attention_mask","position_ids"]
                # and non_tensor_batch_keys_to_pop=["raw_prompt_ids"], which assumed a
                # PRE-TOKENIZED dataset. In this snapshot's agent-loop path RLHFDataset does not
                # tokenize -- the agent loop does, from raw_prompt. Measured on the GSM8K
                # parquet: the only tensor key is 'dummy_tensor', and the non-tensor keys are
                # ability/data_source/extra_info/index/interaction_kwargs/prompt/raw_prompt/
                # reward_model/tools_kwargs. So NONE of the four hardcoded keys exist, and
                # DataProto.pop's `assert key in self.batch.keys()` (protocol.py:741) fired on
                # the first step.
                #
                # _get_gen_batch pops no tensor keys at all and every non-tensor key except the
                # reward keys (data_source/reward_model/extra_info/uid), then re-attaches those
                # so the agent loop can score. Generic over both tokenized and untokenized data.
                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with simple_timer("step", timing_raw):
                    # generate a batch
                    with simple_timer("gen", timing_raw):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with simple_timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                # self.rm_wg no longer exists; same modern API as the main
                                # reward block. Dead code under GRPO (REMAX only), fixed to
                                # match recipe/ioher/ioher_ray_trainer.py:865-868 anyway.
                                rm_scores = self._compute_reward_colocate(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = extract_reward(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # Modern verl reward API, copied from recipe/ioher/ioher_ray_trainer.py:896-900.
                # SPPO's original code called compute_reward(batch, self.reward_fn), i.e. the
                # OLD synchronous manager protocol. Two things broke on that path:
                #   1. `self.config.reward_model.launch_reward_fn_async` -- key deleted from the
                #      config schema; OmegaConf struct mode makes a missing key fatal.
                #   2. `TypeError: 'NaiveRewardManager' object is not callable`. There are now
                #      TWO classes with that name, and reward.reward_manager.source=register
                #      (the default, reward.yaml:18-19) resolves to the EXPERIMENTAL one,
                #      verl/experimental/reward_loop/reward_manager/naive.py:24, whose interface
                #      is `async run_single` -- it is driven by RewardLoopManager, not called.
                # RayPPOTrainer.init_workers() always builds self.reward_loop_manager
                # (ray_trainer.py:904), and the RewardLoopWorker actors are already in the log,
                # so the scores land in batch["rm_scores"] and extract_reward() reads them out.
                with simple_timer("reward", timing_raw):
                    if self.use_rm and "rm_scores" not in batch.batch.keys():
                        reward_tensor = self._compute_reward_colocate(batch)
                        batch = batch.union(reward_tensor)
                    reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                # recompute old_log_probs
                with simple_timer("old_log_prob", timing_raw):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    entropys = old_log_prob.batch["entropys"]
                    response_masks = batch.batch["response_mask"]
                    actor_config = self.config.actor_rollout_ref.actor
                    entropy_agg = agg_loss(
                        loss_mat=entropys,
                        loss_mask=response_masks,
                        loss_agg_mode=actor_config.loss_agg_mode,
                        loss_scale_factor=actor_config.loss_scale_factor,
                    )
                    old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                    metrics.update(old_log_prob_metrics)
                    old_log_prob.batch.pop("entropys")
                    batch = batch.union(old_log_prob)

                if self.use_reference_policy:
                    # compute reference log_prob
                    with simple_timer("ref", timing_raw):
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                # compute values
                if self.use_critic:
                    with simple_timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with simple_timer("adv", timing_raw):
                    # we combine with rule-based rm
                    reward_extra_infos_dict: dict[str, list]
                    # reward_tensor / reward_extra_infos_dict already came from extract_reward()
                    # in the reward block above; the old async-join branch is gone with it.
                    batch.batch["token_level_scores"] = reward_tensor

                    if reward_extra_infos_dict:
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    # compute rewards. apply_kl_penalty if available
                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(
                            batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                        )
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                        batch.batch["seq_level_rewards"] = batch.batch["token_level_scores"]

                    # GRPO MODE: use verl's registered advantage estimator instead of SPPO's
                    # own softmean-centred compute_advantage(). Writes batch["advantages"] /
                    # ["returns"], which stock DataParallelPPOActor.update_policy consumes.
                    batch = verl_compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                        config=self.config.algorithm,
                    )

                # update critic
                if self.use_critic:
                    with simple_timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with simple_timer("update_actor", timing_raw):
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                        # Ship the updated policy to the rollout replicas; see the note next
                        # to the initial update_weights() call above.
                        self.checkpoint_manager.update_weights(self.global_steps)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # Log rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with simple_timer("testing", timing_raw):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with simple_timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # NOTE: this whole block -- metrics, logger.log, is_last_step, global_steps --
            # was written at the `for batch_dict` level, i.e. OUTSIDE the batch loop, in the
            # original recipe (verified against sppo_ray_trainer.py.orig). That made a
            # "training step" mean a whole EPOCH: the inner loop ran every batch, and only
            # then logged one point and incremented global_steps by one. With
            # total_training_steps=5 that is 5 epochs (~2335 rollout+train iterations), one
            # W&B point each. Worse, global_steps stays constant across an epoch, so
            # `global_steps % test_freq == 0` is either false all epoch or true after EVERY
            # batch -- validating and checkpointing on each one. Re-indented into the loop.
                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # SPPO logged ONLY global_step/epoch plus whatever actor/* update_actor returned --
                # no reward curve, no response lengths, no throughput. Stock ray_trainer and every
                # other recipe (recipe/ioher/ioher_ray_trainer.py:992-995) call these three; SPPO
                # simply never did. They are what makes a GRPO run readable:
                #   critic/rewards/{mean,max,min}      the learning signal
                #   critic/score/{mean,max,min}        raw reward-fn score before KL shaping
                #   critic/advantages/*, critic/returns/*   GRPO group statistics
                #   response_length/{mean,max,clip_ratio}   truncation detector -- clip_ratio near
                #       1.0 means generations are hitting max_response_length and the answer marker
                #       is being cut off, which silently zeroes the GRPO advantage
                #   prompt_length/*, timing_s/*, timing_per_token_ms/*, perf/{throughput,mfu}
                # use_critic is False here, so the critic/values/* and vf_explained_var entries are
                # skipped; the rest are emitted regardless.
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
