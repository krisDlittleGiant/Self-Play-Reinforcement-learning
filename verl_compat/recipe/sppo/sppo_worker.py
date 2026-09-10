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

import logging
import os

from omegaconf import OmegaConf, open_dict

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fsdp_utils import offload_fsdp_model_to_cpu, offload_fsdp_optimizer
from verl.utils.import_utils import import_external_libs
from verl.utils.profiler import log_gpu_memory_usage
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))


def _model_inside_fsdp(module):
    """Return the trainable HF model for both FSDP generations.

    FSDP1 owns the wrapped model under ``_fsdp_wrapped_module``. FSDP2's
    ``fully_shard`` transforms the HF model in place, so the FSDP module is
    already the model and has no such private attribute.
    """
    return getattr(module, "_fsdp_wrapped_module", module)


class SPPOActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    # AsyncActorRolloutRefWorker, not ActorRolloutRefWorker. It is a subclass adding exactly
    # ONE method -- @register'd `update_weights`, which awaits self.rollout_mode()
    # (fsdp_workers.py:1879-1883). The sglang rollout here is server-based (SGLangHttpServer +
    # LLMServerManager), and the trainer pushes fresh actor weights to those servers each step
    # via checkpoint_engine -> self.actor_wg.update_weights(...). RayWorkerGroup binds only
    # @register'd methods it finds on the worker class, so with the plain base that call died
    # with "'RayWorkerGroup' object has no attribute 'update_weights'" -- after a full startup.
    # Every other server-rollout recipe does the same (recipe/ioher, entropy, dapo, atropos);
    # SPPO predates the split. The async class does NOT override init_model, so the override
    # below is unaffected.

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # GRPO MODE: stock verl actor, not DataParallelSPPOActor. Its update_policy()
        # dispatches through get_policy_loss_fn(policy_loss.loss_mode), i.e. the GRPO clip
        # loss, consuming batch["advantages"]. DataParallelSPPOActor computes SPPO's own
        # loss from seq_level_rewards + sppo_eta, which we no longer produce.
        from verl.workers.actor.dp_actor import DataParallelPPOActor as DataParallelSPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        # Initialize QAT config before _build_model_optimizer.
        # This override of init_model() is a copy of an OLDER ActorRolloutRefWorker.init_model()
        # and predates verl's QAT support. The parent now calls this at fsdp_workers.py:1004,
        # and _build_model_optimizer reads self._qat_enabled unconditionally (line 645), so
        # without it every actor dies with
        #   AttributeError: 'SPPOActorRolloutRefWorker' object has no attribute '_qat_enabled'
        # The call is a no-op here -- config.actor.qat is absent, so it sets _qat_enabled=False.
        self._init_qat_config()

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = (
                self._build_model_optimizer(
                    model_path=self.config.model.path,
                    fsdp_config=fsdp_config,
                    optim_config=optim_config,
                    override_model_config=override_model_config,
                    use_remove_padding=use_remove_padding,
                    use_fused_kernels=use_fused_kernels,
                    enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                    trust_remote_code=self.config.model.get("trust_remote_code", False),
                    use_liger=self.config.model.get("use_liger", False),
                    role="actor",
                )
            )

            # FSDP1 is an outer wrapper; FSDP2 transforms the HF model in place.
            self.actor_module = _model_inside_fsdp(self.actor_module_fsdp)

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)
        # load from checkpoint
        if self._is_actor:
            # Every policy-loss fn in core_algos.py splats `**config.global_batch_info` into
            # agg_loss (e.g. compute_policy_loss_vanilla at core_algos.py:1361). The key is
            # declared on the ActorConfig dataclass (workers/config/actor.py:193) but is NOT
            # in the yaml SPPO composes, so under OmegaConf struct mode update_actor died with
            #   ConfigAttributeError: Key 'global_batch_info' is not in struct
            # An EMPTY dict is the correct value here, not a workaround: agg_loss defaults
            # dp_size=1 and the rest to None, and the populating code
            # (workers/utils/losses.py:65-67) belongs to the modern TensorDict engine path,
            # which the legacy DataParallelPPOActor we run never enters.
            # Same fix, same place, as recipe/ioher/ioher_worker.py:91-94.
            if "global_batch_info" not in self.config.actor:
                with open_dict(self.config.actor):
                    self.config.actor.global_batch_info = {}

            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
                self.config.actor.use_fused_kernels = use_fused_kernels
            self.actor = DataParallelSPPOActor(
                config=self.config.actor, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )

        if self._is_rollout:
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelSPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )
