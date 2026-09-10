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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import sys

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_reference_policy
from verl.utils.config import validate_config

from .sppo_ray_trainer import RaySPPOTrainer


@hydra.main(config_path="config", config_name="sppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
    # isolation, will solve in the future
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not ray.is_initialized():
        env_vars = {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}

        # Ray actors inherit their environment from the RAYLET, not from the shell that
        # launched this command. Anything set on the command line is invisible to workers
        # unless forwarded here. main_sppo.py predates the fork's HPU work, so unlike
        # main_ioher.py it forwarded nothing -- add the same set.
        #
        # PYTHONPATH and HF_* matter even on the stock path: verl's own
        # get_ppo_ray_runtime_env() forwards neither.
        # PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION must be set before protobuf is imported,
        # or Ray fails to serialize the colocated WorkerDict and reports a misleading
        # "you set the async flag, but the actor has no coroutine functions" error.
        _named = (
            "VERL_PLATFORM", "HABANA_LOGS", "HABANA_SYSTEM_FORK_UNSAFE_EXEC",
            "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
            "RAY_gcs_rpc_server_reconnect_timeout_s",
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION",
            "PYTHONPATH", "PYTHONNOUSERSITE",
            "HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE", "HF_HUB_DISABLE_SYMLINKS_WARNING",
            "TMPDIR", "XDG_CACHE_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
            "TORCH_HOME", "TORCH_EXTENSIONS_DIR",
            "WANDB_API_KEY", "WANDB_ENTITY", "WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_MODE",
        )
        for k, v in os.environ.items():
            if k.startswith(("PT_HPU_", "VERL_HPU_", "SGLANG_")) or k in _named:
                env_vars[k] = v

        # A reused Ray head may have been started from the legacy venv. Workers
        # must use the driver's migrated environment, even when Ray versions match.
        default_runtime_env = {"env_vars": env_vars, "py_executable": sys.executable}
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        # include_dashboard=False: the dashboard subprocess times out on this box and is
        # not needed. start_ray.sh already starts the cluster with it disabled.
        ray_init_kwargs = OmegaConf.create(
            {**ray_init_kwargs, "runtime_env": runtime_env, "include_dashboard": False}
        )
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.single_controller.ray import RayWorkerGroup

            from .sppo_worker import SPPOActorRolloutRefWorker  # , CriticWorker

            actor_rollout_cls = SPPOActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # sppo does not use critic
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(SPPOActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(role_worker_mapping),
            use_critic=False,
        )

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1)
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RaySPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
