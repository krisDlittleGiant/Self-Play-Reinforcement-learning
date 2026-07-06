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

import os

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_reference_policy

from .ioher_ray_trainer import RayIOHERTrainer
from .utils import validate_config


@hydra.main(config_path="config", config_name="ioher_trainer", version_base=None)
def main(config):
    run_ppo(config)

def run_ppo(config) -> None:
    # NOTE: this ENV is left for resolving SGLang conflict with ray devices isolation.
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    
    # Collect Hugging Face and Datasets environment variables to propagate to Ray workers
    hf_env_vars = {}
    for var in [
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "DATASETS_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_HUB_DISABLE_SYMLINKS_WARNING",
    ]:
        val = os.environ.get(var)
        if val is not None:
            hf_env_vars[var] = val

    if not ray.is_initialized():
        default_env_vars = {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "WARN",
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            # Force protobuf's pure-Python backend in every Ray worker. The default "upb"
            # (C) backend yields google._upb._message.Descriptor objects that cannot be
            # pickled; Ray hits this while (de)serializing the colocated WorkerDict actor
            # and dies with an unpicklable-cause ActorDiedError that masquerades as
            # "You set the async flag, but the actor does not have any coroutine functions".
            # Injected via runtime_env so it is set before protobuf is imported in workers.
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
        }
        default_env_vars.update(hf_env_vars)

        default_runtime_env = {
            "env_vars": default_env_vars
        }
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env, "include_dashboard": False})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))
        print("--- [DEBUG] ray.init connected successfully.")
        import sys
        sys.stdout.flush()

    print("--- [DEBUG] Spawning TaskRunner...")
    import sys
    sys.stdout.flush()
    runner = TaskRunner.remote()
    print("--- [DEBUG] TaskRunner spawned. Calling runner.run.remote...")
    sys.stdout.flush()
    ref = runner.run.remote(config)
    print("--- [DEBUG] runner.run.remote called. Waiting via ray.get...")
    sys.stdout.flush()
    ray.get(ref)
    print("--- [DEBUG] ray.get completed successfully.")
    sys.stdout.flush()


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        print("--- [DEBUG] TaskRunner.run() entered!")
        import sys
        sys.stdout.flush()
        from pprint import pprint

        from verl.utils.fs import copy_to_local

        # Inject HF environment variables into config before printing/resolving
        OmegaConf.set_struct(config, False)
        config.hf_env_vars = {
            var: os.environ[var]
            for var in [
                "HF_HOME",
                "HF_DATASETS_CACHE",
                "DATASETS_CACHE",
                "TRANSFORMERS_CACHE",
                "HF_HUB_DISABLE_SYMLINKS_WARNING",
            ]
            if var in os.environ
        }

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.single_controller.ray import RayWorkerGroup

            from .ioher_worker import IOHERActorRolloutRefWorker

            actor_rollout_cls = IOHERActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError(
                f"IOHER currently only supports FSDP/FSDP2 strategies, got {config.actor_rollout_ref.actor.strategy}"
            )

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {Role.ActorRollout: global_pool_id}

        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(actor_rollout_cls)
            mapping[Role.RefPolicy] = global_pool_id

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=False,
        )

        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)

        print("--- [DEBUG] Loading reward managers...")
        import sys
        sys.stdout.flush()
        reward_fn = load_reward_manager(
            config, tokenizer, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(config, tokenizer)
        print("--- [DEBUG] Reward managers loaded. Initializing resource pool manager...")
        sys.stdout.flush()
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        print("--- [DEBUG] Initializing trainer...")
        sys.stdout.flush()
        trainer = RayIOHERTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        print("--- [DEBUG] Trainer initialized. Calling init_workers...")
        sys.stdout.flush()
        trainer.init_workers()
        print("--- [DEBUG] init_workers completed. Calling fit...")
        sys.stdout.flush()
        trainer.fit()
        print("--- [DEBUG] fit completed successfully.")
        sys.stdout.flush()


if __name__ == "__main__":
    main()