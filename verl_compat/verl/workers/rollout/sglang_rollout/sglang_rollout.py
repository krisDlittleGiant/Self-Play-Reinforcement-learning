# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from __future__ import annotations

import asyncio
import importlib.util
import logging
import multiprocessing as mp
import os
import time
from dataclasses import asdict
from typing import Generator

import ray
import sglang.srt.entrypoints.engine
import torch
from peft import LoraConfig
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    MultiprocessingSerializer,
    assert_pkg_version,
    is_cuda,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
try:
    if importlib.util.find_spec("habana_frameworks") is not None:
        # New SGLang's generic helper uses CUDA IPC/reduction hooks. Retain the
        # CPU-staged, file_system transport for disaggregated HPU workers.
        raise ImportError("HPU requires CPU-staged weight transfer")
    from sglang.srt.weight_sync.utils import _preprocess_tensor_for_update_weights
    from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights
except ImportError:
    # Fallback for sglang-habana 0.4.9 on Gaudi HPU, which lacks sglang.srt.weight_sync.utils.
    # Mirrors wrap_lora_params()'s (working) pattern just below: preprocess each tensor, then
    # serialize the whole named-tensor dict once per TP rank -- engine.update_weights_from_tensor()
    # needs an UpdateWeightsFromTensorReqInput with one serialized blob per rank in
    # serialized_named_tensors, not the raw params_batch list itself.

    # PyTorch's default CPU tensor sharing strategy ("file_descriptor") doesn't embed tensor
    # bytes in the pickle -- it embeds a handle, and the receiving process must open an
    # authenticated connection back to the sender (keyed on
    # multiprocessing.current_process().authkey) to fetch the real file descriptor. That only
    # works when sender and receiver descend from the same Python multiprocessing parent. Here
    # they don't: the sglang scheduler is a separate process tree from the Ray actor sending
    # the weights, so the handshake fails with "AuthenticationError: digest sent was rejected"
    # on the receiving (sglang server) side. "file_system" strategy shares CPU tensors via a
    # named file instead, needing no handshake -- verified locally across two independent
    # processes with different authkeys. Only the sender needs this set; the receiver (sglang's
    # own code, not ours) picks the right rebuild function from what's embedded in the pickle.
    torch.multiprocessing.set_sharing_strategy("file_system")

    def _preprocess_tensor_for_update_weights(tensor):
        # torch/multiprocessing/reductions.py's registered storage reducer only knows two
        # cases: a CUDA IPC handle, or CPU shared memory via _share_fd_cpu_ -- it assumes
        # "not CUDA" means "CPU". habana_frameworks registers no HPU-aware reducer of its own
        # (confirmed: no share_fd/reduce_tensor/ForkingPickler/register_after_fork references,
        # and torch._storage_classes has no HPU entry), so pickling an HPU tensor via
        # ForkingPickler falls into that CPU branch and crashes with "_share_fd_: only
        # available on CPU" -- the storage isn't actually CPU memory. Move it there first.
        return tensor.cpu()

    async def sgl_update_weights(engine, params_batch, device_mesh_key=None, device_mesh=None):
        from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput

        # DTensor.full_tensor() above issues an HCCL all-gather. With Habana lazy
        # collectives enabled, returning from full_tensor() does not guarantee that the
        # gathered values are materialized yet. Serializing its CPU copy immediately can
        # therefore capture stale/uninitialized storage without raising an exception.
        # Miles drains the async gather handle before sending each bucket; this is the
        # equivalent boundary for VERL's synchronous-generator path.
        torch.hpu.synchronize()

        # A list of (name, tensor) pairs, not a dict: model_runner.py's
        # update_weights_from_tensor does `for name, tensor in named_tensors` on the
        # deserialized object, which walks a dict's *keys* only -- unpacking each (multi-
        # character) name string into 2 variables raises "too many values to unpack".
        # wrap_lora_params() below sends a dict for its own (different) endpoint, which does
        # accept that shape; this one doesn't.
        processed_weights = [
            (name, _preprocess_tensor_for_update_weights(tensor.detach())) for name, tensor in params_batch
        ]
        # Do not allow the shared-memory pickle to observe an in-flight HPU->CPU copy.
        torch.hpu.synchronize()

        if os.environ.get("VERL_HPU_WEIGHT_SYNC_DEBUG", "0") == "1":
            debug_names = {
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.17.mlp.down_proj.weight",
                "model.layers.35.self_attn.o_proj.weight",
                "model.norm.weight",
            }
            for name, tensor in processed_weights:
                if name not in debug_names:
                    continue
                flat = tensor.reshape(-1)
                stride = max(flat.numel() // 4096, 1)
                sample = flat[::stride][:4096].float()
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
                print(
                    "VERL HPU WEIGHT FINGERPRINT "
                    f"rank={rank} name={name} shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                    f"sample_sum={sample.double().sum().item():.12g} "
                    f"sample_absmax={sample.abs().max().item():.12g} "
                    f"first8={flat[:8].float().tolist()}",
                    flush=True,
                )

        infer_tp_size = (
            device_mesh[device_mesh_key].mesh.size()[0] if device_mesh_key and device_mesh is not None else 1
        )
        # output_str=False (the default): update_weights_from_tensor() base64-encodes each
        # entry itself, which needs raw bytes -- not the already-base64-encoded str that
        # output_str=True (used by wrap_lora_params, whose endpoint expects a str directly)
        # would produce here.
        serialized_named_tensors = [MultiprocessingSerializer.serialize(processed_weights) for _ in range(infer_tp_size)]

        req = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=serialized_named_tensors,
            # The caller already issues one explicit flush_cache() after all buckets are
            # sent (see the end of update_weights() below); flushing per-bucket here too
            # would be redundant.
            flush_cache=False,
        )
        return await engine.update_weights_from_tensor(req)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.sglang_rollout.http_server_engine import AsyncHttpServerAdapter
from verl.workers.rollout.sglang_rollout.utils import (
    SGLANG_LORA_NAME,
    get_named_tensor_buckets,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_original_sglang_set_envs = sglang.srt.entrypoints.engine._set_envs_and_config

# A WorkerDict process may construct more than one ServerAdapter for the same
# actor process group (for example, the checkpoint-manager update followed by
# the actor rollout path).  torch.distributed.new_group() is a collective and
# must be called by every actor rank in exactly the same order.  Keeping these
# handles only on a ServerAdapter instance therefore lets a later adapter issue
# a second, mismatched new_group() and block forever.  Cache them at module
# scope, which is the lifetime of the Ray worker process.
_HPU_WEIGHT_SYNC_GROUP_CACHE: dict[tuple[int, int, int, str], dict[str, object]] = {}

# Under Habana's GPU Migration Toolkit (PT_HPU_GPU_MIGRATION=1), torch.cuda reports as
# available, so sglang's is_cuda() evaluates True on Gaudi too. Gate the CUDA-only checks
# below (sgl_kernel/sglang_kernel package version asserts) behind a real vendor check.
_IS_HPU_HOST = importlib.util.find_spec("habana_frameworks") is not None


# patch to avoid issue https://github.com/sgl-project/sglang/issues/6723
def _set_envs_and_config(server_args: ServerArgs):
    if _IS_HPU_HOST and hasattr(server_args, "cuda_graph_backend_decode"):
        # The pinned Miles source already guards CUDA-only checks on HPU.
        # Keep its current process setup instead of replacing it with the 0.4.9 shim.
        return _original_sglang_set_envs(server_args)
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"
    # Enable faulthandler in subprocesses
    os.environ["PYTHONFAULTHANDLER"] = "1"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer_python",
            "0.2.5",
            "Please uninstall the old version and reinstall the latest version by following the instructions at https://docs.flashinfer.ai/installation.html.",
        )
    # sgl_kernel/sglang_kernel are CUDA-compiled kernel packages with no HPU build; their
    # version check is meaningless (and unsatisfiable) on Gaudi, which never installs them.
    if is_cuda() and not _IS_HPU_HOST:
        try:
            # For sglang 0.5.12 and sglang_kernel > 0.4.2, naming is sglang_kernel
            assert_pkg_version(
                "sglang_kernel",
                "0.1.1",
                "Please reinstall the latest version with `pip install follow https://sgl-project.github.io/get_started/install.html#for-cuda-13`",
            )
        except Exception:
            assert_pkg_version(
                "sgl_kernel",
                "0.1.1",
                "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
            )

    # Set mp start method
    mp.set_start_method("spawn", force=True)


sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config


# because chatCompletion is an async method, it makes the whole ray actor be an async actor
# which can not call loop.run_until_complete. So we need to make the engine to be an async class
class ServerAdapter(BaseRollout):
    """SGLang server adapter used in native http server mode, serve as http client to request SGLang server
    to resume/release/update weights and kv_cache.

    - hybrid mode: reside in each hybrid worker to sync weights between training engine and SGLang server.
    - standalone/colocated mode: just a dummy placeholder to occupy the GPU to prevent ray scheduling new GPU actor.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
        replica_rank: int = -1,
    ):
        super().__init__(config, model_config, device_mesh)
        if self.config.get("quantization", None) == "fp8":
            import sglang
            from packaging import version

            assert version.parse(sglang.__version__) >= version.parse("0.5.5"), (
                "sglang>=0.5.5 is required for FP8 quantization"
            )
            FP8_BLOCK_QUANT_KWARGS = {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }
            fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)
            self.model_config.hf_config.quantization_config = fp8_block_quant_kwargs
        self._engine: AsyncHttpServerAdapter = None
        self._distributed_weight_group = None
        self._distributed_weight_group_name = None
        self._distributed_weight_engines: list[AsyncHttpServerAdapter] = []
        self._hpu_actor_control_group = None

        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        # PD asymmetric layout inflates per-replica footprint; must match
        # agent_loop.py:_initialize_llm_servers or trainer-to-replica mapping breaks.
        disagg = getattr(self.config, "disaggregation", None)
        prefill_tp = self.config.tensor_model_parallel_size
        if disagg is not None and getattr(disagg, "enabled", False):
            # Inline decode_tp default: OmegaConf/Ray serialization drops dataclass methods.
            decode_tp = (
                disagg.decode_tensor_model_parallel_size
                if disagg.decode_tensor_model_parallel_size is not None
                else prefill_tp
            )
            rollout_world_size = (
                (prefill_tp * disagg.prefill_replicas + decode_tp * disagg.decode_replicas)
                * self.config.data_parallel_size
                * self.config.pipeline_model_parallel_size
            )
        else:
            rollout_world_size = prefill_tp * self.config.data_parallel_size * self.config.pipeline_model_parallel_size
        if replica_rank == -1:
            self.replica_rank = rank // rollout_world_size
        else:
            self.replica_rank = replica_rank
        self.rollout_rank = rank % rollout_world_size
        self.node_rank = self.rollout_rank // local_world_size
        self.local_rank = self.rollout_rank % local_world_size

        # Map each trainer rank to its co-located SGLang server so weight-update
        # IPC handles stay on the GPU where they were created. Offset math
        # assumes prefill_replicas == 1 (enforced by SGLangPDReplica); if that
        # ever lifts, update both this block and SGLangPDReplica.launch_servers.
        self._pd_role = None
        self._pd_server_index = None
        self._pd_tp_local_rank = None
        if disagg is not None and getattr(disagg, "enabled", False):
            decode_tp = (
                disagg.decode_tensor_model_parallel_size
                if disagg.decode_tensor_model_parallel_size is not None
                else prefill_tp
            )
            # Modulo by single-group footprint so if DP>1 is ever enabled,
            # each DP group's ranks resolve to the same role offsets.
            footprint = prefill_tp + disagg.decode_replicas * decode_tp
            local = self.rollout_rank % footprint
            if local < prefill_tp:
                self._pd_role = "prefill"
                self._pd_server_index = 0
                self._pd_tp_local_rank = local
            else:
                off = local - prefill_tp
                self._pd_role = "decode"
                self._pd_server_index = off // decode_tp
                self._pd_tp_local_rank = off % decode_tp
        self._has_server = (disagg is None or not getattr(disagg, "enabled", False)) or (self._pd_role is not None)

        # sleep_level controls what gets released during sleep/release:
        #   2 (default) = release weights + kv_cache (full sleep, merge path)
        #   1 = release kv_cache only (keep base weights, adapter path)
        # Set by engine_workers.update_weights() when lora.merge=False.
        self.sleep_level = 2

    async def _init_server_adapter(self):
        if self._engine is not None:
            return

        if not self._has_server:
            return

        # device_mesh is needed to gather cuda ipc handle to update weights.
        if self.device_mesh is None:
            assert torch.distributed.is_initialized(), "torch distributed must be initialized"
            infer_tp = self.config.tensor_model_parallel_size * self.config.data_parallel_size
            infer_pp = self.config.pipeline_model_parallel_size
            infer_world_size = infer_tp * infer_pp
            dp = torch.distributed.get_world_size() // infer_world_size
            self.device_mesh = init_device_mesh(
                "cpu", mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
            )

        # Only the role's TP-rank-0 builds an adapter; others participate in
        # FSDP collectives but skip HTTP dispatch.
        if self._pd_role is not None:
            if self._pd_tp_local_rank != 0:
                return
        else:
            if self.device_mesh["infer_tp"].get_local_rank() != 0:
                return

        if self._pd_role == "prefill":
            actor_name = f"sglang_server_{self.replica_rank}_0"
            timeout_kwargs = {}
        elif self._pd_role == "decode":
            actor_name = f"sglang_server_decode_{self.replica_rank}_{self._pd_server_index}"
            # Decode init on long-prompt workloads can stall past the default
            # (60s × 12); shorter timeout + fewer attempts avoids trainer lockup.
            timeout_kwargs = {"timeout": 10.0, "max_attempts": 2}
        else:
            actor_name = f"sglang_server_{self.replica_rank}_{self.node_rank}"
            timeout_kwargs = {}

        self.server_actor = ray.get_actor(actor_name)
        server_address, server_port = await self.server_actor.get_server_address.remote()
        host = f"[{server_address}]" if is_valid_ipv6_address(server_address) else server_address
        logger.info(
            f"ServerAdapter {self._pd_role or 'colocated'}: "
            f"replica_rank={self.replica_rank}, rollout_rank={self.rollout_rank}, "
            f"server={host}:{server_port}, actor={actor_name}"
        )

        self._engine = AsyncHttpServerAdapter(
            model_path=self.model_config.local_path,
            host=host,
            port=server_port,
            launch_server=False,
            trust_remote_code=self.model_config.trust_remote_code,
            **timeout_kwargs,
        )

    def _is_server_tp_leader(self) -> bool:
        """True if this rank is TP-rank-0 of its server's group.

        In PD, the role's TP (prefill_tp or decode_tp) may differ from the
        config-level TP that device_mesh was built with, so use
        _pd_tp_local_rank when PD is active.
        """
        if self._pd_role is not None:
            return self._pd_tp_local_rank == 0
        return self.device_mesh["infer_tp"].get_local_rank() == 0

    async def _init_hpu_distributed_weight_group(self) -> None:
        """Create the Miles-style trainer-to-rollout HCCL group once.

        FSDP rank 0 is the only sender. Every SGLang TP rank is a receiver. All
        other FSDP ranks still iterate the weight generator because materializing
        a DTensor requires their participation in the FSDP all-gather.
        """
        # Keep weight traffic and control synchronization on separate process
        # groups. Miles deliberately uses Gloo for the barriers around its HCCL
        # weight broadcasts. Reusing the default HCCL/FSDP group here can leave
        # all actor devices blocked after the final full_tensor() collective and
        # eventually trigger Synapse's fatal "No progress error" watchdog.
        # new_group() must be called by every rank in the default actor group.
        debug = os.environ.get("VERL_HPU_WEIGHT_SYNC_DEBUG", "0") == "1"
        rank = torch.distributed.get_rank()
        actor_world_size = torch.distributed.get_world_size()
        infer_tp = self.config.tensor_model_parallel_size
        cache_key = (
            id(torch.distributed.group.WORLD),
            actor_world_size,
            infer_tp,
            str(self.model_config.local_path),
        )
        cached = _HPU_WEIGHT_SYNC_GROUP_CACHE.setdefault(cache_key, {})
        if self._hpu_actor_control_group is None and "control_group" in cached:
            self._hpu_actor_control_group = cached["control_group"]
        if rank == 0 and self._distributed_weight_group is None and "weight_group" in cached:
            self._distributed_weight_group = cached["weight_group"]
            self._distributed_weight_group_name = cached["weight_group_name"]
            self._distributed_weight_engines = cached["engines"]
        if debug:
            print(f"VERL HPU WEIGHT GROUP rank={rank} stage=entry", flush=True)
        if self._hpu_actor_control_group is None:
            if debug:
                print(f"VERL HPU WEIGHT GROUP rank={rank} stage=pre_gloo_new_group", flush=True)
            self._hpu_actor_control_group = torch.distributed.new_group(backend="gloo")
            cached["control_group"] = self._hpu_actor_control_group
            if debug:
                print(f"VERL HPU WEIGHT GROUP rank={rank} stage=post_gloo_new_group", flush=True)
        elif debug:
            print(f"VERL HPU WEIGHT GROUP rank={rank} stage=reuse_gloo_group", flush=True)

        if rank != 0 or self._distributed_weight_group is not None:
            if debug:
                print(f"VERL HPU WEIGHT GROUP rank={rank} stage=non_sender_ready", flush=True)
            return
        if not _IS_HPU_HOST:
            raise RuntimeError("distributed HPU weight sync was requested on a non-HPU host")
        if self._pd_role is not None:
            raise NotImplementedError("distributed HPU weight sync does not yet support SGLang PD disaggregation")
        if self.config.pipeline_model_parallel_size != 1 or self.config.data_parallel_size != 1:
            raise NotImplementedError(
                "distributed HPU weight sync currently requires rollout pipeline/data parallel size 1"
            )

        if actor_world_size % infer_tp != 0:
            raise RuntimeError(f"actor world size {actor_world_size} is not divisible by rollout TP {infer_tp}")
        num_replicas = actor_world_size // infer_tp

        if debug:
            print(f"VERL HPU WEIGHT GROUP rank=0 stage=pre_server_discovery", flush=True)
        engines: list[AsyncHttpServerAdapter] = []
        for replica_rank in range(num_replicas):
            actor_name = f"sglang_server_{replica_rank}_0"
            server_actor = ray.get_actor(actor_name)
            server_address, server_port = await server_actor.get_server_address.remote()
            host = f"[{server_address}]" if is_valid_ipv6_address(server_address) else server_address
            engines.append(
                AsyncHttpServerAdapter(
                    model_path=self.model_config.local_path,
                    host=host,
                    port=server_port,
                    launch_server=False,
                    trust_remote_code=self.model_config.trust_remote_code,
                )
            )
        if debug:
            print(
                f"VERL HPU WEIGHT GROUP rank=0 stage=post_server_discovery engines={len(engines)}",
                flush=True,
            )

        master_address = ray.util.get_node_ip_address().strip("[]")
        master_port, _ = get_free_port(master_address)
        group_name = f"verl_hpu_weights_{os.getpid()}"
        receiver_world_size = num_replicas * infer_tp
        world_size = receiver_world_size + 1

        # The HTTP handlers block while their scheduler ranks join rendezvous, so
        # initialize the trainer process group concurrently rather than awaiting
        # either side first.
        remote_inits = [
            engine.init_weights_update_group(
                master_address=master_address,
                master_port=master_port,
                rank_offset=1 + replica_rank * infer_tp,
                world_size=world_size,
                group_name=group_name,
                backend="hccl",
            )
            for replica_rank, engine in enumerate(engines)
        ]

        from sglang.srt.utils import init_custom_process_group

        init_method_address = (
            f"[{master_address}]" if is_valid_ipv6_address(master_address) else master_address
        )
        local_init = asyncio.to_thread(
            init_custom_process_group,
            backend="hccl",
            init_method=f"tcp://{init_method_address}:{master_port}",
            world_size=world_size,
            rank=0,
            group_name=group_name,
        )
        if debug:
            print(
                f"VERL HPU WEIGHT GROUP rank=0 stage=pre_hccl_rendezvous world_size={world_size}",
                flush=True,
            )
        init_results = await asyncio.gather(local_init, *remote_inits)
        if debug:
            print("VERL HPU WEIGHT GROUP rank=0 stage=post_hccl_rendezvous", flush=True)
        group = init_results[0]
        for replica_rank, result in enumerate(init_results[1:]):
            if not result.get("success", False):
                raise RuntimeError(f"SGLang replica {replica_rank} failed to join HCCL weight group: {result}")

        self._distributed_weight_group = group
        self._distributed_weight_group_name = group_name
        self._distributed_weight_engines = engines
        cached["weight_group"] = group
        cached["weight_group_name"] = group_name
        cached["engines"] = engines
        print(
            "VERL HPU: initialized distributed weight sync "
            f"backend=hccl world_size={world_size} receivers={receiver_world_size}",
            flush=True,
        )

    async def _hpu_distributed_update_bucket(self, params_batch, bucket_index: int) -> None:
        """Broadcast one already-materialized FSDP bucket directly to SGLang."""
        rank = torch.distributed.get_rank()
        debug_weight_sync = os.environ.get("VERL_HPU_WEIGHT_SYNC_DEBUG", "0") == "1"

        # DTensor.full_tensor() is evaluated while every FSDP rank builds this
        # bucket. In HPU lazy mode the non-sender ranks previously returned here
        # without submitting those all-gathers: rank 1 could queue the entire
        # state dict and enter the final Gloo barrier while rank 0 was still
        # broadcasting bucket 0. Besides exposing partially materialized values,
        # that cross-process-group reordering can deadlock on a later full_tensor
        # (Qwen3-4B most often stopped before the final lm_head bucket). Drain the
        # FSDP gather on *every* actor rank before rank 0 starts the rollout-group
        # broadcast and before the other ranks are allowed to advance.
        if _IS_HPU_HOST:
            import habana_frameworks.torch as htorch

            htorch.core.mark_step()
            torch.hpu.synchronize()
        if debug_weight_sync:
            print(
                f"VERL HPU FSDP FULL TENSOR rank={rank} bucket={bucket_index} stage=materialized",
                flush=True,
            )

        if rank != 0:
            return
        if self._distributed_weight_group is None:
            raise RuntimeError("distributed HPU weight group has not been initialized")

        rollout_dtype_name = str(self.config.dtype).removeprefix("torch.")
        rollout_dtype = getattr(torch, rollout_dtype_name, None)
        if not isinstance(rollout_dtype, torch.dtype):
            raise ValueError(f"Unsupported rollout dtype for HPU weight sync: {self.config.dtype!r}")
        # FSDP keeps FP32 master parameters, while SGLang was launched with the
        # rollout dtype (BF16 for this workflow). Miles records an outbound sync
        # dtype and casts after gathering the full parameter. Do the same here so
        # SGLang does not build hundreds of lazy FP32-to-BF16 parameter-copy graphs.
        named_tensors = [
            (
                name,
                tensor.to(dtype=rollout_dtype).contiguous() if tensor.is_floating_point() else tensor.contiguous(),
            )
            for name, tensor in params_batch
        ]
        names = [name for name, _ in named_tensors]
        dtypes = [str(tensor.dtype).removeprefix("torch.") for _, tensor in named_tensors]
        shapes = [list(tensor.shape) for _, tensor in named_tensors]
        nbytes = sum(tensor.numel() * tensor.element_size() for _, tensor in named_tensors)
        started = time.monotonic()

        # The FP32 DTensor full-gather and the FP32->BF16 casts above are lazy
        # HPU operations. HCCL may otherwise read their backing storage before
        # those producers have executed: the broadcasts still complete, but the
        # rollout model receives garbage weights (gibberish generations and a
        # many-orders-of-magnitude trainer/rollout perplexity mismatch). Submit
        # and drain the source tensors before HCCL consumes this bucket.
        if _IS_HPU_HOST:
            import habana_frameworks.torch as htorch

            htorch.core.mark_step()
            torch.hpu.synchronize()

        if debug_weight_sync:
            debug_names = {
                "model.embed_tokens.weight",
                "model.layers.17.mlp.down_proj.weight",
                "model.norm.weight",
                "lm_head.weight",
            }
            for name, tensor in named_tensors:
                if name in debug_names:
                    first_values = tensor.reshape(-1)[:8].float().cpu().tolist()
                    print(
                        "VERL HPU WEIGHT VALUE side=sender "
                        f"name={name} dtype={tensor.dtype} first8={first_values}",
                        flush=True,
                    )
            print(
                "VERL HPU DISTRIBUTED WEIGHT BUCKET START "
                f"index={bucket_index} tensors={len(named_tensors)} bytes={nbytes} "
                f"first={names[0]} last={names[-1]}",
                flush=True,
            )

        # The receiver request must remain live while this actor blocks in HCCL.
        # Run each self-contained aiohttp request on its own event-loop thread;
        # Habana HCCL Work.wait() must stay on the actor's accelerator thread.
        def request_receiver(engine):
            return asyncio.run(
                engine.update_weights_from_distributed(
                    names=names,
                    dtypes=dtypes,
                    shapes=shapes,
                    group_name=self._distributed_weight_group_name,
                    flush_cache=False,
                )
            )

        loop = asyncio.get_running_loop()
        request_futures = [
            loop.run_in_executor(None, request_receiver, engine)
            for engine in self._distributed_weight_engines
        ]
        if debug_weight_sync:
            print(
                f"VERL HPU DISTRIBUTED WEIGHT BUCKET REQUESTED index={bucket_index}",
                flush=True,
            )
        handles = [
            torch.distributed.broadcast(
                tensor,
                src=0,
                group=self._distributed_weight_group,
                async_op=True,
            )
            for _, tensor in named_tensors
        ]
        if debug_weight_sync:
            print(
                f"VERL HPU DISTRIBUTED WEIGHT BUCKET BROADCAST_LAUNCHED index={bucket_index}",
                flush=True,
            )
        # The SGLang receivers synchronize their lazy parameter copies before
        # replying. Complete sender-side HCCL work first, exactly as Miles does,
        # while the HTTP request threads continue independently.
        for handle in handles:
            handle.wait()
        if debug_weight_sync:
            print(
                f"VERL HPU DISTRIBUTED WEIGHT BUCKET BROADCAST_DONE index={bucket_index}",
                flush=True,
            )
        results = await asyncio.gather(*request_futures)
        if debug_weight_sync:
            print(
                f"VERL HPU DISTRIBUTED WEIGHT BUCKET RECEIVERS_DONE index={bucket_index}",
                flush=True,
            )
        for replica_rank, result in enumerate(results):
            if not result.get("success", False):
                raise RuntimeError(f"SGLang replica {replica_rank} rejected weight bucket {bucket_index}: {result}")
        if os.environ.get("VERL_HPU_WEIGHT_SYNC_DEBUG", "0") == "1":
            print(
                "VERL HPU DISTRIBUTED WEIGHT BUCKET "
                f"index={bucket_index} tensors={len(named_tensors)} bytes={nbytes} "
                f"elapsed={time.monotonic() - started:.3f}s first={names[0]} last={names[-1]}",
                flush=True,
            )

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tag: weights or kv_cache.
        """
        await self._init_server_adapter()
        if self._engine is None:
            return
        # free_cache_engine releases/reoccupies GPU memory shared with the colocated training
        # worker (via torch_memory_saver, CUDA-only). The HPU rollout has its own dedicated
        # card (see _IS_HPU_HOST in async_sglang_server.py), so there is nothing to free.
        if self._is_server_tp_leader() and self.config.free_cache_engine and not _IS_HPU_HOST:
            await self._engine.resume_memory_occupation(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory.

        When sleep_level=1 (LoRA adapter mode), only releases kv_cache
        to keep base weights alive across training iterations.
        When sleep_level=2 (default/merge mode), releases everything.
        """
        await self._init_server_adapter()
        if self._engine is None:
            return
        if self._is_server_tp_leader() and self.config.free_cache_engine and not _IS_HPU_HOST:
            if self.sleep_level == 1:
                tags = ["kv_cache"]
            else:
                tags = ["kv_cache", "weights"]
            await self._engine.release_memory_occupation(tags=tags)

    async def update_weights(
        self, weights: Generator[tuple[str, torch.Tensor], None, None], global_steps: int = None, **kwargs
    ):
        """
        Update model weights using tensor buckets, similar to THUDM/slime's implementation.

        Notes:
          - For the best performance of `rebuild_cuda_tensor`, it is recommended to:
              1. Enable `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`.
              2. Manually set `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
            when using Tensor Parallelism (TP >= 8).
          - See reference implementations in SLIME:
            - Main logic: https://github.com/THUDM/slime/blob/fb7605cc5fb09af0f9369d37f7192f12bddee577/slime/ray/ppo_actor.py#L452
            - runtime envs: https://github.com/THUDM/slime/blob/fb7605cc5fb09af0f9369d37f7192f12bddee577/slime/ray/ppo_actor.py#L39
        """
        await self._init_server_adapter()
        # All ranks MUST iterate the weights generator below — DTensor.full_tensor()
        # all_gather's across the FSDP group and skipping deadlocks the others.
        # Only HTTP dispatch is gated on self._engine.

        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        weight_sync_transport = kwargs.get("weight_sync_transport", "tensor")
        use_distributed = weight_sync_transport == "distributed"
        if peft_config and base_sync_done:
            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                # unload lora
                models_result = await self._engine.available_models()
                exists = any(item["id"] == SGLANG_LORA_NAME for item in models_result["data"])
                if exists:
                    await self._engine.unload_lora_adapter(SGLANG_LORA_NAME)

                # load lora by tensor
                serialize_peft_config, serialize_named_tensors = self.wrap_lora_params(peft_config, weights)
                from sglang.srt.managers.io_struct import LoadLoRAAdapterFromTensorsReqInput

                req = LoadLoRAAdapterFromTensorsReqInput(
                    lora_name=SGLANG_LORA_NAME,
                    config_dict=serialize_peft_config,
                    serialized_tensors=serialize_named_tensors,
                )
                # send http request
                await self._engine.load_lora_adapter_from_tensor(req)
        else:
            update_weights_bucket_bytes = int(self.config.checkpoint_engine.update_weights_bucket_megabytes) << 20
            if self.config.get("quantization", None) == "fp8":
                from verl.utils.sglang.sglang_fp8_utils import SGLangFP8QuantizerHelper

                logger.info("Convert bf16 weights to fp8 format before loading")
                fp8_quantizer_helper = SGLangFP8QuantizerHelper(self.model_config.hf_config.quantization_config)
                weights = fp8_quantizer_helper.quant_weights_by_name(
                    weights,
                    dtype=self.model_config.hf_config.dtype,
                )
            else:
                weights = weights

            if use_distributed:
                await self._init_hpu_distributed_weight_group()

            # SGLang >= 0.5.19 requires the tensor transfers to run inside an explicit
            # weight-update session. Without it the scheduler raises
            #   AssertionError: update_weights_from_tensor requires an open
            #   begin_weight_update session
            # and the server process dies, which surfaces here only as "Server
            # disconnected" / "Failed to complete async request". Opened on the same rank
            # that owns the other server-level calls so that exactly one session exists
            # per server. The try/finally matters: a failure partway
            # through the buckets must still close the session, or the server refuses the
            # next step's update with "begin_weight_update called while a weight-update
            # session is already open".
            if use_distributed:
                session_engines = (
                    self._distributed_weight_engines if torch.distributed.get_rank() == 0 else []
                )
            else:
                session_engines = (
                    [self._engine] if self._engine is not None and self._is_server_tp_leader() else []
                )
            weight_session_open = False
            if session_engines:
                # Match Miles' FSDP update ordering: generation is already paused
                # by CheckpointEngine, so clear every rollout cache before opening
                # the session and issuing HCCL weight traffic. A second flush after
                # end_weight_update wedged the HPU schedulers/allocator after all
                # 146 Qwen3-4B buckets and every Gloo barrier had completed.
                if use_distributed:
                    if os.environ.get("VERL_HPU_WEIGHT_SYNC_DEBUG", "0") == "1":
                        print("VERL HPU WEIGHT SYNC rank=0 stage=pre_session_flush_start", flush=True)
                    await asyncio.gather(*(engine.flush_cache() for engine in session_engines))
                    if os.environ.get("VERL_HPU_WEIGHT_SYNC_DEBUG", "0") == "1":
                        print("VERL HPU WEIGHT SYNC rank=0 stage=pre_session_flush_done", flush=True)
                await asyncio.gather(*(engine.begin_weight_update() for engine in session_engines))
                weight_session_open = True
            if use_distributed:
                if os.environ.get("VERL_HPU_WEIGHT_SYNC_DEBUG", "0") == "1":
                    print(
                        f"VERL HPU WEIGHT SYNC rank={torch.distributed.get_rank()} "
                        "stage=pre_stream_gloo_barrier_start",
                        flush=True,
                    )
                torch.distributed.barrier(group=self._hpu_actor_control_group)
                if os.environ.get("VERL_HPU_WEIGHT_SYNC_DEBUG", "0") == "1":
                    print(
                        f"VERL HPU WEIGHT SYNC rank={torch.distributed.get_rank()} "
                        "stage=pre_stream_gloo_barrier_done",
                        flush=True,
                    )
            update_succeeded = False
            try:
                bucket_index = 0
                async for params_batch in get_named_tensor_buckets(weights, update_weights_bucket_bytes):
                    if use_distributed:
                        await self._hpu_distributed_update_bucket(params_batch, bucket_index)
                    else:
                        await sgl_update_weights(
                            engine=self._engine,
                            params_batch=params_batch,
                            device_mesh_key="infer_tp",
                            device_mesh=self.device_mesh,
                        )
                    bucket_index += 1
                # Non-source FSDP ranks can reach the end while rank 0 is still
                # sending its final bucket. Coordinate this control transition on
                # CPU/Gloo, not on the HCCL group used by FSDP and weight traffic.
                if use_distributed:
                    if os.environ.get("VERL_HPU_WEIGHT_SYNC_DEBUG", "0") == "1":
                        print(
                            f"VERL HPU WEIGHT SYNC rank={torch.distributed.get_rank()} "
                            "stage=post_stream_gloo_barrier_start",
                            flush=True,
                        )
                    torch.distributed.barrier(group=self._hpu_actor_control_group)
                    if os.environ.get("VERL_HPU_WEIGHT_SYNC_DEBUG", "0") == "1":
                        print(
                            f"VERL HPU WEIGHT SYNC rank={torch.distributed.get_rank()} "
                            "stage=post_stream_gloo_barrier_done",
                            flush=True,
                        )
                update_succeeded = True
            finally:
                # A timed-out distributed receive may still be blocked inside an
                # HCCL collective. end_weight_update cannot overtake it, and trying
                # only hides the original failure behind another long timeout.
                if weight_session_open and (update_succeeded or not use_distributed):
                    if use_distributed and os.environ.get("VERL_HPU_WEIGHT_SYNC_DEBUG", "0") == "1":
                        print("VERL HPU WEIGHT SYNC rank=0 stage=end_weight_update_start", flush=True)
                    await asyncio.gather(*(engine.end_weight_update() for engine in session_engines))
                    if use_distributed and os.environ.get("VERL_HPU_WEIGHT_SYNC_DEBUG", "0") == "1":
                        print("VERL HPU WEIGHT SYNC rank=0 stage=end_weight_update_done", flush=True)
            if use_distributed:
                if os.environ.get("VERL_HPU_WEIGHT_SYNC_DEBUG", "0") == "1":
                    print(
                        f"VERL HPU WEIGHT SYNC rank={torch.distributed.get_rank()} "
                        "stage=post_session_gloo_barrier_start",
                        flush=True,
                    )
                torch.distributed.barrier(group=self._hpu_actor_control_group)
                if os.environ.get("VERL_HPU_WEIGHT_SYNC_DEBUG", "0") == "1":
                    print(
                        f"VERL HPU WEIGHT SYNC rank={torch.distributed.get_rank()} "
                        "stage=post_session_gloo_barrier_done",
                        flush=True,
                    )

        if self._engine is not None and self._is_server_tp_leader():
            # The distributed HPU path flushed all replicas before the update,
            # following Miles. Keep the legacy post-update flush for the tensor
            # transport, where each adapter owns only its local server.
            if not use_distributed:
                await self._engine.flush_cache()
            if global_steps is not None:
                await self.server_actor.set_global_steps.remote(global_steps)

    def wrap_lora_params(self, peft_config: LoraConfig, weights: Generator[tuple[str, torch.Tensor]]):
        # peft config
        peft_config_json = asdict(peft_config)
        peft_config_json["task_type"] = peft_config_json["task_type"].value
        peft_config_json["peft_type"] = peft_config_json["peft_type"].value
        peft_config_json["target_modules"] = list(peft_config_json["target_modules"])

        # lora weights
        processed_weights: dict[str, torch.Tensor] = {
            name: _preprocess_tensor_for_update_weights(tensor.detach()) for name, tensor in weights
        }

        infer_tp_size = self.device_mesh["infer_tp"].mesh.size()[0]
        serialized_named_tensors = []
        for i in range(infer_tp_size):
            serialized_tensors = MultiprocessingSerializer.serialize(processed_weights, output_str=True)
            serialized_named_tensors.append(serialized_tensors)

        return peft_config_json, serialized_named_tensors
