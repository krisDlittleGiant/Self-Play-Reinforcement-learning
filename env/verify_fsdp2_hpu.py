#!/usr/bin/env python3
"""Single-HPU regression for VERL's FSDP2 wrapping path."""

import os


def main():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29631")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("VERL_PLATFORM", "hpu")

    import habana_frameworks.torch  # noqa: F401
    import habana_frameworks.torch.distributed.hccl as hccl
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from torch.distributed.fsdp._fully_shard import FSDPModule
    from torch.distributed.tensor import DTensor
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    from verl.utils.fsdp_utils import apply_fsdp2, get_shard_placement_fn
    from verl.workers.fsdp_workers import create_device_mesh
    from recipe.sppo.sppo_worker import _model_inside_fsdp

    hccl.initialize_distributed_hpu(world_size=1, rank=0, local_rank=0)
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl", rank=0, world_size=1)

    try:
        mesh = create_device_mesh(world_size=1, fsdp_size=-1, strategy="fsdp2")
        print(f"mesh_device={mesh.device_type} mesh={mesh}", flush=True)
        assert mesh.device_type == "hpu", f"FSDP2 mesh must be hpu, got {mesh.device_type}"

        torch.manual_seed(7)
        config = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
            tie_word_embeddings=True,
        )
        model = Qwen3ForCausalLM(config).to(dtype=torch.float32, device="hpu")
        model.eval()

        input_ids = torch.tensor(
            [[0, 0, 11, 12, 13, 14, 15, 16], [21, 22, 23, 24, 25, 26, 27, 28]],
            dtype=torch.long,
            device="hpu",
        )
        attention_mask = torch.tensor(
            [[0, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]],
            dtype=torch.long,
            device="hpu",
        )
        position_ids = torch.clamp(attention_mask.cumsum(-1) - 1, min=0)

        with torch.no_grad():
            reference = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            ).logits.float().cpu()
        torch.hpu.synchronize()

        fsdp_kwargs = {
            "mesh": mesh,
            "mp_policy": MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32),
            "offload_policy": None,
            "reshard_after_forward": False,
            "shard_placement_fn": get_shard_placement_fn(fsdp_size=1),
        }
        apply_fsdp2(model, fsdp_kwargs, {"wrap_policy": {}, "forward_prefetch": False})
        assert isinstance(model, FSDPModule), f"root model was not converted to FSDPModule: {type(model)}"
        assert _model_inside_fsdp(model) is model, "SPPO selected the wrong model object for FSDP2"

        model.eval()
        with torch.no_grad():
            wrapped = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            ).logits.float().cpu()
        torch.hpu.synchronize()
        torch.testing.assert_close(wrapped, reference, rtol=1e-5, atol=1e-5)
        max_forward_delta = (wrapped - reference).abs().max().item()

        model.train()
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).logits
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, config.vocab_size), input_ids[:, 1:].reshape(-1))
        loss.backward()
        torch.hpu.synchronize()

        grad_sq = 0.0
        grad_tensors = 0
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad
            if isinstance(grad, DTensor):
                grad = grad.to_local()
            assert bool(torch.isfinite(grad).all()), "FSDP2 produced a non-finite gradient"
            grad_sq += grad.float().pow(2).sum().cpu().item()
            grad_tensors += 1
        grad_norm = grad_sq**0.5
        assert grad_tensors > 0 and grad_norm > 0.0

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.step()
        torch.hpu.synchronize()
        print(
            f"PASS FSDP2 HPU forward/backward: loss={loss.item():.6f} "
            f"grad_norm={grad_norm:.6f} grad_tensors={grad_tensors} "
            f"max_forward_delta={max_forward_delta:.3e}",
            flush=True,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
