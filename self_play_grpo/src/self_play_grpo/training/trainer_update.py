"""One synchronized D5 trainer update on already-admitted 16-game shards.

The caller must initialize the four-rank HCCL group, load the authoritative
policy and optimizer, and obtain a ``TrainerShardAdmission`` first. This
module does not launch processes or publish a checkpoint. If any rank fails,
discard its in-memory state and recover from the last committed checkpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from self_play_grpo.policies.llm import (
    checkpoint_qwen3_attention, constrained_log_probs_batched_shape,
)
from self_play_grpo.training.coordinator import PolicyDescriptor
from self_play_grpo.training.distributed import (
    average_trainable_gradients, global_source_max_abs_difference,
    verify_optimizer_replicas,
)
from self_play_grpo.training.identity import (
    optimizer_state_sha256, trainable_parameter_sha256,
)
from self_play_grpo.training.loss import backward_training_loss
from self_play_grpo.training.trainer_handoff import TrainerShardAdmission


@dataclass(frozen=True)
class SynchronizedUpdateResult:
    """One policy update of one trainer rank.

    ``optimizer_steps`` counts policy versions produced (callers rewrite it to
    the cumulative update index). ``gradient_sync_phases`` counts AdamW steps:
    one per whole-match minibatch. With several minibatches, ``loss`` and
    ``grad_norm`` are minibatch means, ratios are owned-token weighted over
    all minibatches, and ``max_abs_log_prob_error`` is the behaviour replay
    error of the first minibatch, taken before any step. ``minibatches``
    records every step and is empty for the single-minibatch baseline.
    """

    rank: int
    source_policy_version: str
    next_policy_version: str
    consumed_manifest_sha256: str
    initial_parameter_sha256: str
    parameter_sha256: str
    optimizer_sha256: str
    loss: float
    policy_loss: float
    kl_loss: float
    mean_ratio: float
    clip_fraction: float
    grad_norm: float
    max_abs_log_prob_error: float
    owned_tokens: int
    turns: int
    changed_parameter_tensors: int
    gradient_sync_phases: int
    optimizer_steps: int
    minibatches: tuple[dict[str, float], ...] = ()


def minibatch_slices(games: int, minibatches: int) -> tuple[slice, ...]:
    """Split a rank's ordered complete matches into equal contiguous minibatches."""

    if type(minibatches) is not int or minibatches <= 0 or games <= 0 or games % minibatches:
        raise ValueError(
            f"{games} matches per trainer rank cannot form {minibatches} equal minibatches"
        )
    size = games // minibatches
    return tuple(slice(index * size, (index + 1) * size) for index in range(minibatches))


def _collective_phase(torch: Any, dist: Any, device: Any, label: str, operation: Callable[[], Any]) -> Any:
    """Agree on local success before any rank enters the next phase."""

    error: Exception | None = None
    result: Any = None
    try:
        result = operation()
    except Exception as exc:
        error = exc
    passed = torch.tensor([0.0 if error else 1.0], dtype=torch.float32, device=device)
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    if float(passed.cpu().item()) != 1.0:
        raise RuntimeError(f"D5 trainer phase {label} failed" + (
            f": {error}" if error is not None else " on another rank"
        )) from error
    return result


def _require_equal_digest(torch: Any, dist: Any, device: Any, digest: str, label: str) -> None:
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"D5 {label} digest is malformed")
    # Every byte is exactly representable in float32 on HCCL. This avoids an
    # object collective and compares the complete content digest, not a schema.
    value = torch.tensor(list(bytes.fromhex(digest)), dtype=torch.float32, device=device)
    if global_source_max_abs_difference(torch, dist, value) != 0.0:
        raise RuntimeError(f"D5 trainer ranks disagree on {label}")


def synchronized_trainer_update(
    trainer: Any,
    admission: TrainerShardAdmission,
    policy: PolicyDescriptor,
    *,
    runtime: Any,
    torch: Any,
    dist: Any,
) -> SynchronizedUpdateResult:
    """Backpropagate 16 games/rank in K minibatches with one AdamW step each.

    K is ``training.minibatches_per_update`` (default 1). Each rank's streamed
    loss divides by its minibatch's game count. Averaging gradients across four
    ranks therefore equals the global objective over that minibatch's games.
    The behavior replay check is per-action and occurs before that action's
    backward; it applies to the first minibatch, which precedes every step.
    Later minibatches see an updated policy, so their ratios leave one and
    clipping becomes active. No rank steps until all ranks pass replay and
    gradient checks.
    """

    if runtime.world_size != 4 or runtime.rank != admission.rank:
        raise ValueError("D5 trainer runtime differs from admitted four-rank shard")
    if len(admission.matches) != 16 or len(admission.match_indices) != 16:
        raise ValueError("D5 trainer update requires 16 complete matches per rank")
    if policy.run_kind != "production":
        raise ValueError("D5 trainer update requires a production policy")
    if (admission.receipt.policy_version != policy.version
            or admission.receipt.adapter_sha256 != policy.adapter_sha256
            or admission.receipt.config_sha256 != policy.config_sha256
            or admission.receipt.match_indices_by_rank[runtime.rank] != admission.match_indices):
        raise ValueError("D5 trainer admission differs from source policy")
    if trainer.policy_version != policy.version or trainer.update_index != policy.update_index:
        raise ValueError("D5 trainer state differs from source policy version")
    if trainer.config.training.kl_beta != 0.0:
        raise ValueError("D5 outcome baseline requires zero KL beta")
    if trainer.config.training.optimizer_epochs_per_batch != 1:
        raise ValueError("D5 outcome baseline requires one optimizer epoch")
    if not math.isfinite(admission.replay_tolerance) or admission.replay_tolerance <= 0.0:
        raise ValueError("D5 admitted replay tolerance must be finite and positive")
    if not dist.is_initialized() or dist.get_world_size() != 4 or dist.get_rank() != runtime.rank:
        raise RuntimeError("D5 four-rank process group is not initialized")

    slices = minibatch_slices(len(admission.matches), trainer.config.training.minibatches_per_update)
    model = trainer.policy.model
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("D5 trainer has no trainable parameters")
    device = torch.device(runtime.device)
    initial_digest = _collective_phase(
        torch, dist, device, "initial policy fingerprint",
        lambda: trainable_parameter_sha256(model),
    )
    _require_equal_digest(torch, dist, device, initial_digest, "initial trainable parameters")
    before = _collective_phase(
        torch, dist, device, "initial parameter snapshot",
        lambda: [parameter.detach().cpu().clone() for parameter in parameters],
    )
    trainer.optimizer.zero_grad(set_to_none=True)
    model.train()

    def backward(matches: Any, replay_tolerance: float | None, prior_tokens: int, final: bool) -> Any:
        with checkpoint_qwen3_attention(model):
            output = backward_training_loss(
                model, matches,
                clip_epsilon=trainer.config.training.clip_epsilon,
                loss_normalizer_per_game=trainer.config.training.loss_normalizer_per_game,
                kl_beta=0.0, log_prob_function=constrained_log_probs_batched_shape,
                replay_tolerance=replay_tolerance,
            )
        covered = prior_tokens + output.owned_tokens
        expected = admission.receipt.owned_tokens_by_rank[runtime.rank]
        if (output.games != len(matches)
                or (covered != expected if final else covered > expected)
                or not math.isfinite(float(output.loss))
                or not math.isfinite(output.max_abs_log_prob_error)):
            raise RuntimeError("D5 streamed backward metrics differ from admitted shard")
        if any(parameter.grad is None for parameter in parameters):
            raise RuntimeError("D5 streamed backward left a trainable gradient missing")
        return output

    outputs: list[Any] = []
    grad_norms: list[float] = []
    sync_phases = 0
    try:
        for index, selected in enumerate(slices):
            label = "" if len(slices) == 1 else f" {index + 1}/{len(slices)}"
            trainer.optimizer.zero_grad(set_to_none=True)
            # Only the first minibatch precedes every step, so only its
            # sampling-time probabilities are an exact replay target.
            tolerance = admission.replay_tolerance if index == 0 else None
            prior_tokens = sum(output.owned_tokens for output in outputs)
            final = index == len(slices) - 1
            output = _collective_phase(
                torch, dist, device, f"streamed backward{label}",
                lambda: backward(admission.matches[selected], tolerance, prior_tokens, final),
            )
            sync = average_trainable_gradients(model, dist, world_size=4)
            sync_phases += int(sync["gradient_sync_phases"])
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                parameters, trainer.config.training.max_grad_norm, error_if_nonfinite=True,
            )
            grad_norm = float(grad_norm_tensor.detach().float().cpu().item())
            if not math.isfinite(grad_norm) or grad_norm <= 0.0:
                raise RuntimeError("D5 synchronized gradient norm must be finite and positive")
            norm_tensor = torch.tensor([grad_norm], dtype=torch.float32, device=device)
            if global_source_max_abs_difference(torch, dist, norm_tensor) != 0.0:
                raise RuntimeError("D5 trainer gradient norms differ after all-reduce")
            _collective_phase(torch, dist, device, f"optimizer step{label}", trainer.optimizer.step)
            if runtime.device == "hpu":
                torch.hpu.synchronize()
            outputs.append(output)
            grad_norms.append(grad_norm)
        replicas = verify_optimizer_replicas(torch, dist, parameters, trainer.optimizer, device=device)
        if any(float(value) != 0.0 for key, value in replicas.items() if key.startswith("global_max_abs_")):
            raise RuntimeError("D5 trainer parameters or optimizer states differ after step")
        changed = sum(
            not torch.equal(old, parameter.detach().cpu())
            for old, parameter in zip(before, parameters)
        )
        if changed <= 0:
            raise RuntimeError("D5 optimizer step changed no trainable tensors")
        parameter_digest = trainable_parameter_sha256(model)
        optimizer_digest = optimizer_state_sha256(trainer.optimizer)
        _require_equal_digest(torch, dist, device, parameter_digest, "updated trainable parameters")
        _require_equal_digest(torch, dist, device, optimizer_digest, "updated optimizer state")
    except Exception:
        trainer.optimizer.zero_grad(set_to_none=True)
        raise
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.update_index += 1
    trainer.policy_version = f"policy-{trainer.update_index:06d}"
    if len(outputs) == 1:
        (output,) = outputs
        loss, policy_loss, kl_loss = float(output.loss), float(output.policy_loss), float(output.kl_loss)
        mean_ratio, clip_fraction, grad_norm = output.mean_ratio, output.clip_fraction, grad_norms[0]
        minibatches: tuple[dict[str, float], ...] = ()
    else:
        tokens = sum(output.owned_tokens for output in outputs)
        loss = math.fsum(float(output.loss) for output in outputs) / len(outputs)
        policy_loss = math.fsum(float(output.policy_loss) for output in outputs) / len(outputs)
        kl_loss = math.fsum(float(output.kl_loss) for output in outputs) / len(outputs)
        mean_ratio = math.fsum(output.mean_ratio * output.owned_tokens for output in outputs) / tokens
        clip_fraction = math.fsum(output.clip_fraction * output.owned_tokens for output in outputs) / tokens
        grad_norm = math.fsum(grad_norms) / len(grad_norms)
        minibatches = tuple(
            {
                "index": index, "games": output.games, "owned_tokens": output.owned_tokens,
                "loss": float(output.loss), "mean_ratio": output.mean_ratio,
                "clip_fraction": output.clip_fraction,
                # Replay error for index 0; policy drift since sampling afterwards.
                "max_abs_log_ratio": output.max_abs_log_prob_error,
                "grad_norm": norm,
            }
            for index, (output, norm) in enumerate(zip(outputs, grad_norms))
        )
    return SynchronizedUpdateResult(
        rank=runtime.rank,
        source_policy_version=policy.version,
        next_policy_version=trainer.policy_version,
        consumed_manifest_sha256=admission.receipt.manifest_sha256,
        initial_parameter_sha256=initial_digest,
        parameter_sha256=parameter_digest,
        optimizer_sha256=optimizer_digest,
        loss=loss,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        mean_ratio=mean_ratio,
        clip_fraction=clip_fraction,
        grad_norm=grad_norm,
        max_abs_log_prob_error=outputs[0].max_abs_log_prob_error,
        owned_tokens=sum(output.owned_tokens for output in outputs),
        turns=sum(len(match.turns) for match in admission.matches),
        changed_parameter_tensors=changed,
        gradient_sync_phases=sync_phases,
        optimizer_steps=1,
        minibatches=minibatches,
    )
