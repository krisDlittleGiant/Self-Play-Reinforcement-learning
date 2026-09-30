"""Masked clipped policy loss with fixed per-game horizon normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from self_play_grpo.policies.llm import constrained_log_probs
from self_play_grpo.rollouts.schema import MatchRecord, ensure_single_policy_version


@dataclass
class LossOutput:
    loss: Any
    policy_loss: Any
    kl_loss: Any
    mean_ratio: float
    clip_fraction: float
    owned_tokens: int
    games: int
    max_abs_log_prob_error: float


def clipped_token_terms(
    new_log_probs: Any,
    old_log_probs: Any,
    advantage: float,
    loss_mask: Any,
    clip_epsilon: float,
) -> tuple[Any, Any]:
    """Return the maximized surrogate terms and their importance ratios."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch is required for the policy objective") from exc
    if new_log_probs.shape != old_log_probs.shape or new_log_probs.shape != loss_mask.shape:
        raise ValueError("Token log probabilities and ownership mask must align")
    ratios = torch.exp(new_log_probs - old_log_probs)
    scalar_advantage = torch.as_tensor(
        advantage, dtype=new_log_probs.dtype, device=new_log_probs.device
    )
    unclipped = ratios * scalar_advantage
    clipped = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * scalar_advantage
    terms = torch.minimum(unclipped, clipped) * loss_mask
    return terms, ratios


def sampled_kl_terms(new_log_probs: Any, reference_log_probs: Any) -> Any:
    """Non-negative sampled reverse-KL estimator used by GRPO implementations."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch is required for KL regularization") from exc
    log_ratio = reference_log_probs - new_log_probs
    return torch.exp(log_ratio) - log_ratio - 1.0


def compute_training_loss(
    model: Any,
    matches: Sequence[MatchRecord],
    *,
    clip_epsilon: float,
    loss_normalizer_per_game: int,
    kl_beta: float = 0.0,
    reference_model: Any | None = None,
    selected_seats: set[int] | None = None,
    selected_seat_importance: float = 1.0,
) -> LossOutput:
    """Build one loss over complete matches without realized-length division."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch is required for training") from exc
    if not matches:
        raise ValueError("At least one complete match is required")
    if loss_normalizer_per_game <= 0:
        raise ValueError("loss_normalizer_per_game must be positive")
    for match in matches:
        match.validate()
    ensure_single_policy_version(matches)
    if kl_beta and reference_model is None:
        raise ValueError("A frozen reference model is required when kl_beta is nonzero")

    numerator = None
    kl_numerator = None
    ratio_values: list[Any] = []
    clip_values: list[Any] = []
    owned_tokens = 0
    replay_errors: list[Any] = []
    for match in matches:
        for turn in match.turns:
            if selected_seats is not None and turn.seat not in selected_seats:
                continue
            sample = turn.policy_sample
            if not sample.completion_token_ids:
                continue
            advantage = turn.credit.training_advantage
            if advantage is None:
                raise ValueError("A trainable turn is missing its training advantage")
            new_log_probs = constrained_log_probs(model, sample)
            old_log_probs = torch.tensor(
                sample.behavior_log_probs,
                dtype=new_log_probs.dtype,
                device=new_log_probs.device,
            )
            replay_errors.append(torch.max(torch.abs(new_log_probs.detach() - old_log_probs)))
            ownership = torch.tensor(
                sample.loss_mask,
                dtype=new_log_probs.dtype,
                device=new_log_probs.device,
            )
            terms, ratios = clipped_token_terms(
                new_log_probs,
                old_log_probs,
                float(advantage) * selected_seat_importance,
                ownership,
                clip_epsilon,
            )
            turn_sum = terms.sum()
            numerator = turn_sum if numerator is None else numerator + turn_sum
            active = ownership > 0
            ratio_values.append(ratios[active].detach())
            clip_values.append(
                ((ratios[active] < 1.0 - clip_epsilon) | (ratios[active] > 1.0 + clip_epsilon))
                .float()
                .detach()
            )
            owned_tokens += int(active.sum().item())
            if reference_model is not None and kl_beta:
                with torch.no_grad():
                    reference = constrained_log_probs(reference_model, sample)
                turn_kl = (sampled_kl_terms(new_log_probs, reference) * ownership).sum()
                kl_numerator = turn_kl if kl_numerator is None else kl_numerator + turn_kl

    if numerator is None:
        raise ValueError("The batch contains no owned language-model tokens")
    denominator = float(len(matches) * loss_normalizer_per_game)
    policy_loss = -numerator / denominator
    if kl_numerator is None:
        kl_loss = torch.zeros((), dtype=policy_loss.dtype, device=policy_loss.device)
    else:
        kl_loss = kl_numerator / denominator
    loss = policy_loss + float(kl_beta) * kl_loss
    all_ratios = torch.cat(ratio_values) if ratio_values else torch.ones(1)
    all_clipped = torch.cat(clip_values) if clip_values else torch.zeros(1)
    return LossOutput(
        loss=loss,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        mean_ratio=float(all_ratios.mean().cpu()),
        clip_fraction=float(all_clipped.mean().cpu()),
        owned_tokens=owned_tokens,
        games=len(matches),
        max_abs_log_prob_error=float(torch.stack(replay_errors).max().cpu()),
    )


def backward_training_loss(
    model: Any,
    matches: Sequence[MatchRecord],
    *,
    clip_epsilon: float,
    loss_normalizer_per_game: int,
    kl_beta: float = 0.0,
    reference_model: Any | None = None,
    action_progress_callback: Callable[[int], None] | None = None,
    log_prob_function: Callable[[Any, Any], Any] | None = None,
    replay_tolerance: float | None = None,
) -> LossOutput:
    """Backpropagate the exact batch objective while retaining one action graph.

    The caller owns gradient reset, clipping, and any optimizer step. Each
    action loss uses the full match-batch denominator before backward, so the
    accumulated gradients equal a single backward over the summed objective.
    """

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch is required for training") from exc
    if not matches:
        raise ValueError("At least one complete match is required")
    if loss_normalizer_per_game <= 0:
        raise ValueError("loss_normalizer_per_game must be positive")
    for match in matches:
        match.validate()
    ensure_single_policy_version(matches)
    if kl_beta and reference_model is None:
        raise ValueError("A frozen reference model is required when kl_beta is nonzero")
    if replay_tolerance is not None and replay_tolerance <= 0.0:
        raise ValueError("replay_tolerance must be positive")
    replay_log_probs = constrained_log_probs if log_prob_function is None else log_prob_function

    denominator = float(len(matches) * loss_normalizer_per_game)

    def backward_one_turn(sample: Any, advantage: float) -> tuple[Any, ...]:
        """Backpropagate one action and return CPU-only detached diagnostics.

        The function-frame boundary is intentional.  Variable-shape cached
        replay can retain a large differentiable KV graph through local Python
        references even after ``backward`` has consumed its saved tensors.
        Returning only tiny CPU diagnostics guarantees that every action graph
        is unreachable before the next replay allocates HPU memory, while the
        parameter gradients themselves remain accumulated on the accelerator.
        """

        new_log_probs = replay_log_probs(model, sample)
        old_log_probs = torch.tensor(
            sample.behavior_log_probs,
            dtype=new_log_probs.dtype,
            device=new_log_probs.device,
        )
        ownership = torch.tensor(
            sample.loss_mask,
            dtype=new_log_probs.dtype,
            device=new_log_probs.device,
        )
        terms, ratios = clipped_token_terms(
            new_log_probs,
            old_log_probs,
            advantage,
            ownership,
            clip_epsilon,
        )
        replay_error_on_device = torch.max(
            torch.abs(new_log_probs.detach() - old_log_probs)
        )
        replay_error = replay_error_on_device.float().cpu()
        replay_error_value = float(replay_error.item())
        if not torch.isfinite(replay_error).item():
            raise RuntimeError("Non-finite behavior replay error before backward")
        if replay_tolerance is not None and replay_error_value > replay_tolerance:
            raise RuntimeError(
                f"Behavior replay mismatch {replay_error_value:.6g} exceeds "
                f"{replay_tolerance:.6g} before backward"
            )
        turn_policy = -terms.sum() / denominator
        turn_kl = torch.zeros((), dtype=turn_policy.dtype, device=turn_policy.device)
        if reference_model is not None and kl_beta:
            with torch.no_grad():
                reference = replay_log_probs(reference_model, sample)
            turn_kl = (sampled_kl_terms(new_log_probs, reference) * ownership).sum()
            turn_kl = turn_kl / denominator
        turn_loss = turn_policy + float(kl_beta) * turn_kl
        turn_loss.backward()

        active = ownership > 0
        detached_loss = turn_loss.detach().float().cpu()
        detached_policy = turn_policy.detach().float().cpu()
        detached_kl = turn_kl.detach().float().cpu()
        active_ratios = ratios[active].detach().float().cpu()
        active_clipped = (
            (ratios[active] < 1.0 - clip_epsilon)
            | (ratios[active] > 1.0 + clip_epsilon)
        ).float().cpu()
        return (
            detached_loss,
            detached_policy,
            detached_kl,
            active_ratios,
            active_clipped,
            replay_error,
            int(active.sum().cpu().item()),
        )

    loss_total = None
    policy_total = None
    kl_total = None
    ratio_values: list[Any] = []
    clip_values: list[Any] = []
    replay_errors: list[Any] = []
    owned_tokens = 0
    completed_actions = 0
    for match in matches:
        for turn in match.turns:
            sample = turn.policy_sample
            if not sample.completion_token_ids:
                continue
            advantage = turn.credit.training_advantage
            if advantage is None:
                raise ValueError("A trainable turn is missing its training advantage")
            (
                detached_loss,
                detached_policy,
                detached_kl,
                active_ratios,
                active_clipped,
                replay_error,
                turn_owned_tokens,
            ) = backward_one_turn(sample, float(advantage))
            loss_total = detached_loss if loss_total is None else loss_total + detached_loss
            policy_total = (
                detached_policy if policy_total is None else policy_total + detached_policy
            )
            kl_total = detached_kl if kl_total is None else kl_total + detached_kl
            ratio_values.append(active_ratios)
            clip_values.append(active_clipped)
            replay_errors.append(replay_error)
            owned_tokens += turn_owned_tokens
            completed_actions += 1
            if action_progress_callback is not None:
                action_progress_callback(completed_actions)

    if loss_total is None or policy_total is None or kl_total is None:
        raise ValueError("The batch contains no owned language-model tokens")
    all_ratios = torch.cat(ratio_values)
    all_clipped = torch.cat(clip_values)
    return LossOutput(
        loss=loss_total,
        policy_loss=policy_total,
        kl_loss=kl_total,
        mean_ratio=float(all_ratios.mean()),
        clip_fraction=float(all_clipped.mean()),
        owned_tokens=owned_tokens,
        games=len(matches),
        max_abs_log_prob_error=float(torch.stack(replay_errors).max()),
    )
