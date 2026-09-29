"""Optional structured-state evaluator for turn-level outcome credit."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Sequence

import numpy as np

from self_play_grpo.envs.observations import BoardState
from self_play_grpo.rewards.progress import evaluator_features
from self_play_grpo.rollouts.schema import MatchRecord


def build_value_network(input_dim: int, hidden_dims: tuple[int, ...] = (256, 128)) -> Any:
    try:
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Torch is required for the optional evaluator") from exc
    layers: list[Any] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, width), nn.GELU()))
        previous = width
    layers.append(nn.Linear(previous, 4))
    return nn.Sequential(*layers)


@dataclass
class OutcomeEvaluator:
    model: Any
    version: str
    device: str = "cpu"

    @classmethod
    def create(cls, example: BoardState, version: str, device: str = "cpu") -> "OutcomeEvaluator":
        input_dim = int(evaluator_features(example).size)
        model = build_value_network(input_dim)
        model.to(device)
        return cls(model=model, version=version, device=device)

    def predict(self, states: Sequence[BoardState]) -> np.ndarray:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Torch is required for evaluator inference") from exc
        if not states:
            return np.empty((0, 4), dtype=np.float32)
        features = np.stack([evaluator_features(state) for state in states])
        tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            predictions = torch.softmax(self.model(tensor), dim=-1)
        return predictions.cpu().numpy()


def evaluator_loss(model: Any, features: Any, targets: Any) -> Any:
    """Soft-target cross entropy supporting one-hot winners and uniform draws."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch is required for evaluator training") from exc
    logits = model(features)
    return -(targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def brier_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    if predictions.shape != targets.shape or predictions.ndim != 2 or predictions.shape[1] != 4:
        raise ValueError("Predictions and targets must both have shape [games, 4]")
    return float(np.mean(np.sum((predictions - targets) ** 2, axis=1)))


def split_complete_matches(
    matches: Sequence[MatchRecord],
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[MatchRecord], list[MatchRecord]]:
    """Split only at match boundaries to keep all dependent views together."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if len(matches) < 2:
        raise ValueError("At least two matches are needed for train/validation splitting")
    indices = list(range(len(matches)))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, round(len(indices) * validation_fraction))
    validation_indices = set(indices[:validation_count])
    training = [match for index, match in enumerate(matches) if index not in validation_indices]
    validation = [match for index, match in enumerate(matches) if index in validation_indices]
    if not training:
        raise ValueError("The requested split left no training matches")
    return training, validation


def _state_examples(matches: Sequence[MatchRecord]) -> tuple[np.ndarray, np.ndarray]:
    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    for match in matches:
        match.validate()
        target = np.asarray(match.final_results, dtype=np.float32)
        for turn in match.turns:
            feature_rows.append(evaluator_features(turn.state_before))
            target_rows.append(target)
    if not feature_rows:
        raise ValueError("Evaluator dataset contains no decision states")
    return np.stack(feature_rows), np.stack(target_rows)


def fit_evaluator(
    evaluator: OutcomeEvaluator,
    training_matches: Sequence[MatchRecord],
    validation_matches: Sequence[MatchRecord],
    *,
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Fit on outcomes while reporting held-out whole-match quality."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch is required for evaluator training") from exc
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    train_x, train_y = _state_examples(training_matches)
    valid_x, valid_y = _state_examples(validation_matches)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    optimizer = torch.optim.AdamW(evaluator.model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    train_features = torch.tensor(train_x, dtype=torch.float32, device=evaluator.device)
    train_targets = torch.tensor(train_y, dtype=torch.float32, device=evaluator.device)
    valid_features = torch.tensor(valid_x, dtype=torch.float32, device=evaluator.device)
    valid_targets = torch.tensor(valid_y, dtype=torch.float32, device=evaluator.device)
    for epoch in range(epochs):
        evaluator.model.train()
        permutation = torch.randperm(len(train_features), generator=generator)
        losses = []
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size].to(evaluator.device)
            loss = evaluator_loss(
                evaluator.model,
                train_features.index_select(0, indices),
                train_targets.index_select(0, indices),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        evaluator.model.eval()
        with torch.no_grad():
            validation_loss = evaluator_loss(
                evaluator.model, valid_features, valid_targets
            )
            validation_predictions = torch.softmax(
                evaluator.model(valid_features), dim=-1
            ).cpu().numpy()
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": float(np.mean(losses)),
                "validation_loss": float(validation_loss.cpu()),
                "validation_brier": brier_score(validation_predictions, valid_y),
                "constant_quarter_brier": brier_score(
                    np.full_like(valid_y, 0.25), valid_y
                ),
            }
        )
    return history
