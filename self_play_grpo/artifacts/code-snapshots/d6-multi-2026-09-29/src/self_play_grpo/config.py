"""Typed configuration with fail-closed experiment invariants."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


PINNED_ENGINE_REVISION = "d0606878b957274cc67a918ed173b36e9fe0fed6"


@dataclass(frozen=True)
class EnvironmentConfig:
    game: str = "quoridor"
    engine_revision: str = PINNED_ENGINE_REVISION
    players: int = 4
    board_size: int = 9
    wall_count: int = 5
    max_joint_actions: int = 120
    horizon_result: str = "uniform_draw"
    action_perspective: str = "absolute"

    def validate(self) -> None:
        if self.game != "quoridor":
            raise ValueError("This implementation intentionally supports only quoridor")
        if self.players != 4:
            raise ValueError("The experiment contract requires exactly four players")
        if self.board_size < 3 or self.board_size > 25:
            raise ValueError("OpenSpiel Quoridor board_size must be in [3, 25]")
        if self.wall_count < 0:
            raise ValueError("wall_count must be non-negative")
        if self.max_joint_actions <= 0:
            raise ValueError("max_joint_actions must be positive")
        if self.horizon_result != "uniform_draw":
            raise ValueError("Only the preregistered uniform horizon draw is supported")
        if self.action_perspective not in {"absolute", "player_relative"}:
            raise ValueError(
                "action_perspective must be 'absolute' or 'player_relative'"
            )
        if self.engine_revision != PINNED_ENGINE_REVISION:
            raise ValueError(
                "Engine revision differs from the validated source contract; "
                "update the adapter tests before changing it"
            )


@dataclass(frozen=True)
class ModelConfig:
    id: str
    revision: str
    local_path: str | None = None
    local_files_only: bool = True
    enable_thinking: bool = False
    device: str = "hpu"
    dtype: str = "bfloat16"
    lora_rank: int = 16
    lora_alpha: int = 32
    dropout: float = 0.0

    @property
    def load_source(self) -> str:
        """Return the local checkpoint path when one is configured."""

        return self.local_path or self.id


@dataclass(frozen=True)
class RolloutConfig:
    games_per_update: int = 16
    parallel_games_per_rank: int = 1
    shared_weights_across_seats: bool = True
    max_new_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    constrain_to_legal_actions: bool = True

    def validate(self) -> None:
        if self.games_per_update <= 0:
            raise ValueError("games_per_update must be positive")
        if self.parallel_games_per_rank <= 0:
            raise ValueError("parallel_games_per_rank must be positive")
        if not self.shared_weights_across_seats:
            raise ValueError("All four seats must share one policy")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if (self.temperature, self.top_p, self.top_k) != (1.0, 1.0, 0):
            raise ValueError("The correctness baseline requires temperature=1, top_p=1, top_k=0")
        if not self.constrain_to_legal_actions:
            raise ValueError("Illegal-action repair is forbidden; constrained decoding is required")


@dataclass(frozen=True)
class TrainingConfig:
    reward_mode: str = "outcome"
    group_key: str = "game_id"
    group_size: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    clip_epsilon: float = 0.2
    optimizer_epochs_per_batch: int = 1
    max_grad_norm: float = 1.0
    kl_beta: float = 0.0
    loss_normalizer_per_game: int = 120

    def validate(self, env: EnvironmentConfig) -> None:
        if self.reward_mode not in {"outcome", "potential", "gae_blend"}:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")
        if self.group_key != "game_id" or self.group_size != 4:
            raise ValueError("Advantages must be grouped as four seats of one game")
        if self.optimizer_epochs_per_batch != 1:
            raise ValueError("The initial on-policy implementation permits exactly one optimizer epoch")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.loss_normalizer_per_game != env.max_joint_actions:
            raise ValueError("loss_normalizer_per_game must equal the configured action horizon")


@dataclass(frozen=True)
class ProcessConfig:
    enabled: bool = False
    proxy_temperature: float = 2.0
    shaping_alpha: float = 1.0
    gamma: float = 1.0
    gae_lambda: float = 0.95
    process_blend_eta: float = 0.25


@dataclass(frozen=True)
class EvaluationConfig:
    fixed_opponent_manifest: str | None = None
    rotate_candidate_through_all_seats: bool = True
    training_seeds: tuple[int, ...] = (11, 22, 33)


@dataclass(frozen=True)
class ExperimentConfig:
    project: str
    seed: int
    environment: EnvironmentConfig
    model: ModelConfig
    rollout: RolloutConfig
    training: TrainingConfig
    process_extension: ProcessConfig = field(default_factory=ProcessConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        self.environment.validate()
        self.rollout.validate()
        self.training.validate(self.environment)
        if not self.model.id or not self.model.revision:
            raise ValueError("Model id and immutable revision are required")
        if self.model.local_path is not None and not Path(self.model.local_path).is_absolute():
            raise ValueError("model.local_path must be absolute when configured")
        if self.model.dropout != 0.0:
            raise ValueError("The first probability-accounting run requires dropout=0")
        if self.process_extension.gamma != 1.0:
            raise ValueError("The specified process-return boundary uses gamma=1")

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"Configuration section {name!r} must be a mapping")
    return value


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise TypeError("Configuration root must be a mapping")

    evaluation_raw = dict(_section(raw, "evaluation"))
    evaluation_raw["training_seeds"] = tuple(evaluation_raw.get("training_seeds", (11, 22, 33)))
    config = ExperimentConfig(
        project=str(raw["project"]),
        seed=int(raw.get("seed", 11)),
        environment=EnvironmentConfig(**_section(raw, "environment")),
        model=ModelConfig(**_section(raw, "model")),
        rollout=RolloutConfig(**_section(raw, "rollout")),
        training=TrainingConfig(**_section(raw, "training")),
        process_extension=ProcessConfig(**_section(raw, "process_extension")),
        evaluation=EvaluationConfig(**evaluation_raw),
    )
    config.validate()
    return config
