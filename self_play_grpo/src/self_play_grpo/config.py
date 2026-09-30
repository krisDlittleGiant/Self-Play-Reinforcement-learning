"""Typed configuration with fail-closed experiment invariants."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


PINNED_ENGINE_REVISION = "d0606878b957274cc67a918ed173b36e9fe0fed6"

# Fields added after the first production run. The canonical dictionary omits
# them while they hold these defaults, so every earlier configuration keeps
# its recorded digest and every recorded environment serializes identically.
_LATER_FIELD_DEFAULTS: dict[str, dict[str, Any]] = {
    "environment": {
        "action_menu_order": "sorted",
        "wall_menu": "full",
        "move_descriptions": "plain",
        "wall_listing": "none",
    },
    "training": {"minibatches_per_update": 1, "advantage_baseline": "game"},
}


def canonical_section(name: str, section: Any) -> dict[str, Any]:
    """Return one section's canonical dictionary without default later fields."""

    from dataclasses import asdict

    value = asdict(section)
    for key, default in _LATER_FIELD_DEFAULTS.get(name, {}).items():
        if value.get(key) == default:
            del value[key]
    return value


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
    # Prompt presentation only; the legal set and the grammar are unchanged.
    # "shuffled" orders moves, then walls, by a hash of the match seed and
    # joint step. "compact" lists wall labels without per-wall descriptions.
    # "directional" states each player-relative move's direction to the goal.
    # "explicit" adds a line naming every placed wall and who placed it.
    action_menu_order: str = "sorted"
    wall_menu: str = "full"
    move_descriptions: str = "plain"
    wall_listing: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return canonical_section("environment", self)

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
        if self.action_menu_order not in {"sorted", "shuffled"}:
            raise ValueError("action_menu_order must be 'sorted' or 'shuffled'")
        if self.wall_menu not in {"full", "compact"}:
            raise ValueError("wall_menu must be 'full' or 'compact'")
        if self.move_descriptions not in {"plain", "directional"}:
            raise ValueError("move_descriptions must be 'plain' or 'directional'")
        if self.move_descriptions == "directional" and self.action_perspective != "player_relative":
            raise ValueError("Directional move descriptions require player_relative actions")
        if self.wall_listing not in {"none", "explicit"}:
            raise ValueError("wall_listing must be 'none' or 'explicit'")


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
    # Optimizer steps per collected batch. Each complete match is used once;
    # the batch is split into this many whole-match minibatches in order.
    minibatches_per_update: int = 1
    # "game": standardize the four seats within each match (recorded credit).
    # "seat_loo": subtract each seat's mean result over the batch's other
    # matches, removing the structural turn-order advantage of some seats.
    advantage_baseline: str = "game"

    def validate(self, env: EnvironmentConfig) -> None:
        if self.reward_mode not in {"outcome", "potential", "gae_blend"}:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")
        if self.group_key != "game_id" or self.group_size != 4:
            raise ValueError("Advantages must be grouped as four seats of one game")
        if self.optimizer_epochs_per_batch != 1:
            raise ValueError("The initial on-policy implementation permits exactly one optimizer epoch")
        if type(self.minibatches_per_update) is not int or self.minibatches_per_update <= 0:
            raise ValueError("minibatches_per_update must be a positive integer")
        if self.advantage_baseline not in {"game", "seat_loo"}:
            raise ValueError("advantage_baseline must be 'game' or 'seat_loo'")
        if self.advantage_baseline == "seat_loo" and self.reward_mode != "outcome":
            raise ValueError("The seat baseline is defined for outcome rewards only")
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

        value = asdict(self)
        for name in _LATER_FIELD_DEFAULTS:
            value[name] = canonical_section(name, getattr(self, name))
        return value


def expected_gradient_sync_phases(config: Any) -> int:
    """One post-backward gradient synchronization per optimizer minibatch.

    Accepts a configuration object or its canonical dictionary; an absent
    field means the single-minibatch baseline.
    """

    if isinstance(config, Mapping):
        training = config.get("training", {})
        value = training.get("minibatches_per_update", 1) if isinstance(training, Mapping) else 1
    else:
        value = getattr(getattr(config, "training", None), "minibatches_per_update", 1)
    if type(value) is not int or value <= 0:
        raise ValueError("Configured minibatches_per_update is invalid")
    return value


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
