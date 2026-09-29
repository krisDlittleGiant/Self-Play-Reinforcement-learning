"""Versioned JSON schema for joint matches and owned policy samples."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from self_play_grpo.envs.observations import BoardState, LegalAction
from self_play_grpo.rewards.outcome import validate_result_vector


SCHEMA_VERSION = 1


@dataclass
class PolicySample:
    chosen_label: str
    prompt_text: str
    completion_text: str
    prompt_token_ids: tuple[int, ...] = ()
    completion_token_ids: tuple[int, ...] = ()
    behavior_log_probs: tuple[float, ...] = ()
    allowed_token_ids: tuple[tuple[int, ...], ...] = ()
    attention_mask: tuple[int, ...] = ()
    loss_mask: tuple[int, ...] = ()
    sampling_config: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        token_count = len(self.completion_token_ids)
        aligned = {
            "behavior_log_probs": len(self.behavior_log_probs),
            "allowed_token_ids": len(self.allowed_token_ids),
            "loss_mask": len(self.loss_mask),
        }
        for name, length in aligned.items():
            if length not in {0, token_count}:
                raise ValueError(f"{name} has length {length}, expected 0 or {token_count}")
        if self.attention_mask and len(self.attention_mask) != len(self.prompt_token_ids) + token_count:
            raise ValueError("attention_mask does not align with prompt and completion tokens")
        for token, allowed in zip(self.completion_token_ids, self.allowed_token_ids):
            if token not in allowed:
                raise ValueError("Sampled token is absent from its recorded allowed-token set")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PolicySample":
        sample = cls(
            chosen_label=str(value["chosen_label"]),
            prompt_text=str(value.get("prompt_text", "")),
            completion_text=str(value.get("completion_text", "")),
            prompt_token_ids=tuple(map(int, value.get("prompt_token_ids", ()))),
            completion_token_ids=tuple(map(int, value.get("completion_token_ids", ()))),
            behavior_log_probs=tuple(map(float, value.get("behavior_log_probs", ()))),
            allowed_token_ids=tuple(
                tuple(map(int, row)) for row in value.get("allowed_token_ids", ())
            ),
            attention_mask=tuple(map(int, value.get("attention_mask", ()))),
            loss_mask=tuple(map(int, value.get("loss_mask", ()))),
            sampling_config=dict(value.get("sampling_config", {})),
        )
        sample.validate()
        return sample


@dataclass
class TurnCredit:
    terminal_result: float | None = None
    outcome_advantage: float | None = None
    training_advantage: float | None = None
    evaluator_version: str | None = None
    progress_before: dict[str, Any] = field(default_factory=dict)
    progress_after: dict[str, Any] = field(default_factory=dict)
    next_own_decision_joint_step: int | None = None


@dataclass
class TurnRecord:
    game_id: str
    joint_step: int
    seat: int
    engine_player: int
    player_local_step: int
    policy_version: str
    state_before: BoardState
    state_after: BoardState
    observation: str
    legal_actions: tuple[LegalAction, ...]
    chosen_engine_action: int
    chosen_notation: str
    policy_sample: PolicySample
    credit: TurnCredit = field(default_factory=TurnCredit)

    def validate(self) -> None:
        if self.seat not in range(4):
            raise ValueError("Turn seat must be in [0, 3]")
        if self.state_before.current_seat != self.seat:
            raise ValueError("Turn owner differs from the state's active canonical seat")
        mapping = {action.label: action for action in self.legal_actions}
        selected = mapping.get(self.policy_sample.chosen_label)
        if selected is None:
            raise ValueError("Chosen label is not in the recorded legal action menu")
        if selected.engine_action != self.chosen_engine_action:
            raise ValueError("Chosen label and engine action disagree")
        if selected.notation != self.chosen_notation:
            raise ValueError("Chosen label and notation disagree")
        self.policy_sample.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "game_id": self.game_id,
            "joint_step": self.joint_step,
            "seat": self.seat,
            "engine_player": self.engine_player,
            "player_local_step": self.player_local_step,
            "policy_version": self.policy_version,
            "state_before": self.state_before.to_dict(),
            "state_after": self.state_after.to_dict(),
            "observation": self.observation,
            "legal_actions": [action.to_dict() for action in self.legal_actions],
            "chosen_engine_action": self.chosen_engine_action,
            "chosen_notation": self.chosen_notation,
            "policy_sample": self.policy_sample.to_dict(),
            "credit": asdict(self.credit),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TurnRecord":
        record = cls(
            game_id=str(value["game_id"]),
            joint_step=int(value["joint_step"]),
            seat=int(value["seat"]),
            engine_player=int(value["engine_player"]),
            player_local_step=int(value["player_local_step"]),
            policy_version=str(value["policy_version"]),
            state_before=BoardState.from_dict(dict(value["state_before"])),
            state_after=BoardState.from_dict(dict(value["state_after"])),
            observation=str(value["observation"]),
            legal_actions=tuple(LegalAction(**item) for item in value["legal_actions"]),
            chosen_engine_action=int(value["chosen_engine_action"]),
            chosen_notation=str(value["chosen_notation"]),
            policy_sample=PolicySample.from_dict(value["policy_sample"]),
            credit=TurnCredit(**dict(value.get("credit", {}))),
        )
        record.validate()
        return record


@dataclass
class MatchRecord:
    game_id: str
    policy_version: str
    seed: int
    environment_config: dict[str, Any]
    engine_manifest: dict[str, Any]
    seat_map: tuple[int, int, int, int]
    initial_state: BoardState
    turns: list[TurnRecord]
    final_results: tuple[float, float, float, float]
    termination_reason: str
    final_environment: str
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported match schema")
        validate_result_vector(self.final_results)
        if self.seat_map != (0, 2, 1, 3):
            raise ValueError("Unexpected four-player OpenSpiel seat map")
        local_steps = [0, 0, 0, 0]
        for expected_step, turn in enumerate(self.turns):
            turn.validate()
            if turn.game_id != self.game_id or turn.policy_version != self.policy_version:
                raise ValueError("A match contains mixed game or policy versions")
            if turn.joint_step != expected_step:
                raise ValueError("Joint steps are not contiguous")
            if turn.player_local_step != local_steps[turn.seat]:
                raise ValueError("Player-local steps are not contiguous")
            local_steps[turn.seat] += 1
            if turn.credit.terminal_result is None:
                raise ValueError("A closed match turn lacks its terminal result")
            if abs(turn.credit.terminal_result - self.final_results[turn.seat]) > 1e-9:
                raise ValueError("A turn received another seat's terminal result")

    def player_views(self) -> tuple[tuple[TurnRecord, ...], ...]:
        return tuple(
            tuple(turn for turn in self.turns if turn.seat == seat)
            for seat in range(4)
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "game_id": self.game_id,
            "policy_version": self.policy_version,
            "seed": self.seed,
            "environment_config": self.environment_config,
            "engine_manifest": self.engine_manifest,
            "seat_map": list(self.seat_map),
            "initial_state": self.initial_state.to_dict(),
            "turns": [turn.to_dict() for turn in self.turns],
            "final_results": list(self.final_results),
            "termination_reason": self.termination_reason,
            "final_environment": self.final_environment,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MatchRecord":
        record = cls(
            schema_version=int(value.get("schema_version", 0)),
            game_id=str(value["game_id"]),
            policy_version=str(value["policy_version"]),
            seed=int(value["seed"]),
            environment_config=dict(value["environment_config"]),
            engine_manifest=dict(value["engine_manifest"]),
            seat_map=tuple(map(int, value["seat_map"])),  # type: ignore[arg-type]
            initial_state=BoardState.from_dict(dict(value["initial_state"])),
            turns=[TurnRecord.from_dict(item) for item in value["turns"]],
            final_results=validate_result_vector(value["final_results"]),
            termination_reason=str(value["termination_reason"]),
            final_environment=str(value["final_environment"]),
        )
        record.validate()
        return record

    @classmethod
    def from_json(cls, payload: str) -> "MatchRecord":
        return cls.from_dict(json.loads(payload))


def write_matches_jsonl(path: str | Path, matches: Sequence[MatchRecord]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for match in matches:
            handle.write(match.to_json())
            handle.write("\n")
    temporary.replace(target)


def read_matches_jsonl(path: str | Path) -> list[MatchRecord]:
    records: list[MatchRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(MatchRecord.from_json(line))
            except Exception as exc:
                raise ValueError(f"Invalid match record on line {line_number}") from exc
    return records


def ensure_single_policy_version(matches: Iterable[MatchRecord]) -> str:
    versions = {match.policy_version for match in matches}
    if len(versions) != 1:
        raise ValueError(f"Rollout batch contains mixed behavior versions: {sorted(versions)}")
    return next(iter(versions))

