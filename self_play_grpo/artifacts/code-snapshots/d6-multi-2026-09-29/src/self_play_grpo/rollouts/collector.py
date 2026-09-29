"""Synchronous four-seat collection against one authoritative environment."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Protocol, Sequence

from self_play_grpo.envs.quoridor import QuoridorEnv
from self_play_grpo.rewards.outcome import outcome_advantages
from self_play_grpo.rewards.progress import progress_features
from self_play_grpo.rollouts.schema import (
    MatchRecord,
    PolicySample,
    TurnCredit,
    TurnRecord,
)


class ActionPolicy(Protocol):
    """A policy that chooses one action without changing the environment."""

    name: str

    def select_action(
        self,
        env: QuoridorEnv,
        *,
        game_id: str,
        rng: random.Random,
    ) -> PolicySample: ...


class BatchedActionPolicy(Protocol):
    """A shared policy that samples one action for each independent match."""

    name: str

    def select_actions_batched(
        self,
        envs: Sequence[QuoridorEnv],
        *,
        game_ids: Sequence[str],
        rngs: Sequence[random.Random],
    ) -> tuple[PolicySample, ...]: ...


@dataclass
class _ActiveMatch:
    index: int
    env: QuoridorEnv
    game_id: str
    seed: int
    rng: random.Random
    initial_state: object
    engine_manifest: dict
    turns: list[TurnRecord]
    local_steps: list[int]


def _close_match(active: _ActiveMatch, policy_version: str) -> MatchRecord:
    results = active.env.terminal_results()
    advantages = outcome_advantages(results)
    for seat in range(4):
        seat_turns = [turn for turn in active.turns if turn.seat == seat]
        for index, turn in enumerate(seat_turns):
            next_turn = seat_turns[index + 1] if index + 1 < len(seat_turns) else None
            turn.credit.terminal_result = results[seat]
            turn.credit.outcome_advantage = advantages[seat]
            turn.credit.training_advantage = advantages[seat]
            turn.credit.next_own_decision_joint_step = (
                next_turn.joint_step if next_turn is not None else None
            )
    reason = active.env.termination_reason
    if reason is None:
        raise RuntimeError("Terminal match is missing its termination reason")
    record = MatchRecord(
        game_id=active.game_id,
        policy_version=policy_version,
        seed=active.seed,
        environment_config=asdict(active.env.config),
        engine_manifest=active.engine_manifest,
        seat_map=active.env.seat_map.engine_players_by_seat,  # type: ignore[arg-type]
        initial_state=active.initial_state,  # type: ignore[arg-type]
        turns=active.turns,
        final_results=results,
        termination_reason=reason,
        final_environment=active.env.serialize(),
    )
    record.validate()
    return record


def _append_turn(
    active: _ActiveMatch,
    sample: PolicySample,
    *,
    policy_name: str,
    policy_version: str,
    collect_progress: bool,
    proxy_temperature: float,
) -> None:
    env = active.env
    seat = env.current_seat
    state_before = env.state_view()
    legal_actions = env.legal_actions()
    observation = env.observation(seat)
    sample.validate()
    selected = next(
        (action for action in legal_actions if action.label == sample.chosen_label),
        None,
    )
    if selected is None:
        raise RuntimeError(
            f"Policy {policy_name!r} returned an illegal action label "
            f"{sample.chosen_label!r}; no substitution was made"
        )
    before_progress = (
        progress_features(state_before, proxy_temperature)
        if collect_progress
        else {}
    )
    state_after = env.step(selected)
    after_progress = (
        progress_features(state_after, proxy_temperature)
        if collect_progress
        else {}
    )
    active.turns.append(
        TurnRecord(
            game_id=active.game_id,
            joint_step=len(active.turns),
            seat=seat,
            engine_player=env.seat_map.engine_player(seat),
            player_local_step=active.local_steps[seat],
            policy_version=policy_version,
            state_before=state_before,
            state_after=state_after,
            observation=observation,
            legal_actions=legal_actions,
            chosen_engine_action=selected.engine_action,
            chosen_notation=selected.notation,
            policy_sample=sample,
            credit=TurnCredit(
                progress_before=before_progress,
                progress_after=after_progress,
            ),
        )
    )
    active.local_steps[seat] += 1


class MatchCollector:
    def __init__(self, *, collect_progress: bool = True, proxy_temperature: float = 2.0) -> None:
        self.collect_progress = collect_progress
        self.proxy_temperature = proxy_temperature

    def collect(
        self,
        env: QuoridorEnv,
        policies: ActionPolicy | Sequence[ActionPolicy],
        *,
        game_id: str,
        seed: int,
        policy_version: str,
    ) -> MatchRecord:
        if isinstance(policies, Sequence):
            if len(policies) != 4:
                raise ValueError("A policy sequence must provide exactly four seats")
            seat_policies = tuple(policies)
        else:
            seat_policies = (policies, policies, policies, policies)

        rng = random.Random(seed)
        initial_state = env.reset(seed=seed)
        engine_manifest = env.contract_manifest()
        turns: list[TurnRecord] = []
        local_steps = [0, 0, 0, 0]

        while not env.is_terminal:
            seat = env.current_seat
            state_before = env.state_view()
            legal_actions = env.legal_actions()
            observation = env.observation(seat)
            sample = seat_policies[seat].select_action(
                env, game_id=game_id, rng=rng
            )
            sample.validate()
            selected = next(
                (action for action in legal_actions if action.label == sample.chosen_label),
                None,
            )
            if selected is None:
                raise RuntimeError(
                    f"Policy {seat_policies[seat].name!r} returned an illegal action label "
                    f"{sample.chosen_label!r}; no substitution was made"
                )
            before_progress = (
                progress_features(state_before, self.proxy_temperature)
                if self.collect_progress
                else {}
            )
            state_after = env.step(selected)
            after_progress = (
                progress_features(state_after, self.proxy_temperature)
                if self.collect_progress
                else {}
            )
            turns.append(
                TurnRecord(
                    game_id=game_id,
                    joint_step=len(turns),
                    seat=seat,
                    engine_player=env.seat_map.engine_player(seat),
                    player_local_step=local_steps[seat],
                    policy_version=policy_version,
                    state_before=state_before,
                    state_after=state_after,
                    observation=observation,
                    legal_actions=legal_actions,
                    chosen_engine_action=selected.engine_action,
                    chosen_notation=selected.notation,
                    policy_sample=sample,
                    credit=TurnCredit(
                        progress_before=before_progress,
                        progress_after=after_progress,
                    ),
                )
            )
            local_steps[seat] += 1

        results = env.terminal_results()
        advantages = outcome_advantages(results)
        for seat in range(4):
            seat_turns = [turn for turn in turns if turn.seat == seat]
            for index, turn in enumerate(seat_turns):
                next_turn = seat_turns[index + 1] if index + 1 < len(seat_turns) else None
                turn.credit.terminal_result = results[seat]
                turn.credit.outcome_advantage = advantages[seat]
                turn.credit.training_advantage = advantages[seat]
                turn.credit.next_own_decision_joint_step = (
                    next_turn.joint_step if next_turn is not None else None
                )

        reason = env.termination_reason
        if reason is None:
            raise RuntimeError("Terminal match is missing its termination reason")
        record = MatchRecord(
            game_id=game_id,
            policy_version=policy_version,
            seed=seed,
            environment_config=asdict(env.config),
            engine_manifest=engine_manifest,
            seat_map=env.seat_map.engine_players_by_seat,  # type: ignore[arg-type]
            initial_state=initial_state,
            turns=turns,
            final_results=results,
            termination_reason=reason,
            final_environment=env.serialize(),
        )
        record.validate()
        return record


class BatchedMatchCollector:
    """Advance independent complete matches through one shared model batch."""

    def __init__(self, *, collect_progress: bool = True, proxy_temperature: float = 2.0) -> None:
        self.collect_progress = collect_progress
        self.proxy_temperature = proxy_temperature

    def collect(
        self,
        envs: Sequence[QuoridorEnv],
        policy: BatchedActionPolicy,
        *,
        game_ids: Sequence[str],
        seeds: Sequence[int],
        policy_version: str,
    ) -> tuple[MatchRecord, ...]:
        if not envs:
            raise ValueError("At least one environment is required")
        if len(envs) != len(game_ids) or len(envs) != len(seeds):
            raise ValueError("envs, game_ids, and seeds must have equal lengths")
        if len(set(game_ids)) != len(game_ids):
            raise ValueError("Batched game IDs must be unique")

        active = [
            _ActiveMatch(
                index=index,
                env=env,
                game_id=str(game_ids[index]),
                seed=int(seeds[index]),
                rng=random.Random(int(seeds[index])),
                initial_state=env.reset(seed=int(seeds[index])),
                engine_manifest=env.contract_manifest(),
                turns=[],
                local_steps=[0, 0, 0, 0],
            )
            for index, env in enumerate(envs)
        ]
        completed: list[MatchRecord | None] = [None] * len(active)
        while active:
            samples = policy.select_actions_batched(
                tuple(item.env for item in active),
                game_ids=tuple(item.game_id for item in active),
                rngs=tuple(item.rng for item in active),
            )
            if len(samples) != len(active):
                raise RuntimeError(
                    f"Batched policy returned {len(samples)} samples for "
                    f"{len(active)} environments"
                )
            survivors: list[_ActiveMatch] = []
            for item, sample in zip(active, samples):
                _append_turn(
                    item,
                    sample,
                    policy_name=policy.name,
                    policy_version=policy_version,
                    collect_progress=self.collect_progress,
                    proxy_temperature=self.proxy_temperature,
                )
                if item.env.is_terminal:
                    completed[item.index] = _close_match(item, policy_version)
                else:
                    survivors.append(item)
            active = survivors
        if any(record is None for record in completed):
            raise RuntimeError("Batched collector exited with incomplete matches")
        return tuple(record for record in completed if record is not None)
