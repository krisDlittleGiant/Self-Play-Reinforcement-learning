"""Exact environment replay for a recorded complete match."""

from __future__ import annotations

from self_play_grpo.config import EnvironmentConfig
from self_play_grpo.envs.quoridor import QuoridorEnv
from self_play_grpo.rollouts.schema import MatchRecord


def replay_match(match: MatchRecord) -> QuoridorEnv:
    """Replay every recorded label and require state-by-state equality."""

    match.validate()
    env = QuoridorEnv(EnvironmentConfig(**match.environment_config))
    initial = env.reset(seed=match.seed)
    if initial != match.initial_state:
        raise RuntimeError("Recorded initial state differs from a fresh environment")

    for turn in match.turns:
        if env.state_view() != turn.state_before:
            raise RuntimeError(f"Replay diverged before joint step {turn.joint_step}")
        action = env.action_for_label(turn.policy_sample.chosen_label)
        if action.engine_action != turn.chosen_engine_action:
            raise RuntimeError(
                f"Engine action changed at joint step {turn.joint_step}: "
                f"{action.engine_action} != {turn.chosen_engine_action}"
            )
        if action.notation != turn.chosen_notation:
            raise RuntimeError(f"Action notation changed at joint step {turn.joint_step}")
        after = env.step(action)
        if after != turn.state_after:
            raise RuntimeError(f"Replay diverged after joint step {turn.joint_step}")

    if not env.is_terminal:
        raise RuntimeError("Recorded complete match replay did not terminate")
    if env.terminal_results() != match.final_results:
        raise RuntimeError("Replayed terminal results differ from the record")
    if env.termination_reason != match.termination_reason:
        raise RuntimeError("Replayed termination reason differs from the record")
    if env.serialize() != match.final_environment:
        raise RuntimeError("Replayed final environment is not byte-identical")
    return env
