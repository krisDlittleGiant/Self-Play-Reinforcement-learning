from dataclasses import dataclass
import random

from self_play_grpo.config import EnvironmentConfig
from self_play_grpo.envs.observations import BoardState, Coordinate, LegalAction
from self_play_grpo.envs.quoridor import SeatMap
from self_play_grpo.rollouts.collector import BatchedMatchCollector
from self_play_grpo.rollouts.schema import PolicySample


class FakeEnv:
    def __init__(self, terminal_after):
        self.config = EnvironmentConfig(max_joint_actions=terminal_after)
        self.seat_map = SeatMap.for_players(4)
        self.terminal_after = terminal_after
        self.steps = 0
        self._seed = 0

    def reset(self, seed=0):
        self.steps = 0
        self._seed = seed
        return self.state_view()

    @property
    def current_seat(self):
        return self.steps % 4

    @property
    def is_terminal(self):
        return self.steps >= self.terminal_after

    @property
    def termination_reason(self):
        return "natural_win" if self.is_terminal else None

    def state_view(self):
        return BoardState(
            board_size=9,
            pawn_positions=(
                Coordinate(4, 8),
                Coordinate(0, 4),
                Coordinate(4, 0),
                Coordinate(8, 4),
            ),
            walls_remaining=(5, 5, 5, 5),
            wall_cells=frozenset(),
            current_seat=None if self.is_terminal else self.current_seat,
            joint_action_index=self.steps,
            max_joint_actions=self.terminal_after,
        )

    def legal_actions(self):
        seat = self.current_seat
        return (
            LegalAction(seat, f"MOVE_A{seat + 1}", f"a{seat + 1}", "move", "move"),
        )

    def observation(self, seat):
        return f"seed={self._seed};seat={seat};step={self.steps}"

    def step(self, action):
        assert action in self.legal_actions()
        self.steps += 1
        return self.state_view()

    def terminal_results(self):
        assert self.is_terminal
        return (1.0, 0.0, 0.0, 0.0)

    def contract_manifest(self):
        return {"fixture": True}

    def serialize(self):
        return f"seed={self._seed};steps={self.steps}"


@dataclass
class FakeBatchedPolicy:
    name: str = "fake_batched"

    def __post_init__(self):
        self.batch_sizes = []

    def select_actions_batched(self, envs, *, game_ids, rngs):
        assert len(envs) == len(game_ids) == len(rngs)
        self.batch_sizes.append(len(envs))
        return tuple(
            PolicySample(
                chosen_label=env.legal_actions()[0].label,
                prompt_text=env.observation(env.current_seat),
                completion_text=f"{env.legal_actions()[0].label}\n",
                sampling_config={"random_probe": rng.random()},
            )
            for env, rng in zip(envs, rngs)
        )


def test_batched_collector_compacts_finished_games_and_preserves_order():
    policy = FakeBatchedPolicy()
    matches = BatchedMatchCollector(collect_progress=False).collect(
        (FakeEnv(2), FakeEnv(3), FakeEnv(4)),
        policy,
        game_ids=("g0", "g1", "g2"),
        seeds=(11, 12, 13),
        policy_version="p0",
    )
    assert policy.batch_sizes == [3, 3, 2, 1]
    assert [match.game_id for match in matches] == ["g0", "g1", "g2"]
    assert [len(match.turns) for match in matches] == [2, 3, 4]
    assert [match.seed for match in matches] == [11, 12, 13]
    assert all(match.policy_version == "p0" for match in matches)
    assert all(match.final_results == (1.0, 0.0, 0.0, 0.0) for match in matches)
    for match in matches:
        expected_rng = random.Random(match.seed)
        observed = [
            turn.policy_sample.sampling_config["random_probe"]
            for turn in match.turns
        ]
        assert observed == [expected_rng.random() for _ in match.turns]


def test_batched_collector_rejects_duplicate_game_ids():
    policy = FakeBatchedPolicy()
    try:
        BatchedMatchCollector(collect_progress=False).collect(
            (FakeEnv(1), FakeEnv(1)),
            policy,
            game_ids=("same", "same"),
            seeds=(1, 2),
            policy_version="p0",
        )
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate batched game IDs were accepted")
