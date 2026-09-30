"""OpenSpiel Quoridor adapter with explicit canonical-seat semantics."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Iterable, Mapping

import numpy as np

from self_play_grpo.config import EnvironmentConfig
from self_play_grpo.envs.observations import (
    BoardState,
    Coordinate,
    LegalAction,
    MenuStyle,
    canonical_actions,
    menu_shuffle_key,
    player_relative_actions,
    render_observation,
    render_player_relative_observation,
)


@dataclass(frozen=True)
class SeatMap:
    """Map OpenSpiel's internal pawn IDs to clockwise canonical seats."""

    engine_players_by_seat: tuple[int, ...]

    @classmethod
    def for_players(cls, players: int) -> "SeatMap":
        orders = {2: (0, 1), 3: (0, 2, 1), 4: (0, 2, 1, 3)}
        try:
            return cls(orders[players])
        except KeyError as exc:
            raise ValueError("Quoridor supports two to four players") from exc

    def engine_player(self, seat: int) -> int:
        return self.engine_players_by_seat[seat]

    def seat(self, engine_player: int) -> int:
        try:
            return self.engine_players_by_seat.index(engine_player)
        except ValueError as exc:
            raise RuntimeError(f"Unknown engine player ID {engine_player}") from exc


class QuoridorEnv:
    """One authoritative capped four-player Quoridor match."""

    def __init__(self, config: EnvironmentConfig, *, game: Any | None = None) -> None:
        config.validate()
        self.config = config
        self.seat_map = SeatMap.for_players(config.players)
        if game is None:
            try:
                import pyspiel  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError(
                    "pyspiel is unavailable; install the pinned 'engine' dependency first"
                ) from exc
            game = pyspiel.load_game(
                "quoridor",
                {
                    "players": config.players,
                    "board_size": config.board_size,
                    "wall_count": config.wall_count,
                    "ansi_color_output": False,
                },
            )
        self._game = game
        self._state: Any = None
        self._joint_actions = 0
        self._seed = 0
        self._horizon_reached = False
        self._termination_reason: str | None = None
        self._action_history: list[dict[str, Any]] = []
        self.reset()

    @property
    def joint_actions(self) -> int:
        return self._joint_actions

    @property
    def action_history(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._action_history)

    @property
    def current_seat(self) -> int:
        if self.is_terminal:
            raise RuntimeError("A terminal match has no current seat")
        return self.seat_map.seat(int(self._state.current_player()))

    @property
    def is_terminal(self) -> bool:
        return bool(self._state.is_terminal()) or self._horizon_reached

    @property
    def termination_reason(self) -> str | None:
        return self._termination_reason

    def reset(self, seed: int = 0) -> BoardState:
        # OpenSpiel Quoridor is deterministic; the seed is provenance for the
        # sampling policies, not hidden environment randomness.
        self._state = self._game.new_initial_state()
        self._joint_actions = 0
        self._seed = int(seed)
        self._horizon_reached = False
        self._termination_reason = None
        self._action_history = []
        return self.state_view()

    def legal_actions(self) -> tuple[LegalAction, ...]:
        if self.is_terminal:
            return ()
        engine_player = int(self._state.current_player())
        actions = canonical_actions(
            self._state, engine_player, self._state.legal_actions()
        )
        if self.config.action_perspective == "player_relative":
            return player_relative_actions(
                actions, self.current_seat, self.config.board_size
            )
        return actions

    def action_for_label(self, label: str) -> LegalAction:
        matches = [action for action in self.legal_actions() if action.label == label.strip()]
        if len(matches) != 1:
            raise ValueError(f"Action label {label!r} is not uniquely legal in the current state")
        return matches[0]

    def menu_style(self) -> MenuStyle:
        return MenuStyle(
            order=self.config.action_menu_order,
            walls=self.config.wall_menu,
            moves=self.config.move_descriptions,
            shuffle_key=menu_shuffle_key(self._seed, self._joint_actions),
        )

    def observation(self, seat: int) -> str:
        board = self.state_view()
        actions = self.legal_actions()
        if self.config.action_perspective == "player_relative":
            return render_player_relative_observation(
                board,
                seat,
                actions,
                self._action_history,
                self.menu_style(),
            )
        return render_observation(
            board,
            seat,
            str(self._state),
            actions,
            self._action_history,
            self.menu_style(),
        )

    def state_view(self) -> BoardState:
        diameter = self.config.board_size * 2 - 1
        plane_count = 2 * self.config.players + 1
        engine_observer = self.seat_map.engine_player(0)
        values = np.asarray(
            self._state.observation_tensor(engine_observer), dtype=np.float32
        ).reshape(plane_count, diameter, diameter)

        positions_by_engine: dict[int, Coordinate] = {}
        for engine_player in range(self.config.players):
            locations = np.argwhere(values[engine_player] > 0.5)
            if len(locations) != 1:
                raise RuntimeError(
                    f"Expected one pawn for engine player {engine_player}, found {len(locations)}"
                )
            y, x = map(int, locations[0])
            if x % 2 or y % 2:
                raise RuntimeError("OpenSpiel placed a pawn off the square lattice")
            positions_by_engine[engine_player] = Coordinate(x=x // 2, y=y // 2)

        pawn_positions = tuple(
            positions_by_engine[self.seat_map.engine_player(seat)]
            for seat in range(self.config.players)
        )
        wall_plane = self.config.players
        wall_cells = frozenset(
            (int(x), int(y))
            for y, x in np.argwhere(values[wall_plane] > 0.5)
        )
        walls_remaining = tuple(
            int(round(float(values[self.config.players + 1 + self.seat_map.engine_player(seat), 0, 0])))
            for seat in range(self.config.players)
        )
        current = None if self.is_terminal else self.seat_map.seat(int(self._state.current_player()))
        return BoardState(
            board_size=self.config.board_size,
            pawn_positions=pawn_positions,  # type: ignore[arg-type]
            walls_remaining=walls_remaining,  # type: ignore[arg-type]
            wall_cells=wall_cells,
            current_seat=current,
            joint_action_index=self._joint_actions,
            max_joint_actions=self.config.max_joint_actions,
        )

    def step(self, action: int | str | LegalAction) -> BoardState:
        if self.is_terminal:
            raise RuntimeError("Cannot step a terminal match")
        legal = self.legal_actions()
        by_id = {item.engine_action: item for item in legal}
        if isinstance(action, LegalAction):
            selected = by_id.get(action.engine_action)
        elif isinstance(action, str):
            selected = next((item for item in legal if item.label == action.strip()), None)
        else:
            selected = by_id.get(int(action))
        if selected is None:
            raise ValueError("The selected action is not legal in the current state")

        seat = self.current_seat
        absolute = None
        if self.config.action_perspective == "player_relative":
            engine_player = int(self._state.current_player())
            absolute = next(
                item
                for item in canonical_actions(
                    self._state, engine_player, self._state.legal_actions()
                )
                if item.engine_action == selected.engine_action
            )
        if hasattr(self._state, "apply_action_with_legality_check"):
            self._state.apply_action_with_legality_check(selected.engine_action)
        else:
            self._state.apply_action(selected.engine_action)
        self._action_history.append(
            {
                "joint_step": self._joint_actions,
                "seat": seat,
                "engine_player": self.seat_map.engine_player(seat),
                "engine_action": selected.engine_action,
                "label": selected.label,
                "notation": selected.notation,
                **(
                    {
                        "absolute_label": absolute.label,
                        "absolute_notation": absolute.notation,
                    }
                    if absolute is not None
                    else {}
                ),
            }
        )
        self._joint_actions += 1

        if self._state.is_terminal():
            returns = tuple(float(item) for item in self._state.returns())
            self._termination_reason = "engine_draw" if all(abs(item) < 1e-12 for item in returns) else "natural_win"
        elif self._joint_actions >= self.config.max_joint_actions:
            self._horizon_reached = True
            self._termination_reason = "horizon_draw"
        return self.state_view()

    def terminal_results(self) -> tuple[float, float, float, float]:
        if not self.is_terminal:
            raise RuntimeError("Terminal results requested before the match ended")
        if self._horizon_reached:
            return (0.25, 0.25, 0.25, 0.25)

        utilities = tuple(float(item) for item in self._state.returns())
        if len(utilities) != self.config.players:
            raise RuntimeError("OpenSpiel returned the wrong number of utilities")
        if all(abs(item) < 1e-12 for item in utilities):
            return (0.25, 0.25, 0.25, 0.25)
        n = self.config.players
        results = tuple(((n - 1) * utility + 1.0) / n for utility in utilities)
        if not math.isclose(sum(results), 1.0, abs_tol=1e-9):
            raise RuntimeError(f"Converted terminal results do not sum to one: {results}")
        if any(item < -1e-9 or item > 1 + 1e-9 for item in results):
            raise RuntimeError(f"Converted terminal result is outside [0, 1]: {results}")
        return results  # type: ignore[return-value]

    def clone(self) -> "QuoridorEnv":
        clone = object.__new__(QuoridorEnv)
        clone.config = self.config
        clone.seat_map = self.seat_map
        clone._game = self._game
        clone._state = self._state.clone()
        clone._joint_actions = self._joint_actions
        clone._seed = self._seed
        clone._horizon_reached = self._horizon_reached
        clone._termination_reason = self._termination_reason
        clone._action_history = [dict(item) for item in self._action_history]
        return clone

    def serialize(self) -> str:
        value = {
            "schema_version": 1,
            "environment": self.config.to_dict(),
            "seed": self._seed,
            "actions": self._action_history,
            "joint_actions": self._joint_actions,
            "horizon_reached": self._horizon_reached,
            "termination_reason": self._termination_reason,
            "engine_state": self._state.serialize(),
        }
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @classmethod
    def deserialize(cls, payload: str, *, game: Any | None = None) -> "QuoridorEnv":
        value = json.loads(payload)
        if value.get("schema_version") != 1:
            raise ValueError("Unsupported serialized environment schema")
        env = cls(EnvironmentConfig(**value["environment"]), game=game)
        env.reset(seed=int(value["seed"]))
        for expected in value["actions"]:
            legal = {item.label: item for item in env.legal_actions()}
            selected = legal.get(expected["label"])
            if selected is None or selected.notation != expected["notation"]:
                raise RuntimeError("Replay diverged from its recorded legal-action mapping")
            env.step(selected)
        if env.serialize() != payload:
            raise RuntimeError("Environment replay was not byte-for-byte reproducible")
        return env

    def contract_manifest(self) -> dict[str, Any]:
        try:
            package_version = metadata.version("open_spiel")
        except metadata.PackageNotFoundError:
            package_version = "unknown"
        parameters: Mapping[str, Any] = self._game.get_parameters()
        return {
            "package_version": package_version,
            "expected_revision": self.config.engine_revision,
            "parameters": {key: str(value) for key, value in parameters.items()},
            "seat_map": list(self.seat_map.engine_players_by_seat),
            "initial_current_engine_player": int(self._state.current_player()),
            "initial_current_seat": self.current_seat,
            "initial_legal_actions": [item.to_dict() for item in self.legal_actions()],
        }

    def apply_labels(self, labels: Iterable[str]) -> None:
        for label in labels:
            self.step(label)
