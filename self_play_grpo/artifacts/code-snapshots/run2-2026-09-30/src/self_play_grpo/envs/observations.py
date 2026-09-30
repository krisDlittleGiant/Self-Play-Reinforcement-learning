"""Canonical public state and action representations for Quoridor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True, order=True)
class Coordinate:
    """A pawn coordinate in zero-based column/row form."""

    x: int
    y: int

    @property
    def notation(self) -> str:
        return f"{chr(ord('a') + self.x)}{self.y + 1}"


@dataclass(frozen=True)
class BoardState:
    """Structured, canonical public state extracted from OpenSpiel."""

    board_size: int
    pawn_positions: tuple[Coordinate, Coordinate, Coordinate, Coordinate]
    walls_remaining: tuple[int, int, int, int]
    wall_cells: frozenset[tuple[int, int]]
    current_seat: int | None
    joint_action_index: int
    max_joint_actions: int

    @property
    def actions_remaining(self) -> int:
        return max(0, self.max_joint_actions - self.joint_action_index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "board_size": self.board_size,
            "pawn_positions": [asdict(position) for position in self.pawn_positions],
            "walls_remaining": list(self.walls_remaining),
            "wall_cells": [list(cell) for cell in sorted(self.wall_cells)],
            "current_seat": self.current_seat,
            "joint_action_index": self.joint_action_index,
            "max_joint_actions": self.max_joint_actions,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BoardState":
        positions = tuple(Coordinate(**position) for position in value["pawn_positions"])
        if len(positions) != 4:
            raise ValueError("A BoardState must contain exactly four pawn positions")
        walls = tuple(int(item) for item in value["walls_remaining"])
        if len(walls) != 4:
            raise ValueError("A BoardState must contain exactly four wall inventories")
        return cls(
            board_size=int(value["board_size"]),
            pawn_positions=positions,  # type: ignore[arg-type]
            walls_remaining=walls,  # type: ignore[arg-type]
            wall_cells=frozenset(tuple(map(int, cell)) for cell in value["wall_cells"]),
            current_seat=value.get("current_seat"),
            joint_action_index=int(value["joint_action_index"]),
            max_joint_actions=int(value["max_joint_actions"]),
        )


@dataclass(frozen=True)
class LegalAction:
    """Stable language-facing action mapped to one engine action ID."""

    engine_action: int
    label: str
    notation: str
    kind: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def action_label(notation: str) -> str:
    """Return an unambiguous, tokenizer-friendly label for an engine move."""

    cleaned = notation.strip().lower()
    if not cleaned:
        raise ValueError("Empty OpenSpiel action notation")
    prefix = "WALL" if cleaned.endswith(("h", "v")) else "MOVE"
    return f"{prefix}_{cleaned.upper()}"


def describe_action(notation: str) -> tuple[str, str]:
    cleaned = notation.strip().lower()
    if cleaned.endswith("h"):
        return "wall", f"place a horizontal wall starting at {cleaned[:-1]}"
    if cleaned.endswith("v"):
        return "wall", f"place a vertical wall starting at {cleaned[:-1]}"
    return "move", f"move the pawn to {cleaned}"


def canonical_actions(state: Any, engine_player: int, action_ids: Iterable[int]) -> tuple[LegalAction, ...]:
    actions: list[LegalAction] = []
    labels: set[str] = set()
    for action_id in action_ids:
        notation = str(state.action_to_string(engine_player, int(action_id))).strip()
        label = action_label(notation)
        if label in labels:
            raise RuntimeError(f"OpenSpiel produced duplicate legal notation {notation!r}")
        labels.add(label)
        kind, description = describe_action(notation)
        actions.append(
            LegalAction(
                engine_action=int(action_id),
                label=label,
                notation=notation,
                kind=kind,
                description=description,
            )
        )
    return tuple(sorted(actions, key=lambda item: item.label))


def rotate_coordinate(
    coordinate: Coordinate, board_size: int, quarter_turns_ccw: int
) -> Coordinate:
    """Rotate one square counter-clockwise around the board center."""

    if board_size <= 0:
        raise ValueError("board_size must be positive")
    x, y = coordinate.x, coordinate.y
    if not (0 <= x < board_size and 0 <= y < board_size):
        raise ValueError(f"Coordinate is outside a {board_size}x{board_size} board")
    for _ in range(quarter_turns_ccw % 4):
        x, y = y, board_size - 1 - x
    return Coordinate(x, y)


def rotate_action_notation(
    notation: str, board_size: int, quarter_turns_ccw: int
) -> str:
    """Rotate a move or wall anchor while preserving its physical action."""

    cleaned = notation.strip().lower()
    if len(cleaned) < 2:
        raise ValueError(f"Invalid Quoridor notation: {notation!r}")
    orientation = cleaned[-1] if cleaned[-1] in {"h", "v"} else ""
    square = cleaned[:-1] if orientation else cleaned
    try:
        x = ord(square[0]) - ord("a")
        y = int(square[1:]) - 1
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid Quoridor notation: {notation!r}") from exc
    coordinate_size = board_size - 1 if orientation else board_size
    if coordinate_size <= 0 or not (0 <= x < coordinate_size and 0 <= y < coordinate_size):
        raise ValueError(f"Action notation is outside a {board_size}x{board_size} board")
    for _ in range(quarter_turns_ccw % 4):
        x, y = y, coordinate_size - 1 - x
        if orientation:
            orientation = "v" if orientation == "h" else "h"
    return f"{chr(ord('a') + x)}{y + 1}{orientation}"


def player_relative_actions(
    actions: Sequence[LegalAction], seat: int, board_size: int
) -> tuple[LegalAction, ...]:
    """Rotate language-facing actions so the acting seat always aims upward."""

    if seat not in range(4):
        raise ValueError("seat must be in [0, 3]")
    relative: list[LegalAction] = []
    labels: set[str] = set()
    for action in actions:
        notation = rotate_action_notation(action.notation, board_size, seat)
        label = action_label(notation)
        if label in labels:
            raise RuntimeError(f"Relative rotation produced duplicate action {label!r}")
        labels.add(label)
        kind, description = describe_action(notation)
        relative.append(
            LegalAction(
                engine_action=action.engine_action,
                label=label,
                notation=notation,
                kind=kind,
                description=description,
            )
        )
    return tuple(sorted(relative, key=lambda item: item.label))


def player_relative_board(board: BoardState, seat: int) -> BoardState:
    """Rotate and renumber a canonical board from one acting player's view."""

    if seat not in range(4):
        raise ValueError("seat must be in [0, 3]")
    order = tuple((seat + offset) % 4 for offset in range(4))
    positions = tuple(
        rotate_coordinate(board.pawn_positions[canonical], board.board_size, seat)
        for canonical in order
    )
    diameter = 2 * board.board_size - 1
    rotated_walls = frozenset(
        (
            rotate_coordinate(Coordinate(x, y), diameter, seat).x,
            rotate_coordinate(Coordinate(x, y), diameter, seat).y,
        )
        for x, y in board.wall_cells
    )
    current = (
        None
        if board.current_seat is None
        else (board.current_seat - seat) % 4
    )
    return BoardState(
        board_size=board.board_size,
        pawn_positions=positions,  # type: ignore[arg-type]
        walls_remaining=tuple(board.walls_remaining[index] for index in order),  # type: ignore[arg-type]
        wall_cells=rotated_walls,
        current_seat=current,
        joint_action_index=board.joint_action_index,
        max_joint_actions=board.max_joint_actions,
    )


def _render_relative_grid(board: BoardState) -> str:
    diameter = 2 * board.board_size - 1
    cells = [[" " for _ in range(diameter)] for _ in range(diameter)]
    for y in range(0, diameter, 2):
        for x in range(0, diameter, 2):
            cells[y][x] = "."
    for x, y in board.wall_cells:
        cells[y][x] = "#"
    for player, position in enumerate(board.pawn_positions):
        cells[2 * position.y][2 * position.x] = str(player)
    header = "  " + " ".join(
        chr(ord("a") + column) for column in range(board.board_size)
    )
    rows = [header]
    for y, row in enumerate(cells):
        prefix = f"{y // 2 + 1:>2}" if y % 2 == 0 else "  "
        rows.append(f"{prefix}{''.join(row)}")
    return "\n".join(rows)


def _render_relative_recent(
    recent_actions: Sequence[dict[str, Any]], seat: int, board_size: int, limit: int = 12
) -> str:
    if not recent_actions:
        return "(none)"
    rows = []
    for action in recent_actions[-limit:]:
        absolute_notation = str(action.get("absolute_notation", action["notation"]))
        notation = rotate_action_notation(absolute_notation, board_size, seat)
        rows.append(
            f"{int(action['joint_step']):03d}: player "
            f"{(int(action['seat']) - seat) % 4} {action_label(notation)} ({notation})"
        )
    return "\n".join(rows)


@dataclass(frozen=True)
class MenuStyle:
    """Prompt presentation of the legal-action menu; never the legal set itself."""

    order: str = "sorted"
    walls: str = "full"
    moves: str = "plain"
    # Hash input for "shuffled": the match seed and joint step, so an order is
    # reproducible from a serialized environment and differs across matches.
    shuffle_key: str = ""

    @property
    def is_default(self) -> bool:
        return (self.order, self.walls, self.moves) == ("sorted", "full", "plain")


def menu_shuffle_key(seed: int, joint_action_index: int) -> str:
    return f"self_play_grpo/menu-order/v1:{int(seed)}:{int(joint_action_index)}"


def _shuffled(items: Sequence[LegalAction], key: str, group: str) -> list[LegalAction]:
    import hashlib
    import random

    digest = hashlib.sha256(f"{key}:{group}".encode("utf-8")).hexdigest()
    result = list(items)
    random.Random(int(digest[:16], 16)).shuffle(result)
    return result


def _move_direction(origin: Coordinate, notation: str) -> str:
    """Name a player-relative pawn move; the goal is always the top edge."""

    x = ord(notation[0]) - ord("a")
    y = int(notation[1:]) - 1
    dx, dy = x - origin.x, y - origin.y
    vertical = "forward" if dy < 0 else "backward" if dy > 0 else ""
    lateral = "left" if dx < 0 else "right" if dx > 0 else ""
    if vertical and lateral:
        direction = f"diagonally {vertical}-{lateral}"
    elif vertical:
        direction = vertical
    else:
        direction = f"sideways {lateral}"
    steps = max(abs(dx), abs(dy))
    suffix = " (jump)" if steps > 1 or (dx and dy) else ""
    return f"{direction}{suffix}"


def render_action_menu(
    legal_actions: Sequence[LegalAction],
    style: MenuStyle,
    *,
    mover_position: Coordinate | None = None,
) -> str:
    """Render the legal-action section of a prompt.

    The default style reproduces the original sorted, fully described menu
    byte for byte. Other styles only change ordering and wording.
    """

    if style.is_default:
        return "Legal actions:\n" + "\n".join(
            f"- {action.label}: {action.description}" for action in legal_actions
        )
    moves = [action for action in legal_actions if action.kind == "move"]
    walls = [action for action in legal_actions if action.kind == "wall"]
    if style.order == "shuffled":
        if not style.shuffle_key:
            raise ValueError("A shuffled action menu needs a match-specific key")
        moves = _shuffled(moves, style.shuffle_key, "moves")
        # A compact wall list stays in board order so squares remain findable.
        if style.walls == "full":
            walls = _shuffled(walls, style.shuffle_key, "walls")

    def move_line(action: LegalAction) -> str:
        if style.moves == "directional":
            if mover_position is None:
                raise ValueError("Directional move descriptions need the mover position")
            return (
                f"- {action.label}: move the pawn {_move_direction(mover_position, action.notation)} "
                f"to {action.notation}"
            )
        return f"- {action.label}: {action.description}"

    sections = []
    if style.walls == "full":
        lines = [move_line(action) for action in moves]
        lines += [f"- {action.label}: {action.description}" for action in walls]
        sections.append("Legal actions:\n" + "\n".join(lines))
    else:
        sections.append(
            "Legal pawn moves:\n"
            + ("\n".join(move_line(action) for action in moves) if moves else "(none)")
        )
        sections.append(
            "Legal wall placements (label WALL_<square><H|V>: a horizontal (H) or "
            "vertical (V) wall starting at that square):\n"
            + (" ".join(action.label for action in walls) if walls else "(none)")
        )
    return "\n\n".join(sections)


def render_player_relative_observation(
    canonical_board: BoardState,
    seat: int,
    legal_actions: Sequence[LegalAction],
    recent_actions: Sequence[dict[str, Any]],
    style: MenuStyle = MenuStyle(),
) -> str:
    """Render a rotation-normalized prompt with the actor always as player 0."""

    if canonical_board.current_seat != seat:
        raise ValueError(f"Observation requested for inactive seat {seat}")
    board = player_relative_board(canonical_board, seat)
    goals = (
        "top edge (row 1)",
        f"right edge (column {chr(ord('a') + board.board_size - 1)})",
        f"bottom edge (row {board.board_size})",
        "left edge (column a)",
    )
    players = "\n".join(
        f"- player {index}: pawn={position.notation}, goal={goals[index]}, "
        f"walls={board.walls_remaining[index]}"
        for index, position in enumerate(board.pawn_positions)
    )
    menu = render_action_menu(legal_actions, style, mover_position=board.pawn_positions[0])
    return (
        "You are player 0 in four-player Quoridor. Choose exactly one legal action.\n"
        "The board and action coordinates are rotated to your perspective: "
        "your goal is always the top edge.\n"
        "Return only the action label, followed by a newline. Do not explain.\n\n"
        "Current mover: player 0\n"
        f"Joint actions used: {board.joint_action_index}\n"
        f"Actions remaining before the experiment cap: {board.actions_remaining}\n"
        f"Board size: {board.board_size}x{board.board_size}\n"
        f"Players:\n{players}\n\n"
        f"Player-relative board (# marks wall material):\n{_render_relative_grid(board)}\n\n"
        "Recent public moves in this perspective:\n"
        f"{_render_relative_recent(recent_actions, seat, board.board_size)}\n\n"
        f"{menu}\n\n"
        "Action: "
    )


def _render_recent(recent_actions: Sequence[dict[str, Any]], limit: int = 12) -> str:
    if not recent_actions:
        return "(none)"
    rows = []
    for action in recent_actions[-limit:]:
        rows.append(
            f"{int(action['joint_step']):03d}: seat {int(action['seat'])} "
            f"{action['label']} ({action['notation']})"
        )
    return "\n".join(rows)


def render_observation(
    board: BoardState,
    seat: int,
    board_text: str,
    legal_actions: Sequence[LegalAction],
    recent_actions: Sequence[dict[str, Any]],
    style: MenuStyle = MenuStyle(),
) -> str:
    """Render a complete action-only prompt without omitting required state."""

    if board.current_seat != seat:
        raise ValueError(f"Observation requested for inactive seat {seat}")
    goals = (
        "top edge (row 1)",
        f"right edge (column {chr(ord('a') + board.board_size - 1)})",
        f"bottom edge (row {board.board_size})",
        "left edge (column a)",
    )
    players = "\n".join(
        f"- seat {index}: pawn={position.notation}, goal={goals[index]}, "
        f"walls={board.walls_remaining[index]}"
        for index, position in enumerate(board.pawn_positions)
    )
    menu = render_action_menu(legal_actions, style)
    return (
        "You are controlling one seat in four-player Quoridor. Choose exactly one legal action.\n"
        "Return only its label, followed by a newline. Do not explain.\n\n"
        f"Canonical seat: {seat}\n"
        f"Current mover: seat {board.current_seat}\n"
        f"Joint actions used: {board.joint_action_index}\n"
        f"Actions remaining before the experiment cap: {board.actions_remaining}\n"
        f"Board size: {board.board_size}x{board.board_size}\n"
        f"Players:\n{players}\n\n"
        f"Authoritative OpenSpiel board:\n{board_text.rstrip()}\n\n"
        f"Recent public moves:\n{_render_recent(recent_actions)}\n\n"
        f"{menu}\n\n"
        "Action: "
    )
