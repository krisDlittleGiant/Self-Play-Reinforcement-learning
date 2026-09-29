"""Explicit physical-device layout for four rollout and four trainer workers."""

from __future__ import annotations

from self_play_grpo.training.coordinator import RoleLayout


def role_layout_from_modules(value: str) -> RoleLayout:
    """Require exactly eight explicitly listed, non-overlapping HPU module IDs.

    The first four modules are rollout ranks 0..3; the next four are trainer
    ranks 0..3.  A launcher must not infer ownership from `hl-smi` visibility.
    """

    parts = value.split(",")
    if len(parts) != 8 or any(not part.strip() for part in parts):
        raise ValueError("D5 requires exactly eight comma-separated HPU module IDs")
    try:
        modules = tuple(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise ValueError("HPU module IDs must be integers") from exc
    return RoleLayout(
        rollout_modules=modules[:4],  # type: ignore[arg-type]
        trainer_modules=modules[4:],  # type: ignore[arg-type]
    )
