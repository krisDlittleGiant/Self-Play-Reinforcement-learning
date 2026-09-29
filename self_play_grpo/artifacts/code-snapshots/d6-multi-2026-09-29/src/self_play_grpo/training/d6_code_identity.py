"""Versioned source identity for the first resumed production update."""

from __future__ import annotations

from pathlib import Path

from self_play_grpo.rollouts.pilot import canonical_sha256, file_sha256
from self_play_grpo.training.production_checkpoint import production_code_identity


_D6_FILES = (
    "training/d6_code_identity.py",
    "training/d6_collection.py",
    "training/d6_recovery.py",
    "training/d6_trainer_resume.py",
    "training/d6_trainer_worker.py",
    "training/d6_refresh_worker.py",
    "training/d6_cycle.py",
)


def d6_code_identity() -> str:
    """Bind D5 core plus the exact D6 worker/orchestration implementation."""

    package = Path(__file__).resolve().parents[1]
    return "source-sha256:" + canonical_sha256({
        "d5_core_identity": production_code_identity(),
        "d6_files": {name: file_sha256(package / name) for name in _D6_FILES},
    })
