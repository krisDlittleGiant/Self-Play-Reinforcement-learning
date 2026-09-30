"""Preserve D5/D6 compatibility while fingerprinting the multi-update driver."""
from pathlib import Path

from self_play_grpo.rollouts.pilot import canonical_sha256, file_sha256
from self_play_grpo.training.d6_code_identity import d6_code_identity
from self_play_grpo.training.production_checkpoint import production_code_identity


FILES = ("multi_identity", "multi_recovery", "multi_resume", "multi_trainer",
         "multi_refresh", "multi_evidence", "multi_supervisor", "multi_run")


def multi_code_identity() -> str:
    directory = Path(__file__).resolve().parent
    return "source-sha256:" + canonical_sha256({
        "previous_generation": d6_code_identity(),
        "files": {name: file_sha256(directory / f"{name}.py") for name in FILES},
    })


def checkpoint_code_identity(update: int) -> str:
    if type(update) is not int or update < 1:
        raise ValueError("Checkpoint update must be positive")
    if update == 1:
        return production_code_identity()
    if update == 2:
        return d6_code_identity()
    return multi_code_identity()
