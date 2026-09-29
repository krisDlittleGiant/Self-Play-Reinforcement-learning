import json

import pytest

from self_play_grpo.rollouts.pilot import PilotManifest, PilotMatchEntry
from self_play_grpo.training.distributed import (
    trainer_match_indices,
    validate_distributed_training_manifest,
    write_rank_update_report,
)


def _config(games: int = 16) -> dict[str, object]:
    return {
        "model": {"revision": "revision"},
        "rollout": {"games_per_update": games},
    }


def _manifest(completed: int = 16, target: int = 16) -> PilotManifest:
    manifest = PilotManifest.create(
        config=_config(),
        adapter_sha256="a" * 64,
        policy_version="pilot:revision:aaaaaaaaaaaa",
        base_seed=11,
        target_games=target,
        replay_tolerance=2e-4,
    )
    for index in range(completed):
        seed = manifest.base_seed + index
        manifest.append(
            PilotMatchEntry(
                index=index,
                game_id=(
                    f"{manifest.policy_version}-game-{index:06d}-seed-{seed}"
                ),
                seed=seed,
                path=f"matches/game-{index:06d}.jsonl",
                sha256=f"{index + 1:064x}",
                policy_version=manifest.policy_version,
                turns=40 + index,
                owned_tokens=160 + index,
                final_results=(1.0, 0.0, 0.0, 0.0),
                termination_reason="natural_win",
                max_abs_log_prob_error=0.0,
            )
        )
    return manifest


def test_trainer_shards_are_equal_disjoint_and_contiguous() -> None:
    shards = [trainer_match_indices(rank, 4, 16) for rank in range(4)]
    assert shards == [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (8, 9, 10, 11),
        (12, 13, 14, 15),
    ]
    assert tuple(index for shard in shards for index in shard) == tuple(range(16))


def test_trainer_sharding_rejects_uneven_match_count() -> None:
    with pytest.raises(ValueError, match="cannot be divided equally"):
        trainer_match_indices(0, 4, 15)


@pytest.mark.parametrize(
    ("rank", "world_size", "total", "message"),
    [
        (0, 1, 16, "at least two"),
        (-1, 4, 16, "outside"),
        (4, 4, 16, "outside"),
        (0, 4, 0, "positive"),
    ],
)
def test_trainer_sharding_rejects_invalid_inputs(
    rank: int,
    world_size: int,
    total: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        trainer_match_indices(rank, world_size, total)


def test_complete_manifest_validates_for_four_trainers() -> None:
    manifest = _manifest()
    assert validate_distributed_training_manifest(
        manifest,
        config=_config(),
        world_size=4,
    ) == 4


def test_incomplete_manifest_is_rejected() -> None:
    manifest = _manifest(completed=15)
    with pytest.raises(ValueError, match="complete rollout manifest"):
        validate_distributed_training_manifest(
            manifest,
            config=_config(),
            world_size=4,
        )


def test_configuration_drift_is_rejected() -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match="configuration differs"):
        validate_distributed_training_manifest(
            manifest,
            config={**_config(), "seed": 12},
            world_size=4,
        )


def test_rank_update_report_is_atomic_json(tmp_path) -> None:
    path = write_rank_update_report(
        tmp_path,
        3,
        {"rank": 3, "indices": [12, 13, 14, 15], "status": "ok"},
    )
    assert path.name == "rank-003.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "rank": 3,
        "indices": [12, 13, 14, 15],
        "status": "ok",
    }

