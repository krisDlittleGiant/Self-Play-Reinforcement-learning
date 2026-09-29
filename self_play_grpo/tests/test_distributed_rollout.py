from self_play_grpo.rollouts.distributed import (
    DistributedRankReport,
    ordered_report_entries,
    rank_match_indices,
    read_rank_report,
    write_rank_report,
)
from self_play_grpo.rollouts.pilot import PilotManifest, PilotMatchEntry


def _manifest() -> PilotManifest:
    return PilotManifest.create(
        config={"model": {"revision": "revision"}},
        adapter_sha256="a" * 64,
        policy_version="pilot:revision:aaaaaaaaaaaa",
        base_seed=11,
        target_games=4,
        replay_tolerance=2e-4,
    )


def _entry(index: int) -> PilotMatchEntry:
    manifest = _manifest()
    seed = manifest.base_seed + index
    return PilotMatchEntry(
        index=index,
        game_id=f"{manifest.policy_version}-game-{index:06d}-seed-{seed}",
        seed=seed,
        path=f"matches/game-{index:06d}.jsonl",
        sha256=f"{index + 1:064x}",
        policy_version=manifest.policy_version,
        turns=10 + index,
        owned_tokens=40 + index,
        final_results=(1.0, 0.0, 0.0, 0.0),
        termination_reason="natural_win",
        max_abs_log_prob_error=0.0,
    )


def _report(rank: int) -> DistributedRankReport:
    manifest = _manifest()
    indices = rank_match_indices(rank, 2, 2)
    return DistributedRankReport(
        rank=rank,
        local_rank=rank,
        world_size=2,
        games_per_rank=2,
        policy_version=manifest.policy_version,
        adapter_sha256=manifest.adapter_sha256,
        matches=tuple(_entry(index) for index in indices),
    )


def test_rank_match_indices_are_disjoint_and_contiguous() -> None:
    shards = [rank_match_indices(rank, 4, 4) for rank in range(4)]
    assert shards[0] == (0, 1, 2, 3)
    assert shards[-1] == (12, 13, 14, 15)
    assert tuple(index for shard in shards for index in shard) == tuple(range(16))


def test_reports_form_one_ordered_global_batch() -> None:
    manifest = _manifest()
    reports = (_report(1), _report(0))
    for report in reports:
        report.validate(manifest)
    entries = ordered_report_entries(reports, world_size=2, games_per_rank=2)
    assert tuple(entry.index for entry in entries) == (0, 1, 2, 3)


def test_rank_report_rejects_wrong_local_rank() -> None:
    manifest = _manifest()
    report = _report(0)
    invalid = DistributedRankReport(
        **{**report.__dict__, "local_rank": 1},
    )
    try:
        invalid.validate(manifest)
    except ValueError as exc:
        assert "local_rank == rank" in str(exc)
    else:
        raise AssertionError("invalid single-node local rank was accepted")


def test_rank_report_round_trip(tmp_path) -> None:
    manifest = _manifest()
    report = _report(1)
    path = write_rank_report(tmp_path, report, manifest)
    assert path.name == "rank-001.json"
    assert read_rank_report(tmp_path, 1, manifest) == report
