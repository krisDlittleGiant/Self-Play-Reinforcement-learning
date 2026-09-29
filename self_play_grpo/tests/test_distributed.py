import pytest

from self_play_grpo.distributed import (
    discover_torchrun_runtime,
    expected_synthetic_average_gradient,
    synthetic_gradient_values,
)


def _environment(**overrides: str) -> dict[str, str]:
    values = {
        "RANK": "1",
        "LOCAL_RANK": "1",
        "WORLD_SIZE": "2",
        "LOCAL_WORLD_SIZE": "2",
    }
    values.update(overrides)
    return values


def test_discovers_valid_single_node_torchrun_runtime() -> None:
    runtime = discover_torchrun_runtime(_environment(), expected_world_size=2)
    assert runtime.rank == 1
    assert runtime.local_rank == 1
    assert runtime.world_size == 2
    assert runtime.local_world_size == 2
    assert runtime.backend == "hccl"
    assert runtime.device == "hpu"
    assert runtime.logical_device == "hpu:1"


@pytest.mark.parametrize(
    "missing", ["RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"]
)
def test_discovery_requires_every_torchrun_variable(missing: str) -> None:
    environ = _environment()
    del environ[missing]
    with pytest.raises(RuntimeError, match=missing):
        discover_torchrun_runtime(environ, expected_world_size=2)


def test_discovery_rejects_non_integer_values() -> None:
    with pytest.raises(RuntimeError, match="RANK must be an integer"):
        discover_torchrun_runtime(_environment(RANK="one"), expected_world_size=2)


def test_discovery_rejects_world_size_mismatch() -> None:
    with pytest.raises(RuntimeError, match="differs from expected"):
        discover_torchrun_runtime(_environment(), expected_world_size=8)


def test_discovery_rejects_multi_node_topology() -> None:
    with pytest.raises(RuntimeError, match="only one-node launches"):
        discover_torchrun_runtime(
            _environment(WORLD_SIZE="4", LOCAL_WORLD_SIZE="2"),
            expected_world_size=4,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"RANK": "2"}, "RANK=2"),
        ({"RANK": "-1"}, "RANK=-1"),
        ({"LOCAL_RANK": "2"}, "LOCAL_RANK=2"),
        ({"LOCAL_RANK": "-1"}, "LOCAL_RANK=-1"),
    ],
)
def test_discovery_rejects_invalid_rank_ranges(
    overrides: dict[str, str], message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        discover_torchrun_runtime(_environment(**overrides), expected_world_size=2)


def test_discovery_rejects_single_process_validation() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        discover_torchrun_runtime(
            {
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
                "LOCAL_WORLD_SIZE": "1",
            },
            expected_world_size=1,
        )


def test_synthetic_gradients_are_rank_distinct_and_deterministic() -> None:
    assert synthetic_gradient_values(0, 4) == (1.0, -2.0, 3.0, -4.0)
    assert synthetic_gradient_values(3, 4) == (4.0, -8.0, 12.0, -16.0)


def test_expected_synthetic_average_gradient_matches_four_ranks() -> None:
    rows = [synthetic_gradient_values(rank, 4) for rank in range(4)]
    observed = tuple(sum(row[index] for row in rows) / 4.0 for index in range(4))
    assert expected_synthetic_average_gradient(4, 4) == observed
    assert observed == (2.5, -5.0, 7.5, -10.0)


@pytest.mark.parametrize("rank", [-1, -2])
def test_synthetic_gradient_rejects_negative_rank(rank: int) -> None:
    with pytest.raises(ValueError, match="rank must be non-negative"):
        synthetic_gradient_values(rank, 4)


@pytest.mark.parametrize("parameter_count", [0, -1])
def test_synthetic_gradient_rejects_invalid_parameter_count(
    parameter_count: int,
) -> None:
    with pytest.raises(ValueError, match="parameter_count must be positive"):
        synthetic_gradient_values(0, parameter_count)


def test_expected_synthetic_average_rejects_one_rank() -> None:
    with pytest.raises(ValueError, match="world_size must be at least 2"):
        expected_synthetic_average_gradient(1, 4)
