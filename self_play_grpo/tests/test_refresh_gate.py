"""CPU-only four-rank refresh aggregation and publication checks."""

from copy import deepcopy

import pytest

from self_play_grpo.training.coordinator import RoleLayout
from self_play_grpo.training.refresh_gate import (
    publish_refresh_evidence, validate_refresh_reports,
)


LAYOUT = RoleLayout((0, 1, 2, 3), (4, 5, 6, 7))
SOURCE = "a" * 64
CHECKPOINT = "b" * 64
ADAPTER = "c" * 64
PARAMETERS = "d" * 64


def reports():
    return {
        rank: {
            "status": "refresh_verified", "rank": rank, "module_id": rank,
            "policy_version": "policy-000001", "source_manifest_sha256": SOURCE,
            "checkpoint_manifest_sha256": CHECKPOINT,
            "adapter_sha256": ADAPTER, "parameter_sha256": PARAMETERS,
            "probe_max_abs_error": 0.0,
        }
        for rank in range(4)
    }


def validate(rows):
    return validate_refresh_reports(
        rows, layout=LAYOUT, source_manifest_sha256=SOURCE,
        checkpoint_manifest_sha256=CHECKPOINT, adapter_sha256=ADAPTER,
        parameter_sha256=PARAMETERS, tolerance=2e-4,
    )


def test_four_matching_refresh_reports_pass():
    rows = reports()
    rows[3]["probe_max_abs_error"] = 1e-5
    assert validate(rows) == pytest.approx(1e-5)


@pytest.mark.parametrize("field,value", [
    ("module_id", 7), ("policy_version", "policy-000000"),
    ("source_manifest_sha256", "e" * 64),
    ("checkpoint_manifest_sha256", "e" * 64),
    ("adapter_sha256", "e" * 64), ("parameter_sha256", "e" * 64),
])
def test_stale_or_different_rank_rejected(field, value):
    rows = deepcopy(reports())
    rows[2][field] = value
    with pytest.raises(ValueError, match="rank 2 identity differs"):
        validate(rows)


def test_missing_or_nonfinite_refresh_report_rejected():
    rows = reports()
    del rows[3]
    with pytest.raises(ValueError, match="exactly four"):
        validate(rows)
    rows = reports()
    rows[0]["probe_max_abs_error"] = float("nan")
    with pytest.raises(ValueError, match="probability probe failed"):
        validate(rows)


def test_summary_publication_is_exclusive(tmp_path):
    evidence = {"status": "refresh_reports_verified"}
    target = publish_refresh_evidence(tmp_path / "refresh", evidence)
    assert target.exists()
    with pytest.raises(FileExistsError):
        publish_refresh_evidence(tmp_path / "refresh", evidence)
