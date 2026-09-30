"""CPU-only stage decisions of the one-command launcher."""

from types import SimpleNamespace

from self_play_grpo.training.launch_run import eval_points, next_step, write_report


def _args(until=10, every=2):
    return SimpleNamespace(until_update=until, eval_every=every)


def test_eval_points_include_start_end_and_period():
    assert eval_points(10, 2) == [0, 2, 4, 6, 8, 10]
    assert eval_points(10, 3) == [0, 3, 6, 9, 10]
    assert eval_points(10, 0) == []


def test_stages_follow_pilot_updates_and_evaluations_in_order():
    args = _args()
    assert next_step(args, pilot_ready=False, commits=-1, evaluated=set()).kind == "pilot"
    assert next_step(args, pilot_ready=True, commits=-1, evaluated=set()).kind == "update1"
    assert next_step(args, pilot_ready=True, commits=0, evaluated=set()).kind == "archive_root"
    step = next_step(args, pilot_ready=True, commits=1, evaluated=set())
    assert (step.kind, step.target) == ("evaluate", 0)
    assert next_step(args, pilot_ready=True, commits=1, evaluated={0}).kind == "update2"
    step = next_step(args, pilot_ready=True, commits=2, evaluated={0})
    assert (step.kind, step.target) == ("evaluate", 2)
    step = next_step(args, pilot_ready=True, commits=2, evaluated={0, 2})
    assert (step.kind, step.target) == ("multi", 4)
    step = next_step(args, pilot_ready=True, commits=10, evaluated={0, 2, 4, 6, 8, 10})
    assert step.kind == "done"


def test_without_evaluation_training_runs_straight_to_the_target():
    args = _args(until=10, every=0)
    step = next_step(args, pilot_ready=True, commits=2, evaluated=set())
    assert (step.kind, step.target) == ("multi", 10)
    assert next_step(args, pilot_ready=True, commits=10, evaluated=set()).kind == "done"


def test_multi_update_segments_never_target_below_update_three():
    step = next_step(_args(until=10, every=1), pilot_ready=True, commits=2, evaluated={0, 1, 2})
    assert (step.kind, step.target) == ("multi", 3)


def test_report_of_an_empty_run_root_is_empty(tmp_path):
    report = write_report(tmp_path)
    assert report["updates"] == [] and report["indicators"] == [] and report["evaluations"] == {}
    assert (tmp_path / "run_report.json").is_file()
