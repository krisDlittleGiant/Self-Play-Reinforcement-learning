import pytest

from self_play_grpo.envs.observations import BoardState, Coordinate, LegalAction
from self_play_grpo.rollouts.schema import (
    MatchRecord,
    PolicySample,
    TurnCredit,
    TurnRecord,
)
from self_play_grpo.rollouts.pilot import (
    PilotManifest,
    PilotMatchEntry,
    directory_sha256,
    initialize_pilot_root,
    read_pilot_manifest,
    validate_manifest_request,
    write_pilot_manifest,
)


def board(current_seat, step) -> BoardState:
    return BoardState(
        board_size=5,
        pawn_positions=(Coordinate(2, 4), Coordinate(0, 2), Coordinate(2, 0), Coordinate(4, 2)),
        walls_remaining=(2, 2, 2, 2),
        wall_cells=frozenset(),
        current_seat=current_seat,
        joint_action_index=step,
        max_joint_actions=100,
    )


def make_match() -> MatchRecord:
    action = LegalAction(4, "MOVE_C4", "c4", "move", "move the pawn to c4")
    sample = PolicySample(
        chosen_label=action.label,
        prompt_text="prompt",
        completion_text="MOVE_C4\n",
        prompt_token_ids=(1, 2),
        completion_token_ids=(3, 4),
        behavior_log_probs=(-0.5, 0.0),
        allowed_token_ids=((3, 7), (4,)),
        attention_mask=(1, 1, 1, 1),
        loss_mask=(1, 1),
    )
    turn = TurnRecord(
        game_id="g",
        joint_step=0,
        seat=0,
        engine_player=0,
        player_local_step=0,
        policy_version="p0",
        state_before=board(0, 0),
        state_after=board(None, 1),
        observation="public",
        legal_actions=(action,),
        chosen_engine_action=4,
        chosen_notation="c4",
        policy_sample=sample,
        credit=TurnCredit(terminal_result=1.0, outcome_advantage=1.0, training_advantage=1.0),
    )
    return MatchRecord(
        game_id="g",
        policy_version="p0",
        seed=11,
        environment_config={},
        engine_manifest={},
        seat_map=(0, 2, 1, 3),
        initial_state=board(0, 0),
        turns=[turn],
        final_results=(1.0, 0.0, 0.0, 0.0),
        termination_reason="natural_win",
        final_environment="serialized",
    )


def test_match_json_round_trip_is_stable() -> None:
    original = make_match()
    payload = original.to_json()
    restored = MatchRecord.from_json(payload)
    assert restored.to_json() == payload
    assert restored.player_views()[0][0].policy_sample.loss_mask == (1, 1)


def test_wrong_owner_is_rejected() -> None:
    match = make_match()
    match.turns[0].seat = 1
    with pytest.raises(ValueError, match="active canonical seat"):
        match.validate()


def test_sampled_token_must_be_in_recorded_grammar() -> None:
    match = make_match()
    match.turns[0].policy_sample.allowed_token_ids = ((9,), (4,))
    with pytest.raises(ValueError, match="allowed-token"):
        match.validate()


PILOT_POLICY_VERSION = "pilot:1234567890ab:bbbbbbbbbbbb"
PILOT_CONFIG = {
    "environment": {"game": "quoridor"},
    "model": {"revision": "1234567890abcdef"},
}


def pilot_entry(index: int, policy_version: str = PILOT_POLICY_VERSION) -> PilotMatchEntry:
    seed = 11 + index
    return PilotMatchEntry(
        index=index,
        game_id=f"{policy_version}-game-{index:06d}-seed-{seed}",
        seed=seed,
        path=f"matches/game-{index:06d}.jsonl",
        sha256="a" * 64,
        policy_version=policy_version,
        turns=100,
        owned_tokens=416,
        final_results=(0.25, 0.25, 0.25, 0.25),
        termination_reason="horizon_draw",
        max_abs_log_prob_error=0.0,
    )


def test_pilot_manifest_round_trip_and_summary(tmp_path) -> None:
    manifest = PilotManifest.create(
        config=PILOT_CONFIG,
        adapter_sha256="b" * 64,
        policy_version=PILOT_POLICY_VERSION,
        base_seed=11,
        target_games=100,
        replay_tolerance=2e-4,
    )
    manifest.append(pilot_entry(0))
    write_pilot_manifest(tmp_path, manifest)
    restored = read_pilot_manifest(tmp_path)
    validate_manifest_request(
        restored,
        config=PILOT_CONFIG,
        base_seed=11,
        replay_tolerance=2e-4,
    )
    assert restored.to_dict() == manifest.to_dict()
    assert restored.summary() == {
        "completed_games": 1,
        "draw_games": 1,
        "fractional_results_by_seat": [0.25, 0.25, 0.25, 0.25],
        "illegal_action_substitutions": 0,
        "max_abs_log_prob_error": 0.0,
        "mean_turns": 100.0,
        "owned_tokens": 416,
        "policy_versions": [PILOT_POLICY_VERSION],
        "status": "incomplete",
        "target_games": 100,
        "termination_counts": {"horizon_draw": 1},
        "turns": 100,
        "wins_by_seat": [0, 0, 0, 0],
    }
    legacy = manifest.to_dict()
    for field in ("fractional_results_by_seat", "mean_turns", "wins_by_seat"):
        legacy["summary"].pop(field)
    assert PilotManifest.from_dict(legacy).to_dict() == manifest.to_dict()


def test_pilot_manifest_rejects_mixed_or_noncontiguous_games() -> None:
    manifest = PilotManifest.create(
        config=PILOT_CONFIG,
        adapter_sha256="b" * 64,
        policy_version=PILOT_POLICY_VERSION,
        base_seed=11,
        target_games=100,
        replay_tolerance=2e-4,
    )
    with pytest.raises(ValueError, match="Expected pilot match index"):
        manifest.append(pilot_entry(1))
    with pytest.raises(ValueError, match="mixed policy versions"):
        manifest.append(pilot_entry(0, policy_version="pilot:other:adapter"))


def test_pilot_target_can_grow_but_cannot_shrink() -> None:
    manifest = PilotManifest.create(
        config=PILOT_CONFIG,
        adapter_sha256="b" * 64,
        policy_version=PILOT_POLICY_VERSION,
        base_seed=11,
        target_games=1,
        replay_tolerance=2e-4,
    )
    manifest.extend_target(100)
    assert manifest.target_games == 100
    with pytest.raises(ValueError, match="cannot shrink"):
        manifest.extend_target(99)


def test_pilot_root_initialization_commits_adapter_and_manifest_together(tmp_path) -> None:
    class FakePeftModel:
        def save_pretrained(self, destination, *, safe_serialization) -> None:
            assert safe_serialization is True
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "adapter_config.json").write_text(
                '{"format":"fixture"}\n', encoding="utf-8"
            )
            (destination / "adapter_model.safetensors").write_bytes(b"fixture")

    root = tmp_path / "pilot"
    manifest = initialize_pilot_root(
        root,
        model=FakePeftModel(),
        config=PILOT_CONFIG,
        model_revision="1234567890abcdef",
        base_seed=11,
        target_games=100,
        replay_tolerance=2e-4,
    )
    assert (root / "matches").is_dir()
    assert read_pilot_manifest(root).to_dict() == manifest.to_dict()
    assert directory_sha256(root / "policy_adapter") == manifest.adapter_sha256
    assert manifest.policy_version == f"pilot:1234567890ab:{manifest.adapter_sha256[:12]}"
    with pytest.raises(FileExistsError, match="already exists"):
        initialize_pilot_root(
            root,
            model=FakePeftModel(),
            config=PILOT_CONFIG,
            model_revision="1234567890abcdef",
            base_seed=11,
            target_games=100,
            replay_tolerance=2e-4,
        )
