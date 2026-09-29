"""Sequential collect-update-checkpoint orchestration.

Nothing in this module starts training at import time. Callers must explicitly
invoke ``run_updates`` after all correctness gates have passed.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from self_play_grpo.config import ExperimentConfig
from self_play_grpo.envs.quoridor import QuoridorEnv
from self_play_grpo.policies.llm import ConstrainedLLMPolicy, max_behavior_replay_error
from self_play_grpo.rewards.progress import (
    blend_process_advantages,
    gae_advantages,
    normalized_potential_advantages,
)
from self_play_grpo.rollouts.collector import MatchCollector
from self_play_grpo.rollouts.schema import MatchRecord, write_matches_jsonl
from self_play_grpo.training.loss import LossOutput, backward_training_loss
from self_play_grpo.training.value import OutcomeEvaluator


@dataclass
class UpdateMetrics:
    update: int
    policy_version: str
    games: int
    turns: int
    owned_tokens: int
    loss: float
    policy_loss: float
    kl_loss: float
    mean_ratio_before_step: float
    clip_fraction_before_step: float
    replay_max_abs_error: float
    grad_norm: float
    optimizer_steps: int


def assign_potential_credit(match: MatchRecord, alpha: float) -> None:
    for seat, turns in enumerate(match.player_views()):
        potentials = [float(turn.credit.progress_before["potentials"][seat]) for turn in turns]
        advantages = normalized_potential_advantages(
            potentials, match.final_results[seat], alpha=alpha
        )
        for turn, advantage in zip(turns, advantages):
            turn.credit.training_advantage = advantage


def assign_evaluator_credit(
    match: MatchRecord,
    evaluator: OutcomeEvaluator,
    *,
    gae_lambda: float,
    eta: float,
) -> None:
    for seat, turns in enumerate(match.player_views()):
        predictions = evaluator.predict([turn.state_before for turn in turns])
        values = [float(row[seat]) for row in predictions]
        gae = gae_advantages(
            values,
            match.final_results[seat],
            gamma=1.0,
            gae_lambda=gae_lambda,
        )
        outcome_advantage = turns[0].credit.outcome_advantage if turns else None
        if outcome_advantage is None:
            continue
        blended = blend_process_advantages(float(outcome_advantage), gae, eta)
        for turn, advantage in zip(turns, blended):
            turn.credit.training_advantage = advantage
            turn.credit.evaluator_version = evaluator.version


class SynchronousTrainer:
    def __init__(
        self,
        config: ExperimentConfig,
        policy: ConstrainedLLMPolicy,
        artifact_dir: str | Path,
        *,
        evaluator: OutcomeEvaluator | None = None,
        reference_model: Any | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Torch is required for training") from exc
        config.validate()
        self.config = config
        self.policy = policy
        self.evaluator = evaluator
        self.reference_model = reference_model
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.update_index = 0
        self.policy_version = "policy-000000"
        parameters = [parameter for parameter in policy.model.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("The policy has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.collector = MatchCollector(
            collect_progress=True,
            proxy_temperature=config.process_extension.proxy_temperature,
        )

    def collect_batch(self) -> list[MatchRecord]:
        matches: list[MatchRecord] = []
        for index in range(self.config.rollout.games_per_update):
            seed = self.config.seed + self.update_index * 100_000 + index
            game_id = f"{self.policy_version}-seed-{seed}"
            env = QuoridorEnv(self.config.environment)
            match = self.collector.collect(
                env,
                self.policy,
                game_id=game_id,
                seed=seed,
                policy_version=self.policy_version,
            )
            if self.config.training.reward_mode == "potential":
                assign_potential_credit(match, self.config.process_extension.shaping_alpha)
            elif self.config.training.reward_mode == "gae_blend":
                if self.evaluator is None:
                    raise RuntimeError("gae_blend requires a frozen outcome evaluator")
                assign_evaluator_credit(
                    match,
                    self.evaluator,
                    gae_lambda=self.config.process_extension.gae_lambda,
                    eta=self.config.process_extension.process_blend_eta,
                )
            matches.append(match)
        write_matches_jsonl(
            self.artifact_dir / "rollouts" / f"{self.policy_version}.jsonl", matches
        )
        return matches

    def update(self, matches: list[MatchRecord], replay_tolerance: float = 2e-4) -> UpdateMetrics:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Torch is required for training") from exc
        samples = [
            turn.policy_sample
            for match in matches
            for turn in match.turns
            if turn.policy_sample.completion_token_ids
        ]
        replay_error = max_behavior_replay_error(self.policy.model, samples)
        if replay_error > replay_tolerance:
            raise RuntimeError(
                f"Behavior replay mismatch {replay_error:.6g} exceeds {replay_tolerance:.6g}"
            )
        self.policy.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        try:
            output: LossOutput = backward_training_loss(
                self.policy.model,
                matches,
                clip_epsilon=self.config.training.clip_epsilon,
                loss_normalizer_per_game=self.config.training.loss_normalizer_per_game,
                kl_beta=self.config.training.kl_beta,
                reference_model=self.reference_model,
            )
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            raise
        trainable = [parameter for parameter in self.policy.model.parameters() if parameter.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, self.config.training.max_grad_norm
        )
        if not torch.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"Non-finite gradient norm: {float(grad_norm)}")
        optimizer_steps = 0
        if float(grad_norm.detach().cpu()) > 0.0:
            self.optimizer.step()
            optimizer_steps = 1
        metrics = UpdateMetrics(
            update=self.update_index,
            policy_version=self.policy_version,
            games=len(matches),
            turns=sum(len(match.turns) for match in matches),
            owned_tokens=output.owned_tokens,
            loss=float(output.loss.detach().cpu()),
            policy_loss=float(output.policy_loss.detach().cpu()),
            kl_loss=float(output.kl_loss.detach().cpu()),
            mean_ratio_before_step=output.mean_ratio,
            clip_fraction_before_step=output.clip_fraction,
            replay_max_abs_error=replay_error,
            grad_norm=float(grad_norm.detach().cpu()),
            optimizer_steps=optimizer_steps,
        )
        self.update_index += 1
        self.policy_version = f"policy-{self.update_index:06d}"
        self.save_checkpoint(metrics)
        return metrics

    def save_checkpoint(
        self,
        metrics: UpdateMetrics,
        *,
        validation_only: bool = False,
        distributed_manifest: Any | None = None,
        rank_rng_staging: str | Path | None = None,
    ) -> Path:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Torch is required for checkpointing") from exc
        destination = self.artifact_dir / "checkpoints" / self.policy_version
        checkpoint_root = destination.parent
        if distributed_manifest is not None:
            from self_play_grpo.training.distributed_checkpoint import (
                CHECKPOINT_FORMAT,
                validate_rank_rng_record,
            )

            if rank_rng_staging is None:
                raise ValueError("Distributed checkpoint requires rank RNG staging")
            if validation_only != (distributed_manifest.run_kind == "validation"):
                raise ValueError("Distributed run kind and validation_only disagree")
            if (distributed_manifest.update_index, distributed_manifest.policy_version) != (
                self.update_index, self.policy_version
            ):
                raise ValueError("Distributed checkpoint trainer version differs")
            if distributed_manifest.checkpoint_format != CHECKPOINT_FORMAT:
                raise ValueError("Distributed checkpoint format differs")
            from self_play_grpo.rollouts.pilot import canonical_sha256

            if distributed_manifest.config_sha256 != canonical_sha256(self.config.to_dict()):
                raise ValueError("Distributed checkpoint config_sha256 differs from active config")
            if (distributed_manifest.model_id, distributed_manifest.model_revision,
                distributed_manifest.dtype) != (
                    self.config.model.id, self.config.model.revision, self.config.model.dtype
                ):
                raise ValueError("Distributed checkpoint model identity differs from active model")
            for rank_state in distributed_manifest.ranks:
                validate_rank_rng_record(rank_rng_staging, rank_state)
        elif rank_rng_staging is not None:
            raise ValueError("Rank RNG staging requires distributed metadata")
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
        with tempfile.TemporaryDirectory(
            dir=checkpoint_root,
            prefix=f".{self.policy_version}-incomplete-",
        ) as temporary_name:
            temporary = Path(temporary_name)
            if not hasattr(self.policy.model, "save_pretrained"):
                raise TypeError("Checkpointing requires a PEFT model with save_pretrained")
            self.policy.model.save_pretrained(
                temporary / "adapter",
                safe_serialization=True,
            )
            torch.save(self.optimizer.state_dict(), temporary / "optimizer_state.pt")
            if distributed_manifest is None:
                rng_state: dict[str, Any] = {"cpu": torch.get_rng_state()}
                if (
                    hasattr(torch, "hpu")
                    and hasattr(torch.hpu, "is_initialized")
                    and torch.hpu.is_initialized()
                    and hasattr(torch.hpu, "get_rng_state_all")
                ):
                    rng_state["hpu"] = torch.hpu.get_rng_state_all()
            else:
                from self_play_grpo.training.distributed_checkpoint import (
                    load_rank_rng_record,
                )

                rank_zero = load_rank_rng_record(rank_rng_staging, distributed_manifest.ranks[0])
                rng_state = {"cpu": rank_zero["cpu"], "hpu": rank_zero["hpu"]}
            torch.save(rng_state, temporary / "torch_rng_state.pt")
            if self.evaluator is not None:
                torch.save(
                    self.evaluator.model.state_dict(),
                    temporary / "evaluator_state.pt",
                )
            state = {
                "checkpoint_format": 2 if distributed_manifest is None else 3,
                "model_storage": "peft_adapter_only",
                "validation_only": validation_only,
                "update_index": self.update_index,
                "policy_version": self.policy_version,
                "config": self.config.to_dict(),
                "last_update": metrics.__dict__,
            }
            if distributed_manifest is not None:
                state["model_training"] = bool(self.policy.model.training)
            (temporary / "trainer_state.json").write_text(
                json.dumps(state, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if distributed_manifest is not None:
                from self_play_grpo.training.distributed_checkpoint import (
                    MANIFEST_PATH,
                    build_checkpoint_inventory,
                    verify_checkpoint_files,
                )

                for rank_state in distributed_manifest.ranks:
                    source = Path(rank_rng_staging) / rank_state.rng_path
                    target = temporary / rank_state.rng_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target, follow_symlinks=False)
                complete_manifest = replace(
                    distributed_manifest,
                    files=build_checkpoint_inventory(temporary),
                )
                complete_manifest.validate()
                manifest_path = temporary / MANIFEST_PATH
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(
                    json.dumps(complete_manifest.to_dict(), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                verify_checkpoint_files(temporary, complete_manifest)
            temporary.replace(destination)
        return destination

    def load_checkpoint(
        self,
        checkpoint: str | Path,
        *,
        allow_validation: bool = False,
        distributed_expected: Any | None = None,
        distributed_rank: int | None = None,
    ) -> None:
        """Restore a batch-boundary checkpoint into an already constructed trainer."""

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Torch is required for checkpoint resume") from exc
        source = Path(checkpoint)
        state = json.loads((source / "trainer_state.json").read_text(encoding="utf-8"))
        distributed = distributed_expected is not None or distributed_rank is not None
        if distributed:
            if distributed_expected is None or distributed_rank is None:
                raise ValueError("Distributed load requires expected identity and rank")
            from self_play_grpo.training.distributed_checkpoint import (
                load_rank_rng_record,
                read_distributed_manifest,
                validate_resume_identity,
                verify_checkpoint_files,
            )

            manifest = read_distributed_manifest(source)
            verify_checkpoint_files(source, manifest)
            validate_resume_identity(
                manifest, expected=distributed_expected, allow_validation=allow_validation
            )
            if not 0 <= distributed_rank < manifest.trainer_world_size:
                raise ValueError("Distributed rank is outside checkpoint world size")
            for rank_state in manifest.ranks:
                load_rank_rng_record(source, rank_state)
            if state.get("checkpoint_format") != 3:
                raise ValueError("Distributed trainer state format differs")
            if state.get("validation_only") != (manifest.run_kind == "validation"):
                raise ValueError("Distributed trainer state run kind differs")
            if (state.get("update_index"), state.get("policy_version")) != (
                manifest.update_index, manifest.policy_version
            ):
                raise ValueError("Distributed trainer state version differs")
            if type(state.get("model_training")) is not bool:
                raise ValueError("Distributed model training mode is missing")
        expected_config = json.dumps(self.config.to_dict(), sort_keys=True)
        recorded_config = json.dumps(state["config"], sort_keys=True)
        if recorded_config != expected_config:
            raise ValueError("Checkpoint configuration differs from the active experiment")
        if state.get("checkpoint_format") != (3 if distributed else 2):
            raise ValueError("Unsupported checkpoint format")
        if state.get("model_storage") != "peft_adapter_only":
            raise ValueError("Checkpoint does not use the required adapter-only format")
        if state.get("validation_only", False) and not allow_validation:
            raise ValueError("Refusing to resume a validation-only checkpoint")
        adapter_path = source / "adapter"
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"Checkpoint adapter is missing: {adapter_path}")
        evaluator_path = source / "evaluator_state.pt"
        if distributed and evaluator_path.exists() != (self.evaluator is not None):
            raise ValueError("Distributed evaluator state presence differs")
        if distributed:
            rank_rng = load_rank_rng_record(source, manifest.ranks[distributed_rank])
            optimizer_state = torch.load(
                source / "optimizer_state.pt", map_location="cpu", weights_only=True
            )
        if not hasattr(self.policy.model, "load_adapter"):
            raise TypeError("Checkpoint resume requires a PEFT model with load_adapter")
        load_result = self.policy.model.load_adapter(
            adapter_path,
            adapter_name="default",
            is_trainable=True,
            torch_device=str(next(self.policy.model.parameters()).device),
            local_files_only=True,
        )
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(
                "Adapter checkpoint keys differ from the active model: "
                f"missing={load_result.missing_keys}, "
                f"unexpected={load_result.unexpected_keys}"
            )
        if not distributed:
            optimizer_state = torch.load(
                source / "optimizer_state.pt", map_location="cpu", weights_only=True
            )
        self.optimizer.load_state_dict(optimizer_state)
        # Optimizer.load_state_dict applies its per-state placement policy. In
        # particular, non-capturable Adam step counters may intentionally stay
        # on CPU while moment tensors follow their HPU parameters.

        if not distributed:
            rng_state = torch.load(
                source / "torch_rng_state.pt", map_location="cpu", weights_only=True
            )
        if evaluator_path.exists():
            if self.evaluator is None:
                raise ValueError("Checkpoint contains an evaluator but the trainer does not")
            evaluator_state = torch.load(
                evaluator_path, map_location="cpu", weights_only=True
            )
            self.evaluator.model.load_state_dict(evaluator_state, strict=True)
        self.update_index = int(state["update_index"])
        self.policy_version = str(state["policy_version"])
        if distributed:
            self.policy.model.train(state["model_training"])
            torch.set_rng_state(rank_rng["cpu"])
            if not hasattr(torch, "hpu") or not hasattr(torch.hpu, "set_rng_state_all"):
                raise RuntimeError("This runtime cannot restore rank HPU RNG state")
            torch.hpu.set_rng_state_all(rank_rng["hpu"])
        else:
            torch.set_rng_state(rng_state["cpu"])
            if "hpu" in rng_state and hasattr(torch, "hpu") and hasattr(torch.hpu, "set_rng_state_all"):
                torch.hpu.set_rng_state_all(rng_state["hpu"])

    def run_updates(
        self,
        count: int,
        on_update: Callable[[UpdateMetrics], None] | None = None,
    ) -> list[UpdateMetrics]:
        """Explicit long-running entry point; never called automatically."""

        if count <= 0:
            raise ValueError("count must be positive")
        history = []
        for _ in range(count):
            metrics = self.update(self.collect_batch())
            history.append(metrics)
            if on_update is not None:
                on_update(metrics)
        return history
