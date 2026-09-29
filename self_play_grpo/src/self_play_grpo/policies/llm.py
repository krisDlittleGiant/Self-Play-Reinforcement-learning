"""Constrained action-only sampling with exact masked log probabilities."""

from __future__ import annotations

import bisect
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from unittest.mock import patch

from self_play_grpo.config import ModelConfig, RolloutConfig
from self_play_grpo.envs.quoridor import QuoridorEnv
from self_play_grpo.rollouts.schema import PolicySample


@contextmanager
def _checkpoint_attention_registry(registry: Any) -> Iterator[None]:
    """Checkpoint attention math without changing the model's KV-cache contract."""

    from torch import is_grad_enabled
    from torch.utils.checkpoint import checkpoint

    original_get = registry.get_interface

    def checkpointed_get(implementation: str, default: Any) -> Any:
        attention = original_get(implementation, default)

        def checkpointed_attention(
            module: Any, query: Any, key: Any, value: Any, attention_mask: Any, **kwargs: Any
        ) -> tuple[Any, Any]:
            if not is_grad_enabled():
                return attention(module, query, key, value, attention_mask, **kwargs)

            def compute(q: Any, k: Any, v: Any) -> Any:
                output, _ = attention(module, q, k, v, attention_mask, **kwargs)
                return output

            return checkpoint(compute, query, key, value, use_reentrant=False), None

        return checkpointed_attention

    with patch.object(registry, "get_interface", side_effect=checkpointed_get):
        yield


@contextmanager
def checkpoint_qwen3_attention(model: Any) -> Iterator[None]:
    """Scope attention-only checkpointing to exact-shape Qwen3 training replay."""

    from transformers.models.qwen3 import modeling_qwen3 as qwen3

    if not model.training:
        raise ValueError("Attention checkpointing requires train mode")
    if not any(isinstance(module, qwen3.Qwen3Attention) for module in model.modules()):
        raise TypeError("Attention checkpointing requires a Qwen3 model")
    with _checkpoint_attention_registry(qwen3.ALL_ATTENTION_FUNCTIONS):
        yield


class TokenTrie:
    """A small exact grammar over complete legal-action token sequences."""

    _END = -1

    def __init__(self, choices: Mapping[str, Sequence[int]]) -> None:
        if not choices:
            raise ValueError("At least one constrained choice is required")
        self._root: dict[int, Any] = {}
        for label, sequence in choices.items():
            if not sequence:
                raise ValueError(f"Choice {label!r} tokenized to an empty sequence")
            node = self._root
            for token in sequence:
                node = node.setdefault(int(token), {})
            if self._END in node:
                raise ValueError("Two labels have the same constrained token sequence")
            node[self._END] = label

    def _node(self, prefix: Sequence[int]) -> dict[int, Any]:
        node = self._root
        for token in prefix:
            child = node.get(int(token))
            if not isinstance(child, dict):
                raise ValueError("Generated prefix is outside the legal-action trie")
            node = child
        return node

    def allowed(self, prefix: Sequence[int]) -> tuple[int, ...]:
        node = self._node(prefix)
        return tuple(sorted(token for token in node if token != self._END))

    def completed_label(self, prefix: Sequence[int]) -> str | None:
        value = self._node(prefix).get(self._END)
        return str(value) if value is not None else None


def _python_categorical(probabilities: Sequence[float], rng: random.Random) -> int:
    cumulative: list[float] = []
    running = 0.0
    for probability in probabilities:
        running += float(probability)
        cumulative.append(running)
    if running <= 0.0:
        raise RuntimeError("Constrained distribution has no probability mass")
    threshold = rng.random() * running
    return min(bisect.bisect_left(cumulative, threshold), len(cumulative) - 1)


@dataclass
class ConstrainedLLMPolicy:
    model: Any
    tokenizer: Any
    rollout_config: RolloutConfig
    name: str = "qwen_constrained"

    @classmethod
    def load(cls, model_config: ModelConfig, rollout_config: RolloutConfig) -> "ConstrainedLLMPolicy":
        """Load the pinned model and attach one trainable shared LoRA adapter."""

        try:
            import torch
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the train dependencies in the Gaudi environment") from exc

        dtype = getattr(torch, model_config.dtype, None)
        if dtype is None:
            raise ValueError(f"Unknown torch dtype {model_config.dtype!r}")
        source = model_config.load_source
        if model_config.local_path is not None and not Path(source).is_dir():
            raise RuntimeError(
                f"Configured local model checkpoint does not exist: {source}"
            )
        load_kwargs: dict[str, Any] = {
            "local_files_only": model_config.local_files_only,
        }
        if model_config.local_path is None:
            load_kwargs["revision"] = model_config.revision
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            **load_kwargs,
        )
        model = AutoModelForCausalLM.from_pretrained(
            source,
            dtype=dtype,
            **load_kwargs,
        )
        model.to(model_config.device)
        lora = LoraConfig(
            r=model_config.lora_rank,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        # PEFT probes its Intel Neural Compressor backend before the generic
        # dense Linear backend whenever INC is installed. The Gaudi image's
        # INC build is not compatible with Transformers 5's Conv1D location,
        # and this unquantized BF16 checkpoint does not need that backend.
        with patch("peft.tuners.lora.inc.is_inc_available", return_value=False):
            model = get_peft_model(model, lora)
        return cls(model=model, tokenizer=tokenizer, rollout_config=rollout_config)

    def _chat_prompt(self, observation: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "Choose one legal Quoridor action. Output only the requested action label.",
            },
            {"role": "user", "content": observation},
        ]
        return str(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    def _choice_tokens(self, labels: Iterable[str]) -> dict[str, tuple[int, ...]]:
        choices: dict[str, tuple[int, ...]] = {}
        for label in labels:
            token_ids = tuple(
                map(
                    int,
                    self.tokenizer.encode(f"{label}\n", add_special_tokens=False),
                )
            )
            if len(token_ids) > self.rollout_config.max_new_tokens:
                raise RuntimeError(
                    f"Legal action {label!r} needs {len(token_ids)} tokens, above max_new_tokens"
                )
            choices[label] = token_ids
        return choices

    def select_action(
        self, env: QuoridorEnv, *, game_id: str, rng: random.Random
    ) -> PolicySample:
        del game_id
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Torch is required for LLM sampling") from exc

        observation = env.observation(env.current_seat)
        prompt = self._chat_prompt(observation)
        prompt_ids = tuple(map(int, self.tokenizer.encode(prompt, add_special_tokens=False)))
        choices = self._choice_tokens(action.label for action in env.legal_actions())
        trie = TokenTrie(choices)
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        completion: list[int] = []
        behavior_log_probs: list[float] = []
        allowed_history: list[tuple[int, ...]] = []
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, use_cache=True)
            logits = outputs.logits[0, -1].float()
            past = outputs.past_key_values
            for _ in range(self.rollout_config.max_new_tokens):
                allowed = trie.allowed(completion)
                if not allowed:
                    raise RuntimeError("Legal-action grammar reached a dead end")
                allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=logits.device)
                allowed_logits = logits.index_select(0, allowed_tensor)
                log_probs = torch.log_softmax(allowed_logits, dim=0)
                probabilities = torch.exp(log_probs).cpu().tolist()
                selected_index = _python_categorical(probabilities, rng)
                token = allowed[selected_index]
                completion.append(token)
                behavior_log_probs.append(float(log_probs[selected_index].cpu()))
                allowed_history.append(allowed)
                label = trie.completed_label(completion)
                if label is not None:
                    break
                next_input = torch.tensor([[token]], dtype=torch.long, device=device)
                outputs = self.model(
                    input_ids=next_input,
                    past_key_values=past,
                    use_cache=True,
                )
                logits = outputs.logits[0, -1].float()
                past = outputs.past_key_values
            else:
                raise RuntimeError("Constrained action exceeded max_new_tokens")

        label = trie.completed_label(completion)
        if label is None:
            raise RuntimeError("Constrained sampler stopped without a complete legal action")
        completion_text = self.tokenizer.decode(completion, skip_special_tokens=False)
        expected_text = f"{label}\n"
        if completion_text != expected_text:
            raise RuntimeError(
                f"Tokenizer round trip changed constrained action: {completion_text!r} != {expected_text!r}"
            )
        return PolicySample(
            chosen_label=label,
            prompt_text=prompt,
            completion_text=completion_text,
            prompt_token_ids=prompt_ids,
            completion_token_ids=tuple(completion),
            behavior_log_probs=tuple(behavior_log_probs),
            allowed_token_ids=tuple(allowed_history),
            attention_mask=(1,) * (len(prompt_ids) + len(completion)),
            loss_mask=(1,) * len(completion),
            sampling_config={
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 0,
                "constrained": True,
                "terminator": "newline",
                "probability_path": "kv_cache",
            },
        )

    def select_actions_batched(
        self,
        envs: Sequence[QuoridorEnv],
        *,
        game_ids: Sequence[str],
        rngs: Sequence[random.Random],
    ) -> tuple[PolicySample, ...]:
        """Sample one action per environment through one padded model batch.

        Samples record the tensor-shape contract needed by replay and training;
        the pinned HPU kernels are not numerically equivalent to single-row
        execution for every prompt shape.
        """

        del game_ids
        if not envs:
            raise ValueError("At least one environment is required")
        if len(envs) != len(rngs):
            raise ValueError("envs and rngs must have equal lengths")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Torch is required for LLM sampling") from exc

        prompts = [self._chat_prompt(env.observation(env.current_seat)) for env in envs]
        prompt_ids = [
            tuple(map(int, self.tokenizer.encode(prompt, add_special_tokens=False)))
            for prompt in prompts
        ]
        choices = [
            self._choice_tokens(action.label for action in env.legal_actions())
            for env in envs
        ]
        tries = [TokenTrie(row) for row in choices]
        device = next(self.model.parameters()).device
        pad_token = self.tokenizer.pad_token_id
        if pad_token is None:
            pad_token = self.tokenizer.eos_token_id
        if pad_token is None:
            raise RuntimeError("Tokenizer has neither a pad token nor an EOS token")
        width = max(map(len, prompt_ids))
        batch_size = len(envs)
        input_ids = torch.full(
            (batch_size, width), int(pad_token), dtype=torch.long, device=device
        )
        attention = torch.zeros(
            (batch_size, width), dtype=torch.long, device=device
        )
        for row, ids in enumerate(prompt_ids):
            input_ids[row, : len(ids)] = torch.tensor(
                ids, dtype=torch.long, device=device
            )
            attention[row, : len(ids)] = 1
        position_ids = attention.cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention == 0, 0)

        completions: list[list[int]] = [[] for _ in envs]
        behavior: list[list[float]] = [[] for _ in envs]
        allowed_rows: list[list[tuple[int, ...]]] = [[] for _ in envs]
        completed = [False] * batch_size
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention,
                position_ids=position_ids,
                use_cache=True,
            )
            logits = torch.stack(
                [outputs.logits[row, len(ids) - 1] for row, ids in enumerate(prompt_ids)]
            ).float()
            past = outputs.past_key_values
            for _ in range(self.rollout_config.max_new_tokens):
                active = [not value for value in completed]
                selected_tokens = [int(pad_token)] * batch_size
                for row in range(batch_size):
                    if not active[row]:
                        continue
                    allowed = tries[row].allowed(completions[row])
                    if not allowed:
                        raise RuntimeError("Legal-action grammar reached a dead end")
                    allowed_tensor = torch.tensor(
                        allowed, dtype=torch.long, device=logits.device
                    )
                    log_probs = torch.log_softmax(
                        logits[row].index_select(0, allowed_tensor), dim=0
                    )
                    probabilities = torch.exp(log_probs).cpu().tolist()
                    selected_index = _python_categorical(probabilities, rngs[row])
                    token = allowed[selected_index]
                    selected_tokens[row] = token
                    completions[row].append(token)
                    behavior[row].append(float(log_probs[selected_index].cpu()))
                    allowed_rows[row].append(allowed)
                    completed[row] = (
                        tries[row].completed_label(completions[row]) is not None
                    )
                if all(completed):
                    break
                next_input = torch.tensor(
                    selected_tokens, dtype=torch.long, device=device
                ).unsqueeze(1)
                next_attention = torch.tensor(
                    active, dtype=torch.long, device=device
                ).unsqueeze(1)
                attention = torch.cat((attention, next_attention), dim=1)
                next_positions = attention.cumsum(dim=-1)[:, -1:] - 1
                next_positions.masked_fill_(next_attention == 0, 0)
                outputs = self.model(
                    input_ids=next_input,
                    attention_mask=attention,
                    position_ids=next_positions,
                    past_key_values=past,
                    use_cache=True,
                )
                logits = outputs.logits[:, -1].float()
                past = outputs.past_key_values
            else:
                raise RuntimeError("Constrained action exceeded max_new_tokens")

        samples: list[PolicySample] = []
        for row in range(batch_size):
            label = tries[row].completed_label(completions[row])
            if label is None:
                raise RuntimeError(
                    "Batched constrained sampler stopped without a complete action"
                )
            completion_text = self.tokenizer.decode(
                completions[row], skip_special_tokens=False
            )
            expected_text = f"{label}\n"
            if completion_text != expected_text:
                raise RuntimeError(
                    "Tokenizer round trip changed constrained action: "
                    f"{completion_text!r} != {expected_text!r}"
                )
            samples.append(
                PolicySample(
                    chosen_label=label,
                    prompt_text=prompts[row],
                    completion_text=completion_text,
                    prompt_token_ids=prompt_ids[row],
                    completion_token_ids=tuple(completions[row]),
                    behavior_log_probs=tuple(behavior[row]),
                    allowed_token_ids=tuple(allowed_rows[row]),
                    attention_mask=(1,) * (
                        len(prompt_ids[row]) + len(completions[row])
                    ),
                    loss_mask=(1,) * len(completions[row]),
                    sampling_config={
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "top_k": 0,
                        "constrained": True,
                        "terminator": "newline",
                        "probability_path": "batched_kv_cache",
                        "sampling_contract_version": 1,
                        "sampling_batch_size": batch_size,
                        "sampling_batch_row": row,
                        "sampling_pad_token_id": int(pad_token),
                        "sampling_prompt_width": width,
                    },
                )
            )
        return tuple(samples)


def constrained_log_probs_full_sequence(model: Any, sample: PolicySample) -> Any:
    """Diagnostic-only full-sequence probability computation.

    This is deliberately not used for behavior replay or training. On the
    pinned HPU BF16 stack, its logits can differ materially from incremental
    KV-cache decoding even when the weights and token sequence are identical.
    """

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch is required to replay policy probabilities") from exc
    sample.validate()
    if not sample.completion_token_ids:
        raise ValueError("Bot samples do not contain trainable token probabilities")
    device = next(model.parameters()).device
    all_ids = sample.prompt_token_ids + sample.completion_token_ids
    input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)
    attention = torch.tensor([sample.attention_mask], dtype=torch.long, device=device)
    outputs = model(input_ids=input_ids, attention_mask=attention, use_cache=False)
    prompt_length = len(sample.prompt_token_ids)
    token_log_probs = []
    for index, (token, allowed) in enumerate(
        zip(sample.completion_token_ids, sample.allowed_token_ids)
    ):
        prediction_position = prompt_length - 1 + index
        logits = outputs.logits[0, prediction_position].float()
        allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=device)
        masked_log_probs = torch.log_softmax(logits.index_select(0, allowed_tensor), dim=0)
        try:
            selected_index = allowed.index(token)
        except ValueError as exc:
            raise ValueError("Recorded completion token is not grammar-legal") from exc
        token_log_probs.append(masked_log_probs[selected_index])
    return torch.stack(token_log_probs)


def constrained_log_probs_cached(model: Any, sample: PolicySample) -> Any:
    """Compute differentiable probabilities through the sampling KV-cache path."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch is required to replay policy probabilities") from exc
    sample.validate()
    if not sample.completion_token_ids:
        raise ValueError("Bot samples do not contain trainable token probabilities")
    device = next(model.parameters()).device
    prompt = torch.tensor([sample.prompt_token_ids], dtype=torch.long, device=device)
    outputs = model(input_ids=prompt, use_cache=True)
    logits = outputs.logits[0, -1].float()
    past = outputs.past_key_values
    token_log_probs = []
    for index, (token, allowed) in enumerate(
        zip(sample.completion_token_ids, sample.allowed_token_ids)
    ):
        allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=device)
        masked = torch.log_softmax(logits.index_select(0, allowed_tensor), dim=0)
        try:
            selected_index = allowed.index(token)
        except ValueError as exc:
            raise ValueError("Recorded completion token is not grammar-legal") from exc
        token_log_probs.append(masked[selected_index])
        if index + 1 < len(sample.completion_token_ids):
            next_input = torch.tensor([[token]], dtype=torch.long, device=device)
            outputs = model(input_ids=next_input, past_key_values=past, use_cache=True)
            logits = outputs.logits[0, -1].float()
            past = outputs.past_key_values
    return torch.stack(token_log_probs)


def _constrained_log_probs_recorded_shape(
    model: Any,
    sample: PolicySample,
    *,
    target_row_only: bool,
) -> Any:
    """Replay one sample in its recorded padding shape and selected batch mode.

    Transformer rows are mathematically independent, so non-target rows may
    duplicate the recorded row. Retaining the original batch dimensions,
    target row, right-padded prompt width, attention mask, and position IDs is
    necessary because the pinned eager HPU BF16 kernels are not numerically
    equivalent across tensor shapes.
    """

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch is required to replay policy probabilities") from exc
    sample.validate()
    if not sample.completion_token_ids:
        raise ValueError("Bot samples do not contain trainable token probabilities")
    config = sample.sampling_config
    required = {
        "sampling_contract_version",
        "sampling_batch_size",
        "sampling_batch_row",
        "sampling_pad_token_id",
        "sampling_prompt_width",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(
            "Batched sample lacks exact shape metadata: " + ", ".join(missing)
        )
    if int(config["sampling_contract_version"]) != 1:
        raise ValueError("Unsupported batched sampling contract version")
    recorded_batch_size = int(config["sampling_batch_size"])
    recorded_target_row = int(config["sampling_batch_row"])
    pad_token = int(config["sampling_pad_token_id"])
    prompt_width = int(config["sampling_prompt_width"])
    prompt_length = len(sample.prompt_token_ids)
    if recorded_batch_size <= 0 or not 0 <= recorded_target_row < recorded_batch_size:
        raise ValueError("Invalid recorded batched sampling dimensions")
    if prompt_width < prompt_length:
        raise ValueError("Recorded prompt width is shorter than the sample prompt")

    batch_size = 1 if target_row_only else recorded_batch_size
    target_row = 0 if target_row_only else recorded_target_row
    device = next(model.parameters()).device
    input_ids = torch.full(
        (batch_size, prompt_width), pad_token, dtype=torch.long, device=device
    )
    attention = torch.zeros(
        (batch_size, prompt_width), dtype=torch.long, device=device
    )
    prompt = torch.tensor(sample.prompt_token_ids, dtype=torch.long, device=device)
    # Duplicate rows keep the original kernel shape without requiring another
    # game's private prompt in every sample artifact. Rows never attend across
    # the batch dimension.
    input_ids[:, :prompt_length] = prompt.unsqueeze(0).expand(batch_size, -1)
    attention[:, :prompt_length] = 1
    position_ids = attention.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention == 0, 0)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention,
        position_ids=position_ids,
        use_cache=True,
    )
    logits = outputs.logits[target_row, prompt_length - 1].float()
    past = outputs.past_key_values
    token_log_probs = []
    for index, (token, allowed) in enumerate(
        zip(sample.completion_token_ids, sample.allowed_token_ids)
    ):
        allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=device)
        masked = torch.log_softmax(logits.index_select(0, allowed_tensor), dim=0)
        try:
            selected_index = allowed.index(token)
        except ValueError as exc:
            raise ValueError("Recorded completion token is not grammar-legal") from exc
        token_log_probs.append(masked[selected_index])
        if index + 1 < len(sample.completion_token_ids):
            next_input = torch.full(
                (batch_size, 1), int(token), dtype=torch.long, device=device
            )
            next_attention = torch.ones(
                (batch_size, 1), dtype=torch.long, device=device
            )
            attention = torch.cat((attention, next_attention), dim=1)
            next_positions = attention.cumsum(dim=-1)[:, -1:] - 1
            outputs = model(
                input_ids=next_input,
                attention_mask=attention,
                position_ids=next_positions,
                past_key_values=past,
                use_cache=True,
            )
            logits = outputs.logits[target_row, -1].float()
            past = outputs.past_key_values
    return torch.stack(token_log_probs)


def constrained_log_probs_batched_shape(model: Any, sample: PolicySample) -> Any:
    """Replay one sample in its exact recorded padded batch tensor shape.

    This remains the authoritative behavior-probability path.
    """

    return _constrained_log_probs_recorded_shape(
        model,
        sample,
        target_row_only=False,
    )


def constrained_log_probs_padded_target_row(model: Any, sample: PolicySample) -> Any:
    """Replay only the target row while preserving recorded padding geometry.

    This diagnostic path can differ from recorded batched behavior probabilities
    and must not be used for the distributed policy update.
    """

    return _constrained_log_probs_recorded_shape(
        model,
        sample,
        target_row_only=True,
    )


def constrained_log_probs(model: Any, sample: PolicySample) -> Any:
    """Authoritative sampling-aligned probability computation.

    Sampling, pre-update replay, policy loss, and reference-policy replay must
    all use this identical incremental KV-cache path. Keeping that invariant is
    more important than equivalence to a differently shaped full-sequence
    forward on accelerator kernels.
    """

    sampling_config = getattr(sample, "sampling_config", {})
    if sampling_config.get("probability_path") == "batched_kv_cache":
        return constrained_log_probs_batched_shape(model, sample)
    return constrained_log_probs_cached(model, sample)


def max_behavior_replay_error(model: Any, samples: Sequence[PolicySample]) -> float:
    """Return max |new-old| before an update; expected to be approximately zero."""

    rows = behavior_replay_diagnostics(model, samples)
    errors = [
        math.inf if row["max_abs_error"] is None else float(row["max_abs_error"])
        for row in rows
    ]
    return max(errors, default=0.0)


def summarize_log_prob_replay(
    behavior: Sequence[float], replayed: Sequence[float]
) -> dict[str, Any]:
    """Build a JSON-safe comparison that treats every non-finite value as failure."""

    if len(behavior) != len(replayed):
        raise ValueError("Behavior and replayed log-probability rows differ in length")
    behavior_values = [float(value) for value in behavior]
    replayed_values = [float(value) for value in replayed]
    bad_behavior = [
        index for index, value in enumerate(behavior_values) if not math.isfinite(value)
    ]
    bad_replayed = [
        index for index, value in enumerate(replayed_values) if not math.isfinite(value)
    ]
    finite = not bad_behavior and not bad_replayed
    return {
        "behavior_log_probs": [
            value if math.isfinite(value) else None for value in behavior_values
        ],
        "finite": finite,
        "max_abs_error": (
            max(
                (abs(new - old) for old, new in zip(behavior_values, replayed_values)),
                default=0.0,
            )
            if finite
            else None
        ),
        "nonfinite_behavior_positions": bad_behavior,
        "nonfinite_replayed_positions": bad_replayed,
        "replayed_log_probs": [
            value if math.isfinite(value) else None for value in replayed_values
        ],
    }


def behavior_replay_diagnostics(
    model: Any, samples: Sequence[PolicySample]
) -> list[dict[str, Any]]:
    """Replay samples and return JSON-safe, fail-closed per-sample comparisons."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch is required to verify policy probabilities") from exc
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for index, sample in enumerate(samples):
            replayed = constrained_log_probs(model, sample).cpu()
            row = summarize_log_prob_replay(
                sample.behavior_log_probs, replayed.tolist()
            )
            rows.append({"sample_index": index, **row})
    return rows
