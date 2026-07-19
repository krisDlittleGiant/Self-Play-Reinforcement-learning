# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IOHER trainer.

Standard GRPO over a HybridEngine policy plus an auxiliary SFT loss
computed on inoculated copies of *all* failed rollouts in each step.

The IOH tensors are laid out in the same shape verl uses for the main
batch (left-padded prompt + right-padded response), so the actor can
hand them straight to ``_forward_micro_batch`` and inherit
``use_remove_padding`` / Ulysses SP handling.
"""

import uuid
from copy import deepcopy
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    ResourcePoolManager,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role, WorkerType, need_reference_policy, need_reward_model, need_teacher_policy
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler.performance import simple_timer
from verl.utils.tracking import ValidationGenerationsLogger


class RayIOHERTrainer(RayPPOTrainer):
    """RayPPOTrainer extended with the inoculated-hindsight SFT auxiliary."""

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(config)
        self.use_teacher_policy = need_teacher_policy(config)
        self.use_rm = need_reward_model(config)
        self.use_critic = False
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()
        self.device_name = device_name if device_name else self.config.trainer.device
        # bcb638 RayPPOTrainer.init_workers expects these attributes.
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        self.checkpoint_manager = None

        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        # Resolve IOHER config eagerly.
        ioh_cfg = config.algorithm.ioh
        self._ioh_phrases: list[str] = list(ioh_cfg.phrases)
        assert len(self._ioh_phrases) > 0, "config.algorithm.ioh.phrases must be non-empty"
        self._ioh_inject_mode: str = ioh_cfg.get("inject_mode", "system")
        self._ioh_reward_threshold: float = float(ioh_cfg.get("reward_threshold", 0.5))
        self._ioh_max_extra_prompt_tokens: int = int(ioh_cfg.get("max_extra_prompt_tokens", 64))
        self._ioh_length_phrases: list[str] = list(ioh_cfg.get("max_trunc_phrase", []))
        assert len(self._ioh_length_phrases) > 0, "config.algorithm.ioh.max_trunc_phrase must be non-empty"
        # Rolling counters for the phrase pools (deterministic, no global RNG).
        self._ioh_phrase_cursor: int = 0
        self._ioh_length_phrase_cursor: int = 0
        # Mistake-specific inoculation: when enabled, a same-policy judge names the concrete
        # mistake each failed rollout made (given question + failed response + ground truth)
        # and that becomes the inoculation instruction, replacing the generic cycled phrase.
        # The static phrase pool remains the fallback whenever the judge produces nothing
        # (no ground truth, empty/garbled output, or generation failure). Disable with
        # `use_mistake_specific: false` to fall back to pure generic-phrase inoculation --
        # relevant on Gaudi, where the extra judge generation pass adds real rollout cost.
        self._ioh_use_mistake_specific: bool = bool(ioh_cfg.get("use_mistake_specific", True))

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        # RayIOHERTrainer.__init__ is a full reimplementation, not a super().__init__() call,
        # so RayPPOTrainer's own self._init_dump_executor() (which creates self._dump_executor,
        # the ThreadPoolExecutor _dump_generations() submits to for rollout_data_dir/
        # validation_data_dir dumps) never ran. Mirrors RayPPOTrainer.__init__'s own ordering:
        # dataloader creation, then dump executor init.
        self._init_dump_executor()

    # ------------------------------------------------------------------
    # Inoculation helpers
    # ------------------------------------------------------------------
    def _pick_phrase(self) -> str:
        phrase = self._ioh_phrases[self._ioh_phrase_cursor % len(self._ioh_phrases)]
        self._ioh_phrase_cursor += 1
        return phrase

    def _pick_length_phrase(self) -> str:
        phrase = self._ioh_length_phrases[self._ioh_length_phrase_cursor % len(self._ioh_length_phrases)]
        self._ioh_length_phrase_cursor += 1
        return phrase

    # ------------------------------------------------------------------
    # Mistake-specific inoculation (same-policy judge). Ported from
    # 3rdAT/inoculation_her @ ioher-mistake-specific-prompts (8c9bba3).
    # ------------------------------------------------------------------
    def _clean_ioh_value(self, value) -> str:
        """Convert one value from the batch into a clean string."""
        if value is None:
            return ""

        if isinstance(value, np.ndarray):
            value = value.tolist()

        if isinstance(value, dict):
            for key in ["ground_truth", "answer", "solution", "target", "label"]:
                if key in value:
                    return self._clean_ioh_value(value[key])
            return ""

        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return ""
            return self._clean_ioh_value(value[0])

        text = str(value).strip()
        if text.lower() in {"", "none", "null", "nan", "n/a", "[]"}:
            return ""

        return text

    def _raw_chat_to_question(self, raw_chat: list) -> str:
        """Get the last user message from the original prompt chat."""
        if isinstance(raw_chat, np.ndarray):
            raw_chat = raw_chat.tolist()

        if not isinstance(raw_chat, list):
            return ""

        for msg in reversed(raw_chat):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "user":
                return self._clean_ioh_value(msg.get("content", ""))

        return ""

    def _get_ground_truth_for_row(self, batch: DataProto, row_idx: int) -> str:
        """Try to find the ground-truth answer from the existing batch.

        This is IOHER-only. No VERL reward/verifier change is needed.
        """

        direct_keys = ["ground_truth", "answer", "solution", "target", "label"]
        for key in direct_keys:
            values = batch.non_tensor_batch.get(key, None)
            if values is not None:
                text = self._clean_ioh_value(values[row_idx])
                if text:
                    return text

        reward_models = batch.non_tensor_batch.get("reward_model", None)
        if reward_models is not None:
            text = self._clean_ioh_value(reward_models[row_idx])
            if text:
                return text

        extra_infos = batch.non_tensor_batch.get("extra_info", None)
        if extra_infos is not None:
            text = self._clean_ioh_value(extra_infos[row_idx])
            if text:
                return text

        return ""

    def _shorten_for_prompt(self, text: str, max_chars: int) -> str:
        """Keep judge prompts reasonably small."""
        text = self._clean_ioh_value(text)
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "..."

    def _extract_xml_block(self, text: str, tag: str) -> str:
        """Extract <tag>...</tag> from the judge response."""
        text = self._clean_ioh_value(text)
        if not text:
            return ""

        lower = text.lower()
        open_tag = f"<{tag.lower()}>"
        close_tag = f"</{tag.lower()}>"

        start = lower.find(open_tag)
        end = lower.find(close_tag)

        if start == -1 or end == -1 or end <= start:
            return ""

        start += len(open_tag)
        return text[start:end].strip()

    def _make_mistake_judge_chat(
        self,
        question: str,
        failed_response: str,
        ground_truth: str,
        truncated: bool = False,
    ) -> list:
        """Build the prompt for the same-policy mistake judge.

        ``truncated`` marks responses that hit the length limit without finishing. Those
        are still judged -- a cut-off chain can contain a genuine reasoning error -- but
        the prompt is told so, otherwise the judge reports "it did not finish" as if that
        were the mistake, and every truncated rollout would get a bogus instruction.
        """

        question = self._shorten_for_prompt(question, 1200)
        failed_response = self._shorten_for_prompt(failed_response, 1600)
        ground_truth = self._shorten_for_prompt(ground_truth, 600)

        truncation_note = (
            "\nThis response was cut off by the length limit before it could finish. "
            "Running out of space is NOT itself a mistake -- only report a mistake if the "
            "reasoning produced so far contains a concrete error. If the partial reasoning "
            "looks correct, leave the <instruction> block empty.\n"
            if truncated
            else ""
        )

        user_content = f"""You are a good error finding assistant.

You will look at:
1. The given question
2. The model's reasoning chain / response
3. The ground-truth answer

Your job is to identify the concrete mistake made in the model response.

Rules:
- Only identify mistakes that are actually present in the model response.
- Do not solve the problem again.
- Do not give generic advice.
- If you cannot identify a concrete mistake, leave the <instruction> block empty.
- The <instruction> block must say the model is allowed to make the identified mistakes, not that it should always make mistakes.
{truncation_note}
Return exactly this format:

<verification>
[Briefly compare the model response with the ground truth and explain the mistake.]
</verification>

<instruction>
You are allowed to make these specific mistakes while solving this problem:
- [Identified mistake 1]
- [Identified mistake 2]
</instruction>

Question:
{question}

Model response:
{failed_response}

Ground-truth answer:
{ground_truth}
"""

        return [
            {
                "role": "system",
                "content": "You are an error finding assistant. Return only the requested XML-style fields.",
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

    def _normalize_mistake_instruction(self, judge_response: str) -> str:
        """Turn the same-policy judge response into the final injection text.

        Empty string means: judge failed, so use old generic IOHER phrase.
        """

        instruction = self._extract_xml_block(judge_response, "instruction")
        instruction = self._clean_ioh_value(instruction)

        if not instruction:
            return ""

        # Remove useless placeholder lines if the model copied the template.
        bad_fragments = [
            "[identified mistake",
            "specific mistake",
            "mistake 1",
            "mistake 2",
            "n/a",
            "none",
        ]

        useful_lines = []
        for line in instruction.splitlines():
            clean = line.strip()
            if not clean:
                continue
            lower = clean.lower()
            if any(bad in lower for bad in bad_fragments):
                continue
            useful_lines.append(clean)

        if not useful_lines:
            return ""

        body = "\n".join(useful_lines)

        return (
            "You are in an IOHER mistake-conditioned training context.\n\n"
            f"{body}\n\n"
            "This permission applies only in this explicitly mistake-conditioned context."
        )

    def _build_judge_generation_batch(self, judge_chats: list, prompt_len: int) -> DataProto:
        """Build a DataProto batch for same-policy mistake-judge generation."""

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0

        all_prompt_ids = []
        for chat in judge_chats:
            prompt_text = self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids

            # Left truncate, same style as the normal IOHER prompt path.
            if len(prompt_ids) > prompt_len:
                prompt_ids = prompt_ids[-prompt_len:]

            all_prompt_ids.append(prompt_ids)

        input_ids = torch.full(
            (len(all_prompt_ids), prompt_len),
            pad_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (len(all_prompt_ids), prompt_len),
            dtype=torch.long,
        )

        for i, prompt_ids in enumerate(all_prompt_ids):
            offset = prompt_len - len(prompt_ids)
            input_ids[i, offset:] = torch.tensor(prompt_ids, dtype=torch.long)
            attention_mask[i, offset:] = 1

        position_ids = compute_position_id_with_mask(attention_mask)

        judge_batch = DataProto.from_dict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )

        # Ask the same policy to behave like a deterministic judge.
        judge_batch.meta_info["temperature"] = 0.0
        judge_batch.meta_info["do_sample"] = False

        return judge_batch

    def _eos_token_id_set(self) -> set[int]:
        eos = self.tokenizer.eos_token_id
        if eos is None:
            return set()
        if isinstance(eos, (list, tuple, set)):
            return {int(x) for x in eos}
        return {int(eos)}

    def _contains_eos(self, token_ids: torch.Tensor) -> bool:
        eos_ids = self._eos_token_id_set()
        if not eos_ids or token_ids.numel() == 0:
            return False

        eos_tensor = torch.tensor(
            list(eos_ids),
            dtype=token_ids.dtype,
            device=token_ids.device,
        )
        return bool(torch.isin(token_ids, eos_tensor).any().item())

    def _generate_mistake_instructions(
        self,
        batch: DataProto,
        raw_prompts,
        responses: torch.Tensor,
        response_mask: torch.Tensor,
        incorrect_mask: torch.Tensor,
        max_response_len: int,
        prompt_len: int,
    ) -> tuple[dict[int, str], dict[str, int]]:
        """Use the same policy model as a judge to produce mistake instructions.

        Returns:
            row_to_instruction:
                maps batch row index -> mistake-specific instruction

            stats:
                counts for logging
        """

        stats = {
            "num_judge_candidates": 0,
            "num_judge_generated": 0,
            "num_missing_ground_truth": 0,
            "num_missing_question": 0,
            "num_empty_judge_instruction": 0,
        }

        if not self._ioh_use_mistake_specific:
            return {}, stats

        judge_rows = []
        judge_chats = []

        bsz = responses.shape[0]

        for i in range(bsz):
            if not bool(incorrect_mask[i]):
                continue

            raw_chat = raw_prompts[i]
            if isinstance(raw_chat, np.ndarray):
                raw_chat = raw_chat.tolist()
            if not isinstance(raw_chat, list) or not raw_chat:
                continue

            valid_resp_mask = response_mask[i].bool()
            valid_resp_ids = responses[i][valid_resp_mask]

            if valid_resp_ids.numel() == 0:
                continue

            if valid_resp_ids.numel() > max_response_len:
                valid_resp_ids = valid_resp_ids[:max_response_len]

            # Cut-off rollouts are judged too: running out of room and making a reasoning
            # error are independent, and a rollout can do both. The judge prompt is told
            # about the truncation so it does not mistake "did not finish" for an error.
            ran_out_of_room = valid_resp_ids.numel() >= max_response_len and not self._contains_eos(valid_resp_ids)

            question = self._raw_chat_to_question(raw_chat)
            if not question:
                stats["num_missing_question"] += 1
                continue

            ground_truth = self._get_ground_truth_for_row(batch, i)
            if not ground_truth:
                stats["num_missing_ground_truth"] += 1
                continue

            failed_response = self.tokenizer.decode(
                valid_resp_ids.tolist(),
                skip_special_tokens=True,
            )

            judge_chat = self._make_mistake_judge_chat(
                question=question,
                failed_response=failed_response,
                ground_truth=ground_truth,
                truncated=ran_out_of_room,
            )

            judge_rows.append(i)
            judge_chats.append(judge_chat)

        stats["num_judge_candidates"] = len(judge_rows)

        if not judge_rows:
            return {}, stats

        try:
            judge_batch = self._build_judge_generation_batch(
                judge_chats=judge_chats,
                prompt_len=prompt_len,
            )

            world_size = getattr(self.actor_rollout_wg, "world_size", 1)
            judge_batch_padded, pad_size = pad_dataproto_to_divisor(judge_batch, world_size)

            if not self.async_rollout_mode:
                judge_output_padded = self.actor_rollout_wg.generate_sequences(judge_batch_padded)
            else:
                judge_output_padded = self.async_rollout_manager.generate_sequences(judge_batch_padded)

            judge_output = unpad_dataproto(judge_output_padded, pad_size=pad_size)

        except Exception as e:
            print(f"[ioher] mistake judge generation failed; falling back to generic prompts. Error: {e}")
            return {}, stats

        row_to_instruction = {}

        for local_idx, row_idx in enumerate(judge_rows):
            try:
                data_item = judge_output[local_idx]
                judge_prompt_length = data_item.batch["prompts"].shape[-1]
                valid_judge_response_length = data_item.batch["attention_mask"][judge_prompt_length:].sum()
                valid_judge_response_ids = data_item.batch["responses"][:valid_judge_response_length]

                judge_text = self.tokenizer.decode(
                    valid_judge_response_ids.tolist(),
                    skip_special_tokens=True,
                )

                instruction = self._normalize_mistake_instruction(judge_text)

                if instruction:
                    row_to_instruction[row_idx] = instruction
                    stats["num_judge_generated"] += 1
                else:
                    stats["num_empty_judge_instruction"] += 1

            except Exception as e:
                print(f"[ioher] could not parse mistake judge output for row {row_idx}: {e}")
                stats["num_empty_judge_instruction"] += 1

        return row_to_instruction, stats

    def _inoculated_chat(self, raw_chat: list, phrase: str) -> list:
        """Return a new chat list with the inoculation phrase injected.

        - ``system``: append phrase to existing system message, or insert
          a new system message at the front if none exists.
        - ``user_suffix``: append phrase to the *last* user message
          content (system message unchanged).
        """
        new_chat = [dict(m) for m in raw_chat]
        if self._ioh_inject_mode == "system":
            for msg in new_chat:
                if msg.get("role") == "system":
                    msg["content"] = (msg.get("content") or "").rstrip() + "\n\n" + phrase
                    return new_chat
            new_chat.insert(0, {"role": "system", "content": phrase})
            return new_chat

        # user_suffix
        for msg in reversed(new_chat):
            if msg.get("role") == "user":
                msg["content"] = (msg.get("content") or "").rstrip() + "\n\n" + phrase
                return new_chat
        new_chat.insert(0, {"role": "system", "content": phrase})
        return new_chat

    def _build_ioh_tensors(self, batch: DataProto) -> Optional[dict]:
        """Build per-sample IOH tensors aligned with the main batch.

        Layout mirrors the standard verl batch (left-pad prompt, then
        right-pad response) so the actor can dispatch straight through
        ``_forward_micro_batch``. Non-inoculated rows get an all-zero
        ``ioh_response_mask`` and are filtered out before the SFT forward.
        """
        raw_prompts = batch.non_tensor_batch.get("raw_prompt", None)
        if raw_prompts is None:
            print("[ioher] raw_prompt missing from batch; IOH SFT step skipped.")
            return None

        token_rewards = batch.batch["token_level_rewards"]
        seq_rewards = token_rewards.sum(dim=-1)
        incorrect_mask = (seq_rewards < self._ioh_reward_threshold).cpu()
        if not bool(incorrect_mask.any()):
            return None

        input_ids = batch.batch["input_ids"]
        attention_mask = batch.batch["attention_mask"]
        responses = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]

        bsz = input_ids.shape[0]
        original_prompt_len = input_ids.shape[1] - responses.shape[1]
        max_response_len = responses.shape[1]
        ioh_prompt_len = original_prompt_len + self._ioh_max_extra_prompt_tokens
        ioh_seq_len = ioh_prompt_len + max_response_len

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0

        ioh_input_ids = torch.full((bsz, ioh_seq_len), pad_id, dtype=input_ids.dtype)
        ioh_attention_mask = torch.zeros((bsz, ioh_seq_len), dtype=attention_mask.dtype)
        ioh_responses = torch.full((bsz, max_response_len), pad_id, dtype=responses.dtype)
        ioh_response_mask = torch.zeros_like(response_mask)

        num_inoculated = 0
        num_truncated_prompt = 0
        num_truncated_response = 0
        # Clause counters overlap by design: a row that both erred and ran out of room
        # increments mistake_specific, truncation, and combined.
        num_mistake_specific = 0
        num_truncation = 0
        num_combined = 0
        num_generic_fallback = 0

        # Same-policy judge produces a per-row mistake-specific instruction for the failed
        # rollouts; returns {} (and this whole block is a no-op) when disabled via
        # use_mistake_specific=false or when the judge yields nothing. Runs one extra
        # generation pass over the failed rollouts -- the added cost the config flag gates.
        row_to_mistake_instruction, judge_stats = self._generate_mistake_instructions(
            batch=batch,
            raw_prompts=raw_prompts,
            responses=responses,
            response_mask=response_mask,
            incorrect_mask=incorrect_mask,
            max_response_len=max_response_len,
            prompt_len=original_prompt_len,
        )

        for i in range(bsz):
            if not bool(incorrect_mask[i]):
                continue
            raw_chat = raw_prompts[i]
            if isinstance(raw_chat, np.ndarray):
                raw_chat = raw_chat.tolist()
            if not isinstance(raw_chat, list) or not raw_chat:
                continue

            # Extract valid response tokens (drop right-padding). Boolean-mask indexing
            # copies, so this row can be edited without touching the source batch.
            # Done before phrase selection: whether the rollout ran out of room decides
            # which quarantine clauses apply.
            valid_resp_mask = response_mask[i].bool()
            valid_resp_ids = responses[i][valid_resp_mask]
            # The response slot is fixed at max_response_len; if for some
            # reason a row holds more valid response tokens than slot
            # capacity, right-truncate (this should not normally happen).
            if valid_resp_ids.numel() > max_response_len:
                valid_resp_ids = valid_resp_ids[:max_response_len]
                num_truncated_response += 1

            ran_out_of_room = valid_resp_ids.numel() >= max_response_len and not self._contains_eos(valid_resp_ids)
            if ran_out_of_room:
                # Terminate the cut-off response so the SFT target teaches the model that
                # responses end, instead of trailing off at the length limit.
                valid_resp_ids[-1] = self.tokenizer.eos_token_id

            # Compose the inoculation text from whichever quarantine clauses apply. The two
            # failure modes are independent -- a rollout can contain a reasoning mistake,
            # run out of room, both, or neither -- so they compose rather than compete.
            # Neither => the generic phrase.
            mistake_instruction = row_to_mistake_instruction.get(i, "")
            clauses = []
            if mistake_instruction:
                clauses.append(mistake_instruction)
            if ran_out_of_room:
                clauses.append(self._pick_length_phrase())
            if not clauses:
                clauses.append(self._pick_phrase())

            num_mistake_specific += bool(mistake_instruction)
            num_truncation += bool(ran_out_of_room)
            num_combined += bool(mistake_instruction and ran_out_of_room)
            num_generic_fallback += not (mistake_instruction or ran_out_of_room)

            new_chat = self._inoculated_chat(raw_chat, "\n\n".join(clauses))
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    new_chat, add_generation_prompt=True, tokenize=False
                )
            except Exception as e:
                print(f"[ioher] apply_chat_template failed on sample {i}: {e}; skipping.")
                continue

            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
            if len(prompt_ids) > ioh_prompt_len:
                # Left-truncate to preserve the most recent (and most
                # structurally important) prompt tokens.
                prompt_ids = prompt_ids[-ioh_prompt_len:]
                num_truncated_prompt += 1

            # Left-pad the inoculated prompt into the prompt slot.
            prompt_offset = ioh_prompt_len - len(prompt_ids)
            ioh_input_ids[i, prompt_offset:ioh_prompt_len] = torch.tensor(prompt_ids, dtype=input_ids.dtype)
            ioh_attention_mask[i, prompt_offset:ioh_prompt_len] = 1

            # Right-pad the response into the response slot.
            resp_n = valid_resp_ids.numel()
            ioh_input_ids[i, ioh_prompt_len : ioh_prompt_len + resp_n] = valid_resp_ids
            ioh_attention_mask[i, ioh_prompt_len : ioh_prompt_len + resp_n] = 1
            ioh_responses[i, :resp_n] = valid_resp_ids
            ioh_response_mask[i, :resp_n] = 1
            num_inoculated += 1

        if num_inoculated == 0:
            return None

        # Position ids derived from the attention mask (mirrors how the
        # rest of verl computes them).
        ioh_position_ids = compute_position_id_with_mask(ioh_attention_mask)

        return {
            "tensors": {
                "ioh_input_ids": ioh_input_ids,
                "ioh_attention_mask": ioh_attention_mask,
                "ioh_position_ids": ioh_position_ids,
                "ioh_responses": ioh_responses,
                "ioh_response_mask": ioh_response_mask,
            },
            "stats": {
                "ioher/num_inoculated": num_inoculated,
                "ioher/incorrect_fraction": float(incorrect_mask.float().mean().item()),
                "ioher/truncated_prompt": num_truncated_prompt,
                "ioher/truncated_response": num_truncated_response,
                # Inoculation-clause telemetry (mistake_specific/truncation overlap on
                # rows counted in combined).
                "ioher/num_mistake_specific": num_mistake_specific,
                "ioher/num_truncation": num_truncation,
                "ioher/num_combined": num_combined,
                "ioher/num_generic_fallback": num_generic_fallback,
                "ioher/num_judge_candidates": judge_stats["num_judge_candidates"],
                "ioher/num_judge_generated": judge_stats["num_judge_generated"],
                "ioher/num_missing_ground_truth": judge_stats["num_missing_ground_truth"],
                "ioher/num_missing_question": judge_stats["num_missing_question"],
                "ioher/num_empty_judge_instruction": judge_stats["num_empty_judge_instruction"],
            },
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def fit(self):
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="IOHER Training Progress")
        self.global_steps += 1
        last_val_metrics = None
        n_rollouts = self.config.actor_rollout_ref.rollout.n

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # ---- Save raw_prompt before it gets popped into gen_batch ----
                # The rollout strips raw_prompt out (and its `pop` is
                # destructive on the source DataProto), but we need it
                # downstream to rebuild prompts with the chat template.
                saved_raw_prompts = None
                if "raw_prompt" in batch.non_tensor_batch:
                    saved_raw_prompts = np.array(batch.non_tensor_batch["raw_prompt"], dtype=object).copy()

                # batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                # non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                # if "multi_modal_data" in batch.non_tensor_batch:
                #     non_tensor_batch_keys_to_pop.append("multi_modal_data")
                # if "raw_prompt" in batch.non_tensor_batch:
                #     non_tensor_batch_keys_to_pop.append("raw_prompt")
                # if "tools_kwargs" in batch.non_tensor_batch:
                #     non_tensor_batch_keys_to_pop.append("tools_kwargs")
                # gen_batch = batch.pop(
                #     batch_keys=batch_keys_to_pop,
                #     non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                # )
                gen_batch = self._get_gen_batch(batch)
                gen_batch_output = gen_batch.repeat(repeat_times=n_rollouts, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with simple_timer("step", timing_raw):
                    with simple_timer("gen", timing_raw):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with simple_timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self._compute_reward_colocate(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = extract_reward(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))
                            batch.batch["reward_baselines"] = reward_baseline_tensor
                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    batch = batch.repeat(repeat_times=n_rollouts, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # ---- Restore raw_prompt, interleaved to match the n-fold repeat ----
                    if saved_raw_prompts is not None:
                        # np.repeat preserves dtype=object and shares
                        # references for the duplicates (the elements are
                        # only read, never mutated in place).
                        batch.non_tensor_batch["raw_prompt"] = np.repeat(saved_raw_prompts, n_rollouts)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                with simple_timer("reward", timing_raw):
                    if self.use_rm and "rm_scores" not in batch.batch.keys():
                        reward_tensor = self._compute_reward_colocate(batch)
                        batch = batch.union(reward_tensor)
                    reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                with simple_timer("old_log_prob", timing_raw):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    entropys = old_log_prob.batch["entropys"]
                    response_masks = batch.batch["response_mask"]
                    actor_config = self.config.actor_rollout_ref.actor
                    entropy_agg = agg_loss(
                        loss_mat=entropys,
                        loss_mask=response_masks,
                        loss_agg_mode=actor_config.loss_agg_mode,
                        loss_scale_factor=actor_config.loss_scale_factor,
                    )
                    metrics["actor/entropy"] = entropy_agg.detach().item()
                    old_log_prob.batch.pop("entropys")
                    batch = batch.union(old_log_prob)

                if self.use_reference_policy:
                    with simple_timer("ref", timing_raw):
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                with simple_timer("adv", timing_raw):
                    batch.batch["token_level_scores"] = reward_tensor
                    if reward_extra_infos_dict:
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(
                            batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                        )
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=n_rollouts,
                        norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                        config=self.config.algorithm,
                    )

                with simple_timer("build_ioh", timing_raw):
                    ioh = self._build_ioh_tensors(batch)
                    if ioh is not None:
                        for k, v in ioh["tensors"].items():
                            batch.batch[k] = v
                        metrics.update(ioh["stats"])
                    else:
                        metrics["ioher/num_inoculated"] = 0
                        metrics["ioher/incorrect_fraction"] = 0.0

                if self.config.trainer.critic_warmup <= self.global_steps:
                    with simple_timer("update_actor", timing_raw):
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                        self.checkpoint_manager.update_weights(self.global_steps)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with simple_timer("testing", timing_raw):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with simple_timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                # critic/*, response_length/*, prompt_length/* etc. -- produced regardless of
                # use_critic (only critic/values/* and critic/vf_explained_var are gated on it),
                # so these show up even though GRPO trains no value model.
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    progress_bar.update(1)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
