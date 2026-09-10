#!/usr/bin/env python3
"""Compare SGLang rollout log-probs with an HF actor forward on one HPU."""

import argparse
import gc
import os
from contextlib import nullcontext


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpu", type=int, default=0)
    parser.add_argument("--max-response", type=int, default=64)
    parser.add_argument("--actor-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--autocast-bfloat16", action="store_true")
    args = parser.parse_args()

    os.environ["HABANA_VISIBLE_MODULES"] = str(args.hpu)
    os.environ["HABANA_VISIBLE_DEVICES"] = str(args.hpu)
    os.environ["PT_HPU_GPU_MIGRATION"] = "0"
    os.environ["PT_HPU_LAZY_MODE"] = "1"
    os.environ["PT_HPU_AUTOLOAD"] = "1"
    os.environ["VERL_HPU_SGLANG_PROCESS"] = "1"
    os.environ["SGLANG_EXPERIMENTAL_HPU_DECODE_GRAPH"] = "1"

    from verify_sglang_miles import check_source

    check_source()

    import habana_frameworks.torch.core as htcore
    import pyarrow.parquet as pq
    import torch
    import torch.nn.functional as F
    from sglang import Engine
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from verl.utils.model import compute_position_id_with_mask
    from verl.utils.tokenizer import normalize_token_ids

    model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen3-4B-Base")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    row = pq.read_table(os.path.join(os.environ["GSM8K_DIR"], "train.parquet")).to_pylist()[0]
    prompt_ids = normalize_token_ids(
        tokenizer.apply_chat_template(
            row["prompt"], tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
    )

    context = ((512 + args.max_response + 127) // 128) * 128
    engine = Engine(
        model_path=model_path,
        device="hpu",
        dtype="bfloat16",
        tp_size=1,
        attention_backend="hpu_fused",
        decode_attention_backend="hpu_fused",
        sampling_backend="pytorch",
        grammar_backend="none",
        cuda_graph_backend_prefill="disabled",
        cuda_graph_backend_decode="full",
        context_length=context,
        max_running_requests=1,
        cuda_graph_max_bs_decode=1,
        mem_fraction_static=0.5,
        disable_overlap_schedule=False,
        enable_memory_saver=False,
    )
    try:
        result = engine.generate(
            input_ids=prompt_ids,
            sampling_params={
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_new_tokens": args.max_response,
            },
            return_logprob=True,
        )
    finally:
        engine.shutdown()

    response_ids = list(result["output_ids"])
    triples = list(result["meta_info"]["output_token_logprobs"])
    assert len(triples) == len(response_ids), (len(triples), len(response_ids))
    rollout_ids = [int(item[1]) for item in triples]
    rollout_logp = torch.tensor([float(item[0]) for item in triples], dtype=torch.float32)
    assert rollout_ids == response_ids, "SGLang returned log-probs for different token IDs"
    print(
        f"SGLANG tokens={len(response_ids)} mean_logp={rollout_logp.mean().item():.6f} "
        f"first_ids={response_ids[:8]} first_logp={rollout_logp[:8].tolist()}",
        flush=True,
    )

    del engine, result, triples
    gc.collect()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=getattr(torch, args.actor_dtype),
        attn_implementation="eager",
        local_files_only=True,
    ).eval().to("hpu")

    def score(input_ids, attention_mask, position_ids, labels, start, label_count):
        with torch.no_grad():
            precision_context = (
                torch.autocast(device_type="hpu", dtype=torch.bfloat16)
                if args.autocast_bfloat16
                else nullcontext()
            )
            with precision_context:
                logits = model(
                    input_ids=input_ids.to("hpu"),
                    attention_mask=attention_mask.to("hpu"),
                    position_ids=position_ids.to("hpu"),
                    use_cache=False,
                ).logits
            selected = logits[:, start : start + label_count, :]
            logp = F.log_softmax(selected, dim=-1).gather(
                -1, labels.to("hpu").unsqueeze(-1)
            ).squeeze(-1)
            htcore.mark_step()
            torch.hpu.synchronize()
        return logp.float().cpu().squeeze(0)

    full_ids = prompt_ids + response_ids
    unpadded_input = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0)
    unpadded_mask = torch.ones_like(unpadded_input)
    unpadded_pos = compute_position_id_with_mask(unpadded_mask)
    labels = torch.tensor(response_ids, dtype=torch.long).unsqueeze(0)
    unpadded_logp = score(
        unpadded_input,
        unpadded_mask,
        unpadded_pos,
        labels,
        len(prompt_ids) - 1,
        len(response_ids),
    )

    prompt_width, response_width = 512, max(args.max_response, len(response_ids))
    pad_id = tokenizer.pad_token_id
    padded_prompt = [pad_id] * (prompt_width - len(prompt_ids)) + prompt_ids
    padded_response = response_ids + [pad_id] * (response_width - len(response_ids))
    padded_input = torch.tensor(padded_prompt + padded_response, dtype=torch.long).unsqueeze(0)
    padded_mask = torch.tensor(
        [0] * (prompt_width - len(prompt_ids))
        + [1] * len(prompt_ids)
        + [1] * len(response_ids)
        + [0] * (response_width - len(response_ids)),
        dtype=torch.long,
    ).unsqueeze(0)
    padded_pos = compute_position_id_with_mask(padded_mask)
    padded_logp = score(
        padded_input,
        padded_mask,
        padded_pos,
        labels,
        prompt_width - 1,
        len(response_ids),
    )

    def report(name, values):
        delta = values - rollout_logp
        print(
            f"{name} mean_logp={values.mean().item():.6f} "
            f"mean_abs_delta={delta.abs().mean().item():.6f} "
            f"max_abs_delta={delta.abs().max().item():.6f} "
            f"first_logp={values[:8].tolist()}",
            flush=True,
        )
        return delta

    unpadded_delta = report("HF_UNPADDED", unpadded_logp)
    padded_delta = report("HF_ACTOR_PADDED", padded_logp)
    padding_delta = padded_logp - unpadded_logp
    print(
        f"PADDING_EFFECT mean_abs_delta={padding_delta.abs().mean().item():.6f} "
        f"max_abs_delta={padding_delta.abs().max().item():.6f}",
        flush=True,
    )
    if padded_delta.abs().mean().item() > 0.25:
        raise SystemExit("FAIL: rollout and HF actor log-probs are not aligned")
    if padding_delta.abs().max().item() > 0.25:
        raise SystemExit("FAIL: actor padding/position construction changes valid-token log-probs")
    print("PASS: SGLang and HF actor token log-probs align", flush=True)


if __name__ == "__main__":
    main()
