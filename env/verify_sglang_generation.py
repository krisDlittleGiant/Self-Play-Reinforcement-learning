#!/usr/bin/env python3
"""Repeated one-HPU GSM8K generation with the same SGLang backend and sampling."""
import argparse
import os
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--hpu", type=int, default=0, help="Physical HPU module ID; must be free")
    parser.add_argument(
        "--overlap",
        action="store_true",
        help="Enable SGLang's overlap scheduler (disabled by default on HPU)",
    )
    parser.add_argument(
        "--require-positive-reward",
        action="store_true",
        help="Fail if none of the generated answers earns a positive GSM8K reward",
    )
    parser.add_argument(
        "--decode-graph-backend",
        choices=("full", "disabled"),
        default="full",
        help="Use full HPU decode graphs or the graph-disabled eager scheduler path",
    )
    args = parser.parse_args()
    assert args.batches > 0 and args.concurrency > 0
    os.environ["HABANA_VISIBLE_MODULES"] = str(args.hpu)
    os.environ["HABANA_VISIBLE_DEVICES"] = str(args.hpu)
    os.environ["PT_HPU_GPU_MIGRATION"] = "0"
    scheduler_lazy = args.decode_graph_backend != "disabled"
    os.environ["PT_HPU_LAZY_MODE"] = "1" if scheduler_lazy else "0"
    os.environ["PT_HPU_AUTOLOAD"] = "1"
    os.environ["VERL_HPU_SGLANG_PROCESS"] = "1"
    os.environ["SGLANG_EXPERIMENTAL_HPU_DECODE_GRAPH"] = "1" if scheduler_lazy else "0"
    from verify_sglang_miles import check_source
    check_source()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from sglang import Engine
    from verl.utils.reward_score.gsm8k import compute_score
    from verl.utils.tokenizer import normalize_token_ids

    model = os.environ.get("MODEL_PATH", "Qwen/Qwen3-4B-Base")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    rows = pq.read_table(os.path.join(os.environ["GSM8K_DIR"], "train.parquet")).to_pylist()
    max_response = int(os.environ.get("MAX_RESPONSE_LENGTH", "1024"))
    context = ((512 + max_response + 127) // 128) * 128
    engine = Engine(
        model_path=model, device="hpu", dtype="bfloat16", tp_size=1,
        attention_backend="hpu_fused", decode_attention_backend="hpu_fused",
        sampling_backend="pytorch", grammar_backend="none",
        cuda_graph_backend_prefill="disabled", cuda_graph_backend_decode=args.decode_graph_backend,
        context_length=context, max_running_requests=args.concurrency,
        cuda_graph_max_bs_decode=args.concurrency, mem_fraction_static=0.5,
        disable_overlap_schedule=not args.overlap, enable_memory_saver=False,
    )
    print(
        f"CONFIG hpu={args.hpu} concurrency={args.concurrency} "
        f"overlap={args.overlap} max_response={max_response} "
        f"decode_graph_backend={args.decode_graph_backend} lazy={int(scheduler_lazy)}",
        flush=True,
    )
    rewards = []
    response_lengths = []
    try:
        for batch in range(args.batches):
            selected = [rows[(batch * args.concurrency + i) % len(rows)] for i in range(args.concurrency)]
            prompts = [normalize_token_ids(tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True,
                enable_thinking=False)) for row in selected]
            assert all(len(prompt) <= 512 for prompt in prompts)
            started = time.monotonic()
            results = engine.generate(input_ids=prompts, sampling_params={
                "temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_new_tokens": max_response,
            })
            assert len(results) == len(prompts), "Missing generation results"
            for row, result in zip(selected, results, strict=True):
                token_ids = result["output_ids"]
                assert token_ids, "Empty generated token sequence"
                assert all(0 <= token < len(tokenizer) for token in token_ids), "Out-of-vocabulary token IDs"
                response_lengths.append(len(token_ids))
                answer = tokenizer.decode(token_ids, skip_special_tokens=True)
                rewards.append(compute_score(answer, row["reward_model"]["ground_truth"]))
            print(f"PASS batch={batch + 1}/{args.batches} sequences={len(results)} "
                  f"elapsed={time.monotonic() - started:.1f}s mean_reward={sum(rewards)/len(rewards):.4f} "
                  f"response_len_min={min(response_lengths)} response_len_max={max(response_lengths)}",
                  flush=True)
        has_positive_reward = any(reward > 0 for reward in rewards)
        if args.require_positive_reward:
            assert has_positive_reward, "All rewards are zero; inspect generated answers before GRPO"
        elif not has_positive_reward:
            print(
                "WARN generation was operational, but this sample set earned zero GSM8K reward; "
                "use --require-positive-reward only for a sufficiently large quality sample",
                flush=True,
            )
        print("PASS repeated generation and token-range checks; FSDP/weight-sync still require the GRPO tests.", flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
