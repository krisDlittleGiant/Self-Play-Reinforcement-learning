#!/usr/bin/env python3
"""Isolate VERL's weight-sync path: generate, push identical weights, generate again.

The GRPO run degrades only after the actor pushes weights into the rollout engine,
while env/verify_sglang_generation.py (which never syncs) is healthy. This script
reproduces just the transfer on one HPU, with no Ray, FSDP or agent loop.

At step 1 the actor's parameters ARE the HF checkpoint, so a correct sync is
semantically a no-op: rewards and log-probs must not move. Any drop here is the
transfer corrupting weights.
"""
import argparse
import os
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=int, default=32)
    parser.add_argument("--hpu", type=int, default=0)
    parser.add_argument("--bucket-mb", type=int, default=128, help="Match VERL's update_weights_bucket_megabytes")
    parser.add_argument("--actor-dtype", default="float32", help="VERL runs the FSDP actor in fp32")
    parser.add_argument(
        "--max-response",
        type=int,
        default=256,
        help="Diagnostic-only greedy token cap; does not alter the GRPO response length",
    )
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

    import torch
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sglang import Engine
    from verl.utils.reward_score.gsm8k import compute_score
    from verl.utils.tokenizer import normalize_token_ids

    model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen3-4B-Base")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    rows = pq.read_table(os.path.join(os.environ["GSM8K_DIR"], "train.parquet")).to_pylist()[: args.prompts]
    max_response = args.max_response
    context = ((512 + max_response + 127) // 128) * 128
    prompts = [normalize_token_ids(tokenizer.apply_chat_template(
        row["prompt"], tokenize=True, add_generation_prompt=True, enable_thinking=False)) for row in rows]

    engine = Engine(
        model_path=model_path, device="hpu", dtype="bfloat16", tp_size=1,
        attention_backend="hpu_fused", decode_attention_backend="hpu_fused",
        sampling_backend="pytorch", grammar_backend="none",
        cuda_graph_backend_prefill="disabled", cuda_graph_backend_decode="full",
        context_length=context, max_running_requests=args.prompts,
        cuda_graph_max_bs_decode=args.prompts, mem_fraction_static=0.5,
        disable_overlap_schedule=False, enable_memory_saver=False,
    )

    def sample(label):
        started = time.monotonic()
        results = engine.generate(input_ids=prompts, sampling_params={
            # Greedy decoding turns the initial weight push into an exact regression:
            # identical checkpoint weights must produce identical output token IDs.
            "temperature": 0.0, "max_new_tokens": max_response,
        })
        rewards, lengths, outputs = [], [], []
        for row, result in zip(rows, results, strict=True):
            token_ids = result["output_ids"]
            outputs.append(token_ids)
            lengths.append(len(token_ids))
            assert all(0 <= t < len(tokenizer) for t in token_ids), f"{label}: out-of-vocab token"
            rewards.append(compute_score(tokenizer.decode(token_ids, skip_special_tokens=True),
                                         row["reward_model"]["ground_truth"]))
        mean_reward = sum(rewards) / len(rewards)
        print(f"{label}: mean_reward={mean_reward:.4f} mean_len={sum(lengths)/len(lengths):.1f} "
              f"elapsed={time.monotonic() - started:.1f}s", flush=True)
        return mean_reward, outputs

    try:
        before_reward, before_outputs = sample("BEFORE sync")

        # Mirror VERL: fp32 actor parameters, staged through CPU, serialized in buckets.
        actor = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=getattr(torch, args.actor_dtype), local_files_only=True).cpu().eval()
        named = [(name, param.detach().cpu()) for name, param in actor.named_parameters()]
        print(f"pushing {len(named)} tensors as {args.actor_dtype} in {args.bucket_mb}MB buckets", flush=True)

        engine.begin_weight_update()
        try:
            bucket, nbytes, sent = [], 0, 0
            limit = args.bucket_mb << 20
            for name, tensor in named:
                bucket.append((name, tensor))
                nbytes += tensor.numel() * tensor.element_size()
                if nbytes >= limit:
                    engine.update_weights_from_tensor(named_tensors=bucket, flush_cache=False)
                    sent += len(bucket); bucket, nbytes = [], 0
            if bucket:
                engine.update_weights_from_tensor(named_tensors=bucket, flush_cache=False)
                sent += len(bucket)
        finally:
            engine.end_weight_update()
        engine.flush_cache()
        print(f"pushed {sent} tensors", flush=True)

        after_reward, after_outputs = sample("AFTER sync")
        exact_matches = sum(before == after for before, after in zip(before_outputs, after_outputs, strict=True))
        print(f"RESULT before={before_reward:.4f} after={after_reward:.4f} "
              f"delta={after_reward - before_reward:+.4f} exact_outputs={exact_matches}/{len(before_outputs)}",
              flush=True)
        if exact_matches != len(before_outputs):
            print("FAIL identical weight sync changed greedy generation", flush=True)
            raise SystemExit(1)
        print("PASS identical weight sync preserved every greedy output", flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
