#!/usr/bin/env python3
"""Read-only source/import/API checks; does not launch Ray or allocate an HPU."""
import argparse
import asyncio
import importlib
import importlib.metadata
import inspect
import os
from pathlib import Path
import subprocess
import sys
import shlex
import tempfile

COMMIT = "cb05a44f35a7c9e27e46d74112cc841ca674ef43"


def check_source():
    root = Path(os.environ["SGLANG_HPU_ROOT"]).resolve()
    patches = Path(__file__).parent / "patches"
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    assert head == COMMIT, f"Wrong SGLang revision: {head}"
    patch_names = (
        "sglang-miles-cb05a44-gaudi.patch",
        "sglang-cb05a44-hpu-container-compat.patch",
        "sglang-cb05a44-verl-hpu-runtime.patch",
    )
    # Reconstruct in patch order. Independent reverse checks are insufficient when
    # an integration patch intentionally edits a line introduced by the Miles layer.
    with tempfile.TemporaryDirectory(prefix="sglang-reconstruct-", dir="/tmp") as tmp:
        reconstructed = Path(tmp) / "sglang"
        subprocess.run(
            ["git", "clone", "--quiet", "--shared", "--no-checkout", str(root), str(reconstructed)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(reconstructed), "checkout", "--quiet", COMMIT],
            check=True,
        )
        for name in patch_names:
            subprocess.run(
                ["git", "-C", str(reconstructed), "apply", str(patches / name)],
                check=True,
            )
        tracked = subprocess.check_output(
            ["git", "-C", str(reconstructed), "diff", "--name-only", "HEAD"],
            text=True,
        ).splitlines()
        added = subprocess.check_output(
            ["git", "-C", str(reconstructed), "ls-files", "--others", "--exclude-standard"],
            text=True,
        ).splitlines()
        for relative in sorted(set(tracked + added)):
            expected = reconstructed / relative
            actual = root / relative
            assert actual.is_file(), f"Live SGLang runtime is missing patched file: {relative}"
            assert actual.read_bytes() == expected.read_bytes(), (
                f"Live SGLang runtime differs from repository patch stack: {relative}"
            )
    print(
        "PASS pinned SGLang reconstructed from repository-owned Miles, "
        f"container, and VERL HPU patches: {root}",
        flush=True,
    )
    scheduler_source = (root / "python/sglang/srt/managers/scheduler.py").read_text()
    manager_utils_source = (root / "python/sglang/srt/managers/utils.py").read_text()
    assert 'if _is_hip or self.server_args.device == "hpu":' in scheduler_source, (
        "HPU result D2H is incorrectly routed through SGLang's CUDA copy stream"
    )
    assert "self._hpu_mark_step()" in scheduler_source
    assert "self._hpu_synchronize()" in scheduler_source, (
        "HPU lazy forward is not explicitly synchronized before result D2H"
    )
    assert 'if t.device.type == "hpu":' in manager_utils_source
    assert 'return t.to("cpu", non_blocking=False)' in manager_utils_source, (
        "HPU result D2H is not forced synchronous"
    )
    print(
        "PASS HPU forward is explicitly synchronized before same-stream result D2H",
        flush=True,
    )
    return root


def check_runtime(root):
    import torch
    import habana_frameworks.torch  # noqa: F401

    assert "+hpu" in torch.__version__, f"Wrong Torch build: {torch.__version__}"
    assert not Path(torch.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), torch.__file__
    print(f"PASS container Habana Torch: {torch.__version__}", flush=True)
    import verl
    import sglang
    from sglang.srt.layers.attention.hpu_fused_backend import HPUFusedAttnBackend
    from sglang.srt.model_executor.runner.hpu_graph_runner import HPUDecodeGraphRunner
    from sglang.srt.layers.sampler import _to_token_id_dtype
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import is_cuda, is_hpu, MultiprocessingSerializer
    from verl.workers.rollout.sglang_rollout import async_sglang_server, sglang_rollout
    import recipe.sppo.main_sppo  # noqa: F401

    assert Path(sglang.__file__).resolve().is_relative_to(root), sglang.__file__
    assert Path(verl.__file__).resolve().is_relative_to(Path(os.environ["VERL_COMPAT"]).resolve())
    assert is_hpu() and not is_cuda(), "SGLang incorrectly selected CUDA under GPU migration"
    assert os.environ.get("PT_HPU_LAZY_MODE") == "0", "Run actor/import checks in eager mode"
    assert "sglang_rollout" in sglang_rollout.sgl_update_weights.__module__, "CUDA IPC weight helper selected"
    # SGLang >= 0.5.19 makes weight updates a session: begin -> update(s) -> end. Missing
    # this killed every rollout server at the first weight sync, ~10 minutes into a run,
    # so assert both ends of the contract here instead of discovering it on hardware.
    from sglang.srt.entrypoints import http_server as sgl_http_server
    from verl.workers.rollout.sglang_rollout.http_server_engine import AsyncHttpServerAdapter
    routes = {getattr(r, "path", None) for r in sgl_http_server.app.routes}
    for endpoint in ("begin_weight_update", "end_weight_update"):
        assert f"/{endpoint}" in routes, f"SGLang has no /{endpoint} route"
        assert callable(getattr(AsyncHttpServerAdapter, endpoint, None)), (
            f"VERL rollout adapter cannot call /{endpoint}"
        )
    weight_updater = inspect.getsource(
        importlib.import_module("sglang.srt.managers.scheduler_components.weight_updater")
    )
    assert "requires an open begin_weight_update session" in weight_updater
    assert "begin_weight_update" in inspect.getsource(sglang_rollout.ServerAdapter.update_weights), (
        "update_weights() does not open a weight-update session"
    )
    # VERL syncs weights with update_weights_from_tensor; miles only ever exercises
    # update_weights_from_distributed, so the tensor path shipped without the HPU lazy
    # execution boundary and generation ran against partially materialized weights.
    runner_updater = inspect.getsource(
        importlib.import_module(
            "sglang.srt.model_executor.model_runner_components.weight_updater"
        )
    )
    tensor_path = runner_updater[runner_updater.index("def update_weights_from_tensor") :]
    assert "_hpu_mark_step()" in tensor_path, (
        "update_weights_from_tensor lacks the HPU execution boundary"
    )
    assert "torch.hpu.synchronize()" in tensor_path, (
        "serialized CPU weights can be released before the lazy HPU copy completes"
    )
    print("PASS weight-update session protocol and HPU tensor-sync boundary", flush=True)
    assert callable(HPUFusedAttnBackend) and callable(HPUDecodeGraphRunner)
    # HPU-specific branch verified without creating a device context.
    class HPUTokenIDs:
        device = type("Device", (), {"type": "hpu"})()
    ids = HPUTokenIDs()
    assert _to_token_id_dtype(ids) is ids

    from huggingface_hub import snapshot_download
    model = os.environ.get("MODEL_PATH", "Qwen/Qwen3-4B-Base")
    if not Path(model).is_dir():
        model = snapshot_download(model, local_files_only=True)
    args = ServerArgs(
        model_path=model, device="hpu", dtype="bfloat16", tp_size=1,
        attention_backend="hpu_fused", decode_attention_backend="hpu_fused",
        sampling_backend="pytorch", grammar_backend="none",
        cuda_graph_backend_prefill="disabled", cuda_graph_backend_decode="full",
        context_length=1536, cuda_graph_max_bs_decode=64, max_running_requests=64,
        mem_fraction_static=0.5, disable_overlap_schedule=False,
        enable_memory_saver=False, skip_server_warmup=True,
    )
    args.check_server_args()
    assert args.device == "hpu" and args.attention_backend == "hpu_fused"
    assert args.sampling_backend == "pytorch" and not args.disable_overlap_schedule
    assert args.cuda_graph_backend_prefill == "disabled"
    assert args.cuda_graph_backend_decode == "full"
    sig = inspect.signature(async_sglang_server.sglang.srt.entrypoints.engine.Engine._launch_subprocesses)
    for name in ("server_args", "init_tokenizer_manager_func", "run_scheduler_process_func", "run_detokenizer_process_func"):
        assert name in sig.parameters, f"Missing launch API: {name}"
    print("PASS VERL imports, HPU detection, token dtype, and SGLang ServerArgs", flush=True)

    class Engine:
        async def update_weights_from_tensor(self, req):
            tensors = MultiprocessingSerializer.deserialize(req.serialized_named_tensors[0])
            assert tensors[0][0] == "probe.weight"
            torch.testing.assert_close(tensors[0][1], torch.tensor([1.25, -2.5]))
            assert not req.flush_cache

    asyncio.run(sglang_rollout.sgl_update_weights(Engine(), [("probe.weight", torch.tensor([1.25, -2.5]))]))
    print("PASS CPU-staged weight serialization round trip", flush=True)
    # Compose the actual launcher command with Hydra, without invoking its trainer.
    from hydra import compose, initialize_config_dir
    repo = Path(os.environ["REPO_ROOT"])
    launched = subprocess.check_output(
        ["bash", str(repo / "env/run_grpo_gsm8k.sh")], text=True, stderr=subprocess.DEVNULL,
        env={**os.environ, "DRY_RUN": "1", "VERL_RUN_INNER": "1", "WANDB": "0"},
    )
    overrides = shlex.split(launched.strip().splitlines()[-1])[3:]
    previous_cwd = Path.cwd()
    try:
        os.chdir(os.environ["VERL_COMPAT"])
        with initialize_config_dir(config_dir=str(Path.cwd() / "recipe/sppo/config"), version_base=None):
            config = compose(config_name="sppo_trainer", overrides=overrides)
    finally:
        os.chdir(previous_cwd)
    assert config.algorithm.adv_estimator == "grpo"
    assert config.trainer.n_gpus_per_node == 4
    assert not config.trainer.val_before_train and config.trainer.test_freq == -1
    assert config.actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend == "hpu_fused"
    print("PASS actual GRPO launcher Hydra composition (4 actors + 4 rollout, no validation)", flush=True)

    # Check Transformers 5 against Habana Torch 2.7 on a tiny CPU model first.
    from transformers import Qwen3Config, Qwen3ForCausalLM
    tiny_config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                             head_dim=16, attn_implementation="sdpa")
    tiny = Qwen3ForCausalLM(tiny_config).cpu()
    tokens = torch.tensor([[1, 2, 3, 4]])
    loss = tiny(input_ids=tokens, labels=tokens).loss
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in tiny.parameters() if p.grad is not None)
    print("PASS tiny Qwen3 CPU forward/backward with migrated Transformers", flush=True)

    import runpy
    import unittest
    # miles moved this file out of test/registered/unit/model_executor/runner and grew it
    # from graph eligibility alone to blockwise warmup shapes, attention-backend graph
    # safety, paged-v2 static inputs and per-layer mark_step hooks. All of it is pure
    # shape/policy logic, so run every case here -- none of it needs an HPU.
    tests = runpy.run_path(str(root / "test/manual/layers/attention/test_hpu_graph_runner.py"))
    suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromTestCase(obj)
        for obj in tests.values()
        if isinstance(obj, type)
        and issubclass(obj, unittest.TestCase)
        and obj is not unittest.TestCase
    )
    assert suite.countTestCases() >= 19, f"Expected miles HPU graph tests, got {suite.countTestCases()}"
    assert unittest.TextTestRunner(verbosity=0).run(suite).wasSuccessful(), "Miles HPU graph regression"
    for package in ("sglang", "transformers", "ray", "triton"):
        print(f"INFO {package}={importlib.metadata.version(package)}")
    print("PASS preflight complete. Generation, gradients, and long-run stability still require HPU tests.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true")
    opts = parser.parse_args()
    source = check_source()
    if not opts.source_only:
        check_runtime(source)
