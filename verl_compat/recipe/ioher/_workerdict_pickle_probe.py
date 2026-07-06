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

"""TEMPORARY diagnostic — safe to delete once the WorkerDict actor issue is fixed.

Why this exists
---------------
Creating the colocated ``WorkerDict`` actor dies with the misleading Ray error
``"You set the async flag, but the actor does not have any coroutine functions"``.
That message is a *symptom*: Ray failed to deserialize the exported actor class on
the worker (via its own vendored ``ray.cloudpickle``), silently swapped in an
async-less ``TemporaryActor`` stub, and the async/stub mismatch produced the error.
The real deserialization traceback is never shown.

This module reproduces Ray's exact export -> load round-trip for ``WorkerDict`` and
prints the true traceback. It does NOT modify any Ray or verl source file: it wraps
``create_colocated_worker_cls`` at runtime and only activates when the environment
variable ``VERL_WORKER_PICKLE_PREFLIGHT`` is truthy. Delete this file (and the two
call sites in ``main_ioher.py``) once the root cause is fixed.
"""

import os
import sys
import traceback


def _enabled() -> bool:
    return os.getenv("VERL_WORKER_PICKLE_PREFLIGHT", "0").lower() in ("1", "true", "yes")


def install_pickle_probe() -> None:
    """Wrap create_colocated_worker_cls to round-trip WorkerDict through ray.cloudpickle.

    No-op unless VERL_WORKER_PICKLE_PREFLIGHT is set. Idempotent.
    """
    if not _enabled():
        return

    import ray
    import verl.trainer.ppo.ray_trainer as rt
    from verl.single_controller.ray.base import _unwrap_ray_remote

    original = rt.create_colocated_worker_cls
    if getattr(original, "_pickle_probe_wrapped", False):
        return

    @ray.remote(num_cpus=0)
    def _loads_on_worker(payload: bytes) -> str:
        import traceback as _tb

        import ray.cloudpickle as _rp

        try:
            _rp.loads(payload)
            return "OK (ray.cloudpickle, fresh worker): WorkerDict deserialized"
        except BaseException:
            return "FAILED (ray.cloudpickle, fresh worker):\n" + _tb.format_exc()

    def wrapped(*args, **kwargs):
        cia = original(*args, **kwargs)
        try:
            import ray.cloudpickle as rp

            worker_cls = _unwrap_ray_remote(cia.cls)
            banner = "=" * 24 + " WorkerDict pickle probe " + "=" * 24
            print(banner, file=sys.stderr, flush=True)

            # (1) Round-trip in this process (matches Ray's export pickler + context).
            try:
                rp.loads(rp.dumps(worker_cls))
                print("OK (ray.cloudpickle, inline): WorkerDict round-tripped", file=sys.stderr, flush=True)
            except BaseException:
                print("FAILED (ray.cloudpickle, inline):\n" + traceback.format_exc(), file=sys.stderr, flush=True)

            # (2) Round-trip on a fresh worker (mirrors Ray's export -> worker load path).
            try:
                payload = rp.dumps(worker_cls)
                print(ray.get(_loads_on_worker.remote(payload)), file=sys.stderr, flush=True)
            except BaseException:
                print("dumps FAILED before worker probe:\n" + traceback.format_exc(), file=sys.stderr, flush=True)

            print("=" * len(banner), file=sys.stderr, flush=True)
        except BaseException:
            print("[pickle probe] internal error:\n" + traceback.format_exc(), file=sys.stderr, flush=True)
        return cia

    wrapped._pickle_probe_wrapped = True
    rt.create_colocated_worker_cls = wrapped
