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

"""Report which sglang symbols verl's sglang rollout imports are present/missing in the
installed sglang build (e.g. the Gaudi ``sglang-habana`` fork), so all version gaps can be
found in one pass instead of hitting them one ImportError at a time.

Run it with the SAME environment your training uses (so sglang and verl resolve, and the
GPU Migration Toolkit is active), e.g.::

    export PYTHONPATH=/workspace/inoculation/verl_gaudi_support/verl_compat:/scratch/sgoli125/sglang-habana/python:$PYTHONPATH
    export PT_HPU_GPU_MIGRATION=1 PT_HPU_LAZY_MODE=0 RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES=1
    python verl_compat/scripts/check_sglang_compat.py
"""

import importlib

# Import verl first so its HPU compatibility hooks (sgl_kernel stub, get_device_capability
# post-import hook) are installed before any sglang module loads.
try:
    import verl  # noqa: F401
except Exception as exc:  # pragma: no cover - diagnostic only
    print(f"[warn] could not import verl (hooks may be missing): {type(exc).__name__}: {exc}")

# (module, symbol) pairs that verl.workers.rollout.sglang_rollout.* import from sglang.
CHECKS = [
    ("sglang.srt.entrypoints.http_server", "ServerArgs"),
    ("sglang.srt.entrypoints.http_server", "app"),
    ("sglang.srt.entrypoints.http_server", "_GlobalState"),
    ("sglang.srt.entrypoints.http_server", "set_global_state"),
    ("sglang.srt.entrypoints.http_server", "Engine"),
    ("sglang.srt.entrypoints.http_server", "_launch_subprocesses"),
    ("sglang.srt.server_args", "ServerArgs"),
    ("sglang.srt.managers.io_struct", "GenerateReqInput"),
    ("sglang.srt.managers.io_struct", "PauseGenerationReqInput"),
    ("sglang.srt.managers.io_struct", "ContinueGenerationReqInput"),
    ("sglang.srt.managers.io_struct", "ReleaseMemoryOccupationReqInput"),
    ("sglang.srt.managers.io_struct", "ResumeMemoryOccupationReqInput"),
    ("sglang.srt.managers.io_struct", "LoadLoRAAdapterFromTensorsReqInput"),
    ("sglang.srt.managers.tokenizer_manager", "ServerStatus"),
    ("sglang.srt.utils.common", "add_prometheus_middleware"),
    ("sglang.srt.weight_sync.utils", "_preprocess_tensor_for_update_weights"),
    ("sglang.srt.weight_sync.utils", "update_weights"),
    ("sglang.srt.layers.moe.routed_experts_capturer", "extract_routed_experts_from_meta_info"),
]

_module_cache = {}
missing = []
mod_failures = {}

for mod_name, sym in CHECKS:
    if mod_name not in _module_cache:
        try:
            _module_cache[mod_name] = importlib.import_module(mod_name)
        except Exception as exc:
            _module_cache[mod_name] = None
            mod_failures[mod_name] = f"{type(exc).__name__}: {exc}"
    module = _module_cache[mod_name]
    if module is None:
        status = f"MODULE-IMPORT-FAILED ({mod_failures[mod_name]})"
        missing.append((mod_name, sym))
    elif hasattr(module, sym):
        status = "OK"
    else:
        status = "MISSING"
        missing.append((mod_name, sym))
    print(f"{status:<40} from {mod_name} import {sym}")

print("\n================ SUMMARY ================")
if missing:
    print(f"{len(missing)} symbol(s) NOT available in this sglang build:")
    for mod_name, sym in missing:
        print(f"  - {mod_name}.{sym}")
else:
    print("All checked sglang symbols are present.")
