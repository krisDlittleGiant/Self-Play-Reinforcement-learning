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

"""Auto-imported by Python's `site` module at the start of every interpreter (PEP 370) --
including sglang's Scheduler/Detokenizer/TP-worker subprocesses, which it spawns with
multiprocessing's "spawn" start method (required because HPU/CUDA device contexts can't be
forked safely). Those subprocesses only import whatever module defines their target function
(e.g. sglang.srt.managers.scheduler) -- never `verl` itself -- so verl's HPU compat patches in
verl/__init__.py (torchao disabling, get_device_capability, ...) never ran there, and the same
"synStatus=8 Device acquire failed" / "'>=' not supported between NoneType and int" errors kept
resurfacing one process deeper each time. Importing verl here applies them everywhere verl_compat
is on PYTHONPATH, no matter which process or entry point starts the interpreter.
"""
try:
    import verl  # noqa: F401
except Exception:
    pass
