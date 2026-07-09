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
resurfacing one process deeper each time.

Only do this inside an actual multiprocessing spawn/forkserver child. Everywhere else --
including Ray's own dashboard/log_monitor/gcs subprocesses, which the raylet launches
directly rather than through Python's multiprocessing module -- stay a no-op: those don't
need verl at all, and importing it drags in torch under PT_HPU_GPU_MIGRATION=1, which was
making the plain dashboard process touch Habana's GPU-migration init and fail to start.

multiprocessing.parent_process() can't be used here: it isn't populated until spawn_main()'s
own bootstrap runs, which happens *after* interpreter startup (i.e. after this file already
ran). What IS available this early is sys.argv: cpython's multiprocessing.spawn.get_command_line()
always appends a literal "--multiprocessing-fork" marker for spawn/forkserver children, present
in argv from process start until multiprocessing's own prepare() step rewrites it later. Verified
locally: argv is ['-c', '--multiprocessing-fork'] at this point in a real spawn child, and plain
['-c'] both in the main process and in multiprocessing's own resource-tracker helper process.
"""
try:
    import sys

    if "--multiprocessing-fork" in sys.argv:
        import verl  # noqa: F401
except Exception:
    pass
