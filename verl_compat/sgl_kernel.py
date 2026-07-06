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

"""Habana/HPU compatibility stub for ``sgl_kernel``.

``sgl_kernel`` ships CUDA-compiled kernels and cannot be installed on Intel Gaudi (HPU);
there is no HPU build (Ascend has ``sgl_kernel_npu``, HPU has nothing). Because the Habana
build enables the GPU Migration Toolkit (``PT_HPU_GPU_MIGRATION=1``), ``torch.cuda`` reports
as available, so sglang's ``is_cuda()`` checks evaluate ``True`` and its CUDA-gated modules
run ``from sgl_kernel import <name>`` at import time — crashing the entire rollout import
with ``ModuleNotFoundError: No module named 'sgl_kernel'`` even when no quantization is used.

This module is a stub placed ahead of the sglang fork on ``PYTHONPATH`` so those imports
resolve. Every requested symbol becomes a placeholder that raises a clear error **only if it
is actually called** — which does not happen on the plain bf16 generation path. If a kernel
*is* invoked at runtime, the message names the exact ``sgl_kernel`` symbol that still needs a
Habana implementation, giving a precise roadmap for the sglang-on-HPU port.

All observed imports are top-level (``from sgl_kernel import X`` / ``import sgl_kernel``), so a
module-level ``__getattr__`` (PEP 562) covers every case. Delete this file once a real HPU
kernel package is available.
"""


def __getattr__(name):  # PEP 562: resolves any `from sgl_kernel import <name>` access.
    def _sgl_kernel_symbol_unavailable_on_hpu(*args, **kwargs):
        raise RuntimeError(
            f"sgl_kernel.{name} is a CUDA-only kernel with no Habana/HPU implementation, "
            f"but it was invoked on the HPU path. This sglang code path needs a Gaudi port "
            f"(replace the sgl_kernel call with an HPU/torch.hpu equivalent or a native "
            f"fallback)."
        )

    return _sgl_kernel_symbol_unavailable_on_hpu
