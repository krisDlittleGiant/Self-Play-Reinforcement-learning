# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import importlib
import logging
import os
import sys

print("=" * 50)
print(f"IMPORTING LOCAL VERL_COMPAT PATH: {__file__}", file=sys.stderr, flush=True)
print("=" * 50)

# --- Surface the REAL error when Ray fails to load an actor class on a worker --------
# When Ray cannot deserialize/load an actor class on a worker it stashes the real
# traceback as a *string* and substitutes a stub (TemporaryActor). That string is then
# lost on the driver because Ray cannot pickle the failing exception's cause (protobuf
# Descriptor, _struct.Struct, grpc/zmq handles, ...), so you only ever see the misleading
# "You set the async flag, but the actor does not have any coroutine functions" error.
# `_create_fake_actor_class` receives that string; we intercept it and print it verbatim
# (a plain string cannot be masked). This runs UNCONDITIONALLY because the function is
# only ever called when an actor class-load ACTUALLY fails -- so it has zero effect on
# healthy runs and needs no env flag (which would not reach Ray-spawned actor workers).
# Installed here, before any heavy verl import below, so it is active even if one of
# those imports is what fails on the worker.
try:
    import ray._private.function_manager as _verl_fm

    _verl_orig_fake = _verl_fm.FunctionActorManager._create_fake_actor_class
    if not getattr(_verl_orig_fake, "_verl_unmask_wrapped", False):

        def _verl_unmasked_fake_actor(self, *args, **kwargs):
            try:
                _cls = args[0] if args else kwargs.get("actor_class_name", "?")
                _tb = args[2] if len(args) >= 3 else kwargs.get("traceback_str", "<no traceback captured>")
                sys.stderr.write("\n" + "=" * 20 + " REAL ACTOR CLASS-LOAD ERROR (verl) " + "=" * 20 + "\n")
                sys.stderr.write(f"actor class: {_cls}\n{_tb}\n")
                sys.stderr.write("=" * 76 + "\n")
                sys.stderr.flush()
            except Exception:
                pass
            return _verl_orig_fake(self, *args, **kwargs)

        _verl_unmasked_fake_actor._verl_unmask_wrapped = True
        _verl_fm.FunctionActorManager._create_fake_actor_class = _verl_unmasked_fake_actor
except Exception:
    pass
# -----------------------------------------------------------------------------------

from packaging.version import parse as parse_version

from .protocol import DataProto
from .utils.device import is_npu_available
from .utils.import_utils import import_external_libs
from .utils.logging_utils import set_basic_config

# --- HPU compat: give torch.cuda.get_device_capability() a sane value on Gaudi -------
# Under the GPU Migration Toolkit, torch.cuda reports as available on HPU, but Gaudi has
# no CUDA compute capability, so torch.cuda.get_device_capability() returns None. sglang /
# vLLM run their CUDA-gated capability checks (e.g. `if major >= 9:` for Hopper fp8/cutlass)
# and crash with "'>=' not supported between NoneType and int". Return an Ampere-class
# (8, 0) capability so those checks resolve -- disabling Hopper/sm90-specific CUDA kernels
# while satisfying minimum-capability guards. Only overrides None/invalid results, so it is
# a no-op on real CUDA GPUs.
try:
    import torch as _verl_torch

    if getattr(_verl_torch, "cuda", None) is not None:
        _verl_orig_get_cap = _verl_torch.cuda.get_device_capability
        if not getattr(_verl_orig_get_cap, "_verl_hpu_wrapped", False):

            def _verl_hpu_get_device_capability(*args, **kwargs):
                try:
                    cap = _verl_orig_get_cap(*args, **kwargs)
                except Exception:
                    cap = None
                if not cap or cap[0] is None:
                    return (8, 0)
                return cap

            _verl_hpu_get_device_capability._verl_hpu_wrapped = True
            _verl_torch.cuda.get_device_capability = _verl_hpu_get_device_capability
except Exception:
    pass
# -----------------------------------------------------------------------------------

# --- HPU compat: patch sglang.srt.utils.get_device_capability() on import ------------
# sglang reads GPU compute capability through its OWN sglang.srt.utils.get_device_capability
# (not torch.cuda's), which returns (None, None) on Gaudi. Its CUDA-gated checks run because
# _is_cuda is True under the GPU Migration Toolkit, so e.g. cutlass_fp8_supported()'s
# `if major >= 9` crashes with "'>=' not supported between NoneType and int" at sglang import.
# verl imports sglang from several files (sglang_rollout, async_sglang_server, ...) across
# different processes, so instead of patching each import site we install a one-time
# post-import hook that patches sglang.srt.utils the moment it is loaded, via any entry point.
# Returns Ampere-class (8, 0) on None/invalid; no-op on real CUDA.
try:
    import importlib.util as _verl_ilu

    class _VerlSglangCapabilityPatcher:
        _target = "sglang.srt.utils"
        _busy = False

        def find_spec(self, fullname, path=None, target=None):
            if fullname != self._target or self._busy:
                return None
            self._busy = True
            try:
                spec = _verl_ilu.find_spec(fullname)
            except Exception:
                spec = None
            finally:
                self._busy = False
            if spec is None or getattr(spec, "loader", None) is None:
                return None
            _orig_exec = spec.loader.exec_module

            def _exec_and_patch(module):
                _orig_exec(module)
                try:
                    _orig_cap = module.get_device_capability
                    if not getattr(_orig_cap, "_verl_hpu_wrapped", False):

                        def _cap(*a, **k):
                            try:
                                c = _orig_cap(*a, **k)
                            except Exception:
                                c = None
                            if not c or c[0] is None:
                                return (8, 0)
                            return c

                        _cap._verl_hpu_wrapped = True
                        module.get_device_capability = _cap
                except Exception:
                    pass

            try:
                spec.loader.exec_module = _exec_and_patch
            except Exception:
                pass
            return spec

    if not any(getattr(_f, "_target", None) == "sglang.srt.utils" for _f in sys.meta_path):
        sys.meta_path.insert(0, _VerlSglangCapabilityPatcher())
except Exception:
    pass
# -----------------------------------------------------------------------------------

# --- HPU compat: disable torch's triton probe (it acquires a device at import time) --
# transformers -> torchao -> torch.sparse._triton_ops calls torch.utils._triton.has_triton()
# at IMPORT time, which probes torch.cuda.current_device() -> (GPU migration) torch.hpu device
# acquire. On any process without a free HPU (e.g. the sglang rollout server, during actor
# argument deserialization, before __init__ runs) this crashes with "Device acquire failed".
# NVIDIA Triton is not usable on Gaudi anyway, so force has_triton() -> False, which skips the
# device probe. verl is imported before torchao in that chain, so the patch is in place in time.
# Gated to hosts where habana_frameworks is installed, so CUDA behavior is untouched.
try:
    import importlib.util as _verl_ilu2

    if _verl_ilu2.find_spec("habana_frameworks") is not None:
        import torch.utils._triton as _verl_triton_mod

        if not getattr(_verl_triton_mod.has_triton, "_verl_hpu_disabled", False):

            def _verl_has_triton_false(*args, **kwargs):
                return False

            _verl_has_triton_false._verl_hpu_disabled = True
            _verl_triton_mod.has_triton = _verl_has_triton_false
except Exception:
    pass
# -----------------------------------------------------------------------------------

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "version/version")) as f:
    __version__ = f.read().strip()


set_basic_config(level=logging.WARNING)


__all__ = ["DataProto", "__version__"]


modules = os.getenv("VERL_USE_EXTERNAL_MODULES", "")
if modules:
    modules = modules.split(",")
    import_external_libs(modules)


# Auto-discover plugins via setuptools entry_points.
# Controlled by VERL_USE_EXTERNAL_PLUGINS:
#   "auto"  — load all entry_points in the "verl.plugins" group (default)
#   "none"  — disable entry_point discovery entirely
#   "pkg1,pkg2" — only load the named entry_points
_plugins_policy = os.getenv("VERL_USE_EXTERNAL_PLUGINS", "auto").strip().lower()
if _plugins_policy != "none":
    from importlib.metadata import entry_points as _entry_points

    _discovered = _entry_points(group="verl.plugins")
    if _plugins_policy == "auto":
        _allowed = None
    else:
        _allowed = {name.strip() for name in _plugins_policy.split(",") if name.strip()}

    for _ep in _discovered:
        if _allowed is not None and _ep.name not in _allowed:
            continue
        try:
            _ep.load()
        except Exception as _e:
            logging.getLogger(__name__).debug("Failed to load plugin '%s': %s", _ep.name, _e)


if os.getenv("VERL_USE_MODELSCOPE", "False").lower() == "true":
    if importlib.util.find_spec("modelscope") is None:
        raise ImportError("You are using the modelscope hub, please install modelscope by `pip install modelscope -U`")
    # Patch hub to download models from modelscope to speed up.
    from modelscope.utils.hf_util import patch_hub

    patch_hub()


if is_npu_available:
    # Workaround for torch-npu's lack of support for creating nested tensors from NPU tensors.
    #
    # ```
    # >>> a, b = torch.arange(3).npu(), torch.arange(5).npu() + 3
    # >>> nt = torch.nested.nested_tensor([a, b], layout=torch.jagged)
    # ```
    # throws "not supported in npu" on Ascend NPU.
    # See https://github.com/Ascend/pytorch/blob/294cdf5335439b359991cecc042957458a8d38ae/torch_npu/utils/npu_intercept.py#L109
    # for details.

    import torch

    try:
        if hasattr(torch.nested.nested_tensor, "__wrapped__"):
            torch.nested.nested_tensor = torch.nested.nested_tensor.__wrapped__
        if hasattr(torch.nested.as_nested_tensor, "__wrapped__"):
            torch.nested.as_nested_tensor = torch.nested.as_nested_tensor.__wrapped__
    except AttributeError:
        pass

    # In verl, the driver process aggregates the computation results of workers via Ray.
    # Therefore, after a worker completes its computation job, it will package the output
    # using tensordict and transfer it to the CPU. Since the `to` operation of tensordict
    # is non-blocking, when transferring data from a device to the CPU, it is necessary to
    # ensure that a batch of data has been completely transferred before being used on the
    # host; otherwise, unexpected precision issues may arise. Tensordict has already noticed
    # this problem and fixed it. Ref: https://github.com/pytorch/tensordict/issues/725
    # However, the relevant modifications only cover CUDA and MPS devices and do not take effect
    # for third-party devices such as NPUs. This patch fixes this issue, and the relevant
    # modifications can be removed once the fix is merged into tensordict.

    import tensordict

    if parse_version(tensordict.__version__) < parse_version("0.10.0"):
        from tensordict.base import TensorDictBase

        def _sync_all_patch(self):
            from torch._utils import _get_available_device_type, _get_device_module

            device_type = _get_available_device_type()
            if device_type is None:
                return

            device_module = _get_device_module(device_type)
            device_module.synchronize()

        TensorDictBase._sync_all = _sync_all_patch


# Register Qwen3 and Qwen3-MoE config/model compatibility for older transformers library versions
try:
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

    if "qwen3" not in AutoConfig.registry:
        AutoConfig.register("qwen3", Qwen2Config)
    if "qwen3moe" not in AutoConfig.registry:
        AutoConfig.register("qwen3moe", Qwen2Config)

    # Inject classes into the transformers module so that they can be loaded by architecture name mapping
    if not hasattr(transformers, "Qwen3ForCausalLM"):
        transformers.Qwen3ForCausalLM = Qwen2ForCausalLM
    if not hasattr(transformers, "Qwen3MoeForCausalLM"):
        transformers.Qwen3MoeForCausalLM = Qwen2ForCausalLM

    # Register the model class mapping for AutoModelForCausalLM
    AutoModelForCausalLM.register(Qwen2Config, Qwen2ForCausalLM)

    # Register tokenizer mapping in TOKENIZER_MAPPING_NAMES if present
    try:
        from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES
        if "qwen3" not in TOKENIZER_MAPPING_NAMES:
            TOKENIZER_MAPPING_NAMES["qwen3"] = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
        if "qwen3moe" not in TOKENIZER_MAPPING_NAMES:
            TOKENIZER_MAPPING_NAMES["qwen3moe"] = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    except Exception:
        pass
except Exception as _e:
    pass
