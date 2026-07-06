# Copyright (c) 2026 BAAI. All rights reserved.
"""Intel Habana Gaudi HPU platform implementation leveraging GPU Migration Toolkit."""

import os
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from .platform_base import PlatformBase
from .platform_manager import PlatformRegistry


class _HpuDeviceModule:
    """Mock/shim device module for torch.cuda to align with PyTorch contract under GPU Migration Toolkit."""
    def __getattr__(self, name):
        return getattr(torch.cuda, name)

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def set_device(self, device_index: int) -> None:
        if torch.cuda.device_count() == 1:
            torch.cuda.set_device(0)
        else:
            torch.cuda.set_device(device_index)


@PlatformRegistry.register(platform="hpu")
@PlatformRegistry.register(platform="intel")
class PlatformHPU(PlatformBase):
    """Platform backend for Intel Habana Gaudi HPUs mapped to CUDA namespace for GPU Migration Toolkit compatibility."""

    @property
    def device_name(self) -> str:
        # Return "cuda" to allow PyTorch native distributed APIs (like init_device_mesh) to run without C++ type errors
        return "cuda"

    @property
    def vendor_name(self) -> str:
        return "intel"

    @property
    def device_module(self) -> ModuleType:
        return _HpuDeviceModule()

    def is_available(self) -> bool:
        try:
            import habana_frameworks.torch.hpu as hthpu
            return hthpu.is_available()
        except ImportError:
            return False

    def is_platform_available(self, use_smi_check=False) -> bool:
        return self.is_available()

    def current_device(self) -> int:
        return torch.cuda.current_device()

    def device_count(self) -> int:
        return torch.cuda.device_count()

    def set_device(self, device_index: int) -> None:
        if torch.cuda.device_count() == 1:
            torch.cuda.set_device(0)
        else:
            torch.cuda.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        torch.cuda.synchronize(device_index)

    def manual_seed(self, seed: int) -> None:
        torch.cuda.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.cuda.manual_seed_all(seed)

    def set_allocator_settings(self, settings: str) -> None:
        pass

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        return torch.cuda.get_device_capability(device_index)

    def communication_backend_name(self) -> str:
        # Intel Habana Gaudi uses HCCL (Habana Collective Communications Library),
        # mirroring Ascend NPU (which also returns "hccl"). Returning "hccl" makes
        # get_nccl_backend() trigger its habana_frameworks.torch.distributed.hccl import
        # guard and lets distributed-init paths that are NOT special-cased for intel
        # (e.g. initialize_global_process_group, megatron_model_merger) select the
        # correct backend on Gaudi instead of falling back to nccl.
        return "hccl"

    def visible_devices_envvar(self) -> str:
        return "HABANA_VISIBLE_DEVICES"

    def ray_resource_name(self) -> str:
        return "HPU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        return {"resources": {"HPU": num_gpus}}

    def ray_noset_envvars(self) -> list[str]:
        return ["RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES"]

    def is_ipc_supported(self) -> bool:
        return False

    def supports_fractional_ray_resources(self) -> bool:
        return False

    @contextmanager
    def nvtx_range(self, msg: str):
        yield

    def profiler_start(self) -> None:
        pass

    def profiler_stop(self) -> None:
        pass

    def cudart(self) -> Any:
        return None
