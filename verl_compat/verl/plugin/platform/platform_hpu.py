# Copyright (c) 2026 BAAI. All rights reserved.
"""Intel Habana Gaudi HPU platform implementation."""

import os
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from .platform_base import PlatformBase
from .platform_manager import PlatformRegistry


class _HpuDeviceModule:
    """Mock/shim device module for habana_frameworks.torch.hpu to align with torch.cuda contract."""
    def __init__(self):
        try:
            import habana_frameworks.torch.hpu as hthpu
            self._module = hthpu
        except ImportError:
            self._module = None

    def __getattr__(self, name):
        if self._module is not None:
            return getattr(self._module, name)
        raise AttributeError("habana_frameworks.torch.hpu is not available")

    def is_available(self) -> bool:
        if self._module is None:
            return False
        return self._module.is_available()

    def empty_cache(self) -> None:
        pass

    def synchronize(self, device: Optional[Any] = None) -> None:
        if self._module is not None:
            self._module.synchronize()

    def get_device_properties(self, device_id: int = 0) -> Any:
        if self._module is not None and hasattr(self._module, "get_device_properties"):
            try:
                return self._module.get_device_properties(device_id)
            except Exception:
                pass
        class MockProperties:
            total_memory = 94 * 1024**3  # 94GB for Gaudi2
            name = "Intel Gaudi HPU"
        return MockProperties()



@PlatformRegistry.register(platform="hpu")
@PlatformRegistry.register(platform="intel")
class PlatformHPU(PlatformBase):
    """Platform backend for Intel Habana Gaudi HPUs."""

    @property
    def device_name(self) -> str:
        return "hpu"

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
        import habana_frameworks.torch.hpu as hthpu
        return hthpu.current_device()

    def device_count(self) -> int:
        import habana_frameworks.torch.hpu as hthpu
        return hthpu.device_count()

    def set_device(self, device_index: int) -> None:
        import habana_frameworks.torch.hpu as hthpu
        hthpu.set_device(0)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        import habana_frameworks.torch.hpu as hthpu
        hthpu.synchronize()

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)

    def set_allocator_settings(self, settings: str) -> None:
        pass

    def empty_cache(self) -> None:
        pass

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        return None, None

    def communication_backend_name(self) -> str:
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
