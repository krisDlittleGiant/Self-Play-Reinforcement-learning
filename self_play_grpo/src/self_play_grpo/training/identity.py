"""Canonical content fingerprints for replicated D5 trainer state.

Unlike a checkpoint schema digest, these hashes include tensor *values*.  The
same model/optimizer on different HPU ranks therefore produces the same hash
even when torch.save uses different pickle/storage ordering.  Hashing copies
tensor data to CPU but does not modify the input state or initialize an HPU.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping
from typing import Any


def _field(hasher: Any, tag: bytes, payload: bytes) -> None:
    hasher.update(tag)
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def _encode(value: Any, hasher: Any, torch: Any) -> None:
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise ValueError("Only dense strided tensors can be fingerprinted")
        detached = value.detach().to(device="cpu").contiguous()
        _field(hasher, b"tensor_dtype", str(detached.dtype).encode("ascii"))
        _encode(tuple(detached.shape), hasher, torch)
        # Viewing raw bytes also supports bfloat16, which NumPy cannot encode
        # directly.  The dtype and shape are hashed separately above.
        raw = detached.reshape(-1).view(torch.uint8).numpy().tobytes()
        _field(hasher, b"tensor_data", raw)
    elif value is None:
        _field(hasher, b"none", b"")
    elif type(value) is bool:
        _field(hasher, b"bool", b"1" if value else b"0")
    elif type(value) is int:
        _field(hasher, b"int", str(value).encode("ascii"))
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Non-finite scalar cannot be fingerprinted")
        _field(hasher, b"float64", struct.pack("!d", value))
    elif type(value) is str:
        _field(hasher, b"str", value.encode("utf-8"))
    elif type(value) is bytes:
        _field(hasher, b"bytes", value)
    elif type(value) in (tuple, list):
        _field(hasher, b"tuple" if type(value) is tuple else b"list", str(len(value)).encode("ascii"))
        for item in value:
            _encode(item, hasher, torch)
    elif isinstance(value, Mapping):
        # Optimizer state dictionaries use integer parameter IDs, while
        # parameter groups and nested metadata use string keys.  Sort by a
        # type-tagged digest to avoid depending on insertion order.
        entries = []
        for key, item in value.items():
            if type(key) not in (int, str):
                raise TypeError(f"Unsupported fingerprint mapping key: {type(key).__name__}")
            key_hasher = hashlib.sha256()
            _encode(key, key_hasher, torch)
            entries.append((key_hasher.digest(), key, item))
        entries.sort(key=lambda entry: entry[0])
        _field(hasher, b"mapping", str(len(entries)).encode("ascii"))
        for _, key, item in entries:
            _encode(key, hasher, torch)
            _encode(item, hasher, torch)
    else:
        raise TypeError(f"Unsupported fingerprint value: {type(value).__name__}")


def trainable_parameter_sha256(model: Any) -> str:
    """Hash names, dtypes, shapes and values of every trainable parameter."""

    import torch

    parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    if len({name for name, _ in parameters}) != len(parameters):
        raise ValueError("Trainable parameter names are not unique")
    hasher = hashlib.sha256()
    _field(hasher, b"domain", b"self-play-grpo/trainable-parameters/v1")
    _encode(tuple(sorted(parameters)), hasher, torch)
    return hasher.hexdigest()


def optimizer_state_sha256(optimizer: Any) -> str:
    """Hash optimizer state values and ordered parameter-group membership."""

    import torch

    state = optimizer.state_dict()
    if set(state) != {"state", "param_groups"}:
        raise ValueError("Optimizer state_dict has unexpected top-level keys")
    if not state["param_groups"]:
        raise ValueError("Optimizer has no parameter groups")
    hasher = hashlib.sha256()
    _field(hasher, b"domain", b"self-play-grpo/optimizer-state/v1")
    _encode(state, hasher, torch)
    return hasher.hexdigest()
