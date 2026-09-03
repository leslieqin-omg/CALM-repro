from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DeviceSpec:
    device: torch.device
    amp_dtype: torch.dtype | None
    name: str
    total_memory_gb: float

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def autocast(self):
        if self.amp_dtype is None:
            return torch.autocast(device_type="cpu", enabled=False)
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)


def _cuda_free_memory(index: int) -> int:
    with torch.cuda.device(index):
        free, _total = torch.cuda.mem_get_info()
    return free


def _pick_cuda_index() -> int:
    count = torch.cuda.device_count()
    if count <= 1:
        return 0
    scored = sorted(((_cuda_free_memory(i), i) for i in range(count)), reverse=True)
    return scored[0][1]


def _cuda_amp_dtype(index: int) -> torch.dtype:
    major, _minor = torch.cuda.get_device_capability(index)
    return torch.bfloat16 if major >= 8 else torch.float16


def _cuda_spec(request: str, allow_amp: bool) -> DeviceSpec:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    index = int(request.split(":", 1)[1]) if ":" in request else _pick_cuda_index()
    props = torch.cuda.get_device_properties(index)
    dtype = _cuda_amp_dtype(index) if allow_amp else None
    memory_gb = props.total_memory / 1024 ** 3
    return DeviceSpec(torch.device("cuda", index), dtype, props.name, memory_gb)


def _mps_available() -> bool:
    return torch.backends.mps.is_available()


def resolve_device(request: str = "auto", allow_amp: bool = True) -> DeviceSpec:
    request = (request or "auto").strip().lower()
    if request == "cpu":
        return DeviceSpec(torch.device("cpu"), None, "cpu", 0.0)
    if request.startswith("cuda"):
        return _cuda_spec(request, allow_amp)
    if request == "mps":
        if not _mps_available():
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")
        return DeviceSpec(torch.device("mps"), None, "mps", 0.0)
    if torch.cuda.is_available():
        return _cuda_spec("cuda", allow_amp)
    if _mps_available():
        return DeviceSpec(torch.device("mps"), None, "mps", 0.0)
    return DeviceSpec(torch.device("cpu"), None, "cpu", 0.0)


def suggest_batch_size(spec: DeviceSpec, requested: int) -> int:
    if not spec.is_cuda:
        return min(requested, 128)
    if spec.total_memory_gb < 16.0:
        return min(requested, 512)
    return requested


def suggest_token_chunk(spec: DeviceSpec, requested: int | None) -> int:
    if requested is not None and requested > 0:
        return requested
    if not spec.is_cuda:
        return 16384
    if spec.total_memory_gb < 16.0:
        return 65536
    return 262144


def describe(spec: DeviceSpec) -> str:
    amp = "off" if spec.amp_dtype is None else str(spec.amp_dtype).replace("torch.", "")
    return f"device={spec.device} name={spec.name} memory={spec.total_memory_gb:.1f}GB autocast={amp}"


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
