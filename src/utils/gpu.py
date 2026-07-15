"""GPU / system resource introspection.

Uses ``pynvml`` when available (NVIDIA only) and gracefully degrades
when the driver / library are missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import psutil

try:
    import pynvml  # type: ignore
    _NVML_OK = True
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore
    _NVML_OK = False


@dataclass
class GPUInfo:
    index: int
    name: str
    total_mb: int
    used_mb: int
    free_mb: int
    utilization: int  # 0-100

    @property
    def used_pct(self) -> float:
        return 0.0 if self.total_mb == 0 else 100.0 * self.used_mb / self.total_mb


@dataclass
class SystemStats:
    cpu_pct: float
    ram_used_gb: float
    ram_total_gb: float
    gpus: List[GPUInfo]


_nvml_initialised = False


def _ensure_nvml() -> bool:
    global _nvml_initialised
    if not _NVML_OK:
        return False
    if _nvml_initialised:
        return True
    try:
        pynvml.nvmlInit()
        _nvml_initialised = True
        return True
    except Exception:  # pragma: no cover
        return False


def read_gpus() -> List[GPUInfo]:
    if not _ensure_nvml():
        return []
    out: List[GPUInfo] = []
    try:
        n = pynvml.nvmlDeviceGetCount()
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            except Exception:
                util = 0
            out.append(
                GPUInfo(
                    index=i,
                    name=name,
                    total_mb=mem.total // (1024 * 1024),
                    used_mb=mem.used // (1024 * 1024),
                    free_mb=mem.free // (1024 * 1024),
                    utilization=int(util),
                )
            )
    except Exception:  # pragma: no cover
        return []
    return out


def read_system() -> SystemStats:
    vm = psutil.virtual_memory()
    return SystemStats(
        cpu_pct=psutil.cpu_percent(interval=None),
        ram_used_gb=(vm.total - vm.available) / (1024**3),
        ram_total_gb=vm.total / (1024**3),
        gpus=read_gpus(),
    )


def cuda_available() -> bool:
    try:
        import torch  # noqa: WPS433
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def torch_device() -> str:
    return "cuda" if cuda_available() else "cpu"


def describe_torch() -> str:
    try:
        import torch  # noqa: WPS433
        parts = [f"torch {torch.__version__}"]
        if torch.cuda.is_available():
            parts.append(f"CUDA {torch.version.cuda}")
            parts.append(f"device: {torch.cuda.get_device_name(0)}")
        else:
            parts.append("CPU only")
        return " | ".join(parts)
    except Exception as e:  # pragma: no cover
        return f"torch unavailable: {e}"
