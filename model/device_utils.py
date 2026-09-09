from __future__ import annotations

import os
import platform
import warnings
from typing import Any

import torch


def _sm_tag(capability: tuple[int, int]) -> str:
    """Purpose: normalize CUDA capability tuples into PyTorch arch tags."""
    major, minor = capability
    return f"sm_{major}{minor}"


def cuda_build_supports_current_gpu() -> bool:
    """Purpose: report whether the installed CUDA build supports GPU0's SM version."""
    if not torch.cuda.is_available():
        return False
    try:
        current_sm = _sm_tag(torch.cuda.get_device_capability(0))
        return current_sm in set(torch.cuda.get_arch_list())
    except Exception:
        return False


def is_xpu_available() -> bool:
    """Purpose: report whether an Intel XPU device is available."""
    return bool(hasattr(torch, "xpu") and torch.xpu.is_available())


def configure_xpu_environment() -> None:
    """Configure recommended environment variables for stable Intel Level-Zero execution.

    Stabilizes Level-Zero queue dispatch and avoids device lost issues on Windows.
    """
    os.environ.setdefault("ZE_ENABLE_PCI_ID_DEVICE_ORDER", "1")
    # Immediate command lists reduce latency and overhead in Level-Zero runtime
    os.environ.setdefault("SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS", "1")


def resolve_runtime_device(env_var: str = "NNKNN_DEVICE") -> torch.device:
    """Purpose: choose a safe runtime device, with optional env override.

    Priority:
    1. Environment variable `NNKNN_DEVICE` if set (e.g. 'cpu', 'cuda', 'xpu', 'xpu:0').
    2. CUDA device if available and supported by PyTorch build.
    3. CPU fallback by default (CPU is optimal for small kNN retrieval latency;
       use NNKNN_DEVICE=xpu or --device xpu explicitly for GPU acceleration).
    """
    requested = os.getenv(env_var)
    if requested:
        dev = torch.device(requested)
        if dev.type == "xpu":
            configure_xpu_environment()
        return dev

    if torch.cuda.is_available():
        try:
            current_sm = _sm_tag(torch.cuda.get_device_capability(0))
            supported_sms = set(torch.cuda.get_arch_list())
            if current_sm in supported_sms:
                return torch.device("cuda")
            warnings.warn(
                f"CUDA GPU0 ({current_sm}) not supported by PyTorch build; falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
        except Exception as exc:
            warnings.warn(
                f"Falling back to CPU because CUDA capability detection failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    return torch.device("cpu")


def adam_kwargs_for_device(device: torch.device | str) -> dict[str, Any]:
    """Purpose: provide safe optimizer parameters for device.

    On Intel XPU (Level-Zero), multi-tensor vectorization (foreach=True) can trigger
    driver-level errors with large parameter lists. Setting foreach=False forces the
    reliable single-tensor update path.
    """
    dev_type = getattr(device, "type", str(device).split(":")[0])
    if dev_type == "xpu":
        return {"foreach": False}
    return {}


def ensure_xpu_tensor_dtype(tensor: torch.Tensor, target_device: torch.device | str) -> torch.Tensor:
    """Purpose: guard against FP64 on Intel Arc GPUs which lack native FP64 units.

    Intel Arc A770 hardware has has_fp64=0. Tensors in torch.float64 must be converted
    to torch.float32 before transfer to avoid 'Required aspect fp64 is not supported' errors.
    """
    dev_type = getattr(target_device, "type", str(target_device).split(":")[0])
    if dev_type == "xpu" and tensor.dtype == torch.float64:
        return tensor.to(dtype=torch.float32, device=target_device)
    return tensor.to(target_device)


def runtime_env_fingerprint() -> dict[str, Any]:
    """Purpose: capture runtime environment details that impact reproducibility and stability."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }
    if is_xpu_available():
        info["xpu_device_name"] = torch.xpu.get_device_name(0)
        try:
            props = torch.xpu.get_device_properties(0)
            info["xpu_driver"] = props.driver_version
            info["xpu_has_fp64"] = bool(props.has_fp64)
            info["xpu_total_memory_mb"] = props.total_memory
        except Exception:
            pass
    elif torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return info
