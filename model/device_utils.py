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
    install_xpu_index_workaround()


_XPU_INDEX_MODE: Any = None


def install_xpu_index_workaround() -> bool:
    """Purpose: work around torch 2.15.0+xpu advanced indexing failures on Intel Arc (idempotent).

    `aten.index`/`aten.index_put` on XPU raise level_zero errors (error 20/39/40/2147483646, later
    ops then fail too) or crash with an access violation, for example `cases[arange(90)]` on a
    (90, 1) float32 tensor or `D[bool_matrix]`. `index_select`, `index_add_` and `index_copy_`
    are fine, so a dispatch mode rewrites single-tensor integer or boolean indexing into those.
    The mode only touches XPU tensors; multi-tensor indexing falls through unchanged.
    Returns True when the mode is active.
    """
    global _XPU_INDEX_MODE
    if _XPU_INDEX_MODE is not None:
        return True
    if not is_xpu_available():
        return False
    from torch.utils._python_dispatch import TorchDispatchMode

    aten = torch.ops.aten

    def single_index(x, indices):
        found = [(d, i) for d, i in enumerate(indices) if i is not None]
        if len(found) != 1 or x.device.type != "xpu":
            return None
        d, i = found[0]
        i = i.to(x.device)
        if i.dtype == torch.bool:
            m = i.dim()
            if m < 1 or tuple(i.shape) != tuple(x.shape[d:d + m]):
                return None
            merged = tuple(x.shape[:d]) + (-1,) + tuple(x.shape[d + m:])
            flat = i.reshape(-1).nonzero().squeeze(-1)
            return merged, d, flat, (flat.numel(),)
        if i.dtype in (torch.int64, torch.int32):
            i = i.long()
            i = torch.where(i < 0, i + x.shape[d], i)
            return tuple(x.shape), d, i.reshape(-1), tuple(i.shape)
        return None

    class XpuIndexFix(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func is aten.index.Tensor:
                found = single_index(args[0], args[1])
                if found:
                    merged, d, flat, mid = found
                    xs = args[0].reshape(merged)
                    return xs.index_select(d, flat).reshape(tuple(xs.shape[:d]) + mid + tuple(xs.shape[d + 1:]))
            elif func in (aten.index_put.default, aten.index_put_.default, aten._index_put_impl_.default):
                x, indices, values = args[0], args[1], args[2]
                accumulate = args[3] if len(args) > 3 else False
                found = single_index(x, indices)
                if found:
                    merged, d, flat, mid = found
                    target = x if func is not aten.index_put.default else x.clone()
                    try:
                        view = target.view(merged)
                    except RuntimeError:
                        view = None
                    if view is not None:
                        shape = tuple(view.shape[:d]) + mid + tuple(view.shape[d + 1:])
                        v = values.to(dtype=x.dtype, device=x.device).expand(shape)
                        v = v.reshape(tuple(view.shape[:d]) + (flat.numel(),) + tuple(view.shape[d + 1:]))
                        (view.index_add_ if accumulate else view.index_copy_)(d, flat, v.contiguous())
                        return target
            return func(*args, **kwargs)

    _XPU_INDEX_MODE = XpuIndexFix()
    _XPU_INDEX_MODE.__enter__()
    return True


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


import platform as _platform
def runtime_env_fingerprint() -> dict:
    """Purpose: capture settings that change results but are not part of cfg.

    Thread count is the motivating case: the nec smoke returns 155.0 at
    default threads and 90.0 under OMP_NUM_THREADS=2, deterministic at each,
    and a PPO ALE run flipped from +9.60 to -21.00 on the same seed purely by
    moving from 3 threads to 2. Without this block, config.json cannot tell
    those runs apart.
    """
    import platform as _platform

    return {
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "torch_num_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "platform": _platform.platform(),
        "python": _platform.python_version(),
    }
