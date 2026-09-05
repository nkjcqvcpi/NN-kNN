import os
import warnings

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


def resolve_runtime_device(env_var: str = "NNKNN_DEVICE") -> torch.device:
    """Purpose: choose a safe runtime device, with optional env override.

    `NNKNN_DEVICE=cpu` forces CPU.
    `NNKNN_DEVICE=cuda` forces CUDA and skips compatibility fallback checks.
    """
    requested = os.getenv(env_var)
    if requested:
        return torch.device(requested)

    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        current_sm = _sm_tag(torch.cuda.get_device_capability(0))
        supported_sms = set(torch.cuda.get_arch_list())
    except Exception as exc:
        warnings.warn(
            f"Falling back to CPU because CUDA capability detection failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")

    if current_sm not in supported_sms:
        warnings.warn(
            "Falling back to CPU because the installed PyTorch CUDA build does not "
            f"support GPU0 ({current_sm}). Supported SMs: {sorted(supported_sms)}. "
            "Install a matching PyTorch CUDA wheel or set NNKNN_DEVICE=cuda to force it.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")

    return torch.device("cuda")


def adam_kwargs_for_device(device: torch.device) -> dict:
    """Purpose: keep Adam off torch's foreach path on Intel XPU.

    A DQN ALE run on an Arc A770 (torch 2.13.0+xpu) crashed at ~26k steps in
    _multi_tensor_adam -> torch._foreach_div_ with
    UR_RESULT_ERROR_UNKNOWN from the level_zero backend. The single-tensor
    path does not use that op. CPU and CUDA are untouched, so their results
    remain bit-identical.
    """
    if getattr(device, "type", None) == "xpu":
        return {"foreach": False}
    return {}


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
