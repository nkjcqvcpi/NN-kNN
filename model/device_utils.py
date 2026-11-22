from __future__ import annotations

import math
import operator
import os
import platform
import threading
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


# --- XPU advanced-indexing workaround ---------------------------------------------------------
#
# torch 2.15.0+xpu (Windows wheel) on Intel Arc: every path into `aten::index`, `aten::index_put`
# (accumulate=False), `masked_select`, and the backward of `index_put` / `masked_scatter` loses the
# device (level_zero error 20/39/40/2147483646 at a later sync) or dies with an access violation.
# `index_select`, `index_copy_`, `index_add_`, `index_fill_`, `nonzero` and their backward are fine.
#
# The fix is a set of Python-level wrappers on the indexing entry points. Each wrapper first checks
# the device (anything not on XPU goes straight to the original) and then for basic indexing
# (ints, slices, None, Ellipsis), so only advanced indexing on an XPU tensor is rewritten. Nothing
# is hooked into the dispatcher, so other ops cost nothing.

_XPU_INDEX_ORIG: dict[str, Any] = {}  # non-empty while the wrappers are installed
_XPU_INDEX_LOCK = threading.RLock()  # install / uninstall / tripwire switches
_XPU_INDEX_TRIPWIRE: Any = None  # the set of broken ops while the tripwire is armed (all threads), else None
_XPU_INDEX_TRIPWIRE_LOCAL = threading.local()  # .mode: the dispatch mode this thread has entered
_XPU_INDEX_WARNED: set[str] = set()
_XPU_INDEX_CHECK_BOUNDS = True
_XPU_INDEX_HOST_MAX = 1 << 16  # index/mask elements up to which range checks and nonzero run on the host
_XPU_INDEX_TENSOR_ATTRS = (
    "__getitem__", "__setitem__", "index_put", "index_put_",
    "masked_select", "masked_scatter", "masked_scatter_",
)
_XPU_INDEX_TORCH_ATTRS = ("index_put", "index_put_", "masked_select", "masked_scatter", "combinations")
_XPU_INT_INDEX_DTYPES = (torch.int64, torch.int32)
_XPU_MASK_DTYPES = (torch.bool, torch.uint8)


class _XpuIndexUnsupported(Exception):
    """Raised (before any write) when a form is not rewritten; the caller then uses the CPU round trip."""


def _xpu_fallback_warn(form: str) -> None:
    if form not in _XPU_INDEX_WARNED:
        _XPU_INDEX_WARNED.add(form)
        warnings.warn(
            f"XPU index workaround: {form} is not rewritten for XPU; it runs through a CPU round trip (slow).",
            RuntimeWarning,
            stacklevel=4,
        )


def _xpu_describe(obj: Any) -> str:
    if isinstance(obj, torch.Tensor):
        return f"Tensor[{str(obj.dtype).replace('torch.', '')},{obj.dim()}d]"
    if isinstance(obj, (tuple, list)):
        return type(obj).__name__ + "(" + ", ".join(_xpu_describe(o) for o in obj[:8]) + ")"
    return type(obj).__name__


def _xpu_to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, tuple):
        return tuple(_xpu_to_cpu(o) for o in obj)
    if isinstance(obj, list):
        return [_xpu_to_cpu(o) for o in obj]
    return obj


def _xpu_norm_indices(idxs: list, sizes: list, dims: list, device: torch.device) -> list:
    """Purpose: return the index tensors as non-negative int64 on `device`; IndexError when out of range.

    CPU-resident indices are checked on the host for free. Indices on the device are copied to the
    host in one transfer for the check (cheaper on Arc than a device reduction plus read-back);
    above `_XPU_INDEX_HOST_MAX` elements the check is a device `aminmax` instead.
    """
    out = list(idxs)
    ranges: list = [None] * len(out)  # (lo, hi) of each index; None = not checked
    pending = []
    for j, t in enumerate(out):
        if t.dtype != torch.int64:
            t = out[j] = t.long()
        if t.numel() == 0:
            ranges[j] = (0, -1)
        elif t.device.type == "cpu":
            lo, hi = torch.aminmax(t)
            ranges[j] = (int(lo), int(hi))
        elif sizes[j] == 0:
            raise IndexError(f"index is out of bounds for dimension {dims[j]} with size 0")
        elif _XPU_INDEX_CHECK_BOUNDS:
            pending.append(j)
    if pending:
        small = [j for j in pending if out[j].numel() <= _XPU_INDEX_HOST_MAX]
        if len(small) == 1:
            host = [out[small[0]].cpu()]
        elif small:
            host = torch.cat([out[j].reshape(-1) for j in small]).cpu().split([out[j].numel() for j in small])
        for j, h in zip(small, host if small else ()):
            lo, hi = torch.aminmax(h)
            ranges[j] = (int(lo), int(hi))
        for j in pending:
            if ranges[j] is None:
                lo, hi = torch.aminmax(out[j])
                ranges[j] = (int(lo), int(hi))
    for j, t in enumerate(out):
        n = sizes[j]
        if ranges[j] is None:  # unchecked (NNKNN_XPU_INDEX_BOUNDS=0): wrap, never read outside the dim
            out[j] = (t if t.device == device else t.to(device)).remainder(n)
            continue
        lo, hi = ranges[j]
        if hi >= lo and (lo < -n or hi >= n):
            raise IndexError(f"index {hi if hi >= n else lo} is out of bounds for dimension {dims[j]} with size {n}")
        if t.device.type == "cpu":
            if lo < 0:
                t = torch.where(t < 0, t + n, t)
            t = t.to(device)
        else:
            if t.device != device:
                t = t.to(device)
            if lo < 0:
                t = t.remainder(n)
        out[j] = t
    return out


def _xpu_list_index(seq: list) -> torch.Tensor:
    """Purpose: flat list of ints or of bools -> CPU index tensor (other lists keep legacy torch rules)."""
    if all(type(v) is int for v in seq):
        return torch.tensor(seq, dtype=torch.int64)
    if all(type(v) is bool for v in seq):
        return torch.tensor(seq, dtype=torch.bool)
    raise _XpuIndexUnsupported


def _xpu_parse_key(x: torch.Tensor, key: Any) -> tuple[torch.Tensor, list]:
    """Purpose: split a Python index into its basic part (applied natively, a view) and tensor indices.

    Returns (view, adv); adv holds (dim in view, index tensor, is_mask) in key order. Follows torch:
    0-dim int tensors and `__index__` objects select like ints, masks consume `mask.dim()` dims.
    """
    if type(key) is not tuple:
        key = (key,)
    items = []
    consumed = 0
    trivial = True
    has_ellipsis = False
    for k in key:
        if k is None:
            items.append(("n", None, None))
            trivial = False
            continue
        if k is Ellipsis:
            if has_ellipsis:
                raise _XpuIndexUnsupported
            has_ellipsis = True
            items.append(("e", Ellipsis, None))
            continue
        if isinstance(k, slice):
            items.append(("s", k, None))
            consumed += 1
            if not (k.start is None and k.stop is None and k.step is None):
                trivial = False
            continue
        if isinstance(k, list):
            k = _xpu_list_index(k)
        elif type(k).__module__ == "numpy" and getattr(k, "ndim", 0) > 0:
            k = torch.as_tensor(k)
        if isinstance(k, torch.Tensor):
            if k.dtype in _XPU_MASK_DTYPES:
                if k.dim() == 0:
                    raise _XpuIndexUnsupported
                items.append(("m", None, k))
                consumed += k.dim()
                continue
            if k.dtype not in _XPU_INT_INDEX_DTYPES:
                raise _XpuIndexUnsupported  # the CPU round trip raises torch's own IndexError
            if k.dim() > 0:
                items.append(("t", None, k))
                consumed += 1
                continue
            k = int(k)
        elif isinstance(k, bool):
            raise _XpuIndexUnsupported
        elif type(k) is not int:
            try:
                k = operator.index(k)
            except TypeError:
                raise _XpuIndexUnsupported from None
        items.append(("i", k, None))
        consumed += 1
        trivial = False
    if consumed > x.dim():
        raise IndexError(f"too many indices for tensor of dimension {x.dim()}")
    basic = []
    adv = []
    pos = 0
    for kind, b, t in items:
        if kind == "t":
            basic.append(slice(None))
            adv.append((pos, t, False))
            pos += 1
        elif kind == "m":
            basic.extend([slice(None)] * t.dim())
            adv.append((pos, t, True))
            pos += t.dim()
        else:
            basic.append(b)
            if kind == "e":
                pos += x.dim() - consumed
            elif kind != "i":
                pos += 1
    return (x if trivial else x[tuple(basic)]), adv


def _xpu_flat_nonzero(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Purpose: row-major positions of the True elements of a bool mask, as a 1-D int64 tensor on `device`."""
    if mask.device.type != "cpu" and mask.numel() <= _XPU_INDEX_HOST_MAX:
        mask = mask.cpu()  # small masks: host nonzero plus one upload beats the device nonzero
    lin = mask.reshape(-1).nonzero().squeeze(-1)
    return lin if lin.device == device else lin.to(device)


def _xpu_mask_shape_check(mask: torch.Tensor, v: torch.Tensor, d: int) -> None:
    if tuple(mask.shape) != tuple(v.shape[d:d + mask.dim()]):
        raise IndexError(
            f"The shape of the mask {list(mask.shape)} at index 0 does not match the shape of the "
            f"indexed tensor {list(v.shape)} at index {d}"
        )


def _xpu_nonzero(mask: torch.Tensor, v: torch.Tensor, d: int, device: torch.device, flat: bool) -> Any:
    """Purpose: indices of the True elements of a mask over dims d.. of `v`, on `device`.

    flat=True gives one 1-D row-major index over the masked dims, else one 1-D index per dim.
    """
    _xpu_mask_shape_check(mask, v, d)
    if flat:
        return _xpu_flat_nonzero(mask if mask.dtype == torch.bool else mask.bool(), device)
    if mask.device.type != "cpu" and mask.numel() <= _XPU_INDEX_HOST_MAX:
        mask = mask.cpu()  # small masks: host nonzero plus one upload beats the device nonzero
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.device.type == "cpu":
        return mask.nonzero().t().to(device).unbind(0)
    return (mask if mask.device == device else mask.to(device)).nonzero(as_tuple=True)


def _xpu_plan(v: torch.Tensor, adv: list, device: torch.device, for_read: bool) -> tuple:
    """Purpose: turn tensor indices into one flat int64 index over adjacent dims of `v`.

    Returns (v, d0, k, lin, bshape): dims d0..d0+k-1 of the returned `v` (permuted when the indexed
    dims are separated, as torch moves them to the front) are addressed by the 1-D index `lin` in
    row-major order, and replace by `bshape` in the result. As in torch, a non-empty index into a
    dim of size 0 is an IndexError, and otherwise a result without elements is not range-checked
    (`lin` is then empty).
    """
    if len(adv) == 1:
        d, t, is_mask = adv[0]
        if is_mask:
            lin = _xpu_nonzero(t, v, d, device, True)
            return v, d, t.dim(), lin, (lin.shape[0],)
        if v.numel() == 0 and t.numel():
            if v.shape[d] == 0:
                raise IndexError("index is out of bounds for dimension with size 0")
            return v, d, 1, torch.empty(0, dtype=torch.int64, device=device), tuple(t.shape)
        lin = _xpu_norm_indices([t], [v.shape[d]], [d], device)[0]
        return v, d, 1, (lin if lin.dim() == 1 else lin.reshape(-1)), tuple(t.shape)
    dims: list[int] = []
    idxs: list[Any] = []
    raw: list[int] = []  # positions in idxs that still need the range check
    for d, t, is_mask in adv:
        if is_mask:
            for j, part in enumerate(_xpu_nonzero(t, v, d, device, False)):
                dims.append(d + j)
                idxs.append(part)
        else:
            raw.append(len(idxs))
            dims.append(d)
            idxs.append(t)
    bshape = tuple(idxs[0].shape)
    for t in idxs:
        if tuple(t.shape) != bshape:
            try:
                bshape = tuple(torch.broadcast_shapes(*[i.shape for i in idxs]))
            except RuntimeError:
                shapes = ", ".join(str(list(i.shape)) for i in idxs)
                raise IndexError(
                    f"shape mismatch: indexing tensors could not be broadcast together with shapes {shapes}"
                ) from None
            break
    k = len(dims)
    empty = 0 in bshape
    if not empty and v.numel() == 0:
        if any(v.shape[d] == 0 for d in dims):
            raise IndexError("index is out of bounds for dimension with size 0")
        empty = True  # another dim has size 0
    if not empty:
        checked = _xpu_norm_indices(
            [idxs[j] for j in raw], [v.shape[dims[j]] for j in raw], [dims[j] for j in raw], device
        )
        for j, t in zip(raw, checked):
            idxs[j] = t
    if dims[-1] - dims[0] != k - 1:  # separated advanced indices: result dims go first
        v = v.permute(dims + [d for d in range(v.dim()) if d not in dims])
        d0 = 0
    else:
        d0 = dims[0]
    if empty:
        return v, d0, k, torch.empty(0, dtype=torch.int64, device=device), bshape
    sizes = tuple(v.shape[d0:d0 + k])
    lin = None
    stride = 1
    for j in range(k - 1, -1, -1):
        if for_read and sizes[j] > 1 and v.stride(d0 + j) == 0:
            # expanded dim: every index reads the same element, so do not materialise the expansion
            v = v.narrow(d0 + j, 0, 1)
            continue
        if lin is None:
            lin = idxs[j] if stride == 1 else idxs[j] * stride
        else:
            lin = torch.add(lin, idxs[j], alpha=stride)  # one kernel per extra index tensor
        stride *= sizes[j]
    if lin is None:
        lin = torch.zeros(bshape, dtype=torch.int64, device=device)
    elif tuple(lin.shape) != bshape:
        lin = lin.expand(bshape)
    return v, d0, k, lin.reshape(-1), bshape


def _xpu_index_read(x: torch.Tensor, key: Any) -> torch.Tensor:
    v, adv = _xpu_parse_key(x, key)
    if not adv:
        return v
    v, d0, k, lin, bshape = _xpu_plan(v, adv, x.device, True)
    shape = tuple(v.shape)
    if k != 1:
        v = v.reshape(shape[:d0] + (math.prod(shape[d0:d0 + k]),) + shape[d0 + k:])
    out = v.index_select(d0, lin)
    if len(bshape) != 1 or lin.shape[0] != bshape[0]:  # the second case: empty result, unchecked index
        out = out.reshape(shape[:d0] + bshape + shape[d0 + k:])
    return out


def _xpu_index_write(
    x: torch.Tensor, v: torch.Tensor, adv: list, value: Any, accumulate: bool, strip_ones: bool
) -> None:
    """Purpose: in-place `v[adv] = value` (or `+=` when accumulate) through index_copy_/index_add_/index_fill_."""
    scalar = not isinstance(value, torch.Tensor)
    if scalar and type(value) not in (int, float, bool):
        raise _XpuIndexUnsupported
    if accumulate and (scalar or x.dtype == torch.bool):
        raise _XpuIndexUnsupported
    if scalar and v.numel() == 0 and len(adv) == 1 and not adv[0][2]:
        d, t, _ = adv[0]
        if v.shape[d] == 0 and t.dtype == torch.int64 and (0 in v.shape[:d] or 0 in v.shape[d + 1:]):
            # torch quirk: one int64 index tensor + Python scalar into a dim of size 0 is a silent no-op
            # when another dim is empty too (tensor values, int32 and several index tensors: IndexError)
            adv = [(d, t.new_empty(0), False)]
    if len(adv) == 1 and adv[0][2] and not accumulate and adv[0][1].dtype == torch.bool and (
        scalar or value.numel() == 1
    ):
        # torch's own shortcut: one bool mask and a one-element value is a masked_fill_ (the value is
        # cast to the target dtype and its shape ignored). One device kernel, no nonzero, no transfer.
        d, mask, _ = adv[0]
        _xpu_mask_shape_check(mask, v, d)
        if not scalar and value.requires_grad:
            # masked_fill_'s backward for a tensor value calls masked_select: keep the value on the
            # index_copy_ path below, as a 0-dim tensor of the target dtype
            value = value.to(x.dtype).reshape(())
        else:
            if not scalar:
                value = value.reshape(())
                if (value if value._base is None else value._base) is (x if x._base is None else x._base):
                    value = value.clone()  # the value aliases the target
            if mask.device != x.device:
                mask = mask.to(x.device)
            tail = v.dim() - d - mask.dim()
            v.masked_fill_(mask.reshape(mask.shape + (1,) * tail) if tail else mask, value)
            return
    if not scalar and value.dtype != x.dtype:
        raise RuntimeError(
            f"Index put requires the source and destination dtypes match, got {x.dtype} for the "
            f"destination and {value.dtype} for the source."
        )
    v, d0, k, lin, bshape = _xpu_plan(v, adv, x.device, False)
    shape = tuple(v.shape)
    head, tail = shape[:d0], shape[d0 + k:]
    vals = None
    if not scalar:
        if value.device != x.device:
            value = value.to(x.device)
        tshape = head + bshape + tail
        if strip_ones and value.dim() > len(tshape):
            lead = 0
            while lead < value.dim() and value.shape[lead] == 1:
                lead += 1
            value = value.reshape(value.shape[lead:])
        try:
            vals = value.expand(tshape)
        except RuntimeError:
            raise RuntimeError(
                f"shape mismatch: value tensor of shape {list(value.shape)} cannot be broadcast to "
                f"indexing result of shape {list(tshape)}"
            ) from None
        vals = vals.reshape(head + (lin.shape[0],) + tail)
        vbase = value if value._base is None else value._base
        if vbase is (x if x._base is None else x._base):
            vals = vals.clone()  # the value aliases the target
    target = v
    direct = True
    if k != 1:
        fshape = head + (math.prod(shape[d0:d0 + k]),) + tail
        try:
            target = v.view(fshape)
        except RuntimeError:  # strides do not allow a view: edit a copy, then copy it back
            target = v.reshape(fshape)
            direct = False
    if scalar:
        try:
            target.index_fill_(d0, lin, value)
        except RuntimeError:
            # torch quirk: with two or more index tensors an overflowing scalar becomes inf in a
            # float16 target (every other form and dtype raises, as index_fill_ does)
            if len(adv) < 2 or x.dtype != torch.float16 or type(value) is bool or value != value:
                raise
            target.index_fill_(d0, lin, math.copysign(math.inf, value))
    elif accumulate:
        target.index_add_(d0, lin, vals)
    else:
        target.index_copy_(d0, lin, vals)
    if not direct:
        v.copy_(target.reshape(shape))


def _xpu_getitem(x: torch.Tensor, key: Any) -> torch.Tensor:
    try:
        return _xpu_index_read(x, key)
    except _XpuIndexUnsupported:
        pass
    out = x.cpu()[_xpu_to_cpu(key)]  # raises torch's own error for an invalid key
    _xpu_fallback_warn(f"getitem with key {_xpu_describe(key)}")
    return out.to(x.device)


def _xpu_setitem(x: torch.Tensor, key: Any, value: Any) -> None:
    try:
        v, adv = _xpu_parse_key(x, key)
        if not adv:
            v[...] = value
        else:
            _xpu_index_write(x, v, adv, value, False, True)
        return
    except _XpuIndexUnsupported:
        pass
    xc = x.cpu()
    xc[_xpu_to_cpu(key)] = _xpu_to_cpu(value)
    _xpu_fallback_warn(f"setitem with key {_xpu_describe(key)} and value {_xpu_describe(value)}")
    x.copy_(xc)


def _xpu_index_put(x: torch.Tensor, indices: Any, values: Any, accumulate: Any, inplace: bool) -> torch.Tensor:
    try:
        if not isinstance(values, torch.Tensor) or not isinstance(indices, (tuple, list)) or not indices:
            raise _XpuIndexUnsupported
        adv = []
        pos = 0
        for t in indices:
            if not isinstance(t, torch.Tensor):
                raise _XpuIndexUnsupported
            if t.dtype in _XPU_MASK_DTYPES and t.dim() > 0:
                adv.append((pos, t, True))
                pos += t.dim()
            elif t.dtype in _XPU_INT_INDEX_DTYPES:
                adv.append((pos, t, False))
                pos += 1
            else:
                raise _XpuIndexUnsupported
        if pos > x.dim():
            raise _XpuIndexUnsupported
        target = x if inplace else x.clone()
        _xpu_index_write(target, target, adv, values, bool(accumulate), False)
        return target
    except _XpuIndexUnsupported:
        pass
    out = x.cpu().index_put(_xpu_to_cpu(indices), _xpu_to_cpu(values), accumulate)
    _xpu_fallback_warn(
        f"index_put(accumulate={accumulate}) with indices {_xpu_describe(indices)} and values {_xpu_describe(values)}"
    )
    return x.copy_(out) if inplace else out.to(x.device)


def _xpu_masked_select(x: torch.Tensor, mask: Any) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        return x.cpu().masked_select(_xpu_to_cpu(mask)).to(x.device)  # always raises torch's own error
    xb, mb = torch.broadcast_tensors(x, mask)
    return xb.reshape(-1).index_select(0, _xpu_flat_nonzero(mb, x.device))


def _xpu_masked_scatter(x: torch.Tensor, mask: Any, source: Any, inplace: bool) -> torch.Tensor:
    """Purpose: masked_scatter through index_copy, whose backward does not call masked_select."""
    ok = isinstance(mask, torch.Tensor) and isinstance(source, torch.Tensor)
    if ok and mask.dtype == torch.bool and source.dtype == x.dtype:
        try:
            xb, mb = (x, mask.expand(x.shape)) if inplace else torch.broadcast_tensors(x, mask)
        except RuntimeError:
            ok = False
        if ok:
            lin = _xpu_flat_nonzero(mb, x.device)
            if source.numel() < lin.shape[0]:
                raise RuntimeError("masked_scatter: number of elements of source < number of ones in mask")
            src = source.reshape(-1).narrow(0, 0, lin.shape[0])
            if src.device != x.device:
                src = src.to(x.device)
            if inplace and x.is_contiguous():
                if (source if source._base is None else source._base) is (x if x._base is None else x._base):
                    src = src.clone()  # the source aliases the target
                x.view(-1).index_copy_(0, lin, src)  # in place: no full-size temporary
                return x
            out = xb.reshape(-1).index_copy(0, lin, src).reshape(xb.shape)
            return x.copy_(out) if inplace else out
    out = x.cpu().masked_scatter(_xpu_to_cpu(mask), _xpu_to_cpu(source))  # raises torch's own error
    _xpu_fallback_warn(f"masked_scatter with mask {_xpu_describe(mask)} and source {_xpu_describe(source)}")
    return x.copy_(out) if inplace else out.to(x.device)


def _xpu_combinations(x: torch.Tensor, r: int, with_replacement: bool) -> torch.Tensor:
    """Purpose: torch.combinations calls aten::index from C++ (access violation on XPU): run it on the CPU."""
    out = _XPU_INDEX_ORIG["torch.combinations"](x.cpu(), r, with_replacement)
    _xpu_fallback_warn("torch.combinations")
    return out.to(x.device)


def _xpu_make_wrappers(orig: dict[str, Any]) -> dict[str, Any]:
    """Purpose: build the wrappers; every one starts with the not-on-XPU fast path."""
    Tensor = torch.Tensor
    o_get, o_set = orig["Tensor.__getitem__"], orig["Tensor.__setitem__"]
    o_put, o_put_, o_tput = orig["Tensor.index_put"], orig["Tensor.index_put_"], orig["torch.index_put"]
    o_tput_, o_comb = orig["torch.index_put_"], orig["torch.combinations"]
    o_msel, o_tmsel = orig["Tensor.masked_select"], orig["torch.masked_select"]
    o_msc, o_msc_, o_tmsc = orig["Tensor.masked_scatter"], orig["Tensor.masked_scatter_"], orig["torch.masked_scatter"]

    def __getitem__(self, key):
        if self.device.type != "xpu":
            return o_get(self, key)
        tk = type(key)
        if tk is int or tk is slice or key is None or key is Ellipsis:
            return o_get(self, key)
        if tk is tuple:
            for k in key:
                tk = type(k)
                if not (tk is int or tk is slice or k is None or k is Ellipsis):
                    break
            else:
                return o_get(self, key)
        return _xpu_getitem(self, key)

    def __setitem__(self, key, value):
        if self.device.type != "xpu":
            return o_set(self, key, value)
        tk = type(key)
        if tk is int or tk is slice or key is None or key is Ellipsis:
            return o_set(self, key, value)
        if tk is tuple:
            for k in key:
                tk = type(k)
                if not (tk is int or tk is slice or k is None or k is Ellipsis):
                    break
            else:
                return o_set(self, key, value)
        return _xpu_setitem(self, key, value)

    def index_put(self, indices, values, accumulate=False):
        if self.device.type != "xpu":
            return o_put(self, indices, values, accumulate)
        return _xpu_index_put(self, indices, values, accumulate, False)

    def index_put_(self, indices, values, accumulate=False):
        if self.device.type != "xpu":
            return o_put_(self, indices, values, accumulate)
        return _xpu_index_put(self, indices, values, accumulate, True)

    def torch_index_put(input, indices, values, accumulate=False):
        if not isinstance(input, Tensor) or input.device.type != "xpu":
            return o_tput(input, indices, values, accumulate)
        return _xpu_index_put(input, indices, values, accumulate, False)

    def torch_index_put_(input, indices, values, accumulate=False):
        if not isinstance(input, Tensor) or input.device.type != "xpu":
            return o_tput_(input, indices, values, accumulate)
        return _xpu_index_put(input, indices, values, accumulate, True)

    def torch_combinations(input, r=2, with_replacement=False):
        if not isinstance(input, Tensor) or input.device.type != "xpu":
            return o_comb(input, r, with_replacement)
        return _xpu_combinations(input, r, with_replacement)

    def masked_select(self, mask):
        if self.device.type != "xpu":
            return o_msel(self, mask)
        return _xpu_masked_select(self, mask)

    def torch_masked_select(input, mask, *, out=None):
        if not isinstance(input, Tensor) or input.device.type != "xpu":
            return o_tmsel(input, mask) if out is None else o_tmsel(input, mask, out=out)
        res = _xpu_masked_select(input, mask)
        return res if out is None else out.resize_(res.shape).copy_(res)

    def masked_scatter(self, mask, source):
        if self.device.type != "xpu":
            return o_msc(self, mask, source)
        return _xpu_masked_scatter(self, mask, source, False)

    def masked_scatter_(self, mask, source):
        if self.device.type != "xpu":
            return o_msc_(self, mask, source)
        return _xpu_masked_scatter(self, mask, source, True)

    def torch_masked_scatter(input, mask, source):
        if not isinstance(input, Tensor) or input.device.type != "xpu":
            return o_tmsc(input, mask, source)
        return _xpu_masked_scatter(input, mask, source, False)

    wrappers = {
        "Tensor.__getitem__": __getitem__, "Tensor.__setitem__": __setitem__,
        "Tensor.index_put": index_put, "Tensor.index_put_": index_put_, "torch.index_put": torch_index_put,
        "torch.index_put_": torch_index_put_, "torch.combinations": torch_combinations,
        "Tensor.masked_select": masked_select, "torch.masked_select": torch_masked_select,
        "Tensor.masked_scatter": masked_scatter, "Tensor.masked_scatter_": masked_scatter_,
        "torch.masked_scatter": torch_masked_scatter,
    }
    for name, fn in wrappers.items():
        # pickle (by reference) and reprs see the public torch name, as for the originals
        fn.__module__ = "torch"
        fn.__name__ = name.split(".", 1)[1]
        fn.__qualname__ = name if name.startswith("Tensor.") else fn.__name__
    return wrappers


def install_xpu_index_workaround() -> bool:
    """Purpose: work around torch 2.15.0+xpu advanced indexing failures on Intel Arc (idempotent).

    Replaces `Tensor.__getitem__`, `Tensor.__setitem__`, `index_put`, `index_put_`, `masked_select`
    and `masked_scatter(_)` (Tensor methods and `torch.*` functions) with wrappers that rewrite
    advanced indexing on XPU tensors into `index_select` / `index_copy_` / `index_add_` /
    `index_fill_` / `masked_fill_`: single and multiple index tensors at any position, masks of any
    rank, lists, CPU-resident indices, `accumulate=True`. Forms that are not recognised run through
    a CPU round trip with a one-time warning, so nothing reaches the broken kernels; so does
    `torch.combinations`, which calls `aten::index` from C++. Tensors on other devices and basic
    indexing go straight to the original function; no other op is touched.

    The wrappers are process-global (all threads). Call it only for runs that use XPU
    (`resolve_runtime_device` does). Environment:
    - `NNKNN_XPU_INDEX_BOUNDS=0` skips the range check of index tensors that live on XPU (the check
      copies the index to the host once per indexing call; without it an out-of-range index wraps
      silently). Index tensors on the CPU, lists and masks are always checked.
    - `NNKNN_XPU_INDEX_TRIPWIRE=1` also enables `enable_xpu_index_tripwire()` on the installing thread.
    Limits: `x[idx] = v` with duplicate indices keeps one of the values, not necessarily the last
    (undefined in torch too); `torch.func.vmap` over a batched index tensor raises (the range check
    reads the index); TorchScript and other C++ callers of `aten::index` are not covered (use the
    tripwire to find them). Returns True when the wrappers are active.
    """
    global _XPU_INDEX_CHECK_BOUNDS
    with _XPU_INDEX_LOCK:
        if _XPU_INDEX_ORIG:
            return True
        if not is_xpu_available():
            return False
        _XPU_INDEX_CHECK_BOUNDS = os.environ.get("NNKNN_XPU_INDEX_BOUNDS", "1") != "0"
        orig: dict[str, Any] = {}
        for name in _XPU_INDEX_TENSOR_ATTRS:
            orig["Tensor." + name] = getattr(torch.Tensor, name)
            orig["own." + name] = name in torch.Tensor.__dict__
        for name in _XPU_INDEX_TORCH_ATTRS:
            orig["torch." + name] = getattr(torch, name)
        wrappers = _xpu_make_wrappers(orig)
        for name in _XPU_INDEX_TENSOR_ATTRS:
            setattr(torch.Tensor, name, wrappers["Tensor." + name])
        for name in _XPU_INDEX_TORCH_ATTRS:
            setattr(torch, name, wrappers["torch." + name])
        _XPU_INDEX_ORIG.update(orig)
        if os.environ.get("NNKNN_XPU_INDEX_TRIPWIRE") == "1":
            enable_xpu_index_tripwire()
        return True


def uninstall_xpu_index_workaround() -> None:
    """Purpose: restore the original torch indexing functions and switch the tripwire off (for tests)."""
    with _XPU_INDEX_LOCK:
        disable_xpu_index_tripwire()
        if not _XPU_INDEX_ORIG:
            return
        for name in _XPU_INDEX_TENSOR_ATTRS:
            if _XPU_INDEX_ORIG["own." + name]:
                setattr(torch.Tensor, name, _XPU_INDEX_ORIG["Tensor." + name])
            else:
                delattr(torch.Tensor, name)
        for name in _XPU_INDEX_TORCH_ATTRS:
            setattr(torch, name, _XPU_INDEX_ORIG["torch." + name])
        _XPU_INDEX_ORIG.clear()


def xpu_index_workaround_installed() -> bool:
    """Purpose: report whether the XPU indexing wrappers are active in this process."""
    return bool(_XPU_INDEX_ORIG)


def _xpu_tripwire_ops() -> frozenset:
    aten = torch.ops.aten
    names = (
        ("index", "Tensor"), ("index", "Tensor_out"), ("_unsafe_index", "Tensor"),
        ("index_put", "default"), ("index_put", "out"), ("index_put_", "default"),
        ("_index_put_impl_", "default"), ("_unsafe_index_put", "default"),
        ("masked_select", "default"), ("masked_select", "out"), ("masked_scatter_backward", "default"),
    )
    broken = set()
    for packet, overload in names:
        op = getattr(getattr(aten, packet, None), overload, None)
        if op is not None:
            broken.add(op)
    return frozenset(broken)


def enable_xpu_index_tripwire() -> bool:
    """Purpose: debug aid that raises when a broken index kernel is about to run on an XPU tensor.

    Enters a TorchDispatchMode on the calling thread, which raises RuntimeError before `aten::index`,
    `index_put(_)`, `masked_select`, their `out=` / `_unsafe_` variants or `masked_scatter_backward`
    runs on an XPU tensor. It catches what the wrappers cannot see: C++ callers inside torch and
    backward formulas. Dispatch modes are per thread: backward passes started from a covered thread
    are covered, any other thread has to call this function itself (it returns True only when the
    calling thread is covered). It adds a Python callback to every op, so it is off unless
    `NNKNN_XPU_INDEX_TRIPWIRE=1` or this function is called.
    """
    global _XPU_INDEX_TRIPWIRE
    with _XPU_INDEX_LOCK:
        if getattr(_XPU_INDEX_TRIPWIRE_LOCAL, "mode", None) is not None:
            if _XPU_INDEX_TRIPWIRE is None:
                _XPU_INDEX_TRIPWIRE = _xpu_tripwire_ops()  # re-arm the mode this thread never left
            return True
        if not is_xpu_available():
            return False
        from torch.utils._python_dispatch import TorchDispatchMode

        class XpuIndexTripwire(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                broken = _XPU_INDEX_TRIPWIRE  # None once disabled from any thread: pass through
                if (
                    broken is not None and func in broken
                    and args and isinstance(args[0], torch.Tensor) and args[0].device.type == "xpu"
                ):
                    raise RuntimeError(
                        f"XPU index tripwire: {func} reached an XPU tensor of shape {tuple(args[0].shape)}; "
                        "this kernel is broken in torch 2.15.0+xpu, rewrite the caller (index_select / "
                        "index_copy / gather / scatter) or move it to the CPU."
                    )
                return func(*args, **(kwargs or {}))

        if _XPU_INDEX_TRIPWIRE is None:
            _XPU_INDEX_TRIPWIRE = _xpu_tripwire_ops()
        mode = XpuIndexTripwire()
        mode.__enter__()
        _XPU_INDEX_TRIPWIRE_LOCAL.mode = mode
        return True


def disable_xpu_index_tripwire() -> None:
    """Purpose: switch the tripwire off for every thread; safe to call from any thread.

    The calling thread also leaves its dispatch mode. A mode entered by another thread cannot be
    removed from here: it stays on that thread as a pass-through until that thread calls this
    function (or `enable_xpu_index_tripwire()` again, which re-arms it).
    """
    global _XPU_INDEX_TRIPWIRE
    with _XPU_INDEX_LOCK:
        _XPU_INDEX_TRIPWIRE = None
        mode = getattr(_XPU_INDEX_TRIPWIRE_LOCAL, "mode", None)
        if mode is not None:
            _XPU_INDEX_TRIPWIRE_LOCAL.mode = None
            mode.__exit__(None, None, None)


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
