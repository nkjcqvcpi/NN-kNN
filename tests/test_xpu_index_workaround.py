"""CPU-vs-XPU equivalence of the XPU advanced-indexing workaround (model.device_utils).

Every XPU test runs with the debug tripwire on, so a rewrite that still reached `aten::index`,
`index_put`, `masked_select` or `masked_scatter_backward` on the device (forward or backward)
fails here instead of losing the device.

Tests named `test_cpu_*` also run on machines without XPU: they call the rewrite functions
directly on CPU tensors and compare them with native torch indexing.
"""
import math
import os
import pickle
import subprocess
import sys
import threading
import warnings
from pathlib import Path

import pytest
import torch

from model import device_utils as du
from model.device_utils import install_xpu_index_workaround, is_xpu_available

pytestmark = [
    pytest.mark.filterwarnings("ignore:indexing with dtype torch.uint8:UserWarning"),
]
DEV = "xpu"
HAS_XPU = is_xpu_available()


@pytest.fixture(autouse=True)
def xpu_index_patch(request):
    if not HAS_XPU:
        if not request.node.name.startswith("test_cpu_"):
            pytest.skip("no XPU device")
        yield
        return
    was_installed = du.xpu_index_workaround_installed()
    assert install_xpu_index_workaround()
    assert du.enable_xpu_index_tripwire()
    yield
    du.disable_xpu_index_tripwire()
    if was_installed:
        assert install_xpu_index_workaround()
    else:
        du.uninstall_xpu_index_workaround()


def T(data, dtype=torch.int64):
    return torch.tensor(data, dtype=dtype)


def B(data):
    return torch.tensor(data, dtype=torch.bool)


def to_dev(obj):
    if isinstance(obj, torch.Tensor):
        return obj.to(DEV)
    if isinstance(obj, tuple):
        return tuple(to_dev(o) for o in obj)
    return obj


def base(shape, dtype=torch.float32):
    n = math.prod(shape)
    return (torch.arange(n, dtype=torch.float32) * 0.5 - 3.0).reshape(shape).to(dtype).clone()


def mask_of(shape, mod=3):
    return (torch.arange(math.prod(shape)) % mod == 1).reshape(shape)


S_ = slice(None)
# (id, shape of x, key). Shapes: (6, 4) and (5, 6, 4).
READ_CASES = [
    ("idx1d_dups", (6, 4), T([3, 0, 5, 0])),
    ("idx1d_on_1d", (7,), T([6, 0, 0, 2])),
    ("idx_negative", (6, 4), T([-1, 0, -6])),
    ("idx_int32", (6, 4), T([1, 2], torch.int32)),
    ("idx2d", (6, 4), T([[0, 1], [5, 2], [3, 3]])),
    ("idx3d", (6, 4), T([[[0], [1]], [[5], [2]]])),
    ("idx_list", (6, 4), [0, 5, 0]),
    ("idx_list_bool", (6, 4), [True, False, True, False, False, True]),
    ("idx_empty", (6, 4), T([])),
    ("idx_empty_list", (6, 4), []),
    ("idx_0dim_is_select", (6, 4), T(2)),
    ("slice_idx", (6, 4), (S_, T([3, 0]))),
    ("slice_idx_negative", (6, 4), (S_, T([-1, -4]))),
    ("slice_list", (6, 4), (S_, [3, 0])),
    ("stepslice_idx2d", (5, 6, 4), (slice(1, 4), T([[0, 5], [2, 2]]))),
    ("idx_stepslice", (5, 6, 4), (T([0, 4]), slice(None, None, 2))),
    ("ellipsis_idx", (5, 6, 4), (Ellipsis, T([0, 3]))),
    ("idx_ellipsis", (5, 6, 4), (T([0, 3]), Ellipsis)),
    ("none_idx", (6, 4), (None, T([0, 3]))),
    ("idx_none", (6, 4), (T([0, 3]), None)),
    ("int_idx", (5, 6, 4), (2, T([0, 3]))),
    ("idx_int", (5, 6, 4), (T([0, 3]), 1)),
    ("idx_0dim_idx", (5, 6, 4), (T([0, 1]), T(1), T([3, 0]))),
    ("int_slice_idx", (5, 6, 4), (0, S_, T([1, 2, 3, 0, 0]))),
    ("two_paired", (6, 4), (T([0, 1, 2, 0]), T([3, 0, 1, 3]))),
    ("two_broadcast", (6, 4), (T([[0], [1], [4]]), T([[0, 1], [2, 3], [3, 3]]))),
    ("two_cover_all_dims_3d", (5, 6, 4), (T([0, 4]), T([5, 1]), T([3, 3]))),
    ("two_trailing_slice", (5, 6, 4), (T([[0], [4]]), T([[5, 1, 1], [0, 2, 3]]), S_)),
    ("two_leading_slice", (5, 6, 4), (S_, T([[0], [4]]), T([[3, 1, 1], [0, 2, 3]]))),
    ("two_negative", (5, 6, 4), (T([-1, -5]), T([-6, 5]))),
    ("two_int32_and_int64", (6, 4), (T([0, 5], torch.int32), T([3, 0]))),
    ("two_separated", (5, 6, 4), (T([0, 1, 4]), S_, T([0, 3, 3]))),
    ("two_separated_broadcast", (5, 6, 4), (T([[0], [1]]), slice(1, 5), T([0, 3, 3]))),
    ("two_separated_by_none", (6, 4), (T([0, 5]), None, T([1, 2]))),
    ("idx_list_pair", (6, 4), ([0, 5], [1, 2])),
    ("mask1d", (6, 4), mask_of((6,), 2)),
    ("mask1d_on_1d", (7,), mask_of((7,), 2)),
    ("mask2d_full_rank", (6, 4), mask_of((6, 4))),
    ("mask2d_leading_of_3d", (5, 6, 4), mask_of((5, 6))),
    ("mask3d_full_rank", (5, 6, 4), mask_of((5, 6, 4))),
    ("slice_mask", (6, 4), (S_, mask_of((4,), 2))),
    ("slice_mask2d", (5, 6, 4), (S_, mask_of((6, 4)))),
    ("int_mask", (5, 6, 4), (2, mask_of((6,), 2))),
    ("mask_int", (5, 6, 4), (mask_of((5,), 2), 2)),
    ("mask_0dim", (5, 6, 4), (mask_of((5,), 2), T(2))),
    ("mask_idx", (5, 6, 4), (B([True, False, True, False, True]), T([0, 1, 5]))),
    ("mask_slice_idx", (5, 6, 4), (B([True, False, True, False, True]), S_, T([0, 1, 3]))),
    ("mask_uint8", (6, 4), mask_of((6,), 2).to(torch.uint8)),
    ("mask_all_false", (6, 4), torch.zeros(6, dtype=torch.bool)),
    ("mask_all_true", (6, 4), torch.ones(6, 4, dtype=torch.bool)),
    ("zero_rows_empty_idx", (0, 4), T([])),
    ("zero_rows_mask", (0, 4), torch.zeros(0, dtype=torch.bool)),
    ("zero_cols_idx", (5, 0), T([0, 4])),
    ("zero_cols_mask2d", (5, 0), torch.zeros(5, 0, dtype=torch.bool)),
    ("zero_cols_two", (5, 0, 3), (T([0, 4]), S_, T([2, 0]))),
]


def _ids(cases):
    return [c[0] for c in cases]


@pytest.mark.parametrize("where", ["xpu_index", "cpu_index"])
@pytest.mark.parametrize("case", READ_CASES, ids=_ids(READ_CASES))
def test_getitem_matches_cpu(case, where):
    _, shape, key = case
    x = base(shape)
    xd = x.to(DEV)
    ref = x[key]
    out = xd[to_dev(key) if where == "xpu_index" else key]
    assert out.device.type == DEV and out.dtype == ref.dtype and out.shape == ref.shape
    assert torch.equal(out.cpu(), ref)
    # copy-vs-view: writing into the result must change the source exactly when it does on the CPU
    ref.add_(1.0)
    out.add_(1.0)
    assert torch.equal(xd.cpu(), x)


@pytest.mark.parametrize("case", READ_CASES, ids=_ids(READ_CASES))
def test_getitem_backward_matches_cpu(case):
    _, shape, key = case
    grads = []
    for device, k in (("cpu", key), (DEV, to_dev(key))):
        w = base(shape).to(device).requires_grad_(True)
        out = (w * 2.0)[k]
        coef = (torch.arange(out.numel(), dtype=torch.float32).reshape(out.shape) + 1.0).to(device)
        (out * coef).sum().backward()
        grads.append(w.grad.cpu())
    assert torch.equal(grads[0], grads[1])


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.float16])
def test_getitem_other_dtypes(dtype):
    x = base((6, 4), dtype) if dtype != torch.bool else mask_of((6, 4))
    xd = x.to(DEV)
    for key in (T([5, 0, 0]), (T([[0], [5]]), T([[1, 2], [3, 3]])), mask_of((6,), 2), (S_, T([3, 1]))):
        out = xd[to_dev(key)]
        assert out.dtype == dtype and torch.equal(out.cpu(), x[key])


def test_getitem_non_contiguous_and_expanded_sources():
    x = base((6, 5, 4))
    xd = x.to(DEV)
    views = [
        lambda t: t.transpose(0, 2),
        lambda t: t[::2, 1:],
        lambda t: t.permute(1, 2, 0)[:, 1:3],
        lambda t: t[:1].expand(7, 5, 4),
        lambda t: t[:, :1, :1].expand(6, 5, 3),
    ]
    for view in views:
        v, vd = view(x), view(xd)
        n0, n1, n2 = v.shape
        keys = [
            T([n0 - 1, 0, 0]),
            (S_, T([0, n1 - 1])),
            (T([[0], [n0 - 1]]), T([[0, 1], [n1 - 1, 0]])),
            (T([0, n0 - 1]), S_, T([n2 - 1, 0])),
            (S_, T([0, n1 - 1]), T([n2 - 1, 0])),
            mask_of((n0,), 2),
            mask_of((n0, n1)),
            mask_of((n0, n1, n2)),
        ]
        for key in keys:
            out = vd[to_dev(key)]
            assert out.shape == v[key].shape and torch.equal(out.cpu(), v[key])


def test_expanded_source_backward():
    b, i = T([[0], [1], [2]]), T([[0, 3], [3, 3], [1, 2]])
    grads = []
    for device in ("cpu", DEV):
        w = base((4, 2)).to(device).requires_grad_(True)
        out = (w * 3.0).unsqueeze(0).expand(3, 4, 2)[b.to(device), i.to(device)]
        (out * out).sum().backward()
        grads.append(w.grad.cpu())
    assert torch.equal(grads[0], grads[1])


# (id, shape, key); written positions are unique, so the result does not depend on write order
WRITE_CASES = [
    ("idx1d", (6, 4), T([3, 0, 5])),
    ("idx1d_on_1d", (7,), T([6, 0, 2])),
    ("idx_negative", (6, 4), T([-1, 0, -5])),
    ("idx_int32", (6, 4), T([1, 2], torch.int32)),
    ("idx2d", (6, 4), T([[0, 1], [5, 2]])),
    ("idx_list", (6, 4), [0, 5]),
    ("idx_empty", (6, 4), T([])),
    ("idx_0dim_is_select", (6, 4), T(2)),
    ("slice_idx", (6, 4), (S_, T([3, 0]))),
    ("stepslice_idx", (5, 6, 4), (slice(1, 4), T([0, 5, 2]))),
    ("idx_stepslice", (5, 6, 4), (T([0, 4]), slice(None, None, 2))),
    ("ellipsis_idx", (5, 6, 4), (Ellipsis, T([0, 3]))),
    ("int_idx", (5, 6, 4), (2, T([0, 3]))),
    ("idx_int", (5, 6, 4), (T([0, 3]), 1)),
    ("two_paired", (6, 4), (T([0, 1, 2, 5]), T([3, 0, 1, 3]))),
    ("two_broadcast", (6, 4), (T([[0], [1], [4]]), T([[0, 1], [2, 3], [3, 0]]))),
    ("two_trailing_slice", (5, 6, 4), (T([[0], [4]]), T([[5, 1, 2], [0, 2, 3]]), S_)),
    ("two_leading_slice", (5, 6, 4), (S_, T([[0], [4]]), T([[3, 1, 2], [0, 2, 3]]))),
    ("two_separated", (5, 6, 4), (T([0, 1, 4]), S_, T([0, 3, 2]))),
    ("two_separated_broadcast", (5, 6, 4), (T([[0], [1]]), slice(1, 5), T([0, 3, 2]))),
    ("three", (5, 6, 4), (T([0, 4]), T([5, 1]), T([3, 3]))),
    ("mask1d", (6, 4), mask_of((6,), 2)),
    ("mask2d_full_rank", (6, 4), mask_of((6, 4))),
    ("mask2d_leading_of_3d", (5, 6, 4), mask_of((5, 6))),
    ("slice_mask", (6, 4), (S_, mask_of((4,), 2))),
    ("int_mask", (5, 6, 4), (2, mask_of((6,), 2))),
    ("mask_idx", (5, 6, 4), (B([True, False, True, False, True]), T([0, 1, 5]))),
    ("mask_all_false", (6, 4), torch.zeros(6, dtype=torch.bool)),
    ("zero_rows_mask", (0, 4), torch.zeros(0, dtype=torch.bool)),
    ("zero_cols_idx", (5, 0), T([0, 4])),
]
VALUE_KINDS = ["scalar", "int_scalar", "full", "zero_dim", "last_dim", "leading_ones"]


def make_value(kind, result_shape):
    n = math.prod(result_shape)
    full = (torch.arange(n, dtype=torch.float32) + 100.0).reshape(result_shape)
    if kind == "scalar":
        return 2.5
    if kind == "int_scalar":
        return 7
    if kind == "full":
        return full
    if kind == "zero_dim":
        return torch.tensor(-9.0)
    if kind == "last_dim":
        return full.reshape(-1, result_shape[-1])[:1].reshape(result_shape[-1]).clone() if n else torch.tensor(1.0)
    return full[None, None]


@pytest.mark.parametrize("kind", VALUE_KINDS)
@pytest.mark.parametrize("case", WRITE_CASES, ids=_ids(WRITE_CASES))
def test_setitem_matches_cpu(case, kind):
    _, shape, key = case
    x = base(shape)
    xd = x.to(DEV)
    value = make_value(kind, tuple(x[key].shape))
    x[key] = value
    xd[to_dev(key)] = to_dev(value)
    assert torch.equal(xd.cpu(), x)
    y = base(shape).to(DEV)
    y[key] = value  # index and value both resident on the CPU
    assert torch.equal(y.cpu(), x)


@pytest.mark.parametrize("case", WRITE_CASES, ids=_ids(WRITE_CASES))
def test_setitem_backward_matches_cpu(case):
    _, shape, key = case
    grads = []
    for device, k in (("cpu", key), (DEV, to_dev(key))):
        w = base(shape).to(device).requires_grad_(True)
        y = w * 2.0
        result_shape = tuple(y[k].shape)
        v = make_value("full", result_shape).to(device).requires_grad_(True)
        y[k] = v * 3.0
        coef = (torch.arange(y.numel(), dtype=torch.float32).reshape(y.shape) + 1.0).to(device)
        (y * coef).sum().backward()
        grads.append((w.grad.cpu(), v.grad.cpu()))
    assert torch.equal(grads[0][0], grads[1][0]) and torch.equal(grads[0][1], grads[1][1])


def test_setitem_non_contiguous_targets():
    def views(t):
        return [t.transpose(0, 2), t[::2, 1:], t.permute(1, 2, 0)[:, 1:3], t[1:5, :, 2]]

    for n in range(4):
        probe = views(base((6, 5, 4)))[n]
        dims = probe.shape
        keys = [
            T([dims[0] - 1, 0]),
            (S_, T([0, dims[1] - 1])),
            (T([[0], [dims[0] - 1]]), T([[0, 1], [dims[1] - 1, 0]])),
            mask_of((dims[0],), 2),
            mask_of(tuple(dims[:2])),
            mask_of(tuple(dims)),
        ]
        if len(dims) == 3:
            keys += [(T([0, dims[0] - 1]), S_, T([dims[2] - 1, 0])), (S_, T([0, dims[1] - 1]), T([dims[2] - 1, 0]))]
        for key in keys:
            for kind in ("scalar", "full"):
                x = base((6, 5, 4))
                xd = x.to(DEV)
                v, vd = views(x)[n], views(xd)[n]
                value = make_value(kind, tuple(v[key].shape))
                v[key] = value
                vd[to_dev(key)] = to_dev(value)
                assert torch.equal(xd.cpu(), x), (n, key, kind)


def test_setitem_non_contiguous_backward():
    key = (T([[0], [3]]), T([[0, 1], [2, 0]]))
    grads = []
    for device in ("cpu", DEV):
        w = base((4, 3, 5)).to(device).requires_grad_(True)
        v = base((2, 2, 4)).to(device).requires_grad_(True)
        y = w * 2.0
        y.transpose(0, 2)[to_dev(key) if device == DEV else key] = v * 3.0
        (y * y).sum().backward()
        grads.append((w.grad.cpu(), v.grad.cpu()))
    assert torch.equal(grads[0][0], grads[1][0]) and torch.equal(grads[0][1], grads[1][1])


def test_setitem_other_dtypes_and_project_forms():
    # bool target, two index tensors, python scalar (match_mask[arange, exclude] = False)
    m = torch.ones(5, 6, dtype=torch.bool)
    md = m.to(DEV)
    rows, cols = torch.arange(5), T([0, 5, 2, 2, 1])
    m[rows, cols] = False
    md[rows.to(DEV), cols.to(DEV)] = False
    assert torch.equal(md.cpu(), m)
    m[:, T([3])] = False
    md[:, T([3]).to(DEV)] = False
    assert torch.equal(md.cpu(), m)
    m[T([0, 1])] = torch.ones(2, 6, dtype=torch.bool)
    md[T([0, 1])] = torch.ones(2, 6, dtype=torch.bool, device=DEV)
    assert torch.equal(md.cpu(), m)
    # int64 target, float python scalar is truncated like on the CPU
    a = torch.arange(12).reshape(3, 4)
    ad = a.to(DEV)
    a[T([0, 2])] = 1.7
    ad[T([0, 2]).to(DEV)] = 1.7
    a[T([1]), T([1])] = T([99])
    ad[T([1]), T([1])] = T([99]).to(DEV)
    assert ad.dtype == torch.int64 and torch.equal(ad.cpu(), a)
    # b[mask] = b[mask] * c, d[mask2d] = inf, and a value that is a view of the target
    b = base((8,))
    bd = b.to(DEV)
    mask = b > 0
    b[mask] = b[mask] * 0.5
    bd[mask.to(DEV)] = bd[mask.to(DEV)] * 0.5
    assert torch.equal(bd.cpu(), b)
    d = base((6, 4))
    dd = d.to(DEV)
    d[d > 0] = float("inf")
    dd[dd > 0] = float("inf")
    assert torch.equal(dd.cpu(), d)
    e = base((6, 4))
    ed = e.to(DEV)
    e[T([3, 4, 5])] = e[:3].clone()
    ed[T([3, 4, 5]).to(DEV)] = ed[:3]
    assert torch.equal(ed.cpu(), e)


PUT_CASES = [
    ("one", (6, 4), (T([3, 0, 5, 3]),), (4, 4)),
    ("one_2d", (6, 4), (T([[3, 0], [5, 3]]),), (2, 2, 4)),
    ("one_0dim", (6, 4), (T(2),), (4,)),
    ("two", (6, 4), (T([0, 1, 2, 0]), T([3, 0, 1, 3])), (4,)),
    ("two_broadcast", (5, 6, 4), (T([[0], [4]]), T([[5, 1, 5], [0, 2, 0]])), (2, 3, 4)),
    ("three", (5, 6, 4), (T([0, 4, 0]), T([5, 1, 5]), T([3, 3, 3])), (3,)),
    ("mask", (6, 4), (mask_of((6,), 2),), (3, 4)),
    ("mask2d", (6, 4), (mask_of((6, 4)),), (8,)),
    ("mask_and_idx", (6, 4), (mask_of((6,), 2), T([0, 3, 0])), (3,)),
]


def _unique_positions(indices):
    return tuple(i.clone() for i in indices)


@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("case", PUT_CASES, ids=_ids(PUT_CASES))
def test_index_put_matches_cpu(case, accumulate):
    name, shape, indices, vshape = case
    if not accumulate and name in ("one", "one_2d", "two", "two_broadcast", "three"):
        # duplicates are only defined with accumulate=True; make the positions unique
        indices = {
            "one": (T([3, 0, 5, 1]),), "one_2d": (T([[3, 0], [5, 1]]),),
            "two": (T([0, 1, 2, 0]), T([3, 0, 1, 2])), "two_broadcast": (T([[0], [4]]), T([[5, 1, 3], [0, 2, 4]])),
            "three": (T([0, 4, 0]), T([5, 1, 5]), T([3, 3, 2])),
        }[name]
    x = base(shape)
    values = base(vshape) + 50.0
    ref = x.clone().index_put_(indices, values, accumulate=accumulate)
    dind, dval = to_dev(tuple(indices)), values.to(DEV)
    xd = x.to(DEV)
    assert xd.index_put_(dind, dval, accumulate=accumulate) is xd
    assert torch.equal(xd.cpu(), ref)
    xd = x.to(DEV)
    out = xd.index_put(list(dind), dval, accumulate)
    assert torch.equal(out.cpu(), ref) and torch.equal(xd.cpu(), x)
    out = torch.index_put(xd, dind, dval, accumulate=accumulate)
    assert torch.equal(out.cpu(), ref) and torch.equal(xd.cpu(), x)
    out = torch.index_put(input=xd, indices=tuple(indices), values=values[..., :1].to(DEV) if vshape[-1] == 4 else dval,
                          accumulate=accumulate)
    assert torch.equal(out.cpu(), x.index_put(tuple(indices), values[..., :1] if vshape[-1] == 4 else values, accumulate))


@pytest.mark.parametrize("case", PUT_CASES, ids=_ids(PUT_CASES))
def test_index_put_accumulate_backward_matches_cpu(case):
    _, shape, indices, vshape = case
    grads = []
    for device in ("cpu", DEV):
        w = base(shape).to(device).requires_grad_(True)
        v = (base(vshape) + 50.0).to(device).requires_grad_(True)
        ind = tuple(i.to(device) for i in indices)
        out = torch.index_put(w * 2.0, ind, v * 3.0, accumulate=True)
        coef = (torch.arange(out.numel(), dtype=torch.float32).reshape(out.shape) + 1.0).to(device)
        (out * coef).sum().backward()
        grads.append((w.grad.cpu(), v.grad.cpu()))
    assert torch.equal(grads[0][0], grads[1][0]) and torch.equal(grads[0][1], grads[1][1])


def test_masked_select_and_masked_scatter_match_cpu():
    x = base((6, 4))
    xd = x.to(DEV)
    for mask in (mask_of((6, 4)), mask_of((4,), 2), mask_of((6, 1), 2)):
        assert torch.equal(xd.masked_select(mask.to(DEV)).cpu(), x.masked_select(mask))
        assert torch.equal(torch.masked_select(xd, mask.to(DEV)).cpu(), torch.masked_select(x, mask))
        src = base((30,)) + 70.0
        assert torch.equal(xd.masked_scatter(mask.to(DEV), src.to(DEV)).cpu(), x.masked_scatter(mask, src))
        assert torch.equal(torch.masked_scatter(xd, mask.to(DEV), src.to(DEV)).cpu(), torch.masked_scatter(x, mask, src))
        y = xd.clone()
        assert y.masked_scatter_(mask.to(DEV), src.to(DEV)) is y
        assert torch.equal(y.cpu(), x.clone().masked_scatter_(mask, src))
    mask = mask_of((6, 4))
    grads = []
    for device in ("cpu", DEV):
        w = base((6, 4)).to(device).requires_grad_(True)
        s = (base((30,)) + 70.0).to(device).requires_grad_(True)
        a = torch.masked_select(w * 2.0, mask.to(device))
        b = torch.masked_scatter(w * 3.0, mask.to(device), s * 5.0)
        ((a * a).sum() + (b * b).sum()).backward()
        grads.append((w.grad.cpu(), s.grad.cpu()))
    assert torch.equal(grads[0][0], grads[1][0]) and torch.equal(grads[0][1], grads[1][1])


def test_errors_match_cpu():
    x = base((6, 4))
    xd = x.to(DEV)
    bad = [
        T([0, 6]), T([-7]), (S_, T([4])), (T([0, 1]), T([0, 4])), (T([0, 1]), T([0, 1, 2])),
        B([True, False]), (T([0]), T([0]), T([0])), torch.tensor([0.0, 1.0]), T([0, 1], torch.int16),
    ]
    for key in bad:
        with pytest.raises(IndexError):
            x[key]
        with pytest.raises(IndexError):
            xd[to_dev(key)]
        with pytest.raises(IndexError):
            xd[key]  # CPU-resident index
        with pytest.raises(IndexError):
            xd[to_dev(key)] = 1.0
    for key, value in [(T([0, 1]), torch.ones(3)), (T([0]), torch.ones(4, dtype=torch.int64)),
                       (T([0]), torch.ones(4, dtype=torch.float64))]:
        with pytest.raises(RuntimeError):
            x.clone()[key] = value
        with pytest.raises(RuntimeError):
            xd[to_dev(key)] = value.to("cpu")
    with pytest.raises(RuntimeError):
        xd.index_put_((T([0, 1]).to(DEV),), torch.ones(1, 1, 2, 4, device=DEV))
    with pytest.raises(TypeError):
        xd[T([0]).to(DEV)] = [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(TypeError):
        xd.index_put((T([0]).to(DEV),), 1.0)
    with pytest.raises(RuntimeError):
        xd.masked_select(T([1, 0, 1, 0]).to(DEV))
    w = torch.zeros(4, device=DEV, requires_grad=True)
    with pytest.raises(RuntimeError):
        w[T([0]).to(DEV)] = 1.0  # in-place on a leaf that requires grad
    assert torch.equal(xd.cpu(), x)
    torch.xpu.synchronize()


def test_bounds_check_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(du, "_XPU_INDEX_CHECK_BOUNDS", False)
    x = base((6, 4))
    xd = x.to(DEV)
    for key in (T([-1, 0, 5]), (T([-6, 2]), T([-1, 0])), (S_, T([-4, 3]))):
        assert torch.equal(xd[to_dev(key)].cpu(), x[key])
    with pytest.raises(IndexError):
        xd[T([6])]  # CPU-resident indices are still checked


def test_large_indices_and_masks_use_device_side_checks(monkeypatch):
    monkeypatch.setattr(du, "_XPU_INDEX_HOST_MAX", 4)  # force the paths used above 65536 elements
    x = base((6, 5, 4))
    xd = x.to(DEV)
    keys = [T([5, 0, -1, 2, 2, 3]), (T([[0], [5], [1]]), T([[0, 1, -1], [4, 4, 0], [2, 3, 1]])), mask_of((6,), 2),
            mask_of((6, 5)), (mask_of((6,), 2), T([0, 4, 1])), (S_, mask_of((5, 4)))]
    for key in keys:
        assert torch.equal(xd[to_dev(key)].cpu(), x[key])
        y, yd = x.clone(), xd.clone()
        y[key] = 1.5
        yd[to_dev(key)] = 1.5
        assert torch.equal(yd.cpu(), y)
    for key in (T([0, 1, 2, 3, 4, 6]), (T([0, 1, 2, 3, 4, 5]), T([0, 1, 2, 3, 4, 5]))):
        with pytest.raises(IndexError):
            xd[to_dev(key)]


def test_unrecognised_forms_use_cpu_round_trip_and_warn_once(monkeypatch):
    monkeypatch.setattr(du, "_XPU_INDEX_WARNED", set())
    x = base((3, 4))
    xd = x.to(DEV)
    with pytest.warns(RuntimeWarning, match="CPU round trip"):
        out = xd[True]
    assert out.device.type == DEV and torch.equal(out.cpu(), x[True])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert torch.equal(xd[True].cpu(), x[True])  # second use: no warning
    with pytest.warns(RuntimeWarning, match="CPU round trip"):
        xd[torch.tensor(True), T([0, 2]).to(DEV)] = 4.0
    x[torch.tensor(True), T([0, 2])] = 4.0
    assert torch.equal(xd.cpu(), x)
    m = torch.zeros(5, dtype=torch.bool)
    md = m.to(DEV)
    with pytest.warns(RuntimeWarning, match="CPU round trip"):
        md.index_put_((T([1, 1, 3]).to(DEV),), torch.tensor(True, device=DEV), accumulate=True)
    assert torch.equal(md.cpu(), m.index_put_((T([1, 1, 3]),), torch.tensor(True), accumulate=True))
    grads = []
    for device in ("cpu", DEV):  # the round trip is differentiable
        w = base((3, 4)).to(device).requires_grad_(True)
        y = w * 2.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            y[True, T([0, 2]).to(device)] = (w * 3.0)[:2]
        ((y[True]) ** 2).sum().backward()
        grads.append(w.grad.cpu())
    assert torch.equal(grads[0], grads[1])


def test_wrappers_pass_cpu_tensors_through():
    orig_get = du._XPU_INDEX_ORIG["Tensor.__getitem__"]
    x = base((6, 4))
    keys = [2, slice(1, 3), (S_, 1), T([3, 0]), (T([0, 1]), T([1, 2])), mask_of((6,), 2), [0, 5], (None, Ellipsis, T([0]))]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for key in keys:
            assert torch.equal(x[key], orig_get(x, key))
        assert x[1:3]._base is x and x[2]._base is x
        y = x.clone()
        y[T([0, 0])] = 1.0
        y[mask_of((6, 4))] = base((8,))
        y.index_put_((T([1, 1]),), torch.ones(4), accumulate=True)
        z = x.clone()
        du._XPU_INDEX_ORIG["Tensor.__setitem__"](z, T([0, 0]), 1.0)
        du._XPU_INDEX_ORIG["Tensor.__setitem__"](z, mask_of((6, 4)), base((8,)))
        du._XPU_INDEX_ORIG["Tensor.index_put_"](z, (T([1, 1]),), torch.ones(4), True)
        assert torch.equal(y, z)
        assert torch.equal(torch.index_put(x, (T([1]),), torch.ones(4)), du._XPU_INDEX_ORIG["torch.index_put"](x, (T([1]),), torch.ones(4)))
        assert torch.equal(torch.masked_select(x, x > 0), x[x > 0])
    # basic indexing on XPU is not rewritten either: views stay views
    xd = x.to(DEV)
    assert xd[1:3]._base is xd and xd[2]._base is xd and xd[..., 1]._base is xd and xd[None, 1:, 0]._base is xd
    xd[1:3, 0] = 5.0
    xd[2] = torch.ones(4, device=DEV)
    x[1:3, 0] = 5.0
    x[2] = torch.ones(4)
    assert torch.equal(xd.cpu(), x)


def test_install_only_for_xpu_and_uninstall_restores(monkeypatch):
    for name in ("ZE_ENABLE_PCI_ID_DEVICE_ORDER", "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.delenv("NNKNN_XPU_INDEX_TRIPWIRE", raising=False)
    patched_get, patched_put = torch.Tensor.__getitem__, torch.index_put
    du.uninstall_xpu_index_workaround()
    assert not du.xpu_index_workaround_installed() and du._XPU_INDEX_TRIPWIRE is None
    for name in du._XPU_INDEX_TENSOR_ATTRS:
        assert name not in torch.Tensor.__dict__
    assert torch.Tensor.__getitem__ is not patched_get and torch.index_put is not patched_put
    assert torch.Tensor.__getitem__ is torch._C.TensorBase.__getitem__

    monkeypatch.setenv("NNKNN_DEVICE", "cpu")
    assert du.resolve_runtime_device().type == "cpu"
    assert not du.xpu_index_workaround_installed()  # CPU runs are not patched at all

    monkeypatch.setenv("NNKNN_DEVICE", "xpu")
    assert du.resolve_runtime_device().type == "xpu"
    assert du.xpu_index_workaround_installed() and du._XPU_INDEX_TRIPWIRE is None  # tripwire is opt-in
    first = torch.Tensor.__getitem__
    assert install_xpu_index_workaround() and torch.Tensor.__getitem__ is first  # idempotent

    du.uninstall_xpu_index_workaround()
    monkeypatch.setenv("NNKNN_XPU_INDEX_TRIPWIRE", "1")
    assert install_xpu_index_workaround() and du._XPU_INDEX_TRIPWIRE is not None


def test_wrappers_are_active_in_other_threads():
    import threading

    x = base((90, 1))
    xd = x.to(DEV)
    idx = torch.arange(90, device=DEV)
    result = {}

    def work():
        result["out"] = xd[idx].cpu()

    du.disable_xpu_index_tripwire()  # the tripwire is thread-local, the wrappers are not
    t = threading.Thread(target=work)
    t.start()
    t.join()
    torch.xpu.synchronize()
    assert torch.equal(result["out"], x)


def test_tripwire_raises_before_the_broken_kernels():
    xd = base((6, 4)).to(DEV)
    idx = T([0, 1]).to(DEV)
    with pytest.raises(RuntimeError, match="tripwire"):
        du._XPU_INDEX_ORIG["Tensor.__getitem__"](xd, idx)
    with pytest.raises(RuntimeError, match="tripwire"):
        du._XPU_INDEX_ORIG["Tensor.index_put_"](xd, (idx,), torch.ones(2, 4, device=DEV), False)
    with pytest.raises(RuntimeError, match="tripwire"):
        du._XPU_INDEX_ORIG["torch.masked_select"](xd, xd > 0)
    # backward formulas run on autograd worker threads; the tripwire still sees them
    w = torch.zeros(6, device=DEV, requires_grad=True)
    s = torch.ones(6, device=DEV, requires_grad=True)
    y = du._XPU_INDEX_ORIG["torch.masked_scatter"](w * 1.0, mask_of((6,), 2).to(DEV), s * 2.0)
    with pytest.raises(RuntimeError, match="tripwire"):
        y.sum().backward()
    torch.xpu.synchronize()
    assert torch.equal((xd + 0).cpu(), base((6, 4)))  # the device is still healthy


def _tiny_retrieval_step(device):
    g = torch.Generator().manual_seed(7)
    cases = torch.randn(40, 6, generator=g).to(device)
    labels = torch.randn(40, generator=g).to(device)
    queries = torch.randn(9, 6, generator=g).to(device)
    w = torch.ones(6, device=device, requires_grad=True)
    bias = torch.zeros(40, device=device, requires_grad=True)
    lin = torch.nn.Linear(6, 6).to(device)
    with torch.no_grad():
        lin.weight.copy_(torch.eye(6) + 0.01 * torch.randn(6, 6, generator=g))
        lin.bias.zero_()
    n = cases.shape[0]
    case_indices = torch.arange(n, device=device)
    feats = lin(cases[case_indices]) * w                       # x[arange]
    d = torch.cdist(lin(queries) * w, feats) * 1.0
    far = d > d.detach().mean()
    d[far] = d[far] * 2.0                                      # 2-D mask read and write, grad flows
    v, top = d.topk(5, dim=1, largest=False)
    batch = torch.arange(queries.shape[0], device=device).unsqueeze(1)
    lab = labels.unsqueeze(0).expand(queries.shape[0], n)[batch, top]   # two tensors on an expanded view
    lab = lab + bias[case_indices[top]]                        # 2-D index into a parameter
    adapted = torch.zeros(queries.shape[0], n, 1, device=device)
    adapted[batch, top, :] = (torch.softmax(-v, dim=1) * lab).unsqueeze(-1)  # two tensors + slice, setitem
    keep = torch.ones(queries.shape[0], n, dtype=torch.bool, device=device)
    keep[batch.squeeze(1), top[:, 0]] = False
    pred = adapted.squeeze(-1).masked_fill(~keep, 0.0).sum(1)
    loss = ((pred - 0.3) ** 2).mean()
    loss.backward()
    return loss.detach().cpu(), w.grad.cpu(), bias.grad.cpu(), lin.weight.grad.cpu(), top.cpu()


def test_tripwire_on_tiny_model_forward_backward():
    assert du._XPU_INDEX_TRIPWIRE is not None
    ref = _tiny_retrieval_step("cpu")
    out = _tiny_retrieval_step(DEV)
    torch.xpu.synchronize()
    assert torch.equal(ref[4], out[4])
    for a, b in zip(ref[:4], out[:4]):
        assert torch.allclose(a, b, rtol=1e-3, atol=1e-5)
    assert ref[1].abs().sum() > 0 and ref[2].abs().sum() > 0


@pytest.mark.parametrize("shape", [(90, 1), (1000, 5), (5, 90), (16, 1), (90, 8), (4096, 64)])
def test_integer_indexing_matches_cpu(shape):
    c = torch.randn(*shape)
    idx = torch.arange(shape[0])
    assert torch.equal(c.to("xpu")[idx.to("xpu")].cpu(), c[idx])
    assert torch.equal(c.to("xpu")[idx].cpu(), c[idx])  # CPU index into an XPU tensor


def test_boolean_masks_and_assignment_match_cpu():
    d = torch.randn(40, 40)
    mask = torch.isfinite(d) & (d > 0)
    assert torch.equal(d.to("xpu")[mask.to("xpu")].cpu(), d[mask])
    col = d[:, 3].contiguous()
    assert torch.equal(col.to("xpu")[(col > 0).to("xpu")].cpu(), col[col > 0])
    x, y = d.clone(), d.to("xpu").clone()
    rows = torch.tensor([3, 7, 11])
    x[rows] = 0.0
    y[rows.to("xpu")] = 0.0
    assert torch.equal(y.cpu(), x)
    z = torch.zeros(10, requires_grad=True, device="xpu")
    z[torch.tensor([1, 1, 2], device="xpu")].sum().backward()
    assert z.grad.cpu().tolist() == [0, 2, 1, 0, 0, 0, 0, 0, 0, 0]


# --- regression tests for the review findings -------------------------------------------------
# Modes: "cpu_direct" calls the rewrite functions on CPU tensors (runs anywhere); "xpu" goes through
# the installed wrappers with index and value on the device; "xpu_cpu_args" keeps them on the CPU.
MODES = ["cpu_direct", "xpu", "xpu_cpu_args"]


def _need_xpu(mode):
    if mode != "cpu_direct" and not HAS_XPU:
        pytest.skip("no XPU device")


def _args(mode, *objs):
    return tuple(to_dev(o) for o in objs) if mode == "xpu" else objs


def _get(mode, x, key):
    if mode == "cpu_direct":
        return du._xpu_getitem(x, key)
    (key,) = _args(mode, key)
    return x.to(DEV)[key].cpu()


def _set(mode, x, key, value):
    if mode == "cpu_direct":
        y = x.clone()
        du._xpu_setitem(y, key, value)
        return y
    key, value = _args(mode, key, value)
    y = x.to(DEV)
    y[key] = value
    return y.cpu()


def _put(mode, x, indices, values, accumulate):
    if mode == "cpu_direct":
        return du._xpu_index_put(x.clone(), indices, values, accumulate, True)
    indices, values = _args(mode, indices, values)
    return x.to(DEV).index_put_(indices, values, accumulate=accumulate).cpu()


def _native_set(x, key, value):
    y = x.clone()
    y[key] = value
    return y


def _outcome(fn):
    try:
        return "ok", fn()
    except Exception as exc:  # noqa: BLE001 - the exception type is what is compared
        return "error", type(exc)


def _same_outcome(native, ours):
    assert native[0] == ours[0], (native, ours)
    if native[0] == "error":
        assert native[1] is ours[1], (native, ours)
    else:
        assert native[1].dtype == ours[1].dtype and native[1].shape == ours[1].shape
        if native[1].is_floating_point():  # nan == nan for this comparison
            assert torch.equal(torch.nan_to_num(native[1], nan=1.5), torch.nan_to_num(ours[1], nan=1.5))
        else:
            assert torch.equal(native[1], ours[1])


_M4 = B([True, False, True, False])
_I2 = T([0, 2])
ONE_ELEMENT_KEYS = [
    ("mask", _M4), ("slice_mask", (S_, _M4)), ("mask_slice", (_M4, S_)), ("int_mask", (1, _M4)),
    ("mask_int", (_M4, 1)), ("none_mask", (None, _M4)), ("ellipsis_mask", (Ellipsis, _M4)),
    ("mask2d", torch.eye(4, dtype=torch.bool)), ("bool_list", [True, False, True, False]),
    ("stepslice_mask", (slice(0, 2), _M4)),
    # controls: not the shortcut, a dtype mismatch raises on the CPU too
    ("mask_idx", (_M4, _I2)), ("mask_mask", (_M4, _M4)), ("idx", _I2), ("idx_slice", (_I2, S_)),
]
ONE_ELEMENT_VALUES = [
    (torch.float32, torch.tensor(7)), (torch.float32, torch.tensor([[7]])), (torch.float32, torch.tensor(True)),
    (torch.float32, torch.tensor(1.5, dtype=torch.float64)), (torch.float32, torch.tensor([[[7.0]]])),
    (torch.float32, torch.tensor(float("inf"), dtype=torch.float64)), (torch.float32, torch.tensor([3, 9]).max()),
    (torch.int64, torch.tensor(-1, dtype=torch.int32)), (torch.int64, torch.tensor(2.7)),
    (torch.bool, torch.tensor(1)), (torch.float16, torch.tensor(3)),
    (torch.float32, torch.tensor([7, 8])),  # two elements: raises on both sides
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", ONE_ELEMENT_KEYS, ids=_ids(ONE_ELEMENT_KEYS))
def test_cpu_mask_write_of_one_element_value_matches_native(case, mode):
    """Finding: one bool mask + one-element tensor value of another dtype / with extra dims."""
    _need_xpu(mode)
    _, key = case
    for dtype, value in ONE_ELEMENT_VALUES:
        x = base((4, 4), dtype) if dtype != torch.bool else mask_of((4, 4))
        native = _outcome(lambda: _native_set(x, key, value))
        _same_outcome(native, _outcome(lambda: _set(mode, x, key, value)))


@pytest.mark.parametrize("mode", MODES)
def test_cpu_mask_index_put_of_one_element_value_matches_native(mode):
    _need_xpu(mode)
    x = base((4, 4))
    for indices in ((_M4,), (torch.eye(4, dtype=torch.bool),), (_M4, _I2), (_I2,)):
        for value in (torch.tensor(7), torch.tensor([[[7.0]]]), torch.ones(1, 1, 1, 1, 1), torch.tensor(2.5)):
            for accumulate in (False, True):
                native = _outcome(lambda: x.clone().index_put_(indices, value, accumulate=accumulate))
                _same_outcome(native, _outcome(lambda: _put(mode, x, indices, value, accumulate)))
                if mode == "cpu_direct":
                    ours = _outcome(lambda: du._xpu_index_put(x, indices, value, accumulate, False))
                    _same_outcome(native, ours)


@pytest.mark.parametrize("device", ["cpu_direct", "xpu"])
def test_cpu_mask_write_of_one_element_value_backward(device):
    """The value keeps its gradient (and stays off masked_fill_'s masked_select backward)."""
    _need_xpu(device)
    dev = "cpu" if device == "cpu_direct" else DEV
    for vdtype in (torch.float32, torch.float64 if dev == "cpu" else torch.float32):
        ref = []
        for native in (True, False):
            d = "cpu" if native else dev
            w = base((4, 4)).to(d).requires_grad_(True)
            v = torch.tensor(2.0, dtype=vdtype, device=d, requires_grad=True)
            y = w * 2.0
            if native:
                y[_M4] = v * 3.0
            elif device == "cpu_direct":
                du._xpu_setitem(y, _M4, v * 3.0)
            else:
                y[_M4.to(d)] = v * 3.0
            coef = (torch.arange(16.0).reshape(4, 4) + 1.0).to(d)
            (y * coef).sum().backward()
            ref.append((w.grad.cpu(), v.grad.cpu()))
        assert torch.equal(ref[0][0], ref[1][0]) and torch.equal(ref[0][1], ref[1][1]) and ref[0][1] != 0
    # a value without grad: masked_fill_, the target still back-propagates
    for d, direct in (("cpu", True),) + (((DEV, False),) if HAS_XPU else ()):
        w = base((4, 4)).to(d).requires_grad_(True)
        y = w * 2.0
        if direct:
            du._xpu_setitem(y, (S_, _M4), torch.tensor(9))
        else:
            y[:, _M4.to(d)] = torch.tensor(9, device=d)
        (y * y).sum().backward()
        expect = base((4, 4)) * 8.0
        expect[:, _M4] = 0.0
        assert torch.equal(w.grad.cpu(), expect)


_E = torch.zeros(0, dtype=torch.int64)
EMPTY_CASES = [
    ("zero_cols", (3, 0), (T([5]),)),
    ("zero_rows_and_empty_idx", (0, 4), (T([0]), _E)),
    ("empty_second_idx", (3, 4), (T([9]), _E)),
    ("empty_slice", (3, 4), (T([9]), slice(0, 0))),
    ("mask_selects_nothing", (3, 4), (T([9]), torch.zeros(4, dtype=torch.bool))),
    ("mask_first_selects_nothing", (3, 4), (torch.zeros(3, dtype=torch.bool), T([9]))),
    ("zero_tail", (3, 4, 0), (T([9]), T([7]))),
    ("empty_2d_idx", (3, 4), (torch.zeros(0, 1, dtype=torch.int64), T([9, 9]))),
    ("negative_out_of_range", (2, 0, 3), (S_, S_, T([-9]))),
    # controls: the result has elements, or the index shapes do not broadcast -> IndexError on both
    ("control_nonempty", (3, 4), (T([9]),)),
    ("control_zero_indexed_dim", (0, 4), (T([0]),)),
    ("control_separated", (3, 4, 0), (T([9]), S_, T([0]))),
    ("control_no_broadcast", (3, 4), (_E, T([9, 9]))),
    # a non-empty index into a dim of size 0 raises even though the result would be empty
    ("control_zero_dim_empty_result", (0, 3, 0), (None, [0, 0, 0])),
    ("control_zero_dim_empty_slice", (0, 4, 1), (T([0, 0]), slice(0, 0), slice(1, 3))),
    ("control_zero_dim_two", (4, 0, 3, 0), (T([1, 2]), T([0, 0]))),
    ("control_zero_dim_2d_idx", (0, 0), (T([[0, 0, 0]]),)),
    ("control_zero_dim_int32", (1, 0), (slice(1, None), T([0, 0], torch.int32))),
    ("control_zero_dim_after_slice", (2, 0, 0), (S_, S_, T([0, 0]))),
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", EMPTY_CASES, ids=_ids(EMPTY_CASES))
def test_cpu_out_of_range_index_with_empty_result_matches_native(case, mode):
    """Finding: torch does not range-check an index when the indexing result has no elements."""
    _need_xpu(mode)
    name, shape, key = case
    x = torch.zeros(shape)
    native = _outcome(lambda: x[key])
    _same_outcome(native, _outcome(lambda: _get(mode, x, key)))
    assert native[0] == ("error" if name.startswith("control") else "ok")
    if name == "control_zero_indexed_dim":
        return  # native setitem fails with an internal setStorage error here; the rewrite raises IndexError
    _same_outcome(_outcome(lambda: _native_set(x, key, 1.0)), _outcome(lambda: _set(mode, x, key, 1.0)))
    if not name.startswith("control"):
        with pytest.raises(RuntimeError, match="shape mismatch"):
            _set(mode, x, key, torch.ones(7, 7, 7))  # a value that cannot broadcast still raises
    if all(isinstance(k, torch.Tensor) for k in key):
        for accumulate in (False, True):
            native = _outcome(lambda: x.clone().index_put_(key, torch.tensor(1.0), accumulate=accumulate))
            _same_outcome(native, _outcome(lambda: _put(mode, x, key, torch.tensor(1.0), accumulate)))
    if mode == "cpu_direct" and not name.startswith("control"):
        y = x.clone()
        version = y._version
        du._xpu_setitem(y, key, 1.0)
        assert y._version == version + 1  # still an in-place op, as in torch
        w = torch.zeros(shape, requires_grad=True)
        assert du._xpu_getitem(w * 1.0, key).requires_grad


@pytest.mark.parametrize("mode", MODES)
def test_cpu_float16_overflowing_scalar_matches_native(mode):
    """Finding: with two or more index tensors torch writes inf into a float16 target; other forms raise."""
    _need_xpu(mode)
    keys = [(_I2, _I2), (_I2, None, _I2), (_M4, _I2), (_M4, _M4), _I2, (S_, _I2), (_I2, S_), (0, _I2), _M4, (S_, _M4)]
    for dtype in (torch.float16, torch.float32, torch.int64):
        for key in keys:
            for value in (1e40, -1e40, 2 ** 40, float("nan"), True, 2.5):
                x = torch.zeros(4, 4, dtype=dtype)
                native = _outcome(lambda: _native_set(x, key, value))
                _same_outcome(native, _outcome(lambda: _set(mode, x, key, value)))


@pytest.mark.parametrize("case", READ_CASES, ids=_ids(READ_CASES))
def test_cpu_rewriter_read_matches_native(case):
    """The rewrite logic itself, on CPU tensors: values, shape, dtype, copy-vs-view and gradient."""
    _, shape, key = case
    x, x2 = base(shape), base(shape)
    ref, out = x[key], du._xpu_getitem(x2, key)
    assert out.dtype == ref.dtype and out.shape == ref.shape and torch.equal(out, ref)
    ref.add_(1.0)
    out.add_(1.0)
    assert torch.equal(x, x2)  # the result shares memory with the source exactly when torch's does
    grads = []
    for fn in (lambda t: t[key], lambda t: du._xpu_getitem(t, key)):
        w = base(shape).requires_grad_(True)
        out = fn(w * 2.0)
        (out * (torch.arange(out.numel(), dtype=torch.float32).reshape(out.shape) + 1.0)).sum().backward()
        grads.append(w.grad)
    assert torch.equal(grads[0], grads[1])


@pytest.mark.parametrize("kind", VALUE_KINDS)
@pytest.mark.parametrize("case", WRITE_CASES, ids=_ids(WRITE_CASES))
def test_cpu_rewriter_write_matches_native(case, kind):
    _, shape, key = case
    x = base(shape)
    value = make_value(kind, tuple(x[key].shape))
    assert torch.equal(_set("cpu_direct", x, key, value), _native_set(x, key, value))
    if kind != "full":
        return
    grads = []
    for native in (True, False):
        w = base(shape).requires_grad_(True)
        v = value.clone().requires_grad_(True)
        y = w * 2.0
        if native:
            y[key] = v * 3.0
        else:
            du._xpu_setitem(y, key, v * 3.0)
        (y * (torch.arange(y.numel(), dtype=torch.float32).reshape(y.shape) + 1.0)).sum().backward()
        grads.append((w.grad, v.grad))
    assert torch.equal(grads[0][0], grads[1][0]) and torch.equal(grads[0][1], grads[1][1])


@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("case", PUT_CASES, ids=_ids(PUT_CASES))
def test_cpu_rewriter_index_put_matches_native(case, accumulate):
    name, shape, indices, vshape = case
    if not accumulate and name in ("one", "one_2d", "two", "two_broadcast", "three"):
        return  # duplicate positions: only defined with accumulate=True
    x = base(shape)
    values = base(vshape) + 50.0
    ref = x.index_put(indices, values, accumulate=accumulate)
    assert torch.equal(du._xpu_index_put(x.clone(), indices, values, accumulate, True), ref)
    out = du._xpu_index_put(x, list(indices), values, accumulate, False)
    assert torch.equal(out, ref) and torch.equal(x, base(shape))


_VIEW_OPS = {
    "reshape", "view", "expand", "permute", "narrow", "unbind", "squeeze", "t", "split", "split_with_sizes",
    "alias", "slice", "select", "unsqueeze", "_reshape_alias", "expand_as", "detach", "lift_fresh", "transpose",
    "as_strided", "_unsafe_view", "view_as", "broadcast_to",
}


def _trace(fn, on_device=None):
    """Run fn under a logging dispatch mode. Returns (kernels, downloads, readbacks).

    kernels: non-view ops (on tensors of `on_device` when given); downloads: copies from that
    device to the CPU; readbacks: `.item()` on a tensor of that device.
    """
    from torch.utils._python_dispatch import TorchDispatchMode

    log = []

    class Trace(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            flat = []
            for a in list(args) + list(kwargs.values()):
                flat.extend(a if isinstance(a, (list, tuple)) else [a])
            devices = {a.device.type for a in flat if isinstance(a, torch.Tensor)}
            log.append((func._schema.name.replace("aten::", ""), devices, str(kwargs.get("device", ""))))
            return func(*args, **kwargs)

    with Trace():
        fn()
    mine = [(n, kw) for n, devs, kw in log if on_device is None or on_device in devs]
    kernels = [n for n, _ in mine if n not in _VIEW_OPS and n not in ("_to_copy", "_local_scalar_dense")]
    downloads = sum(1 for n, kw in mine if n == "_to_copy" and kw.startswith("cpu"))
    kernels += [n for n, kw in mine if n == "_to_copy" and not kw.startswith("cpu")]
    readbacks = sum(1 for n, _ in mine if n == "_local_scalar_dense")
    return kernels, downloads, readbacks


def test_cpu_rewriter_kernel_counts():
    """Findings: mask + scalar is one masked_fill_; index linearisation is one add per extra index."""
    g = torch.Generator().manual_seed(0)
    d = torch.randn(8, 50, generator=g)
    m2 = d > 0
    big = torch.randn(8, 50, 3, generator=g)
    b, i = torch.arange(8).unsqueeze(1), torch.randint(0, 50, (8, 5), generator=g)
    for value in (float("inf"), torch.tensor(float("inf")), torch.tensor([3])):
        y = d.clone()
        kernels, _, _ = _trace(lambda: du._xpu_setitem(y, m2, value))
        assert kernels == ["masked_fill_"], kernels
        assert torch.equal(y, _native_set(d, m2, value))
    y = big.clone()
    kernels, _, _ = _trace(lambda: du._xpu_setitem(y, (S_, m2[0]), 0.0))  # mask after a slice, trailing dim
    assert kernels == ["masked_fill_"] and torch.equal(y, _native_set(big, (S_, m2[0]), 0.0))
    kernels, _, _ = _trace(lambda: du._xpu_getitem(big, (b, i)))
    assert "mul" not in kernels and kernels.count("add") == 1 and kernels[-1] == "index_select", kernels
    kernels, _, _ = _trace(lambda: du._xpu_getitem(big, (b, i, T([0, 2, 2, 1, 0]))))
    assert "mul" not in kernels and kernels.count("add") == 2 and kernels[-1] == "index_select", kernels
    y, src = d.clone(), torch.randn(400, generator=g)
    kernels, _, _ = _trace(lambda: du._xpu_masked_scatter(y, m2, src, True))
    assert kernels == ["nonzero", "index_copy_"], kernels  # in place: no full-size copy
    assert torch.equal(y, d.clone().masked_scatter_(m2, src))


def test_cpu_masked_scatter_in_place_variants_match_native():
    x = base((6, 5, 4))
    mask = mask_of((6, 5, 4))
    src = base((200,)) + 70.0
    for view in (lambda t: t, lambda t: t.transpose(0, 2), lambda t: t[::2, 1:]):  # contiguous or not
        a, b = x.clone(), x.clone()
        m = view(mask)
        ref = view(a).masked_scatter_(m, src)
        assert du._xpu_masked_scatter(view(b), m, src, True).shape == ref.shape and torch.equal(a, b)
    a, b = x.clone(), x.clone()
    a.masked_scatter_(mask, a.clone().flip(0))
    du._xpu_masked_scatter(b, mask, b.flip(0), True)
    c = x.clone()
    du._xpu_masked_scatter(c, mask, c.reshape(-1), True)  # the source is a view of the target
    assert torch.equal(a, b) and torch.equal(c, x.clone().masked_scatter_(mask, x.reshape(-1).clone()))
    w = base((6, 5, 4)).requires_grad_(True)
    s = src.clone().requires_grad_(True)
    grads = []
    for native in (True, False):
        w.grad = s.grad = None
        y = w * 2.0
        y.masked_scatter_(mask, s * 3.0) if native else du._xpu_masked_scatter(y, mask, s * 3.0, True)
        (y * y).sum().backward()
        grads.append((w.grad.clone(), s.grad.clone()))
    assert torch.equal(grads[0][0], grads[1][0]) and torch.equal(grads[0][1], grads[1][1])


def test_cpu_fresh_cpu_process_installs_nothing():
    """A CPU run does not patch torch and does not initialise XPU."""
    code = (
        "import torch\n"
        "get, put = torch.Tensor.__getitem__, torch.index_put\n"
        "from model import device_utils as du\n"
        "dev = du.resolve_runtime_device()\n"
        "assert dev.type != 'xpu', dev\n"
        "assert not du.xpu_index_workaround_installed() and du._XPU_INDEX_TRIPWIRE is None\n"
        "assert torch.Tensor.__getitem__ is get and torch.index_put is put\n"
        "assert not any(n in torch.Tensor.__dict__ for n in du._XPU_INDEX_TENSOR_ATTRS)\n"
        "xpu = getattr(torch, 'xpu', None)\n"
        "assert xpu is None or not getattr(xpu, 'is_initialized', lambda: False)()\n"
        "assert torch.arange(6.0)[torch.tensor([1, 2])].tolist() == [1.0, 2.0]\n"
        "print('CLEAN')\n"
    )
    root = Path(__file__).resolve().parents[1]
    for requested in (None, "cpu"):
        env = {k: v for k, v in os.environ.items() if k not in ("NNKNN_DEVICE", "NNKNN_XPU_INDEX_TRIPWIRE")}
        if requested:
            env["NNKNN_DEVICE"] = requested
        run = subprocess.run([sys.executable, "-B", "-c", code], cwd=root, env=env, capture_output=True, text=True, timeout=180)
        assert run.returncode == 0 and "CLEAN" in run.stdout, run.stderr[-2000:]


def test_cpu_concurrent_installs_are_serialised(monkeypatch):
    """Finding: a second thread installing during the first install made uninstall keep the wrappers."""
    monkeypatch.delenv("NNKNN_XPU_INDEX_TRIPWIRE", raising=False)
    du.uninstall_xpu_index_workaround()
    native_get, native_put = torch.Tensor.__getitem__, torch.index_put
    entered, release, calls = threading.Event(), threading.Event(), []

    def slow_is_xpu_available():
        calls.append(1)
        if len(calls) == 1:  # the first installer pauses in the middle of its install
            entered.set()
            release.wait(20)
        return True

    monkeypatch.setattr(du, "is_xpu_available", slow_is_xpu_available)
    first = threading.Thread(target=du.install_xpu_index_workaround)
    second = threading.Thread(target=du.install_xpu_index_workaround)
    first.start()
    try:
        assert entered.wait(20)
        second.start()
        second.join(0.3)
        assert second.is_alive()  # it waits for the first install instead of interleaving with it
    finally:
        release.set()
        first.join(20)
        second.join(20)
    try:
        assert du.xpu_index_workaround_installed() and torch.Tensor.__getitem__ is not native_get
        assert len(calls) == 1  # the second call saw the finished install
    finally:
        du.uninstall_xpu_index_workaround()
    assert not du.xpu_index_workaround_installed()
    assert torch.Tensor.__getitem__ is native_get and torch.index_put is native_put
    assert not any(n in torch.Tensor.__dict__ for n in du._XPU_INDEX_TENSOR_ATTRS)


def test_cpu_wrapped_functions_pickle_by_reference(monkeypatch):
    """Finding: pickle.dumps(torch.masked_select) failed while the wrappers were installed."""
    monkeypatch.delenv("NNKNN_XPU_INDEX_TRIPWIRE", raising=False)
    monkeypatch.setattr(du, "is_xpu_available", lambda: True)
    was_installed = du.xpu_index_workaround_installed()
    assert install_xpu_index_workaround()
    try:
        for name in du._XPU_INDEX_TORCH_ATTRS:
            fn = getattr(torch, name)
            assert fn is not du._XPU_INDEX_ORIG["torch." + name]
            assert pickle.loads(pickle.dumps(fn)) is fn
        for name in du._XPU_INDEX_TENSOR_ATTRS:
            fn = getattr(torch.Tensor, name)
            assert pickle.loads(pickle.dumps(fn)) is fn
        x = base((6, 4))  # CPU tensors pass through every wrapper, including the two added ones
        assert torch.index_put_(x.clone(), (T([1]),), torch.ones(4))[1].tolist() == [1.0] * 4
        assert torch.equal(torch.combinations(x[:, 0], r=3), du._XPU_INDEX_ORIG["torch.combinations"](x[:, 0], 3))
    finally:
        if not was_installed:
            du.uninstall_xpu_index_workaround()


def test_torch_index_put__function_is_rewritten():
    """Finding: torch.index_put_ (module-level) went straight to aten::index_put_ (backward: aten::index)."""
    x = base((6, 4))
    idx, values = T([3, 0, 5, 3]), base((4, 4)) + 50.0
    xd = x.to(DEV)
    assert torch.index_put_(xd, (T([3, 0, 5, 1]).to(DEV),), values.to(DEV)) is xd
    assert torch.equal(xd.cpu(), x.clone().index_put_((T([3, 0, 5, 1]),), values))
    xd = x.to(DEV)
    torch.index_put_(input=xd, indices=(idx.to(DEV),), values=values.to(DEV), accumulate=True)
    assert torch.equal(xd.cpu(), x.clone().index_put_((idx,), values, accumulate=True))
    grads = []
    for device in ("cpu", DEV):
        w = base((6, 4)).to(device).requires_grad_(True)
        v = values.clone().to(device).requires_grad_(True)
        out = torch.index_put_(w * 2.0, (idx.to(device),), v * 3.0, accumulate=True)
        (out * out).sum().backward()
        grads.append((w.grad.cpu(), v.grad.cpu()))
    torch.xpu.synchronize()
    assert torch.equal(grads[0][0], grads[1][0]) and torch.equal(grads[0][1], grads[1][1])


def test_combinations_runs_on_the_cpu_and_warns_once(monkeypatch):
    """Finding: torch.combinations calls aten::index from C++ (access violation on this device)."""
    monkeypatch.setattr(du, "_XPU_INDEX_WARNED", set())
    v = torch.arange(5.0) * 1.5
    with pytest.warns(RuntimeWarning, match="torch.combinations"):
        out = torch.combinations(v.to(DEV))
    assert out.device.type == DEV and torch.equal(out.cpu(), torch.combinations(v))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = torch.combinations(v.to(DEV), r=3, with_replacement=True)
    assert torch.equal(out.cpu(), torch.combinations(v, r=3, with_replacement=True))
    w = v.to(DEV).requires_grad_(True)
    torch.combinations(w * 2.0, 2).prod(1).sum().backward()
    wc = v.clone().requires_grad_(True)
    torch.combinations(wc * 2.0, 2).prod(1).sum().backward()
    torch.xpu.synchronize()
    assert torch.equal(w.grad.cpu(), wc.grad)


def test_tripwire_covers_unsafe_and_out_overloads():
    xd = base((6, 4)).to(DEV)
    idx = T([0, 1]).to(DEV)
    calls = [
        lambda: torch.ops.aten._unsafe_index.Tensor(xd, [idx]),
        lambda: torch.ops.aten._unsafe_index_put(xd, [idx], torch.ones(2, 4, device=DEV), False),
        lambda: torch.ops.aten.index.Tensor_out(xd, [idx], out=torch.empty(0, device=DEV)),
        lambda: du._XPU_INDEX_ORIG["torch.masked_select"](xd, xd > 0, out=torch.empty(0, device=DEV)),
        lambda: du._XPU_INDEX_ORIG["torch.index_put_"](xd, (idx,), torch.ones(2, 4, device=DEV)),
    ]
    for call in calls:
        with pytest.raises(RuntimeError, match="tripwire"):
            call()
    torch.xpu.synchronize()
    assert torch.equal((xd + 0).cpu(), base((6, 4)))


def test_tripwire_is_enabled_per_thread_and_disabled_from_any_thread():
    """Finding: enable() claimed coverage on a thread without a mode; disable() from another thread raised."""
    xd = base((6, 4)).to(DEV)
    idx = T([0, 1]).to(DEV)
    native_get = du._XPU_INDEX_ORIG["Tensor.__getitem__"]
    depth = torch._C._len_torch_dispatch_stack
    main_depth = depth()
    seen = {}

    def other_thread():
        seen["depth_before"] = depth()
        seen["enabled"] = du.enable_xpu_index_tripwire()
        seen["depth_after"] = depth()
        if seen["enabled"] and seen["depth_after"] == seen["depth_before"] + 1:
            try:
                native_get(xd, idx)
                seen["raised"] = False
            except RuntimeError as exc:
                seen["raised"] = "tripwire" in str(exc)
        try:
            du.disable_xpu_index_tripwire()
            seen["disable_error"] = None
        except Exception as exc:  # noqa: BLE001
            seen["disable_error"] = exc
        seen["depth_end"] = depth()

    t = threading.Thread(target=other_thread)
    t.start()
    t.join()
    assert seen["enabled"] and seen["depth_before"] == 0 and seen["depth_after"] == 1  # really entered there
    assert seen["raised"] is True
    assert seen["disable_error"] is None and seen["depth_end"] == 0
    # switched off for every thread: this thread's mode is still on the stack, as a pass-through
    assert du._XPU_INDEX_TRIPWIRE is None and depth() == main_depth
    assert torch.equal(xd[idx].cpu(), base((6, 4))[:2])
    assert du.enable_xpu_index_tripwire() and depth() == main_depth  # re-armed, not pushed twice
    with pytest.raises(RuntimeError, match="tripwire"):
        native_get(xd, idx)
    du.disable_xpu_index_tripwire()
    assert depth() == main_depth - 1
    du.disable_xpu_index_tripwire()  # idempotent
    torch.xpu.synchronize()


def test_basic_keys_never_enter_the_rewriter(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("basic indexing reached the rewriter")

    xd = base((5, 6, 4)).to(DEV)
    monkeypatch.setattr(du, "_xpu_getitem", boom)
    monkeypatch.setattr(du, "_xpu_setitem", boom)
    for key in (2, -1, slice(1, 3), Ellipsis, None, (S_, 1), (1, slice(None, None, 2), -1), (None, Ellipsis, 0), (Ellipsis, slice(0, 2))):
        assert xd[key]._base is xd
        xd[key] = 1.0
        xd[key] = xd[key] * 2.0
    with pytest.raises(AssertionError, match="rewriter"):
        xd[T([0])]


def _device_tensors():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(90, 4, generator=g)
    big = torch.randn(64, 500, generator=g)
    idx = torch.randint(0, 90, (16,), generator=g)
    b, i = torch.arange(64).unsqueeze(1), torch.randint(0, 500, (64, 5), generator=g)
    return x, big, idx, b, i


@pytest.mark.parametrize("bounds", [False, True])
def test_device_kernel_counts(monkeypatch, bounds):
    """Findings: what each rewritten form costs on the device; no host transfer with the bounds check off."""
    monkeypatch.setattr(du, "_XPU_INDEX_CHECK_BOUNDS", bounds)
    x, big, idx, b, i = _device_tensors()
    xd, bigd, idxd, bd, idd = (t.to(DEV) for t in (x, big, idx, b, i))
    rows, cols = torch.arange(64, device=DEV), idd[:, 0].contiguous()
    keep = torch.ones(64, 500, dtype=torch.bool, device=DEV)
    vals = torch.randn(16, 4).to(DEV)
    wrap = [] if bounds else ["remainder"]  # unchecked indices are wrapped into range, one kernel each
    expected = [
        ("x[idx]", lambda: xd[idxd], wrap + ["index_select"], 1),
        ("x[:, idx]", lambda: xd[:, idxd % 4], ["remainder"] + wrap + ["index_select"], 1),
        ("X[b, i]", lambda: bigd[bd, idd], (["cat"] if bounds else wrap * 2) + ["add", "index_select"], 1),
        ("x[idx] = v", lambda: xd.__setitem__(idxd, vals), wrap + ["index_copy_"], 1),
        ("x[idx] = 0.0", lambda: xd.__setitem__(idxd, 0.0), wrap + ["index_fill_"], 1),
        ("m[arange, t] = False", lambda: keep.__setitem__((rows, cols), False),
         (["cat"] if bounds else wrap * 2) + ["add", "index_fill_"], 1),
    ]
    for name, fn, kernels, downloads in expected:
        got = _trace(fn, DEV)
        assert got == (kernels, downloads if bounds else 0, 0), (name, got)
    # masks never need the range check: mask + scalar is a single kernel without any transfer
    mask = bigd > 0
    for value in (float("inf"), torch.tensor(2.0, device=DEV), torch.tensor(3)):
        got = _trace(lambda: bigd.__setitem__(mask, value), DEV)
        assert got == (["masked_fill_"], 0, 0), got
    src = torch.randn(64 * 500).to(DEV)
    got = _trace(lambda: bigd.masked_scatter_(mask, src), DEV)
    assert got == (["index_copy_"], 1, 0), got  # one mask download for the host nonzero, written in place
    torch.xpu.synchronize()
    ref = big.clone()
    cm = (ref > 0)
    ref[cm] = float("inf")
    ref[cm] = 2.0
    ref[cm] = 3.0
    ref.masked_scatter_(cm, src.cpu())
    assert torch.equal(bigd.cpu(), ref)
