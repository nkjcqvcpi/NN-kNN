import pytest
import torch

from model.device_utils import install_xpu_index_workaround, is_xpu_available

pytestmark = pytest.mark.skipif(not is_xpu_available(), reason="no XPU device")


@pytest.mark.parametrize("shape", [(90, 1), (1000, 5), (5, 90), (16, 1), (90, 8)])
def test_integer_indexing_matches_cpu(shape):
    assert install_xpu_index_workaround()
    c = torch.randn(*shape)
    idx = torch.arange(shape[0])
    assert torch.equal(c.to("xpu")[idx.to("xpu")].cpu(), c[idx])
    assert torch.equal(c.to("xpu")[idx].cpu(), c[idx])  # CPU index into an XPU tensor


def test_boolean_masks_and_assignment_match_cpu():
    assert install_xpu_index_workaround()
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
