import copy

import torch

from model.nnknn_rl_workflow import (
    NNKNNValueNetwork, _align_nnknn_target_case_store,
    _build_nnknn_target_value_model, _sync_nnknn_target_value_model,
)


def critic():
    m = NNKNNValueNetwork(2, case_capacity=3, top_k=3)
    m.add_cases(torch.tensor([[1., 2.], [3., 4.]]), torch.tensor([10., 20.]))
    return m


def assert_ids(m, expected):
    assert m.active_case_ids().tolist() == expected
    assert m.nnknn_model.active_case_ids().tolist() == expected
    assert torch.equal(m.case_ids, m.nnknn_model.case_ids)
    assert torch.equal(m.next_case_id, m.nnknn_model.next_case_id)


def test_target_compaction_preserves_lagged_values_by_retrieval_identity():
    m = critic()
    t = _build_nnknn_target_value_model(m, device=torch.device('cpu'))
    assert_ids(m, [0, 1])
    with torch.no_grad():
        t.nnknn_model.labels[:2].copy_(torch.tensor([[100.], [200.]]))
        t.nnknn_model.biases[:2].copy_(torch.tensor([4., 8.]))
    m._compact_cases([1])
    m.add_cases(torch.tensor([[5., 6.]]), torch.tensor([30.]))
    _align_nnknn_target_case_store(m, t)
    assert_ids(m, [1, 2])
    assert_ids(t, [1, 2])
    assert t.nnknn_model.cases.data_ptr() == m.nnknn_model.cases.data_ptr()
    assert t.nnknn_model.labels.data_ptr() != m.nnknn_model.labels.data_ptr()
    assert t.nnknn_model.labels[:2].flatten().tolist() == [200., 30.]
    assert t.nnknn_model.biases[0].item() == 8.
    r = t.nnknn_model.retrieve(torch.tensor([[3., 4.]]), query_case_ids=torch.tensor([1]))
    ids = t.nnknn_model.case_ids[r['case_indices']]
    assert r['weights'][0, ids == 1].sum().item() == 0.
    _sync_nnknn_target_value_model(m, t, mode='ema', ema_tau=0.5)
    assert_ids(t, [1, 2])
    assert t.nnknn_model.labels[:2].flatten().tolist() == [110., 30.]


def test_loading_legacy_checkpoint_repairs_core_ids_without_changing_values():
    m = critic()
    state = copy.deepcopy(m.state_dict())
    state['nnknn_model.case_ids'] = torch.tensor([3, 4, 2])
    state['nnknn_model.next_case_id'] = torch.tensor(5)
    restored = NNKNNValueNetwork(2, case_capacity=3, top_k=3)
    restored.load_state_dict(state)
    assert restored.case_entries == 2
    assert_ids(restored, [0, 1])
    assert torch.equal(restored(torch.tensor([[2., 3.]])), m(torch.tensor([[2., 3.]])))
    restored._compact_cases([1])
    restored.add_cases(torch.tensor([[5., 6.]]), torch.tensor([30.]))
    assert_ids(restored, [1, 2])


def test_full_capacity_replacement_never_reuses_active_ids():
    m = critic()
    m.add_cases(torch.tensor([[5., 6.], [7., 8.], [9., 10.]]), torch.tensor([30., 40., 50.]))
    ids = m.active_case_ids().tolist()
    assert len(ids) == len(set(ids)) == 3
    assert_ids(m, ids)
    assert m.next_case_id.item() == 5
