import pytest
import torch
from model.common.core import CoreConfig, build_model, kth_neighbor_learned_distance


def test_initializer_counts_distinct_duplicate_neighbors():
    x = torch.tensor([[0.], [0.], [1.], [1.], [2.], [2.]])
    model = build_model(x, torch.arange(6).float(), CoreConfig(task_type="regression", bias_init_k=1), None)
    assert kth_neighbor_learned_distance(model, 1) == 0.
    assert model.t1_bias_init["value"] == 0.
    with pytest.raises(ValueError, match="positive"):
        kth_neighbor_learned_distance(model, 0)
