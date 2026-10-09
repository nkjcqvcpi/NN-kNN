"""Both sides of post-selection compression retain a restorable predictor contract."""
from pathlib import Path
import copy
from types import SimpleNamespace

import torch

from tools.t1_run import Runner, load_config
from model.common.core import CoreConfig, build_model, make_optimizer
from model.nn_cdh import NNCDHAdapter


def test_before_and_after_metadata_and_frozen_pair_optimizer_restore(tmp_path):
    cfg = load_config(Path("configs/t1/alignment_sync_capacity_match.yaml"))
    cfg["core"].update(epochs=2, patience=2, batch_size=3, top_k=3, bias_init_k=3)
    cfg["sync"].update(epochs=4, block_epochs=1, maintenance_epochs=[2], patience=4)
    cfg["conditions"] = [{"schedule": "alternating_rr"}]
    cfg["statistics"]["regression_success_tolerance"] = .5
    cfg["retention"]["min_per_cohort"] = 0
    X = torch.tensor([[0.], [.1], [.3], [.8], [.9], [1.2]])
    y = torch.tensor([0., .1, .2, .8, 1., 1.2])
    from model.t1.data import T1Data
    data = T1Data(name="tiny_regression", task_type="regression", X_train=X, y_train=y,
        X_val=X + .01, y_val=y, X_test=X + .02, y_test=y, reg_bins=None)
    runner = Runner(cfg, tmp_path / "runs")
    runner.data_for = lambda ds, seed: data
    runner.run_sync({"name": data.name, "task_type": "regression"}, 5)
    path = next((tmp_path / "runs").rglob("checkpoint.pt"))
    before = torch.load(path.parent / "before_capacity_match.pt", weights_only=True)
    after = torch.load(path, weights_only=True)
    assert before["active_case_count"] == 6 and after["active_case_count"] == 4
    assert before["model_description"] == after["model_description"]
    desc = before["model_description"]
    assert desc["near_scale"] > 0 and desc["free_radius_selected"] is not None
    assert before["optimizer_checkpoint_epoch"] == after["optimizer_checkpoint_epoch"]
    assert before["adapter_requires_grad"] == after["adapter_requires_grad"]
    assert any(not value for value in after["adapter_requires_grad"].values())
    for ck in (before, after):
        cc = CoreConfig(**ck["model_description"]["core"])
        m = build_model(X, y, cc, None)
        m.set_active_case_count(ck["active_case_count"])
        m.load_state_dict(ck["model_state"])
        for name, p in m.named_parameters(): p.requires_grad_(ck["requires_grad"][name])
        ad = NNCDHAdapter(1, 1, tuple(desc["sync"]["hidden_dims"]))
        ad.load_state_dict(ck["adapter_state"])
        for name, p in ad.named_parameters(): p.requires_grad_(ck["adapter_requires_grad"][name])
        ropt = make_optimizer(m, cc)
        ropt.load_state_dict(copy.deepcopy(ck["optimizer_state"]))
        aopt = torch.optim.Adam([p for p in ad.parameters() if p.requires_grad])
        aopt.load_state_dict(copy.deepcopy(ck["adapter_optimizer_state"]))
        assert len(aopt.param_groups[0]["params"]) == len(ck["adapter_optimizer_state"]["param_groups"][0]["params"])
