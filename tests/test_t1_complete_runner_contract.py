import json
from pathlib import Path

import torch
from model.t1.data import T1Data
from tools.t1_run import Runner, load_config


def test_regression_reuse_is_wired_to_main_runner(tmp_path):
    cfg = load_config(Path("configs/t1/p2_reuse.yaml"))
    cfg["core"].update(epochs=2, patience=2, batch_size=3, top_k=3, bias_init_k=3)
    cfg["reuse"].update(epochs=2, patience=2, batch_size=3, hidden_dims=[8, 4])
    cfg["conditions"] = [{"output_mode": "nominal_residual_scores", "loss": "combined"}]
    x = torch.tensor([[0.], [.1], [.3], [.8], [.9], [1.2]])
    y = x.flatten()
    data = T1Data("tiny", "regression", x, y, x + .01, y, x + .02, y)
    runner = Runner(cfg, tmp_path)
    runner.data_for = lambda *args: data
    runner.run_reuse(dict(name="tiny", task_type="regression"), 1)
    manifest = json.loads(next(tmp_path.rglob("run_manifest.json")).read_text())
    assert "aggregate_regression" in manifest["components"]["reuse_or_adapter"]
    metrics = json.loads(next(tmp_path.rglob("metrics.json")).read_text())
    assert metrics["test_rmse_post"] >= 0


def test_dataset_roles_have_separate_artifact_names(tmp_path):
    cfg = load_config(Path("configs/t1/p2_reuse.yaml"))
    cfg["datasets"] = [dict(name="wine", task_type=task) for task in ("classification", "regression")]
    runner = Runner(cfg, tmp_path)
    assert runner.dataset_key(cfg["datasets"][0]) == "classification__wine"
    assert runner.dataset_key(cfg["datasets"][1]) == "regression__wine"
