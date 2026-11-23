import json
from pathlib import Path

import torch

from tools.t1_run import load_config
from model.t1 import adapted_retraining

from t1_units import study_runner, unit


def test_actual_paired_reference_routes_share_initial_state_and_record_denominators(tmp_path, monkeypatch):
    cfg = load_config(Path("configs/t1/alignment_adapted_reference_guard.yaml"))
    cfg["experiment"] = "reference_route_test"
    cfg["core"].update(epochs=2, patience=2)
    cfg["reuse"].update(epochs=2, patience=2)
    cfg["K"] = [17]
    cfg["seeds"] = [8]
    cfg["conditions"] = [dict(scope="train", retrieval_epochs=1, adapter_epochs=0, reference_stream="train_loo"),
                         dict(scope="maintenance", retrieval_epochs=1, adapter_epochs=0, reference_stream="maintenance_split")]
    ds = dict(name="iris", task_type="classification", max_train=18)
    calls = []
    original = adapted_retraining.run_adapted_retrained_removal
    def wrapped(*args, **kwargs):
        reference = args[7]
        calls.append(dict(stream=reference.stream, denominator=len(reference.y), query_ids=reference.query_case_ids))
        return original(*args, **kwargs)
    monkeypatch.setattr(adapted_retraining, "run_adapted_retrained_removal", wrapped)
    study_runner(cfg, tmp_path, monkeypatch).run_adapted_retrained_removal(ds, 8)
    assert [c["stream"] for c in calls] == ["train_loo", "maintenance_split"]
    assert calls[0]["denominator"] == 18 and calls[0]["query_ids"].tolist() == list(range(18))
    assert calls[1]["denominator"] != 18 and calls[1]["query_ids"] is None
    starts = []
    for scope in ("train", "maintenance"):
        run = unit(tmp_path / "runs", f"{scope}_K17")
        starts.append(run.torch("initial.pt"))
        manifest = run.json("run_manifest.json")
        trace = run.json("maintenance_scoring_trace.json")
        expected = "train_loo" if scope == "train" else "maintenance_split"
        assert manifest["maintenance"]["reference_stream"] == expected
        assert trace and all(r["reference_stream"] == expected for r in trace)
        assert all(r["reference_denominator"] == calls[0 if scope == "train" else 1]["denominator"] for r in trace)
    for key in ("model_state", "adapter_state"):
        assert all(torch.equal(v, starts[1][key][n]) for n, v in starts[0][key].items())
    # The global defaults are not overwritten by a per-condition treatment.
    assert cfg["statistics"]["source"] == "train_loo"
