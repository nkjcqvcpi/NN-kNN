"""Query-chunked retrieval equals the single call (exact for raw features)."""
import pytest
import torch

import model.t1.chunked as chunked
from model.common.core import CoreConfig, build_model
from model.t1.chunked import chunk_rows, retrieve_chunked
from model.t1.data import T1Data
from model.t1.mcb import StabilityTracker
from model.t1.reuse import neighborhood_inputs
from model.t1.sync import _forward_terms
from model.nn_cdh import ClassificationNNCDHAdapter


def toy(representation="raw", n=40):
    g = torch.Generator().manual_seed(0)
    X = torch.randn(n, 6, generator=g)
    y = (X[:, 0] + X[:, 1] > 0).long()
    cc = CoreConfig(task_type="classification", representation=representation, top_k=5, glocal_fw_set_num=2)
    model = build_model(X, y, cc, 2)
    model.eval()
    return model, X, y


def test_under_budget_is_the_plain_call():
    model, X, _ = toy()
    assert chunk_rows(model, len(X)) is None
    r = retrieve_chunked(model, X, exclude_identical=False)
    assert r["feature_distance_contributions"] is not None and "chunk_rows" not in r


def test_raw_chunks_reproduce_the_single_call_exactly_including_leave_one_out():
    model, X, _ = toy()
    ids = torch.arange(len(X))
    with torch.no_grad():
        full = model.retrieve(X, exclude_identical=False, query_case_ids=ids)
        part = retrieve_chunked(model, X, exclude_identical=False, query_case_ids=ids, budget=7 * model.case_count() * 6)
    assert part["chunk_rows"] == 7
    for key in ("weights", "distances", "query_features", "excluded"):
        assert torch.equal(part[key], full[key]), key
    assert torch.equal(part["case_indices"], full["case_indices"]) and part["feature_distance_contributions"] is None


def test_mlp_chunks_match_the_single_call():
    model, X, _ = toy("mlp")
    with torch.no_grad():
        full = model.retrieve(X, exclude_identical=False)
        part = retrieve_chunked(model, X, exclude_identical=False, budget=1)
    assert part["chunk_rows"] == 1
    for key in ("weights", "distances", "query_features"):
        assert torch.allclose(part[key], full[key], atol=1e-6), key


def test_evaluation_paths_use_chunks_above_the_budget(monkeypatch):
    model, X, y = toy()
    adapter = ClassificationNNCDHAdapter(6, 2, 0, (8, 4), "nominal_residual_scores")
    adapter.eval()
    with torch.no_grad():
        nb_full = neighborhood_inputs(model, X, exclude_identical=False)
        T_full = _forward_terms(model, adapter, X, y, None, output_mode="nominal_residual_scores", near_scale=1.0)
        monkeypatch.setattr(chunked, "RETRIEVE_BUDGET", 5 * model.case_count() * 6)
        assert chunk_rows(model, len(X)) == 5
        nb = neighborhood_inputs(model, X, exclude_identical=False)
        T = _forward_terms(model, adapter, X, y, None, output_mode="nominal_residual_scores", near_scale=1.0)
    for key in ("dz", "p0", "weights"):
        assert torch.equal(nb[key], nb_full[key]), key
    for key in T_full:
        assert torch.allclose(T[key], T_full[key], atol=1e-6), key
    tracker = StabilityTracker(X, k=3)
    assert tracker.snapshot(model, 0)["n_cases"] == len(X)
