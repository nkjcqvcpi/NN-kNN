"""Contract tests for the shared NN-kNN core (identical on every branch)."""
from __future__ import annotations

import ast
import importlib
import os
import sys

os.environ.setdefault("NNKNN_DEVICE", "cpu")

import pytest
import torch

from tools.check_nnknn_core import CORE_FILES, ROOT, STAGE_MODULES, compare, load_manifest, working_tree_hashes

CORE_PYTHON = [path for path in CORE_FILES if path.endswith(".py")]


def _imported_modules(path: str) -> set[str]:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):  # walk covers imports nested inside functions
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _is_stage(module: str) -> bool:
    return any(module == stage or module.startswith(stage + ".") for stage in STAGE_MODULES)


class _StageBlocker:
    """Meta-path finder that fails any stage import, even where the stage exists."""

    def find_spec(self, name, path=None, target=None):
        if _is_stage(name):
            raise ImportError(f"core code imported stage module {name}")
        return None


@pytest.fixture
def stage_blocked():
    saved = {name: module for name, module in sys.modules.items() if _is_stage(name)}
    for name in saved:
        del sys.modules[name]
    blocker = _StageBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)


def test_core_files_match_manifest():
    problems = compare(load_manifest()["files"], working_tree_hashes())
    assert not problems, "shared core drifted from tools/nnknn_core_manifest.json:\n" + "\n".join(problems)


@pytest.mark.parametrize("path", CORE_PYTHON)
def test_core_file_does_not_import_stage_code(path):
    stage = sorted(module for module in _imported_modules(path) if _is_stage(module))
    assert not stage, f"{path} imports stage modules {stage}"


def test_core_modules_import_without_stage_packages(stage_blocked):
    for module in ("model.nnknn_model", "model.nn_cdh", "model.common.core", "model.common.outcomes",
                   "model.common.nominal", "model.classification_workflow", "model.regression_workflow"):
        importlib.import_module(module)


def test_classification_adapter_forward_without_stage_packages(stage_blocked):
    from model.common.core import CoreConfig, build_model
    from model.nn_cdh import ClassificationNNCDHAdapter

    torch.manual_seed(0)
    X = torch.tensor([[0., 0.], [.1, .2], [.3, .4], [1., .9], [1.1, .8], [1.3, .6]])
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    model = build_model(X, y, CoreConfig(task_type="classification", top_k=2), 3)
    pre = model(X[:2] + .03)[0]
    model.classification_adapter = ClassificationNNCDHAdapter(2, 3, nominal_dim=0, hidden_dims=(8, 4))
    post = model(X[:2] + .03)[0]
    assert pre.shape == post.shape == (2, 3)
    assert torch.isfinite(post).all()


@pytest.mark.parametrize("normalize", [False, True])
def test_eval_feature_cache_matches_previous_loop_and_fills_only_requested_rows(normalize):
    import torch.nn.functional as F
    from model.common.core import CoreConfig, build_model

    torch.manual_seed(0)
    X = torch.randn(30, 5)
    y = torch.randint(0, 3, (30,))
    model = build_model(X, y, CoreConfig(task_type="classification", representation="mlp", top_k=4,
                                         normalize_embeddings=normalize), 3)
    model.eval()

    def encode(rows):
        f = model.feature_extractor(model.cases[rows])
        return F.normalize(f, p=2, dim=-1) if normalize else f

    def previous_loop(requests):  # the per-case loop this replaced
        cache = torch.zeros((30, model.feature_dim))
        out = []
        for part in requests:
            todo = [i.item() for i in part if cache[i].sum() == 0]
            if todo:
                cache[todo] = encode(todo)
            out.append(cache[part].clone())
        return out

    requests = [torch.tensor([3, 7, 7, 1]), torch.arange(30), torch.tensor([29, 0])]
    with torch.no_grad():
        reference = previous_loop(requests)
        model._invalidate_case_cache()  # build_model filled it for the bias initializer
        got = [model._extract_features(part).clone() for part in requests[:1]]
        assert (model.cached_features.sum(dim=-1) != 0).nonzero().view(-1).tolist() == [1, 3, 7]
        got += [model._extract_features(part).clone() for part in requests[1:]]
        assert all(torch.equal(g, r) for g, r in zip(got, reference))
        assert torch.allclose(got[1], encode(torch.arange(30)), atol=1e-6)
        model._invalidate_case_cache()
        assert model.cached_features is None


def test_build_model_puts_every_module_on_the_case_device_and_keeps_cpu_adam_default():
    from model.common.core import CoreConfig, build_model, make_optimizer

    torch.manual_seed(0)
    X = torch.randn(12, 4)
    y = torch.randn(12)
    cfg = CoreConfig(task_type="regression", representation="mlp", mcb_enabled=True, top_k=3)
    model = build_model(X, y, cfg, None)
    devices = {p.device for p in model.parameters()} | {b.device for b in model.buffers()}
    assert devices == {model.cases.device}
    opt = make_optimizer(model, cfg)
    if model.cases.device.type == "cpu":
        assert all(g.get("foreach") is None for g in opt.param_groups)
