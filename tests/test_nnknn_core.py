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
