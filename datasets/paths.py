"""Where the shared NN-kNN data live.

Datasets and preprocessed caches are not tracked by git. One directory, shared
by every worktree (main, rl, llm, neural-cbr), holds them: by default the
``datasets`` directory next to the worktrees (``<NN-KNN>/datasets``), or the
directory named by the ``NNKNN_DATASETS`` environment variable. This package
(``datasets/*.py``) only holds the loader code.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "NNKNN_DATASETS"

# <NN-KNN>/<worktree>/datasets/paths.py -> parents[2] is <NN-KNN>.
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "datasets"


def datasets_root() -> Path:
    """Return the shared data directory; raise if it does not exist.

    It never creates the directory, so a missing or mistyped location fails
    loudly instead of silently re-downloading data into a fresh empty folder.
    """
    env = os.environ.get(ENV_VAR, "").strip()
    root = Path(env).expanduser().resolve() if env else DEFAULT_ROOT
    if not root.is_dir():
        source = f"${ENV_VAR}" if env else f"the default (${ENV_VAR} is not set)"
        raise FileNotFoundError(
            f"NN-kNN data directory {root} does not exist (from {source}). "
            f"Create it with the shared datasets, or set {ENV_VAR} to the directory "
            "that holds them."
        )
    return root


def dataset_file(*parts: str) -> Path:
    """Path of a file under the shared data directory."""
    return datasets_root().joinpath(*parts)
