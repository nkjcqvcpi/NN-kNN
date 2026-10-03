"""Run manifests and event artifacts (shared contract sections 2-6)."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    return o


def config_id(cfg: dict[str, Any]) -> str:
    blob = json.dumps(_jsonable(cfg), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def git_state(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return ""

    dirty = run("status", "--porcelain", "--", "model", "tools", "tests", "datasets", "configs")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"), "dirty_code_worktree": bool(dirty), "remote": run("remote", "get-url", "origin")}


def environment_info() -> dict[str, Any]:
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "torch_threads": torch.get_num_threads(),
        "environment_lock": "uv.lock",
    }


def source_fingerprint(repo: Path) -> str:
    """Identify dirty local implementations as well as committed versions."""
    digest = hashlib.sha256()
    for directory in ("model", "tools", "configs", "tests", "datasets"):
        for path in sorted((repo / directory).rglob("*")):
            if path.suffix not in {".py", ".yaml", ".ps1"} or "__pycache__" in path.parts:
                continue
            digest.update(path.relative_to(repo).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2, sort_keys=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(_jsonable(r)) + "\n")


def build_manifest(*, run_id: str, repo: Path, cfg: dict[str, Any], data_desc: dict[str, Any], components: dict[str, Any], budgets: dict[str, Any], maintenance: dict[str, Any], evaluation: dict[str, Any], model_desc: dict[str, Any]) -> dict[str, Any]:
    return {
        "run": {
            "id": run_id,
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "code_repository": str(repo),
            **git_state(repo),
            "source_fingerprint_sha256": source_fingerprint(repo),
            "specification_commit": cfg.get("config", {}).get("specification_commit"),
            "pipeline_revision": cfg.get("config", {}).get("pipeline_revision"),
            "environment": environment_info(),
            "seed": cfg.get("seed"),
            "configuration_id": config_id(cfg),
        },
        "task": data_desc,
        "model": model_desc,
        "components": components,
        "budgets": budgets,
        "maintenance": maintenance,
        "evaluation": evaluation,
        "resolved_config": cfg,
    }


def case_statistics_rows(model, store, archive, scores=None) -> list[dict[str, Any]]:
    rows = []
    active = set(int(c) for c in model.active_case_ids().tolist())
    for cid, st in store.items():
        d = st.to_dict()
        d["archive_state"] = "active" if cid in active else ("archived" if cid in archive.entries else "unknown")
        rows.append(d)
    return rows
