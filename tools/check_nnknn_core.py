"""Verify that the shared NN-kNN core is byte-identical to its manifest.

The core is maintained on ``main`` and copied unchanged to the neural-cbr, rl
and LLM branches. Each branch carries ``tools/nnknn_core_manifest.json``; this
tool compares the working tree (or other branches) against it.

    python tools/check_nnknn_core.py                    # check this checkout
    python tools/check_nnknn_core.py --branches main neural-cbr rl LLM
    python tools/check_nnknn_core.py --write            # main only: refresh manifest

Hashes are SHA-256 of the file bytes with CRLF normalized to LF, so a Windows
checkout with ``core.autocrlf`` still matches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tools" / "nnknn_core_manifest.json"

# The shared core: model, adaptation, device handling, construction/geometry/
# artifact helpers, the supervised classification/regression workflows and
# their loaders, the supervised benchmark tools, and this contract itself.
CORE_FILES = (
    "model/__init__.py",
    "model/nnknn_model.py",
    "model/nn_cdh.py",
    "model/device_utils.py",
    "model/feature_extractors.py",
    "model/common/__init__.py",
    "model/common/artifacts.py",
    "model/common/core.py",
    "model/common/geometry.py",
    "model/common/nominal.py",
    "model/common/outcomes.py",
    "model/classification_workflow.py",
    "model/regression_workflow.py",
    "model/mlkr.py",
    "datasets/__init__.py",
    "datasets/classification_data.py",
    "datasets/cls_small_data.py",
    "datasets/reg_data.py",
    "tools/run_classification_benchmarks.py",
    "tools/run_table1_kfold.py",
    "tools/resume_table1_kfold.py",
    "tools/table1_nnknn_kfold.py",
    "tools/check_nnknn_core.py",
    "tests/test_nnknn_core.py",
    "docs/NNKNN_CORE.md",
)

# Stage packages and stage workflows that core files must never import.
STAGE_MODULES = (
    "model.t1",
    "model.t2",
    "model.t3",
    "model.rl_workflow",
    "model.nec_workflow",
    "model.nnknn_rl_workflow",
    "model.ppo_workflow",
    "model.td3_workflow",
    "model.cnn_encoders",
    "datasets.rl_tasks",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def working_tree_hashes(root: Path = ROOT) -> dict[str, str | None]:
    return {
        path: digest((root / path).read_bytes()) if (root / path).is_file() else None
        for path in CORE_FILES
    }


def branch_hashes(branch: str, root: Path = ROOT) -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    for path in CORE_FILES:
        result = subprocess.run(
            ["git", "show", f"{branch}:{path}"], cwd=root, capture_output=True
        )
        hashes[path] = digest(result.stdout) if result.returncode == 0 else None
    return hashes


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def compare(expected: dict[str, str], actual: dict[str, str | None]) -> list[str]:
    problems = []
    for path in CORE_FILES:
        want, have = expected.get(path), actual.get(path)
        if want is None:
            problems.append(f"{path}: not listed in manifest")
        elif have is None:
            problems.append(f"{path}: missing")
        elif have != want:
            problems.append(f"{path}: differs from manifest")
    for path in sorted(set(expected) - set(CORE_FILES)):
        problems.append(f"{path}: listed in manifest but not in CORE_FILES")
    return problems


def write_manifest() -> None:
    hashes = working_tree_hashes()
    missing = [path for path, value in hashes.items() if value is None]
    if missing:
        raise SystemExit(f"cannot write manifest; missing core files: {missing}")
    payload = {
        "description": "SHA-256 (CRLF normalized to LF) of the shared NN-kNN core. "
        "Edit the core on main, refresh with --write, then copy the core files and "
        "this manifest unchanged to neural-cbr, rl and LLM.",
        "origin_branch": "main",
        "files": hashes,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)} ({len(hashes)} files)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--branches", nargs="+", help="check these local branches instead of the working tree")
    parser.add_argument("--write", action="store_true", help="refresh the manifest from this working tree")
    args = parser.parse_args()
    if args.write:
        write_manifest()
        return 0
    expected = load_manifest()["files"]
    targets = {name: branch_hashes(name) for name in args.branches} if args.branches else {"working tree": working_tree_hashes()}
    failed = False
    for name, actual in targets.items():
        problems = compare(expected, actual)
        failed |= bool(problems)
        print(f"{name}: {'OK' if not problems else f'{len(problems)} problem(s)'}")
        for problem in problems:
            print(f"  - {problem}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
