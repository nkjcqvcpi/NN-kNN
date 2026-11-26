"""Download and check the pinned TabPFN v2 checkpoints of the grid's ``tabpfn`` family (model/t1/tabpfn_baseline.py).

    python tools/t1_tabpfn_weights.py [--check-only]

Writes ``tabpfn-v2-classifier.ckpt`` and ``tabpfn-v2-regressor.ckpt`` into ``model.t1.tabpfn_baseline.weights_dir()``
(``<NN-KNN root>/models/tabpfn``, or ``NNKNN_TABPFN_DIR``) from Hugging Face (Prior-Labs/TabPFN-v2-clf, -reg; ungated),
then checks each file's sha256 against the pinned value and the installed tabpfn version. A present file is not
downloaded again. Exit 0 only when both checkpoints are present and match. Trains and records nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from model.t1 import tabpfn_baseline as tp  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check-only", action="store_true", help="do not download; only check what is there")
    args = ap.parse_args(argv)
    target = tp.weights_dir()
    report = {"weights_dir": str(target), "checkpoints": {}}
    ok = True
    try:
        report["tabpfn_version"] = tp.check_version()
    except Exception as exc:  # noqa: BLE001 - reported, exit 1
        report["tabpfn_version_error"] = str(exc)
        ok = False
    for task, (repo_id, name, expected) in tp.CHECKPOINTS.items():
        path = target / name
        entry = {"file": str(path), "expected_sha256": expected}
        if not path.is_file() and not args.check_only:
            from huggingface_hub import hf_hub_download
            target.mkdir(parents=True, exist_ok=True)
            hf_hub_download(repo_id=repo_id, filename=name, local_dir=str(target))
        try:
            entry["sha256"] = tp.checkpoint(task)[1]
            entry["ok"] = True
        except Exception as exc:  # noqa: BLE001 - reported, exit 1
            entry["error"] = str(exc)
            entry["ok"] = False
            ok = False
        report["checkpoints"][task] = entry
    report["ok"] = ok
    print(json.dumps(report, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
