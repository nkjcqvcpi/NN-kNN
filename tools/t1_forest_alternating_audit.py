"""Independent saved-prediction audit; optionally prove completed-unit reuse."""
import argparse
from collections import Counter
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
from model.common import tracking
from model.t1 import records
from tools import t1_forest_alternating as d


def measured(saved, kind):
    final, target = np.asarray(saved["final"]), np.asarray(saved["target"])
    if kind == "classification":
        error = float(np.count_nonzero(final.argmax(1) != target) / len(target))
        loss = float(-np.log(final[np.arange(len(target)), target.astype(int)].clip(1e-12, 1.)).mean())
    else:
        loss = float(((final.reshape(-1) - target.reshape(-1)) ** 2).mean())
        error = float(np.sqrt(loss))
    return error, loss


def forbid(*args, **kwargs):
    raise AssertionError("complete study unexpectedly attempted fitting, inference, or data reconstruction")


def audit(study, tasks, seeds, *, reuse=False):
    checks, rows = Counter(), []
    for task in tasks:
        kind = task.split("__")[0]
        for seed in seeds:
            context_unit, context = d.a.source_context(task, seed)
            for family in d.FAMILIES:
                source = d.source_family(task, seed, family, context_unit)
                options = []
                for cycle in range(d.PROTOCOL["cycles"] + 1):
                    candidate = source["candidate"] if cycle == 0 else tracking.load_unit(
                        d.key_for(study, task, seed, f"f{d.TAGS[family]}{cycle}"), config=None, arrays=False)
                    if candidate is None:
                        raise AssertionError("missing completed candidate")
                    saved = d.p.numpy_state(candidate.load_state("val_predictions"))
                    np.testing.assert_array_equal(saved["target"], context["val"]["target"])
                    error, loss = measured(saved, kind)
                    row = records.record(candidate, "row")
                    if not np.allclose([error, loss], [row["val_error"], row["val_task_loss"]], rtol=1e-12, atol=1e-12):
                        raise AssertionError("independent validation metric mismatch")
                    options.append((error, loss, cycle))
                    checks["validation_metric_pairs"] += 1
                chosen = min(options)[2]
                unit = tracking.load_unit(d.key_for(study, task, seed, "e" + d.TAGS[family]), config=None, arrays=False)
                if unit is None or unit.config["protocol"] != d.PROTOCOL:
                    raise AssertionError("matching complete selection required")
                row = records.record(unit, "row")
                if row["selected_cycle"] != chosen:
                    raise AssertionError("independent validation choice differs")
                saved = d.p.numpy_state(unit.load_state("test_predictions"))
                before = d.p.numpy_state(source["selected"].load_state("test_predictions"))
                np.testing.assert_array_equal(saved["target"], before["target"])
                error, loss = measured(saved, kind)
                if not np.allclose([error, loss], [row["test_error"], row["test_task_loss"]], rtol=1e-12, atol=1e-12):
                    raise AssertionError("independent test metric mismatch")
                checks["selected_test_metric_pairs"] += 1
                checks["independent_validation_choices"] += 1
                rows.append(dict(task=task, task_type=kind, seed=seed, family=family, selected_cycle=chosen,
                    alternating_error=error, sequential_error=measured(before, kind)[0]))
            if reuse:
                with ExitStack() as stack:
                    for target, name in ((d, "train_retrieval_block"), (d, "fit_forest"), (d, "predict_pairs"),
                            (d, "extract_context"), (d.p.s.n, "load_data")):
                        stack.enter_context(patch.object(target, name, forbid))
                    with redirect_stdout(io.StringIO()):
                        d.run_split(study, task, seed)
                checks["completed_units_reused_without_fit_or_inference"] += len(d.FAMILIES) * (2 * d.PROTOCOL["cycles"] + 1)
            print(f"[audit] {task} s{seed}", flush=True)
    means = []
    for kind in ("classification", "regression"):
        for family in d.FAMILIES:
            group = [r for r in rows if r["task_type"] == kind and r["family"] == family]
            if not group:
                continue
            names = sorted({r["task"] for r in group})
            task_means = [{key: float(np.mean([r[key] for r in group if r["task"] == task]))
                for key in ("sequential_error", "alternating_error")} for task in names]
            means.append(dict(task_type=kind, family=family,
                **{key: float(np.mean([r[key] for r in task_means])) for key in task_means[0]},
                task_wins=sum(r["alternating_error"] < r["sequential_error"] - 1e-8 for r in task_means),
                task_losses=sum(r["alternating_error"] > r["sequential_error"] + 1e-8 for r in task_means),
                seed_wins=sum(r["alternating_error"] < r["sequential_error"] - 1e-8 for r in group),
                seed_losses=sum(r["alternating_error"] > r["sequential_error"] + 1e-8 for r in group),
                cycle_zero=sum(r["selected_cycle"] == 0 for r in group)))
    out = REPO / "results" / "adaptation_improvement" / study
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(study=study, checks=dict(checks), means=means, rows=rows)
    (out / "alternating_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(checks=dict(checks), means=means), indent=2), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--tasks", nargs="+", choices=d.p.q.prior.TASKS, default=list(d.p.q.prior.TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(d.p.q.prior.SEEDS))
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    records.check_study(args.study)
    return audit(args.study, args.tasks, args.seeds, reuse=args.reuse)


if __name__ == "__main__":
    raise SystemExit(main())
