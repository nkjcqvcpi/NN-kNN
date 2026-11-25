"""Q-B agreement on fabricated retention units with hand-computed answers."""
import json
import math

import pandas as pd
import pytest

from t1_units import units_from_files
from tools.t1_qb_agreement import main, qb_stats

# five evidenced cases (Q ranks 5..1, B ranks 5,1,4,2,3: sum d^2 = 14, rho = 1 - 6*14/120 = 0.3)
# and two unevidenced cases with extreme values that must be ignored
Q = [0.9, 0.8, 0.7, 0.2, 0.1, 0.0, 1.0]
B = [0.9, 0.1, 0.8, 0.2, 0.3, 1.0, 0.0]
EV = [True] * 5 + [False] * 2


def test_spearman_and_median_quadrants():
    s = qb_stats(Q, B, EV)
    assert s["n_cases"] == 7 and s["n_evidenced"] == 5
    assert s["spearman"] == pytest.approx(0.3)
    # medians Q 0.7, B 0.3; high = strictly above
    assert (s["n_hiQ_hiB"], s["n_loQ_loB"], s["n_hiQ_loB"], s["n_loQ_hiB"]) == (1, 2, 1, 1)
    assert s["agree_share"] == pytest.approx(0.6)


def test_degenerate_inputs():
    assert math.isnan(qb_stats([0.5, 0.5, 0.5], [0.1, 0.2, 0.3], [True] * 3)["spearman"])
    s = qb_stats([0.1, 0.2], [0.3, 0.4], [False, False])
    assert s["n_evidenced"] == 0 and math.isnan(s["spearman"]) and math.isnan(s["agree_share"])


def write_unit(root, ds, cond, seed, q, b, ev):
    d = root / ds / cond / f"s{seed}"
    d.mkdir(parents=True)
    (d / "run_manifest.json").write_text(json.dumps({"run": {"seed": seed}}))
    events = [{"case_id": i, "cohort": "class_0", "Q": qi, "B": bi, "evidenced": e, "action": "keep"}
              for i, (qi, bi, e) in enumerate(zip(q, b, ev))]
    (d / "case_maintenance.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))


def test_cli_reads_units_and_averages_seeds(tmp_path):
    files = tmp_path / "files" / "retention__classification"
    write_unit(files, "toy", "full_memory_K0.5", 30, Q, B, EV)
    write_unit(files, "toy", "provenance_only_K0.5", 30, Q, B, EV)
    write_unit(files, "toy", "full_memory_K0.5", 31, Q[:5], Q[:5], EV[:5])  # identical columns: rho 1
    write_unit(files, "toy", "provenance_only_K0.5", 31, Q[:5], [x + 0.25 for x in Q[:5]], EV[:5])
    (files / "runs__toy.jsonl").write_text("".join(json.dumps(
        {"dataset": "toy", "condition": c, "seed": s, "task_type": "classification"}) + "\n"
        for s in (30, 31) for c in ("full_memory_K0.5", "provenance_only_K0.5")))
    tree = units_from_files(files, tmp_path / "runs")
    out = tmp_path / "out"
    assert main([str(tree), "--out", str(out)]) == 0
    per_seed = pd.read_csv(out / "qb_agreement_per_seed.csv")
    assert list(per_seed.seed) == [30, 31]
    assert per_seed.spearman.tolist() == pytest.approx([0.3, 1.0])
    assert per_seed.check_max_abs_dQ.tolist() == pytest.approx([0.0, 0.0])
    assert per_seed.check_max_abs_dB.tolist() == pytest.approx([0.0, 0.25])
    ds = pd.read_csv(out / "qb_agreement_by_dataset.csv")
    assert len(ds) == 1 and ds.n_seeds[0] == 2 and ds.task_type[0] == "classification"
    assert ds.spearman[0] == pytest.approx(0.65)
    assert "| classification | toy |" in (out / "qb_agreement.md").read_text()
