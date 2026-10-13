"""Pre-registered confirmation analysis: known effects, REML, Holm and the refusals."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from tools.t1_confirm_report import (Evidence, Task, bh, compare, condition_id, holm, main, plan_tasks, pool,
                                     reml_tau2)

PREREG = Path("configs/t1/prereg_20261013.json")
SEEDS = list(range(40, 50))
LEGACY = [("iris", "classification"), ("wine", "classification"), ("wine", "regression"),
          ("yacht", "regression"), ("digits", "classification")]
TRANSFER = [("tabarena_363702", "classification"), ("tabarena_363705", "regression"),
            ("tabarena_363686", "regression")]
N_TEST = 2000
F = "tuned::joint_summed_w5"


def write_prereg(tmp_path):
    data = json.loads(PREREG.read_text(encoding="utf-8"))
    data["tasks"]["primary_legacy"] = [{"name": n, "task_type": t} for n, t in LEGACY]
    data["tasks"]["primary_transfer"] = [{"name": n, "task_type": t, "max_train": 3000} for n, t in TRANSFER]
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def run_row(name, tt, cond, seed, post, pre, **extra):
    key = f"{tt}__{name}" if name == "wine" else name  # the runner prefixes names shared across task types
    r = {"dataset": key, "dataset_name": name, "task_type": tt, "seed": seed, "arm": "tuned",
         "condition": f"tuned__{cond}", **extra}
    metric = "accuracy" if tt == "classification" else "rmse"
    conv = (lambda e: 1.0 - e) if tt == "classification" else (lambda e: e)
    r[f"test_{metric}_pre"] = conv(pre)
    if post is not None:
        r[f"test_{metric}_post"] = conv(post)
    return r


def synth(tmp_path, *, f_ratio=0.9, f_pre=1.0, catboost=1.3, tabm=1.0, noise=0.02, seeds=SEEDS):
    """Shards and baselines with known effects on every task (catboost/tabm: F / baseline error); returns an argv builder."""
    rng = np.random.default_rng(0)
    runs, alt, grid = tmp_path / "runs", tmp_path / "runs_default", tmp_path / "grid"
    shards = {}
    for i, (name, tt) in enumerate(LEGACY + TRANSFER):
        ref = 0.15 + 0.03 * i
        cls = tt == "classification"
        jobs = {fam: [] for fam in ("catboost", "tabm", "mlp", "knn")}
        for s in seeds:
            err = lambda m: ref * m * math.exp(rng.normal(0, noise))
            e_ro, e_def, e_f, e_fpre = err(1.0), err(1.0), err(f_ratio), err(f_pre)
            common = {"test_n": N_TEST, "test_correction_l1_mean": 0.1, "test_locality_input": 1.0, "n_cases": 100}
            flips = {"test_flip_wrong_to_correct": 8, "test_flip_correct_to_wrong": 2, "test_ece_pre": 0.05,
                     "test_ece_post": 0.08, "test_ece_post_T": 0.06} if cls else {}
            shards.setdefault((runs, name, tt), []).append(
                run_row(name, tt, "retrieval_only", s, None, e_ro, setting="retrieval_only", n_cases=100))
            shards[runs, name, tt].append(run_row(name, tt, "joint_summed_w5", s, e_f, e_fpre, schedule="joint_rr",
                                                  **common, **flips))
            shards.setdefault((alt, name, tt), []).append(
                run_row(name, tt, "alternating_default", s, e_def, e_ro, schedule="alternating_rr", **common, **flips))
            f_true = ref * f_ratio
            for fam, mult in (("catboost", f_true / catboost / ref), ("tabm", f_true / tabm / ref), ("mlp", 1.1), ("knn", 1.2)):
                e = ref * mult * math.exp(rng.normal(0, noise))
                jobs[fam].append({"status": "completed", "seed": s, "source_sha256": "grid",
                                  "test_metrics": {"accuracy": 1 - e} if cls else {"rmse_standardized": e},
                                  "dataset": {"n_test": N_TEST}})
        for fam, entries in jobs.items():
            job = grid / f"{tt}__{name}" / fam
            job.mkdir(parents=True)
            if fam == "tabm":  # evaluate-tool layout: a bare entry list
                (job / "evaluation.json").write_text(json.dumps(entries))
            else:
                (job / "completed.json").write_text(json.dumps({"status": "completed", "evaluation": entries}))
    for (d, name, tt), rows in shards.items():
        key = f"{tt}__{name}" if name == "wine" else name
        d.mkdir(exist_ok=True)
        with (d / f"runs__{key}.jsonl").open("a", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
    prereg, sha = write_prereg(tmp_path)

    def argv(*extra, sha256=sha):
        return ["--prereg", str(prereg), "--prereg-sha256", sha256, "--runs", str(runs), str(alt),
                "--baselines", str(grid), "--final", F, "--out", str(tmp_path / "out"), *extra]
    argv.runs, argv.alt, argv.grid = runs, alt, grid
    return argv


def test_known_effects_decide_each_endpoint(tmp_path):
    rep = main(synth(tmp_path)())
    ep = rep["primary"]["endpoints"]
    assert rep["primary"]["tasks"] and len(rep["primary"]["tasks"]) == 8 and not rep["missing"]
    assert ep["E1"]["gmr"] == pytest.approx(0.9, abs=0.02) and ep["E1"]["supported"]
    assert ep["E2"]["gmr"] == pytest.approx(0.9, abs=0.02) and ep["E2"]["supported"]
    assert rep["primary"]["guards"]["hold"] and rep["primary"]["guards"]["pre_adaptation"]["value"] == pytest.approx(1.0, abs=0.02)
    # F 30% worse than CatBoost: non-inferiority fails and the prereg wording reports the gap.
    assert ep["E3a"]["gmr"] == pytest.approx(1.3, abs=0.03) and not ep["E3a"]["supported"]
    assert ep["E3a"]["wording"].startswith("trails tuned CatBoost by +")
    assert ep["E3b"]["gmr"] == pytest.approx(1.0, abs=0.02) and ep["E3b"]["supported"]
    assert ep["E3b"]["gmr_upper95_one_sided"] < 1.1 and ep["E3b"]["sensitivity"]["0.2"]["upper_bound_below_margin"]
    assert all(ep[k]["p_holm"] >= ep[k]["p_one_sided"] for k in ep)
    assert set(rep["groups"]) == {"legacy", "transfer"} and rep["groups"]["transfer"]["endpoints"]["E1"]["k"] == 3
    assert rep["primary"]["transfer_check"]["E1"]["opposite"] is False
    sec = rep["secondary"]["pooled"]
    assert sec["flips"]["ratio"] == 4.0 and sec["flips"]["holds"] and sec["ece"]["holds"]
    ranks = sec["ranks"]["mean_rank"]
    assert sec["ranks"]["tasks"] == 8 and ranks["CatBoost"] == 1.0 and ranks["kNN"] == 6.0
    assert set(rep["secondary_bh"]["adjusted"]) >= {"pooled/ranks", "legacy/flips"}
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "trails tuned CatBoost by" in md and json.loads((tmp_path / "out" / "report.json").read_text())


def test_failed_endpoints_use_the_preregistered_wording(tmp_path):
    rep = main(synth(tmp_path, f_ratio=1.0, catboost=1.0)())
    ep = rep["primary"]["endpoints"]
    assert not ep["E1"]["supported"] and ep["E1"]["wording"] == "no detectable gain from reuse on tuned cores"
    assert not ep["E2"]["supported"] and ep["E2"]["wording"] == "no detectable schedule effect"
    assert ep["E3a"]["supported"]


def test_a_failed_guard_reports_gate_5(tmp_path):
    rep = main(synth(tmp_path, f_pre=1.1)())
    ep, guards = rep["primary"]["endpoints"], rep["primary"]["guards"]
    assert guards["pre_adaptation"]["value"] == pytest.approx(1.1, abs=0.02) and not guards["hold"]
    assert ep["E1"]["supported"] and ep["E2"]["criteria"]["holm_significant"]
    assert not ep["E2"]["supported"] and ep["E2"]["wording"] == "gate 5 fails"


def test_reml_matches_the_equal_variance_closed_form():
    rng = np.random.default_rng(1)
    v = np.full(12, 0.01)
    y = rng.normal(0.1, 0.2, size=12)  # heterogeneous: tau^2 = S^2 - v
    assert reml_tau2(y, v) == pytest.approx(np.var(y, ddof=1) - 0.01, rel=1e-6)
    y = 0.05 + rng.normal(0, 0.01, size=12)  # spread below the within-task variance: tau^2 = 0
    assert np.var(y, ddof=1) < 0.01 and reml_tau2(y, v) == 0.0
    agree = pool(np.full(5, -0.1), np.array([.01, .02, .03, .01, .05]))
    assert agree.tau2 == 0.0 and agree.mu == pytest.approx(-0.1) and agree.se == pytest.approx(0, abs=1e-12)
    assert agree.p_below(0) == pytest.approx(0, abs=1e-12)


def test_reml_solves_its_estimating_equation_with_unequal_variances():
    rng = np.random.default_rng(2)
    v = rng.uniform(0.001, 0.02, size=15)
    y = rng.normal(0, 0.15, size=15) + rng.normal(0, np.sqrt(v))
    t2 = reml_tau2(y, v)
    w = 1 / (v + t2)
    mu = np.sum(w * y) / np.sum(w)
    assert t2 > 0 and np.sum(w) - np.sum(w**2) / np.sum(w) == pytest.approx(np.sum(w**2 * (y - mu) ** 2), rel=1e-5)


def test_hartung_knapp_interval_by_hand():
    y, v = np.array([-0.2, -0.05, -0.12, 0.01, -0.3]), np.array([0.004, 0.01, 0.002, 0.02, 0.008])
    p = pool(y, v)
    w = 1 / (v + p.tau2)
    mu = np.sum(w * y) / np.sum(w)
    se = math.sqrt(np.sum(w * (y - mu) ** 2) / (len(y) - 1) / np.sum(w))
    assert p.mu == pytest.approx(mu) and p.se == pytest.approx(se)
    assert p.p_below(0.0) == pytest.approx(stats.t.cdf(mu / se, 4))
    assert p.ci()[1] == pytest.approx(mu + stats.t.ppf(0.975, 4) * se)
    assert p.upper() == pytest.approx(mu + stats.t.ppf(0.95, 4) * se)


def test_holm_and_bh_adjust_in_order():
    p = {"E1": 0.01, "E2": 0.04, "E3a": 0.03, "E3b": 0.005}
    assert holm(p) == pytest.approx({"E3b": 0.02, "E1": 0.03, "E3a": 0.06, "E2": 0.06})
    assert bh(p) == pytest.approx({"E3b": 0.02, "E1": 0.02, "E3a": 0.04, "E2": 0.04})
    assert holm({"a": 0.5, "b": 0.6}) == {"a": 1.0, "b": 1.0}


def test_a_changed_prereg_is_refused(tmp_path):
    argv = synth(tmp_path)
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        main(argv(sha256="0" * 64))


def test_seeds_outside_the_evaluation_seeds_are_refused(tmp_path):
    argv = synth(tmp_path)
    shard = argv.runs / "runs__iris.jsonl"
    rows = [json.loads(l) for l in shard.read_text().splitlines()]
    with shard.open("a") as fh:
        fh.write(json.dumps({**rows[1], "seed": 30}) + "\n")  # an F row on a retired seed
    with pytest.raises(SystemExit, match="outside the pre-registered evaluation seeds"):
        main(argv())


def test_baseline_seeds_outside_the_evaluation_seeds_are_refused(tmp_path):
    argv = synth(tmp_path)
    path = argv.grid / "regression__yacht" / "catboost" / "completed.json"
    data = json.loads(path.read_text())
    data["evaluation"].append({**data["evaluation"][0], "seed": 31})
    path.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="outside the evaluation seeds"):
        main(argv())


def test_small_task_seeds_need_a_small_task_listing(tmp_path):
    argv = synth(tmp_path)
    shard = argv.runs / "runs__iris.jsonl"
    rows = [json.loads(l) for l in shard.read_text().splitlines()]
    with shard.open("a") as fh:
        fh.writelines(json.dumps({**r, "seed": 55}) + "\n" for r in rows[:2])
    with pytest.raises(SystemExit, match="outside"):
        main(argv())
    rep = main(argv("--small-tasks", "iris", "--allow-incomplete"))
    iris = next(t for t in rep["tasks"] if t["key"] == "classification__iris")
    assert iris["small"] and len(iris["expected_seeds"]) == 30 and rep["missing"]
    with pytest.raises(SystemExit, match="OpenML"):
        main(argv("--small-tasks", "tabarena_363702"))


def test_mixed_run_code_is_refused_unless_allowed(tmp_path):
    argv = synth(tmp_path)
    for d, cond, fp in ((argv.runs, "tuned__joint_summed_w5", "a" * 64), (argv.alt, "tuned__alternating_default", "b" * 64)):
        for s in SEEDS[:3]:
            path = d / "iris" / cond / f"s{s}" / "run_manifest.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"run": {"source_fingerprint_sha256": fp, "device": "cpu"}}))
    with pytest.raises(SystemExit, match="fingerprints"):
        main(argv())
    pv = main(argv("--allow-mixed-code"))["provenance"]
    assert pv["run_manifests_found"] == 6 and pv["run_manifests_absent"] == pv["rows_analysed"] - 6
    assert pv["source_fingerprints"]["a" * 64] == {F: 3}


def test_incomplete_evidence_is_refused_unless_allowed(tmp_path):
    argv = synth(tmp_path)
    shard = argv.alt / "runs__yacht.jsonl"
    shard.write_text("\n".join(shard.read_text().splitlines()[:-1]) + "\n")  # drop the default's seed 49
    with pytest.raises(SystemExit, match="incomplete"):
        main(argv())
    rep = main(argv("--allow-incomplete"))
    assert rep["missing"] == [{"task": "regression__yacht", "method": "tuned::alternating_default", "missing_seeds": [49]}]
    assert next(p for p in rep["primary"]["endpoints"]["E2"]["per_task"] if p["task"] == "regression__yacht")["n_seeds"] == 9


def test_identical_reruns_merge_and_differing_ones_are_refused(tmp_path):
    argv = synth(tmp_path)
    rows = [json.loads(l) for l in (argv.runs / "runs__digits.jsonl").read_text().splitlines()]
    with (argv.alt / "runs__digits.jsonl").open("a") as fh:
        fh.write(json.dumps({**rows[0], "run_id": "rerun"}) + "\n")
    assert main(argv())["provenance"]["duplicate_rows"] == 1
    with (argv.alt / "runs__digits.jsonl").open("a") as fh:
        fh.write(json.dumps({**rows[0], "test_accuracy_pre": 0.0}) + "\n")
    with pytest.raises(SystemExit, match="differ"):
        main(argv())


def test_classification_eps_uses_the_test_size():
    tasks = [Task(f"classification__t{i}", f"t{i}", "classification", "legacy", False, (40, 41)) for i in range(2)]
    ev = Evidence()
    for t in tasks:
        for s, (f, r) in zip((40, 41), ((0.0, 0.1), (0.0, 0.1))):
            ev.put(t.key, "F", s, f)
            ev.put(t.key, "R", s, r)
            ev.n_test[t.key, s] = 10
    c = compare(ev, tasks, "F", "R")
    # (0 + 0.05) / (0.1 + 0.05): a perfect F is not an infinite gain; no seed variation hits the floor.
    assert c["per_task"][0]["log_ratio"] == pytest.approx(math.log(1 / 3))
    assert c["variance_floor"] == 1e-12 and c["per_task"][0]["variance_floored"]
    assert c["pooled"].mu == pytest.approx(math.log(1 / 3))


def test_condition_ids_match_the_schedule_search():
    from tools.t1_schedule_search import condition_id as search_id
    rows = [{"arm": "tuned", "condition": "tuned__retrieval_only"},
            {"arm": "tuned", "condition": "tuned__nominal_residual_scores_combined", "loss": "combined"},
            {"arm": "tuned", "condition": "tuned__joint_summed_w5__w_post=0.25", "schedule": "joint_rr"},
            {"condition": "alternating_default", "schedule": "alternating_rr"}]
    assert [condition_id(r) for r in rows] == [search_id(r) for r in rows]
    assert condition_id(rows[1]) == "tuned::frozen_reuse__combined"


def test_the_frozen_prereg_plans_thirty_tasks():
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    tasks, unknown = plan_tasks(prereg, ["legacy", "transfer"], {"iris", "nonexistent"})
    assert len(tasks) == 30 and len({t.key for t in tasks}) == 30 and unknown == ["nonexistent"]
    assert {"classification__wine", "regression__wine"} <= {t.key for t in tasks}
    iris = next(t for t in tasks if t.name == "iris")
    assert iris.seeds == tuple(range(40, 70)) and all(t.seeds == tuple(range(40, 50)) for t in tasks if t is not iris)
