"""Pre-registered confirmation analysis: known effects, REML, Holm and the refusals."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from tools.t1_confirm_report import (Evidence, Task, bh, compare, condition_id, device_type, expected_fingerprints,
                                     holm, macro_f1, main, plan_tasks, pool, reml_tau2, small_task_names, wording)

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


def synth(tmp_path, *, f_ratio=0.9, f_pre=1.0, catboost=1.3, tabm=1.0, noise=0.02, seeds=SEEDS, n_test=None):
    """Shards and baselines with known effects on every task (catboost/tabm: F / baseline error); returns an argv builder.

    ``n_test`` maps a task name to its test-set size (default N_TEST)."""
    rng = np.random.default_rng(0)
    runs, alt, grid = tmp_path / "runs", tmp_path / "runs_default", tmp_path / "grid"
    shards = {}
    for i, (name, tt) in enumerate(LEGACY + TRANSFER):
        ref = 0.15 + 0.03 * i
        cls = tt == "classification"
        n = (n_test or {}).get(name, N_TEST)
        jobs = {fam: [] for fam in ("catboost", "tabm", "mlp", "knn")}
        for s in seeds:
            err = lambda m: ref * m * math.exp(rng.normal(0, noise))
            e_ro, e_def, e_f, e_fpre = err(1.0), err(1.0), err(f_ratio), err(f_pre)
            common = {"test_n": n, "test_correction_l1_mean": 0.1, "test_locality_input": 1.0, "n_cases": 100}
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
                                  "dataset": {"n_test": n}})
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
    argv = synth(tmp_path, n_test={"iris": 30})
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


def rewrite_rows(directory, fn):
    for shard in Path(directory).glob("runs__*.jsonl"):
        rows = [fn(json.loads(l)) for l in shard.read_text().splitlines() if l.strip()]
        shard.write_text("".join(json.dumps(r) + "\n" for r in rows))


def write_manifests(directory, dataset, condition, device, seeds=SEEDS):
    for s in seeds:
        path = directory / dataset / condition / f"s{s}" / "run_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        run = {"source_fingerprint_sha256": "a" * 64, **({"device": device} if device is not None else {})}
        path.write_text(json.dumps({"run": run}))


def test_e3_is_decided_by_its_holm_adjusted_noninferiority_p(tmp_path):
    # Prereg addendum 1: E3 belongs to the Holm family. Here the unadjusted bound is below
    # 1 + delta but the Holm-adjusted p exceeds 0.05, so non-inferiority is not shown.
    rep = main(synth(tmp_path / "weak", f_ratio=1.0, catboost=1.04, tabm=1.06, noise=0.15)())
    e3a = rep["primary"]["endpoints"]["E3a"]
    assert e3a["gmr_upper95_one_sided"] < 1.1 and e3a["upper_bound_below_margin"] and e3a["p_holm"] > 0.05
    assert e3a["criteria"] == {"holm_significant": False} and not e3a["supported"]
    assert "not shown" in e3a["wording"] or e3a["wording"].startswith("trails")
    assert e3a["decision_rule"] == rep["decision_rules"]["noninferiority"] and "addendum 1" in e3a["decision_rule"]
    md = (tmp_path / "weak" / "out" / "report.md").read_text(encoding="utf-8")
    assert "| E3a | CatBoost |" in md and "addendum 1" in md
    strong = main(synth(tmp_path / "strong", f_ratio=1.0, catboost=1.0, tabm=1.0, noise=0.02)())
    e3a = strong["primary"]["endpoints"]["E3a"]
    assert e3a["criteria"] == {"holm_significant": True} and e3a["supported"]
    assert e3a["wording"].startswith("non-inferior to tuned CatBoost at delta = 0.1")


def test_a_failed_e3_with_a_lower_f_error_does_not_say_trails():
    spec = {"vs": "tabm", "test": "noninferiority", "delta": 0.1}
    r = {"k": 5, "gmr": 0.871, "gmr_ci95": [0.6, 1.26], "gmr_upper95_one_sided": 1.2, "supported": False}
    text = wording("E3b", r, spec, {"hold": True})
    assert text.startswith("non-inferiority to tuned TabM not shown at delta = 0.1: GMR 0.871") and "trails" not in text
    assert wording("E3b", {**r, "gmr": 1.05}, spec, {"hold": True}).startswith("trails tuned TabM by +5.0% [95% CI -40.0%")
    assert wording("E3b", {**r, "k": 1}, spec, {"hold": True}).startswith("not evaluable")


def test_zero_flips_and_infinite_ratios_still_render(tmp_path):
    argv = synth(tmp_path)
    out = tmp_path / "out"

    def flips(w, c):
        return lambda r: {**r, "test_flip_wrong_to_correct": w, "test_flip_correct_to_wrong": c} \
            if "test_flip_correct_to_wrong" in r else r
    rewrite_rows(argv.runs, flips(8, 0))  # F never turns a correct answer wrong
    fl = main(argv())["secondary"]["pooled"]["flips"]
    assert fl["wrong_to_correct"] == 4 * 10 * 8 and fl["correct_to_wrong"] == 0 and fl["ratio"] == "inf" and fl["holds"]
    assert "| pooled | flips | 320 / 0 = inf | yes |" in (out / "report.md").read_text(encoding="utf-8")
    rewrite_rows(argv.runs, flips(0, 0))  # no flips at all: undefined ratio, criterion not evaluable
    fl = main(argv())["secondary"]["pooled"]["flips"]
    assert fl["ratio"] is None and fl["holds"] is None
    assert json.loads((out / "report.json").read_text())["secondary"]["pooled"]["flips"]["holds"] is None
    assert "| pooled | flips | 0 / 0 = n/a | n/a (no flips) |" in (out / "report.md").read_text(encoding="utf-8")
    # A default with zero correction makes the correction-size ratio infinite: the guard fails.
    rewrite_rows(argv.alt, lambda r: {**r, "test_correction_l1_mean": 0.0})
    rep = main(argv())
    cor = rep["primary"]["guards"]["correction"]
    assert cor["value"] == "inf" and cor["holds"] is False and rep["primary"]["endpoints"]["E2"]["wording"] == "gate 5 fails"
    assert "| correction | inf | 1.25 | no |" in (out / "report.md").read_text(encoding="utf-8")


def test_a_bare_small_task_name_shared_by_two_tasks_is_refused(tmp_path):
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    for items in (["wine"], [str(tmp_path / "audit.json")]):
        (tmp_path / "audit.json").write_text(json.dumps({"small_tasks": ["wine", "classification__iris"]}))
        with pytest.raises(SystemExit, match="several analysed tasks"):
            plan_tasks(prereg, ["legacy", "transfer"], small_task_names(items, prereg))
    tasks, _ = plan_tasks(prereg, ["legacy", "transfer"], small_task_names(["classification__wine"], prereg))
    assert [(t.key, t.small, len(t.seeds)) for t in tasks if t.name == "wine"] == [
        ("classification__wine", True, 30), ("regression__wine", False, 10)]


def test_small_task_declarations_are_checked_against_test_sizes(tmp_path):
    argv = synth(tmp_path, n_test={"iris": 30})
    with pytest.raises(SystemExit, match="not declared small"):  # its seeds 50-69 would be silently missing
        main(argv())
    rep = main(argv("--allow-incomplete"))
    assert rep["provenance"]["small_task_check"]["small_but_undeclared"] == ["classification__iris (30 test rows)"]
    assert any("not declared small" in w for w in rep["warnings"])
    with pytest.raises(SystemExit, match="100 or more test rows"):
        main(argv("--small-tasks", "classification__iris", "digits", "--allow-incomplete"))
    rep = main(argv("--small-tasks", "classification__iris", "--allow-incomplete"))
    assert rep["inputs"]["small_tasks"] == ["classification__iris"] and not rep["provenance"]["small_task_check"]["small_but_undeclared"]


def test_mixed_devices_within_a_comparison_are_refused(tmp_path):
    argv = synth(tmp_path)
    write_manifests(argv.runs, "iris", "tuned__joint_summed_w5", "cpu")
    write_manifests(argv.alt, "iris", "tuned__alternating_default", "xpu:0")
    with pytest.raises(SystemExit, match="different device types"):
        main(argv())
    rep = main(argv("--allow-mixed-device"))
    assert rep["provenance"]["device_check"]["mixed_comparisons"] == ["classification__iris E2: ['cpu', 'xpu']"]
    assert rep["inputs"]["allow_mixed_device"] and any("--allow-mixed-device" in w for w in rep["warnings"])
    # One retrieval-only seed on the XPU mixes E1 too.
    write_manifests(argv.runs, "iris", "tuned__retrieval_only", "xpu:0", seeds=SEEDS[:1])
    with pytest.raises(SystemExit, match="E1"):
        main(argv())
    # Device indices do not matter; manifests without a device (older runs) count as unknown, not 'None'.
    write_manifests(argv.runs, "iris", "tuned__joint_summed_w5", "XPU:1")
    write_manifests(argv.runs, "iris", "tuned__retrieval_only", None, seeds=SEEDS[:1])
    pv = main(argv())["provenance"]
    assert pv["devices"]["xpu"] == 20 and pv["devices"]["unknown"] == pv["rows_analysed"] - 20 and "None" not in pv["devices"]
    assert pv["devices_by_task"]["classification__iris"][F] == ["xpu"] and not pv["device_check"]["mixed_comparisons"]
    assert [device_type(x) for x in (None, "None", "", "cpu", "xpu:0", "cuda:1")] == [
        "unknown", "unknown", "unknown", "cpu", "xpu", "cuda"]


def test_the_frozen_prereg_plans_thirty_tasks():
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    tasks, unknown = plan_tasks(prereg, ["legacy", "transfer"], {"iris", "nonexistent"})
    assert len(tasks) == 30 and len({t.key for t in tasks}) == 30 and unknown == ["nonexistent"]
    assert {"classification__wine", "regression__wine"} <= {t.key for t in tasks}
    iris = next(t for t in tasks if t.name == "iris")
    assert iris.seeds == tuple(range(40, 70)) and all(t.seeds == tuple(range(40, 50)) for t in tasks if t is not iris)


def test_guard_fallback_rows_are_reported(tmp_path):
    argv = synth(tmp_path)
    rewrite_rows(argv.runs, lambda r: {**r, "sync_no_eligible_checkpoint": True} if r.get("schedule") and r["seed"] == SEEDS[0] else r)
    rep = main(argv())
    flagged = [w for w in rep["warnings"] if "no eligible checkpoint" in w]
    assert flagged and all(f"seed {SEEDS[0]}" in w for w in flagged)


def test_final_map_uses_another_condition_as_f_on_one_task(tmp_path):
    argv = synth(tmp_path)
    base = main(argv())
    shard = next(Path(argv.runs).glob("runs__iris.jsonl"))
    rows = [json.loads(l) for l in shard.read_text().splitlines() if l.strip()]
    extra = [{**r, "condition": "tuned__iris_only", "test_accuracy_post": 1 - 2 * (1 - r["test_accuracy_post"])}
             for r in rows if r["condition"] == "tuned__joint_summed_w5"]
    shard.write_text("".join(json.dumps(r) + "\n" for r in rows + extra))
    fm = tmp_path / "final_map.json"
    fm.write_text(json.dumps({"classification__iris": "tuned::iris_only"}))
    rep = main(argv("--final-map", str(fm)))
    # F's iris errors now come from the mapped condition (twice F's error), raising E1's pooled ratio.
    assert rep["primary"]["endpoints"]["E1"]["gmr"] > base["primary"]["endpoints"]["E1"]["gmr"]


def test_the_final_map_is_recorded_and_stated(tmp_path):
    argv = synth(tmp_path)
    assert main(argv())["inputs"]["final_map"] is None
    shard = argv.runs / "runs__iris.jsonl"
    rows = [json.loads(l) for l in shard.read_text().splitlines() if l.strip()]
    shard.write_text("".join(json.dumps(r) + "\n" for r in rows + [
        {**r, "condition": "tuned__iris_only"} for r in rows if r["condition"] == "tuned__joint_summed_w5"]))
    fm = tmp_path / "final_map.json"
    contents = {"classification__iris": "tuned::iris_only", "classification__digits": "tuned::absent",
                "regression__nowhere": "tuned::x"}
    fm.write_text(json.dumps(contents))
    rep = main(argv("--final-map", str(fm), "--allow-incomplete"))  # digits' F is mapped to nothing
    assert rep["inputs"]["final_map"] == {"path": str(fm), "sha256": hashlib.sha256(fm.read_bytes()).hexdigest(),
                                          "contents": contents}
    assert rep["provenance"]["final_map_rows"] == {"classification__iris": 10}
    assert any("classification__digits -> tuned::absent matched no analysed rows" in w for w in rep["warnings"])
    assert any("regression__nowhere -> tuned::x is not an analysed task" in w for w in rep["warnings"])
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert f"F = `{F}` (per-task F from --final-map `{fm}`, sha256 `{rep['inputs']['final_map']['sha256'][:16]}...`: " \
           "classification__iris: `tuned::iris_only` (10 rows); classification__digits: `tuned::absent` (0 rows);" in md


def write_all_manifests(argv, fingerprint):
    """A run manifest for every row of every shard; ``fingerprint(task name)`` gives its source fingerprint."""
    for d in (argv.runs, argv.alt):
        for shard in d.glob("runs__*.jsonl"):
            for r in (json.loads(l) for l in shard.read_text().splitlines() if l.strip()):
                path = d / r["dataset"] / r["condition"] / f"s{r['seed']}" / "run_manifest.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"run": {"source_fingerprint_sha256": fingerprint(r["dataset_name"]),
                                                    "device": "cpu"}}))


def test_fingerprints_are_recorded_per_task_and_can_be_required(tmp_path):
    argv = synth(tmp_path)
    write_all_manifests(argv, lambda name: ("b" if name.startswith("tabarena_") else "a") * 64)
    with pytest.raises(SystemExit, match="run source fingerprints"):
        main(argv())
    rep = main(argv("--allow-mixed-code"))
    by_task = rep["provenance"]["source_fingerprints_by_task"]
    assert by_task["classification__iris"] == {"a" * 64: 30} and by_task["regression__tabarena_363705"] == {"b" * 64: 30}
    assert len(by_task) == 8 and not rep["provenance"]["rows_without_manifest_by_task"]
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8").splitlines()
    line = next(l for l in md if l.startswith("  - `aaaaaaaaaaaa`"))
    assert line.startswith("  - `aaaaaaaaaaaa`: 5 task(s), 150 rows: ") and "classification__iris" in line
    assert any(l.startswith("  - `bbbbbbbbbbbb`: 3 task(s), 90 rows: ") for l in md)
    # An exact key or name decides first, then the first matching glob in the order given.
    spec = tmp_path / "expect.json"
    spec.write_text(json.dumps({"tabarena_*": ["BBBB"], "*": "aaaa"}))
    rep = main(argv("--allow-mixed-code", "--expect-fingerprints", str(spec)))
    fx = rep["provenance"]["fingerprint_expectations"]
    assert fx["rows_checked"] == rep["provenance"]["rows_analysed"] == 240
    assert fx["pattern_by_task"]["classification__iris"] == "*" and fx["pattern_by_task"]["regression__tabarena_363686"] == "tabarena_*"
    assert rep["inputs"]["expect_fingerprints"] == {"tabarena_*": ["bbbb"], "*": ["aaaa"]}
    assert "- --expect-fingerprints held for all 240 analysed rows" in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="does not allow"):  # '*' now comes first and decides every task
        main(argv("--allow-mixed-code", "--expect-fingerprints", '{"*": "aaaa", "tabarena_*": "bbbb"}'))
    main(argv("--allow-mixed-code", "--expect-fingerprints", "*=aaaa", "tabarena_363702=bbbb",
              "regression__tabarena_363705=cccc,bbbb", "tabarena_363686=bbbb"))
    with pytest.raises(SystemExit, match="no --expect-fingerprints pattern matches"):
        main(argv("--allow-mixed-code", "--expect-fingerprints", "tabarena_*=bbbb"))
    with pytest.raises(SystemExit, match="run source fingerprints"):  # expectations do not replace --allow-mixed-code
        main(argv("--expect-fingerprints", str(spec)))
    (argv.runs / "yacht" / "tuned__retrieval_only" / f"s{SEEDS[0]}" / "run_manifest.json").unlink()
    with pytest.raises(SystemExit, match="no run manifest"):
        main(argv("--allow-mixed-code", "--expect-fingerprints", str(spec)))
    assert main(argv("--allow-mixed-code"))["provenance"]["rows_without_manifest_by_task"] == {"regression__yacht": 1}


def test_fingerprint_pattern_precedence_is_key_then_name_then_glob(tmp_path):
    """Two prereg tasks share the name 'wine': the key entry decides its task in either order, and a
    pattern that matches no task (a mistyped key falling through to '*') is warned about."""
    cw, rw = (Task(f"{t}__wine", "wine", t, "legacy", False, (40,)) for t in ("classification", "regression"))
    for spec in ({"wine": ["aaaa"], "regression__wine": ["bbbb"]}, {"regression__wine": ["bbbb"], "wine": ["aaaa"]}):
        assert expected_fingerprints({"*": ["cccc"], **spec}, rw) == ("regression__wine", ("bbbb",))
        assert expected_fingerprints({"*": ["cccc"], **spec}, cw) == ("wine", ("aaaa",))
    assert expected_fingerprints({"*": ["cccc"], "regression__*": ["dddd"]}, rw) == ("*", ("cccc",))
    argv = synth(tmp_path)
    write_all_manifests(argv, lambda name: "a" * 64)
    rep = main(argv("--expect-fingerprints", "*=aaaa", "regresion__yacht=bbbb", "yacht=aaaa"))
    fx = rep["provenance"]["fingerprint_expectations"]
    assert fx["unmatched_patterns"] == ["regresion__yacht"] and fx["pattern_by_task"]["regression__yacht"] == "yacht"
    assert "--expect-fingerprints pattern 'regresion__yacht' matches no analysed task" in rep["warnings"]


def test_macro_f1_matches_sklearn():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(4)
    y, p = rng.integers(0, 4, 60), rng.integers(0, 5, 60)
    p[p == 3] = 2  # class 3 only in the targets, class 4 only in the predictions
    assert macro_f1(y, p) == pytest.approx(f1_score(y, p, average="macro", zero_division=0))


def test_log_loss_and_macro_f1_on_churn_and_mic(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from sklearn.metrics import f1_score
    import tools.t1_confirm_report as report_tool
    extra = {"classification__digits": "MIC", "classification__tabarena_363702": "churn"}  # stand-ins
    monkeypatch.setattr(report_tool, "EXTRA_METRIC_TASKS", extra)
    n, rng, f1 = 200, np.random.default_rng(3), {}
    argv = synth(tmp_path, n_test={"digits": n, "tabarena_363702": n})

    def predictions(acc, k=3):
        """Targets and predictions of n rows, round(acc * n) of them correct."""
        y = rng.integers(0, k, n)
        p, wrong = y.copy(), rng.choice(n, n - round(acc * n), replace=False)
        p[wrong] = (y[wrong] + rng.integers(1, k, len(wrong))) % k
        return y, p

    def add(d):
        def fn(r):
            if r["dataset_name"] not in ("digits", "tabarena_363702"):
                return r
            ro = r["condition"].endswith("retrieval_only")
            y, p = predictions(r["test_accuracy_pre" if ro else "test_accuracy_post"])
            path = d / r["dataset"] / r["condition"] / f"s{r['seed']}" / "retrieval_events.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            events = [{"stream": "test", "query_index": i, "target": int(a), "final_prediction": int(b),
                       "pre_adaptation_prediction": int(b) if ro else 0} for i, (a, b) in enumerate(zip(y, p))]
            path.write_text("".join(json.dumps(e) + "\n" for e in reversed(events)))  # any order
            f1[r["dataset_name"], r["condition"], r["seed"]] = f1_score(y, p, average="macro")
            return {**r, **({"test_loss_pre": 0.5} if ro else
                            {"test_L_post": 0.3 if "joint" in r["condition"] else 0.35, "test_nll_post_T": 0.2})}
        return fn
    rewrite_rows(argv.runs, add(argv.runs))
    rewrite_rows(argv.alt, add(argv.alt))
    for task in ("classification__digits", "classification__tabarena_363702"):
        for fam in ("catboost", "tabm", "knn"):
            path = argv.grid / task / fam / ("evaluation.json" if fam == "tabm" else "completed.json")
            data = json.loads(path.read_text())
            entries = data if fam == "tabm" else data["evaluation"]
            for e in entries:
                e.update(stage="evaluation", candidate_id=f"{fam}0")
                e["test_metrics"]["loss"] = 0.25 + e["seed"] / 1000
                if fam == "knn":  # no saved predictions
                    continue
                y, p = predictions(e["test_metrics"]["accuracy"])
                prob = np.full((n, 3), 0.1)
                prob[np.arange(n), p] = 0.8
                out = argv.grid / task / fam / "evaluation" / f"{fam}0" / f"s{e['seed']}" / "test_predictions.pt"
                out.parent.mkdir(parents=True)
                torch.save({"prediction": torch.as_tensor(prob), "target": torch.as_tensor(y)}, out)
                f1[task, fam, e["seed"]] = f1_score(y, p, average="macro")
            path.write_text(json.dumps(data))
    churn_f = argv.runs / "tabarena_363702" / "tuned__joint_summed_w5"
    lines = (churn_f / f"s{SEEDS[1]}" / "retrieval_events.jsonl").read_text().splitlines()
    (churn_f / f"s{SEEDS[1]}" / "retrieval_events.jsonl").write_text("".join(  # every answer right: not this run's
        json.dumps({**json.loads(l), "final_prediction": json.loads(l)["target"]}) + "\n" for l in lines))
    (argv.alt / "tabarena_363702" / "tuned__alternating_default" / f"s{SEEDS[2]}" / "retrieval_events.jsonl").unlink()
    rep = main(argv())
    x = rep["secondary"]["pooled"]["log_loss_macro_f1"]
    assert x["tasks"] == ["classification__digits", "classification__tabarena_363702"] and len(x["per_method"]) == 14
    get = {(r["task_label"], r["method"]): r for r in x["per_method"]}
    mean_f1 = lambda name, cond, seeds: pytest.approx(np.mean([f1[name, cond, s] for s in seeds]))
    r = get["churn", "F"]
    assert (r["log_loss"], r["log_loss_T"]) == (pytest.approx(0.3), pytest.approx(0.2)) and r["log_loss_seeds"] == 10
    assert r["macro_f1_seeds"] == 9 and r["macro_f1"] == mean_f1(
        "tabarena_363702", "tuned__joint_summed_w5", [s for s in SEEDS if s != SEEDS[1]])
    assert r["macro_f1_missing"] == {"not used: saved predictions do not reproduce the recorded accuracy": [SEEDS[1]]}
    r = get["churn", "alternating default"]
    assert r["log_loss"] == pytest.approx(0.35) and r["macro_f1_seeds"] == 9
    assert r["macro_f1_missing"] == {"not recorded: no retrieval_events.jsonl": [SEEDS[2]]}
    r = get["MIC", "NN-kNN retrieval-only"]
    assert r["log_loss"] == pytest.approx(0.5) and r["log_loss_T"] is None and r["macro_f1"] == mean_f1(
        "digits", "tuned__retrieval_only", SEEDS)
    r = get["MIC", "CatBoost"]
    assert r["log_loss"] == pytest.approx(0.25 + np.mean(SEEDS) / 1000) and r["macro_f1"] == mean_f1(
        "classification__digits", "catboost", SEEDS)
    assert get["churn", "TabM"]["macro_f1"] == mean_f1("classification__tabarena_363702", "tabm", SEEDS)
    r = get["churn", "kNN"]
    assert r["macro_f1"] is None and r["macro_f1_missing"] == {"not recorded: no test_predictions.pt": SEEDS}
    r = get["churn", "MLP"]  # entries without a test loss or a candidate directory
    assert r["n_seeds"] == 10 and r["log_loss"] is None and r["macro_f1_missing"] == {"not recorded: no test_predictions.pt": SEEDS}
    assert rep["secondary"]["transfer"]["log_loss_macro_f1"]["tasks"] == ["classification__tabarena_363702"]
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "| churn | F | 0.3000 (10) | 0.2000 (10) | " in md and " (9) |" in md
    assert "| churn | kNN | 0.2945 (10) | not recorded | not recorded |" in md
    assert f"- churn kNN macro-F1 not recorded: no test_predictions.pt (seeds {', '.join(map(str, SEEDS))})" in md
    assert "- log_loss_T: adapted neural-CBR arms' test_nll_post_T" in md
