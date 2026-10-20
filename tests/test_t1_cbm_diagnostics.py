"""CBM diagnostics on fabricated case statistics with hand-computed answers (hub and tie-dominated runs)."""
import json
from dataclasses import replace

import numpy as np
import pytest

from model.t1.retention import protection_requirements, select_active_set
from tools.t1_cbm_diagnostics import (_cfg, analyse_run, diagnose, fill, load_scores, main, parse_K, ranking,
                                      resolve_K, retention_kept, run_config)

STATS = {"source": "train_loo", "accumulation": "per_checkpoint", "smoothing": 1.0, "min_retrieval_count": 2,
         "min_activation_mass": 0.1, "bias_normalization": "within_cohort_percentile", "bias_source": "current"}


def case(cid, cohort, R=0.0, A=0.0, C=0.0, H=0.0, bias=0.0, protected=False, state="active"):
    return {"case_id": cid, "case_role": "classification", "cohort": cohort, "initial_bias": bias, "current_bias": bias,
            "retrieval_count": R, "activation_mass": A, "positive_support": C, "harmful_support": H, "cf_pos": 0.0,
            "cf_neg": 0.0, "error_support": H, "audits": 1, "protected": protected, "insertion_step": 0,
            "archive_state": state}


def manifest(task="t", seed=20, min_per_cohort=2, condition=None):
    return {"run": {"seed": seed}, "task": {"name": task},
            "resolved_config": {"condition": condition or {"arm": "tuned", "name": "a"},
                                "config": {"statistics": STATS, "retention": {
                                    "policy": "provenance_only", "min_per_cohort": min_per_cohort,
                                    "protect_rare_cohort_below": 0, "boundary_protect_fraction": 0.0}}}}


def hub_rows(bias_sign=1.0, protect=()):
    """20 cases, 15 never retrieved; ids 3 and 15 hold 18 of the 20.05 activation mass."""
    hubs = {3: (30, 12, 12, 0), 15: (20, 6, 4, 2), 7: (3, 1.5, 1.0, 0.5), 12: (2, .5, .5, 0), 1: (1, .05, .05, 0)}
    return [case(i, f"class_{i // 10}", *hubs.get(i, (0, 0, 0, 0)), bias=bias_sign * i / 100, protected=i in protect)
            for i in range(20)]


def tie_rows():
    """12 cases, all biases equal; a harmful evidenced hub (id 0, Q = 2/7) and ten 0.5 ties."""
    hot = {0: (10, 5, 1, 4), 6: (4, 2, 2, 0), 11: (1, .2, .2, 0), 9: (2, .05, .05, 0)}
    return [case(i, f"class_{i // 6}", *hot.get(i, (0, 0, 0, 0))) for i in range(12)]


def ids(mask):
    return set(np.flatnonzero(mask).tolist())


def write_run(root, task, cond, seed, rows, events=(), history=None, **man):
    d = root / task / cond / f"s{seed}"
    d.mkdir(parents=True)
    (d / "run_manifest.json").write_text(json.dumps(manifest(task, seed, **man)))
    (d / "case_statistics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "case_maintenance.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (d / "history.json").write_text(json.dumps(history or {"core": [], "sync": []}))
    return d


def test_capacity_resolution_matches_runner():
    assert [resolve_K(0.5, 90), resolve_K(0.5, 45), resolve_K(1.0, 90), resolve_K(60, 90)] == [45, 22, 90, 60]
    assert parse_K("0.5") == 0.5 and parse_K("45") == 45 and isinstance(parse_K("45"), int)


def test_hub_run_exposure_kept_sets_and_tie_breaks():
    m = diagnose(hub_rows() + [case(20, "class_0", 50, 40, 40, 0, state="archived")], manifest(), 0.5)
    ctx, m = m["context"], m["metrics"]
    assert ctx["n_archived"] == 1 and m["n"] == 20 and m["K"] == 10
    assert m["never_retrieved_share"] == 0.75 and m["evidenced_share"] == 0.2
    assert m["top10_activation_share"] == pytest.approx(18 / 20.05)
    # Evidenced ids 3, 15, 7, 12: Q ranks 4, 3, 1, 2 vs A ranks 4, 3, 2, 1; H = 0 for ids 3 and 12.
    assert m["spearman_q_a_evidenced"] == pytest.approx(0.8) and m["evidenced_h0_share"] == 0.5
    # Floors keep 3, 7 and 15, 12; the other six slots go to the lowest ids among sixteen 0.5 ties.
    assert m["tie_decided_share"] == 1.0
    q, b, e = (m["policies"][p] for p in ("provenance_only", "bias_current", "exposure_only"))
    assert q["cut_tie_share"] == 1.0 and b["cut_tie_share"] == 0.0 and e["cut_tie_share"] == 1.0
    assert q["archived_activation_share"] == 0.0 and e["archived_activation_share"] == 0.0
    assert b["archived_activation_share"] == pytest.approx(13.55 / 20.05)  # bias keeps ids 8-19
    assert m["jaccard"]["provenance_only|bias_current"] == pytest.approx({"all": 1 / 9, "retrieved": 0.4})
    assert m["jaccard"]["provenance_only|exposure_only"] == {"all": 1.0, "retrieved": 1.0}
    assert m["jaccard"]["bias_current|exposure_only"] == pytest.approx({"all": 1 / 9, "retrieved": 0.4})
    # Case-ID ties keep class_0 at 0.8; random ties draw 6 of 16 (8 per class): mean 0.5, sd 0.1.
    c0, c1 = q["cohort_kept"]["class_0"], q["cohort_kept"]["class_1"]
    assert (c0["case_id"], c1["case_id"]) == (0.8, 0.2)
    assert c0["random_mean"] == pytest.approx(0.5, abs=0.03) and c1["random_mean"] == pytest.approx(0.5, abs=0.03)
    assert 0.07 < c0["random_sd"] < 0.13 and c0["random_min"] >= 0.2 and c0["random_max"] <= 0.8
    assert q["tie_shift_max_abs"] == pytest.approx(0.3, abs=0.03) and q["random_tie_jaccard_mean"] < 1
    # Exposure: id 1 (R = 1) outranks the fifteen never-retrieved ties; random mean (2 + 1 + 5 * 7/15) / 10.
    assert e["cohort_kept"]["class_0"]["case_id"] == 0.8
    assert e["cohort_kept"]["class_0"]["random_mean"] == pytest.approx(16 / 30, abs=0.03)
    assert b["tie_shift_max_abs"] == 0.0 and b["random_sd_max"] == 0.0 and b["random_tie_jaccard_mean"] == 1.0


def test_tie_share_counts_the_tie_value_while_cut_ties_need_a_split_block():
    m = diagnose(hub_rows(), manifest(), 0.2)["metrics"]  # K = 4: the four floor picks fill capacity
    q = m["policies"]["provenance_only"]
    assert m["K"] == 4 and m["tie_decided_share"] == 1.0 and q["cut_tie_share"] == 0.0
    assert q["tie_shift_max_abs"] == 0.0 and q["random_sd_max"] == 0.0


def test_tie_dominated_run_with_a_harmful_hub():
    m = diagnose(tie_rows(), manifest(min_per_cohort=0), 0.5)["metrics"]
    assert m["K"] == 6 and m["never_retrieved_share"] == pytest.approx(8 / 12)
    assert m["evidenced_share"] == pytest.approx(2 / 12) and m["evidenced_h0_share"] == 0.5
    assert m["spearman_q_a_evidenced"] is None  # two evidenced cases
    assert m["top10_activation_share"] == pytest.approx(7 / 7.25)
    # Q keeps id 6 then ids 1-5 of ten 0.5 ties; the harmful hub id 0 (Q = 2/7) is archived without a tie.
    q, b, e = (m["policies"][p] for p in ("provenance_only", "bias_current", "exposure_only"))
    assert m["tie_decided_share"] == pytest.approx(5 / 6) and q["cut_tie_share"] == pytest.approx(5 / 6)
    assert q["archived_activation_share"] == pytest.approx(5.25 / 7.25)
    assert b["archived_activation_share"] == pytest.approx(2.25 / 7.25) and b["cut_tie_share"] == 1.0
    assert e["archived_activation_share"] == 0.0
    assert m["jaccard"]["provenance_only|bias_current"] == pytest.approx({"all": 5 / 7, "retrieved": 0.0})
    assert m["jaccard"]["provenance_only|exposure_only"] == pytest.approx({"all": 1 / 3, "retrieved": 0.25})
    assert m["jaccard"]["bias_current|exposure_only"] == pytest.approx({"all": 1 / 3, "retrieved": 0.25})
    c0 = q["cohort_kept"]["class_0"]
    assert c0["case_id"] == pytest.approx(5 / 6) and c0["random_mean"] == pytest.approx(2.5 / 6, abs=0.03)
    assert q["tie_shift_max_abs"] == pytest.approx(2.5 / 6, abs=0.03)


def test_kept_sets_come_from_select_active_set_with_protection_and_floors():
    man = manifest()
    scfg, rcfg = run_config(man)
    scores, _ = load_scores(hub_rows(protect={19}), scfg)
    kept = {p: ids(retention_kept(scores, 10, rcfg, p)) for p in ("provenance_only", "bias_current", "exposure_only")}
    # id 19 is human-protected, so class_1 needs one floor pick (15); id 12 (Q = 0.6) leads the pool.
    assert kept["provenance_only"] == {0, 1, 2, 3, 4, 5, 7, 12, 15, 19}
    assert kept["exposure_only"] == {0, 1, 2, 3, 4, 5, 7, 12, 15, 19}
    assert kept["bias_current"] == {8, 9, 12, 13, 14, 15, 16, 17, 18, 19}
    assert diagnose(hub_rows(protect={19}), man, 0.5)["metrics"]["n_protected"] == 1
    # The random tie-break replica is select_active_set with permuted tie IDs.
    protected, floors, _ = protection_requirements(scores, _cfg(rcfg, "provenance_only"), None)
    cohorts = np.asarray(scores.cohorts)
    rng = np.random.default_rng(1)
    for perm in (rng.permutation(20) for _ in range(5)):
        for p in ("provenance_only", "bias_current", "exposure_only"):
            assert np.array_equal(fill(ranking(scores, p, rcfg), perm, cohorts, protected, floors, 10)[0],
                                  retention_kept(replace(scores, case_ids=perm), 10, rcfg, p))


def test_logged_maintenance_is_replayed(tmp_path):
    man = manifest()
    scfg, rcfg = run_config(man)
    scores, _ = load_scores(hub_rows(), scfg)

    def logged(policy, step, flip=None):
        sel = select_active_set(scores, 10, replace(rcfg, policy=policy), None)
        out = []
        for j, d in enumerate(sel.decisions):
            action = d["action"] if j != flip else ("keep" if d["action"] == "archive" else "archive")
            out.append({**d, "action": action, "run_id": "alternating_rrr", "step": step,
                        "policy_version": f"{policy}/pi-20260920-v2", "capacity_before": 20, "capacity_after": 10})
        return out

    events = logged("provenance_only", 40) + logged("bias_current", 80, flip=3) + [
        {**e, "step": 90, "policy_version": "kcenter/pi-20260920-v2"} for e in logged("provenance_only", 90)]
    history = {"core": [], "sync": [{"epoch": 40, "maintenance": {"n_after": 10}}, {"epoch": 41}, {"epoch": 80, "maintenance": None}]}
    run = analyse_run(write_run(tmp_path, "hub", "tuned__a", 20, hub_rows(), events, history), tmp_path, 0.5, 20, 0)
    assert run["context"]["maintenance_steps"] == 1
    assert run["context"]["replay"] == {
        "provenance_only": {"steps": 1, "replayed": 1, "agree": 1, "case_mismatches": 0},
        "bias_current": {"steps": 1, "replayed": 1, "agree": 0, "case_mismatches": 1},
        "kcenter": {"steps": 1, "replayed": 0, "agree": 0, "case_mismatches": 0}}


def test_cli_filters_averages_seeds_and_writes_tables(tmp_path):
    root = tmp_path / "runs"
    write_run(root, "hub", "tuned__a", 20, hub_rows())
    write_run(root, "hub", "tuned__a", 21, hub_rows(bias_sign=-1.0))  # bias keeps ids 0-7, 10, 11
    write_run(root, "hub", "tuned__a", 30, tie_rows())  # not a development seed
    write_run(root, "hub", "tuned__retrieval_only", 20, [])
    write_run(root, "tie", "tuned__b", 22, tie_rows(), min_per_cohort=0)
    out = tmp_path / "out"
    assert main([str(root), "--draws", "10", "--json", str(out / "d.json"), "--md", str(out / "d.md")]) == 0
    rep = json.loads((out / "d.json").read_text())
    a = rep["summary"]["hub"]["tuned__a"]
    assert a["seeds"] == [20, 21] and a["n"] == 20
    assert a["jaccard.provenance_only|bias_current.all"] == pytest.approx((1 / 9 + 8 / 12) / 2)
    assert a["jaccard.provenance_only|bias_current.retrieved"] == pytest.approx(0.5)
    assert rep["summary"]["tie"]["tuned__b"]["tie_decided_share"] == pytest.approx(5 / 6)
    assert [s["run_dir"] for s in rep["skipped"]] == ["hub/tuned__retrieval_only/s20"]
    assert sorted((r["task"], r["seed"]) for r in rep["runs"]) == [("hub", 20), ("hub", 21), ("tie", 22)]
    md = (out / "d.md").read_text()
    assert "## hub" in md and "## tie" in md and "| tuned__a | 2 | 20 | 0.75 | 0.20 |" in md
    assert main([str(root), "--condition", "tuned__b", "--draws", "0", "--json", str(out / "b.json"), "--md", str(out / "b.md")]) == 0
    assert list(json.loads((out / "b.json").read_text())["summary"]) == ["tie"]
    assert main([str(root), "--all-seeds", "--condition", "tuned__a", "--draws", "0", "--json", str(out / "c.json"),
                 "--md", str(out / "c.md")]) == 0
    assert json.loads((out / "c.json").read_text())["summary"]["hub"]["tuned__a"]["seeds"] == [20, 21, 30]
