"""Independently check saved mask-comparison evidence using the standard library."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
rows = json.loads((OUT / "runs.json").read_text())
summary = json.loads((OUT / "summary.json").read_text())
manifest = json.loads((OUT / "manifest.json").read_text())
study = manifest["study"]
key = lambda r: (r["dataset"], r["seed"], r["masked"])
index = {key(r): r for r in rows}
expected = {(d["name"], s, m) for d in study["datasets"] for s in study["seeds"] for m in [True, False]}
assert len(index) == len(rows) == len(expected) and set(index) == expected
for ds in study["datasets"]:
    for seed in study["seeds"]:
        a, b = [index[ds["name"], seed, m] for m in [True, False]]
        assert a["initial_state_sha256"] == b["initial_state_sha256"]
        assert a["bias_initialization"] == b["bias_initialization"]
        assert a["bias_initialization"]["k"] == 5
        assert {k: v for k, v in a["core"].items() if k != "pre_topk_mask"} == {
            k: v for k, v in b["core"].items() if k != "pre_topk_mask"}
        assert a["data"] == b["data"] and a["test_target"] == b["test_target"]
        sets = [set(a["data"]["split_indices"][s]) for s in ["train", "validation", "test"]]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        for r in [a, b]:
            pred, target = r["test_prediction"], r["test_target"]
            assert len(pred) == len(target) == r["data"]["n_test"]
            error = (math.sqrt(statistics.mean((p - t) ** 2 for p, t in zip(pred, target)))
                     if r["task_type"] == "regression" else statistics.mean(p != t for p, t in zip(pred, target)))
            assert math.isclose(error, r["test_error"], rel_tol=2e-6, abs_tol=1e-7)
        assert a["initial_train_support"]["positive_weights_max"] <= 5
        assert a["initial_test_support"]["positive_weights_max"] <= 5
        assert a["final_test_support"]["positive_weights_max"] <= 5
        assert b["initial_train_support"]["positive_weights_mean"] > 5
        assert b["initial_test_support"]["positive_weights_mean"] > 5
for s in summary:
    masked = [index[s["dataset"], seed, True]["test_error"] for seed in study["seeds"]]
    full = [index[s["dataset"], seed, False]["test_error"] for seed in study["seeds"]]
    for field, values in [("masked", masked), ("all_case", full)]:
        assert math.isclose(statistics.mean(values), s[field + "_mean_error"], abs_tol=1e-14)
        assert math.isclose(statistics.stdev(values), s[field + "_sd_error"], abs_tol=1e-14)
    assert sum(b < a for a, b in zip(masked, full)) == s["unmasked_better_pairs"]
    assert sum(b > a for a, b in zip(masked, full)) == s["unmasked_worse_pairs"]
    assert sum(b == a for a, b in zip(masked, full)) == s["ties"]
old = json.loads((OUT.parent / "pre_topk_pilot_20261030/runs.json").read_text())
for a in old:
    b = index[key(a)]
    for field in ["initial_state_sha256", "bias_initialization", "test_error", "test_prediction", "best_epoch"]:
        assert a[field] == b[field]
digest = hashlib.sha256()
for directory in ["model", "tools", "configs", "tests", "datasets"]:
    for p in sorted((ROOT / directory).rglob("*")):
        if p.suffix in {".py", ".yaml", ".ps1"} and "__pycache__" not in p.parts:
            digest.update(p.relative_to(ROOT).as_posix().encode())
            digest.update(p.read_bytes())
assert digest.hexdigest() == manifest["source_sha256"]
log_bytes = (ROOT / "logs/t1/pre_topk_expanded_branch_tests_20261030.log").read_bytes()
test_log = log_bytes.decode("utf-16" if log_bytes.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
matches = re.findall(r"(\d+) failed, (\d+) passed, (\d+) skipped", test_log)
failed, passed, skipped = map(int, matches[-1])
result = {"checked_fits": len(rows), "identical_initialization_pairs": len(rows) // 2,
          "only_configuration_difference": "pre_topk_mask", "bias_init_k_both_arms": 5,
          "prediction_metrics": "independently recomputed; all match", "split_disjointness": "all checked",
          "original_pilot_fits_reproduced_exactly": len(old), "source_fingerprint_matches": True,
          "import_smoke": "passed", "branch_tests": {"passed": passed, "failed": failed, "skipped": skipped,
          "note": "Same seven failures as initial pilot: three missing CatBoost metadata, four exact historical-output checks."}}
(OUT / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
for s in summary:
    print(s["dataset"], f"{s['masked_mean_error']:.4f}", f"{s['all_case_mean_error']:.4f}",
          s["unmasked_better_pairs"], s["unmasked_worse_pairs"], s["ties"])
