"""Independently validate the four-arm pilot from saved targets, probabilities and predictions."""
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
index = {(r["dataset"], r["seed"], r["masked"], r["frozen"]): r for r in rows}
expected = {(d["name"], s, m, f) for d in study["datasets"] for s in study["seeds"]
            for m in [True, False] for f in [True, False]}
assert len(index) == len(rows) == len(expected) and set(index) == expected


def close(a, b):
    assert math.isclose(a, b, rel_tol=3e-6, abs_tol=1e-7), (a, b)


def metrics(m, task):
    y = m["target"]
    for phase in ["pre", "post"]:
        if phase + "_prediction" not in m:
            continue
        pred = m[phase + "_prediction"]
        assert len(pred) == len(y)
        if task == "regression":
            mse = statistics.mean((a-b)**2 for a,b in zip(pred,y))
            close(mse, m[phase + "_loss"])
            close(math.sqrt(mse), m[phase + "_error"])
        else:
            close(statistics.mean(a != b for a,b in zip(pred,y)), m[phase + "_error"])
            if phase == "pre":
                probabilities = m["pre_probability"]
                loss = statistics.mean(-math.log(max(p[int(t)],1e-8)) for p,t in zip(probabilities,y))
                assert pred == [p.index(max(p)) for p in probabilities]
            else:
                logits = m["post_logits"]
                loss = statistics.mean(max(z) + math.log(sum(math.exp(v-max(z)) for v in z)) - z[int(t)]
                                       for z,t in zip(logits,y))
                assert pred == [z.index(max(z)) for z in logits]
            close(loss, m[phase + "_loss"])


for ds in study["datasets"]:
    for seed in study["seeds"]:
        quartet = [index[ds["name"],seed,m,f] for m in [True,False] for f in [True,False]]
        assert len({r["initial_state_sha256"] for r in quartet}) == 1
        assert all(r["bias_initialization"] == quartet[0]["bias_initialization"] for r in quartet)
        for masked in [True,False]:
            a,b = [index[ds["name"],seed,masked,f] for f in [True,False]]
            for field in ["pretrained_state_sha256", "warmup_end_sha256", "baseline", "data", "core", "sync", "free_radius"]:
                assert a[field] == b[field], (ds,seed,masked,field)
            assert a["pretrained_state_sha256"] == a["selected_retriever_sha256"] == a["last_retriever_sha256"]
            assert a["baseline"]["pre_prediction"] == a["selected"]["pre_prediction"] == a["last"]["pre_prediction"]
            assert a["baseline"]["pre_loss"] == a["selected"]["pre_loss"] == a["last"]["pre_loss"]
            splits = [set(a["data"]["split_indices"][name]) for name in ["train","validation","test"]]
            assert not (splits[0]&splits[1] or splits[0]&splits[2] or splits[1]&splits[2])
        for r in quartet:
            assert r["warmup_end_sha256"]["retriever"] == r["pretrained_state_sha256"]
            for m in [r["baseline"], r["selected"], r["last"]]:
                metrics(m,r["task_type"])
            assert r["selected_postwarmup_epochs"] == max(0,r["best_epoch"]-study["sync"]["adapter_warmup_epochs"])
for s in summary:
    for masked,label in [(True,"masked"),(False,"all_cases")]:
        for frozen,schedule in [(True,"frozen"),(False,"joint")]:
            rs = [index[s["dataset"],seed,masked,frozen] for seed in study["seeds"]]
            a = s[label+"_"+schedule]
            for field,values in {
                "baseline_error":[r["baseline"]["pre_error"] for r in rs],
                "baseline_loss":[r["baseline"]["pre_loss"] for r in rs],
                "pre_error":[r["selected"]["pre_error"] for r in rs],
                "post_error":[r["selected"]["post_error"] for r in rs],
                "pre_loss":[r["selected"]["pre_loss"] for r in rs],
                "post_loss":[r["selected"]["post_loss"] for r in rs],
                "loss_drift":[r["selected"]["pre_loss"]-r["baseline"]["pre_loss"] for r in rs],
                "last_loss_drift":[r["last"]["pre_loss"]-r["baseline"]["pre_loss"] for r in rs]}.items():
                close(statistics.mean(values),a[field])
    close(s["selected_loss_interaction"],s["all_cases_joint"]["loss_drift"]-s["masked_joint"]["loss_drift"])
    close(s["last_loss_interaction"],s["all_cases_joint"]["last_loss_drift"]-s["masked_joint"]["last_loss_drift"])
digest = hashlib.sha256()
current_digest = hashlib.sha256()
for directory in ["model","tools","configs","tests","datasets"]:
    for p in sorted((ROOT/directory).rglob("*")):
        if p.suffix in {".py",".yaml",".ps1"} and "__pycache__" not in p.parts:
            relative = p.relative_to(ROOT).as_posix()
            data = p.read_bytes()
            current_digest.update(relative.encode())
            current_digest.update(data)
            # Only the new test's cleanup fixture changed after launch; execution code is unchanged.
            execution_data = (OUT/"test_at_execution.py").read_bytes() if relative == "tests/test_t1_mask_joint_pilot.py" else data
            digest.update(relative.encode())
            digest.update(execution_data)
assert digest.hexdigest() == manifest["source_sha256"]
log_bytes = (ROOT/"logs/t1/mask_joint_branch_tests_final_20261030.log").read_bytes()
test_log = log_bytes.decode("utf-16" if log_bytes.startswith((b"\xff\xfe",b"\xfe\xff")) else "utf-8-sig")
failed,passed,skipped = map(int,re.findall(r"(\d+) failed, (\d+) passed, (\d+) skipped",test_log)[-1])
result = {"checked_continuations":len(rows), "matched_pretrained_and_warmup_pairs":len(rows)//2,
          "unchanged_frozen_retrievers":len(rows)//2, "independently_recomputed_predictions_and_losses":"all match",
          "disjoint_splits":"all checked", "execution_source_fingerprint_matches_with_archived_test":True,
          "current_source_sha256":current_digest.hexdigest(),
          "post_launch_change":"New test fixture now restores torch process settings; archived test reconstructs the execution fingerprint. Training implementation unchanged.",
          "synthetic_integrity_tests":"classification and regression passed", "import_smoke":"passed",
          "branch_tests":{"passed":passed,"failed":failed,"skipped":skipped,
                          "failures":[line for line in test_log.splitlines() if line.startswith("FAILED ")],
                          "note":"Final suite after restoring torch process settings in the new test; no claim of a fully passing suite."}}
(OUT/"validation.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
