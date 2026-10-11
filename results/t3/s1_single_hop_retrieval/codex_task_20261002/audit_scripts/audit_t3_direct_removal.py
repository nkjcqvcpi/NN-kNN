from pathlib import Path
import hashlib
import json
import sys
import zipfile

work = Path(__file__).resolve().parent
archive = work / "t3_prompt_direct_removal_source_archive"
sys.path.insert(0, str(archive))
from model.t1.artifacts import source_fingerprint
from model.t3.retrieval import conditional_credit
out = work.parent / "outputs/t3-prompt-pilot-20261004-direct-removal"
freeze = json.loads((work / "t3_prompt_direct_removal_freeze.json").read_text())
protocol = json.loads((out / "protocol.json").read_text())
assert source_fingerprint(archive) == protocol["source_fingerprint"] == freeze["fingerprint"]
with zipfile.ZipFile(freeze["zip"]) as z:
    assert z.testzip() is None and len(z.namelist()) == freeze["entries"]
assert hashlib.sha256(Path(freeze["zip"]).read_bytes()).hexdigest() == freeze["zip_sha256"]
trials = [json.loads(r) for r in (out / "trials.jsonl").read_text().splitlines()]
counterfactuals = [json.loads(r) for r in (out / "direct_set_counterfactuals.jsonl").read_text().splitlines()]
assert len(trials) == 16 and len(counterfactuals) == 4
records = []
for cf in counterfactuals:
    assert "error" not in cf and cf["no_refill"] and cf["no_new_retrieval"]
    trial = next(t for t in trials if t["entity"] == cf["entity"] and t["condition"] == "nnknn_one_shot")
    result = trial["result"]
    assert cf["full_evidence"] == result["evidence"] and cf["full_answer"] == result["answer"]
    assert cf["removed_evidence"] == [c for c in result["evidence"] if c["case_id"] != cf["case_id"]]
    event = next(e for e in result["events"] if e["event_id"] == cf["source_event_id"])
    weight = next(c["weight"] for c in event["audit"]["candidates"] if c["case_id"] == cf["case_id"])
    full_json = json.dumps(cf["full_evidence"], ensure_ascii=False)
    removed_json = json.dumps(cf["removed_evidence"], ensure_ascii=False)
    full_prompt = trial["host_calls"][-1]["prompt"]
    assert full_prompt.count(full_json) == 1
    assert cf["host_calls"][0]["prompt"] == full_prompt.replace(full_json, removed_json)
    assert len(cf["host_calls"]) == 1
    output = cf["host_calls"][0]["output"].strip()
    if output.startswith("```json") and output.endswith("```"):
        output = output[7:-3].strip()
    decision = json.loads(output)
    assert decision["ready"] is True and decision["answer"] == cf["removed_answer"]
    full_loss = float(cf["full_answer"].strip().lower() != cf["expected"])
    removed_loss = float(cf["removed_answer"].strip().lower() != cf["expected"])
    assert cf["full_loss"] == full_loss and cf["removed_loss"] == removed_loss
    credit = conditional_credit(weight, full_loss, removed_loss)
    assert credit == cf["credit"]
    records.append(dict(entity=cf["entity"], case_id=cf["case_id"], full_answer=cf["full_answer"],
        removed_answer=cf["removed_answer"], full_loss=full_loss, removed_loss=removed_loss,
        credit=credit, same_prompt_except_evidence=True, passed=True))
summary = json.loads((out / "summary.json").read_text())
calls = [c for t in trials for c in t["host_calls"]]+[c for t in counterfactuals for c in t["host_calls"]]
assert len(calls) == summary["host_calls"] == 32
assert sum(c["input_tokens"] for c in calls) == summary["input_tokens"] == 4160
assert sum(c["output_tokens"] for c in calls) == summary["output_tokens"] == 852
report = dict(source=freeze, all_checks_passed=True, direct_removals=4, records=records,
    limits="four constructed single-case evidence sets; stored actual host generations and exact-set intervention audit; no global update or broad causal claim")
(work.parent / "outputs/neural-cbr-t3-direct-removal-20261004-verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(dict(direct_removals=4, all_checks_passed=True)))
