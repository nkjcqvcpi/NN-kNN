from pathlib import Path
import hashlib
import json
import re
import sys
from dataclasses import replace
import zipfile
import torch

work = Path(__file__).resolve().parent
archive = work / "t3_version_binding_source_archive"
sys.path.insert(0, str(archive))
from model.t1.core import CoreConfig, build_model
from model.t1.artifacts import source_fingerprint
from model.t3.retrieval import Case, Access, Request, Retriever

torch.set_num_threads(1)
out = work.parent / "outputs/t3-version-binding-host-20261005"
freeze = json.loads((work / "t3_version_binding_freeze.json").read_text())
protocol = json.loads((out / "protocol.json").read_text())
assert protocol["source_fingerprint"] == freeze["fingerprint"] == source_fingerprint(archive)
zip_path = Path(freeze["zip"])
assert hashlib.sha256(zip_path.read_bytes()).hexdigest() == freeze["zip_sha256"]
with zipfile.ZipFile(zip_path) as z:
    assert z.testzip() is None and len(z.namelist()) == freeze["entries"]
state = torch.load(out / "retriever.pt", weights_only=True)
cases = [Case(**r) for r in state["cases"]]
model = build_model(state["model"]["cases"], torch.zeros(len(cases)), CoreConfig(**state["core_config"]), None)
model.load_state_dict(state["model"])
def encode(request):
    v = torch.zeros(64)
    for word in re.findall(r"[a-z]+", request.need.lower()):
        v[int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 64] += 1
    return (v/v.norm().clamp_min(1e-9)).unsqueeze(0)
trials = [json.loads(r) for r in (out / "trials.jsonl").read_text().splitlines()]
assert len(trials) == 16 and all("error" not in r for r in trials)
event_ids, records = [], []
for trial in trials:
    result = trial["result"]
    assert trial["exact_correct"] == ((result["answer"] or "").lower().strip() == trial["expected"])
    assert trial["abstained"] == ((result["answer"] or "").lower().strip() == "unknown")
    target = next(c.case_id for c in cases if trial["entity"] in c.content)
    bank = [replace(c, quarantined=True) if trial["condition"] == "quarantine_target" and c.case_id == target else c for c in cases]
    retriever = Retriever(model, bank, encode, model_version="fixed-fixture-0", encoder_version=state["encoder"])
    for event in result["events"]:
        audit = event["audit"]
        assert event["event_id"] == audit["event_id"]
        event_ids.append(event["event_id"])
        assert trial["entity"].lower() in audit["request"]["need"].lower()
        replay = retriever.retrieve(Request(**audit["request"]), Access("fixture-session", "fixture-user"), now=audit["timestamp"])
        assert replay["evidence"] == event["evidence"]
        for key in ("model_state_sha256", "encoder_state_sha256", "encoder_verification", "bank_version"):
            assert replay["audit"][key] == audit[key]
        assert audit["encoder_verification"] == "declared_only" and audit["encoder_state_sha256"] is None
        assert replay["audit"]["selected_ids"] == audit["selected_ids"]
        for actual, saved in zip(replay["audit"]["candidates"], audit["candidates"]):
            assert actual["case_id"] == saved["case_id"]
            for key in ("distance", "bias", "activation", "weight", "feature_distance_contributions"):
                torch.testing.assert_close(torch.tensor(actual[key]), torch.tensor(saved[key]))
        if trial["condition"] == "quarantine_target":
            assert target not in audit["eligible_ids"] and target not in audit["selected_ids"]
    for call in trial["host_calls"][1 if trial["condition"] != "no_retrieval" else len(trial["host_calls"]):]:
        output = call["output"].strip()
        if output.startswith("```json") and output.endswith("```"):
            output = output[7:-3].strip()
        decision = json.loads(output)
        assert decision["ready"] is True and decision["answer"] == result["answer"]
    records.append(dict(entity=trial["entity"], condition=trial["condition"], exact_correct=trial["exact_correct"],
        abstained=trial["abstained"], rounds=len(result["events"]), passed=True))
assert len(event_ids) == len(set(event_ids)) == 12
assert all(torch.equal(v, model.state_dict()[k]) for k,v in state["model"].items())
report = dict(trials=16, events=12, source=freeze, records=records, checks_passed=True,
    limits="fresh retrieval replay, stored host decisions and outcome checks; no new LLM generations or task-quality confidence claim")
(work.parent / "outputs/neural-cbr-t3-version-host-verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(dict(trials=16, events=12, checks_passed=True)))
