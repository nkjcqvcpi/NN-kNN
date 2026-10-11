import json
from pathlib import Path
import subprocess
import sys

work = Path(__file__).resolve().parent
if len(sys.argv) == 1:
    records = []
    for archive in ('t3_prompt_direct_removal_source_archive','t3_version_binding_source_archive'):
        completed = subprocess.run([sys.executable,__file__,str(work/archive)],capture_output=True,text=True,check=True)
        records.append(json.loads(completed.stdout.splitlines()[-1]))
    old,new = records
    assert old['string_validation_admitted'] and not new['string_validation_admitted']
    assert old['metadata_replaced'] and not new['metadata_replaced']
    assert old['bias_drift_delivered'] and not new['bias_drift_delivered']
    assert not old['active_only_constructed'] and new['active_only_constructed']
    report = dict(records=records,all_checks_passed=True,scope='actual old/new core contract counterexamples; no downstream benchmark')
    (work.parent/'outputs/neural-cbr-t3-version-binding-probes.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report)); sys.exit()
sys.path.insert(0,sys.argv[1])
import torch
from model.t1.core import CoreConfig,build_model
from model.t3.retrieval import Access,Case,Request,Retriever
result = dict(archive=sys.argv[1])
try:
    Case(0,'original','evidence','fixture','global',validated='false')
    result['string_validation_admitted'] = True
except ValueError:
    result['string_validation_admitted'] = False
def bank():
    model = build_model(torch.tensor([[0.],[1.],[2.]]),torch.zeros(3),CoreConfig(task_type='regression',bias_init='manual'),None)
    cases = [Case(i,str(i),'evidence','fixture','global',validated=True) for i in range(3)]
    return model,cases
model,cases = bank()
r = Retriever(model,cases,lambda request:torch.tensor([[0.]]),model_version='v1',encoder_version='v1')
try:
    r.cases[0] = Case(0,'replacement','evidence','different','global',validated=True)
    result['metadata_replaced'] = True
except TypeError:
    result['metadata_replaced'] = False
with torch.no_grad(): model.biases[0].add_(2)
try:
    r.retrieve(Request('need',('evidence',)),Access('s','u'),now=1)
    result['bias_drift_delivered'] = True
except ValueError:
    result['bias_drift_delivered'] = False
model,cases = bank(); model.set_active_case_count(2); model.case_ids[2:] = -1
try:
    r = Retriever(model,cases[:2],lambda request:torch.tensor([[0.]]),model_version='v1',encoder_version='v1')
    r.retrieve(Request('need',('evidence',)),Access('s','u'),now=1)
    result['active_only_constructed'] = True
except ValueError:
    result['active_only_constructed'] = False
print(json.dumps(result))
