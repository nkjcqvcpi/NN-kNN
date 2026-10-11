import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import torch

work=Path(__file__).resolve().parent
freeze=json.loads((work/'t3_train_need_queries_freeze.json').read_text());archive=Path(freeze['archive']);sys.path.insert(0,str(archive))
from model.t1.artifacts import source_fingerprint
from model.t3.host_schema import request_prompt
assert source_fingerprint(archive)==freeze['fingerprint']
assert hashlib.sha256(Path(freeze['zip']).read_bytes()).hexdigest()==freeze['zip_sha256']
with zipfile.ZipFile(freeze['zip']) as bundle:
    assert bundle.testzip() is None and len(bundle.namelist())==147
    assert all(bundle.read(name)==(archive/name).read_bytes() for name in bundle.namelist())
new=work.parent/'outputs/t3-need-query-default-compat-20261005/s8'
old=work/'t3-softmax-recall-20261005/s8'
current=torch.load(new/'metric.pt',weights_only=True);reference=torch.load(old/'metric.pt',weights_only=True)
def same(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b
for field in ('selected_metric','selected_optimizer','initial_metric'):same(current[field],reference[field])
same(current['final_metric'],reference['selected_metric']);same(current['final_optimizer'],reference['selected_optimizer'])
cur_summary=json.loads((new/'summary.json').read_text());old_summary=json.loads((old/'summary.json').read_text())
assert cur_summary['history']==old_summary['history'][:2]
cur_events=json.loads((new/'training_events.json').read_text());old_events=json.loads((old/'training_events.json').read_text())
assert len(cur_events)==128 and cur_events==old_events[:128]
old_source=work/'t3_format_enforced_source_archive/tools/t3_public_host_pilot.py'
tree=ast.parse(old_source.read_text())
callback=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='callback')
expression=callback.body[0].body[0].value
assert all(isinstance(n,(ast.BinOp,ast.Add,ast.Constant,ast.Name,ast.Load)) for n in ast.walk(expression))
compiled=compile(ast.Expression(expression),'frozen_request_prompt','eval')
proto=json.loads((work.parent/'outputs/t3-public-host-20261005-format-enforced/protocol.json').read_text())
questions=[q['question'] for group in proto['questions'].values() for q in group]
for question in questions:assert eval(compiled,{'__builtins__':{}},{'task':question})==request_prompt(question)
report=dict(source=freeze,actual_nonzero_updates=sum(e['gradient_norm']>0 for e in cur_events),training_events=128,
    exact_state_checks=['selected/final metric','selected/final Adam','initial metric','two history rows','128 training events'],
    frozen_prompt_equivalence_questions=len(questions),all_recorded_checks_passed=True,
    boundary='one actual seed8/one-epoch default-query retraining reproduces frozen old epoch1 learning states; literal prompt extraction equivalence,not actual-need fitting or public benefit')
(work.parent/'outputs/neural-cbr-t3-need-query-compat-verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='source'}))
