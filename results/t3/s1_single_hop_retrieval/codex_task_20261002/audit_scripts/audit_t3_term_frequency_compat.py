import hashlib
import json
from pathlib import Path
import torch
work=Path(__file__).resolve().parent;root=work.parent/'outputs'
new=root/'t3-term-frequency-default-compat-20261006'
old=root/'t3-need-query-default-compat-20261005'
def equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
a,b=[torch.load(path/'s8/metric.pt',weights_only=True) for path in (new,old)]
assert equal(a,b)
a,b=[json.loads((path/'s8/summary.json').read_text()) for path in (new,old)]
for row in (a,b):row.pop('training_seconds')
assert a==b and a['actual_training_steps']==a['nonzero_gradient_steps']==128
assert (new/'s8/training_events.json').read_bytes()==(old/'s8/training_events.json').read_bytes()
a,b=[json.loads((path/'protocol.json').read_text()) for path in (new,old)]
a.pop('source_fingerprint');b.pop('source_fingerprint');assert a.pop('term_frequency')=='count';assert a==b
report=dict(actual_training_updates=128,all_recorded_checks_passed=True,
    scope='real default count epoch; exact selected/final/initial/Adam/event/curve equality against archived pre-mode run; protocol only declares new source/explicit count',
    inputs={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for folder in (new,old) for path in (folder/'protocol.json',folder/'s8/metric.pt',folder/'s8/summary.json',folder/'s8/training_events.json')})
(root/'neural-cbr-t3-term-frequency-compat-verification.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='inputs'}))
