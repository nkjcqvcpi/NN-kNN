import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
work=Path(__file__).resolve().parent
freeze=json.loads((work/'t3_internal_audited_freeze.json').read_text())
preflight=work.parent/'outputs/neural-cbr-t3-internal-preflight-verification.json'
assert json.loads(preflight.read_text())['source_fingerprint']==freeze['fingerprint']
plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),freeze=freeze,alpha=.1,conditions=['none','prompt','internal','combined'],
    questions=16,answer_calls=64,answer_token_ceiling=128,maximum_new_tokens=8192,
    host='Qwen3-0.6B c1899de289a04d12100db370d81485cdf75e47ca',trainable_host_parameters=0,trainable_retrieval_parameters=0,
    query_source='shared actual source80 learned request outputs; no new request generation or driver rewrite',
    retrievals='16 fresh maintained-core retrievals reproduce shared original sets; unchanged old need-s8 loss-selected metric',
    preflight_sha256=hashlib.sha256(preflight.read_bytes()).hexdigest(),
    scope='existing exploratory questions; source failures retained; no reserved question or alpha/seed selection on new outputs',
    cost_boundary='one shared archived request/fresh retrieval per question; new answer-only opportunities equal but prompt/interface compute differs',
    scope_limits='candidate lexical internal interface; no fresh full-loop or stronger host/internal architecture confirmation')
with (work/'t3_internal_experiment_plan.json').open('x') as file:json.dump(plan,file,indent=2)
print('Prespecified four internal mechanism arms before new host answers')
