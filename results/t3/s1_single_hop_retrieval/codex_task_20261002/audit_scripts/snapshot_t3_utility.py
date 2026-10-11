import hashlib
import json
from pathlib import Path
import shutil
work=Path(__file__).resolve().parent
out=work.parent/'outputs/t3-public-set-utility-20261005'
def lines(name):
    raw=(out/name).read_bytes();complete=raw[:raw.rfind(b'\n')+1]
    return raw,complete,[json.loads(line) for line in complete.splitlines()]
raw,complete,rows=lines('set_audits.jsonl')
keys={(r['dataset'],r['question_id'],r['condition']) for r in rows}
snapshot=work.parent/f'outputs/t3-utility-prefix-{len(rows)}';snapshot.mkdir(exist_ok=False)
shutil.copyfile(out/'protocol.json',snapshot/'protocol.json')
(snapshot/'set_audits.jsonl').write_bytes(complete)
included={};excluded={}
for name in ('host_calls.jsonl','outcome_events.jsonl'):
    _,_,records=lines(name)
    chosen=[r for r in records if (r['dataset'],r['question_id'],r['condition']) in keys]
    included[name]=chosen;excluded[name]=len(records)-len(chosen)
    (snapshot/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen),encoding='utf-8')
calls=included['host_calls.jsonl']
summary=dict(provisional_snapshot=True,sets=len(rows),host_calls=len(calls),input_tokens=sum(c['input_tokens'] for c in calls),
    output_tokens=sum(c['output_tokens'] for c in calls),invalid_outcomes=sum(o['outcome'].get('task_failure',False) for r in rows for o in r.get('audit',{}).get('outcomes',[])))
(snapshot/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
manifest=dict(source=str(out),snapshot=str(snapshot),completed_sets=len(rows),completed_host_calls=len(calls),excluded_inflight=excluded,
    set_prefix_sha256=hashlib.sha256(complete).hexdigest(),set_complete_bytes=len(complete),set_read_bytes=len(raw),
    scope='completed-set prefix snapshot,not terminal32-set results; in-flight call/outcome records excluded explicitly')
(snapshot/'snapshot.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest))
