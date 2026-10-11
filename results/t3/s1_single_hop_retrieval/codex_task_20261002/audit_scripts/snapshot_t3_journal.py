import hashlib
import json
from pathlib import Path
import shutil

work=Path(__file__).resolve().parent
out=work.parent/'outputs/t3-public-host-20261005-format-enforced'
def lines(path):
    raw=path.read_bytes();complete=raw[:raw.rfind(b'\n')+1]
    return raw,complete,[json.loads(line) for line in complete.splitlines()]
trialraw,trialprefix,rows=lines(out/'trials.jsonl')
journalraw,journalprefix,journal=lines(out/'retrieval_events.jsonl')
keys={(r['dataset'],r['question_id'],r['condition']) for r in rows}
included=[r for r in journal if (r['dataset'],r['question_id'],r['condition']) in keys]
snapshot=work.parent/f'outputs/t3-format-prefix-{len(rows)}'
snapshot.mkdir(exist_ok=False)
shutil.copyfile(out/'protocol.json',snapshot/'protocol.json')
(snapshot/'trials.jsonl').write_bytes(trialprefix)
(snapshot/'retrieval_events.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in included),encoding='utf-8')
summary=dict(provisional_snapshot=True,trials=len(rows),host_calls=sum(len(r['host_calls']) for r in rows),
    input_tokens=sum(c['input_tokens'] for r in rows for c in r['host_calls']),
    output_tokens=sum(c['output_tokens'] for r in rows for c in r['host_calls']))
(snapshot/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
manifest=dict(source=str(out),snapshot=str(snapshot),completed_trials=len(rows),journal_events=len(included),
    excluded_inflight_events=len(journal)-len(included),trial_bytes=len(trialraw),complete_trial_bytes=len(trialprefix),
    trial_prefix_sha256=hashlib.sha256(trialprefix).hexdigest(),journal_complete_sha256=hashlib.sha256(journalprefix).hexdigest(),
    boundary='completed-trial prefix snapshot while source runs; excludes in-flight events; not terminal80-trial results')
(snapshot/'snapshot.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest))
