"""Bound actual train-only public host needs; no implicit query replacement."""
import hashlib
import json
from pathlib import Path
import re

from .host_schema import request_prompt,request_decode,validate_decision


def load_need_queries(path,tasks,*,data_sha256,sample_sha256):
    raw=Path(path).read_bytes();artifact=json.loads(raw)
    protocol=artifact['protocol']
    if protocol.get('version') not in (1,2) or protocol.get('dataset')!='squad_train_v11.json' or protocol.get('data_sha256')!=data_sha256 or protocol.get('sample_sha256')!=sample_sha256:
        raise ValueError('need data/sample/version binding mismatch')
    ceiling=128
    if protocol['version']==2:
        ceiling=protocol.get('request_token_ceiling')
        if protocol.get('decode')!=request_decode(protocol.get('decode',{}),ceiling):
            raise ValueError('need request decode ceiling mismatch')
        if protocol.get('maximum_calls')!=len(tasks) or protocol.get('maximum_generated_tokens')!=len(tasks)*ceiling:
            raise ValueError('need request total budget mismatch')
    if protocol.get('query_source')!='actual frozen host request' or protocol.get('public_dev_annotations_used') is not False:
        raise ValueError('actual train-only needs required')
    host=protocol.get('host',{})
    if not isinstance(host,dict) or not isinstance(host.get('revision'),str) or not re.fullmatch('[0-9a-f]{40}',host['revision']) or not isinstance(host.get('files'),list) or not host['files']:
        raise ValueError('frozen host revision/file binding required')
    names=set()
    for file in host['files']:
        if not isinstance(file,dict) or not isinstance(file.get('name'),str) or Path(file['name']).name!=file['name'] or file['name'] in names or not isinstance(file.get('sha256'),str) or not re.fullmatch('[0-9a-f]{64}',file['sha256']) or type(file.get('bytes')) is not int or file['bytes']<1:
            raise ValueError('complete unique host file bindings required')
        names.add(file['name'])
    expected={t.question_id:t.question for t in tasks}
    if len(expected)!=len(tasks):raise ValueError('task IDs must be unique')
    queries={}
    for record in artifact['records']:
        if set(record)!={'question_id','question','request','call','error'}:raise ValueError('exact public need record fields required')
        qid=record['question_id']
        if qid not in expected or qid in queries or record['question']!=expected[qid]:raise ValueError('need question identity mismatch')
        if record['error'] is not None or record['request'] is None:raise ValueError('unobserved need cannot be filled,filtered or replaced')
        call=record['call']
        if call['stage']!='request' or call['source_prompt']!=request_prompt(expected[qid]) or call['source_prompt'] not in call['prompt']:
            raise ValueError('need prompt/stage binding mismatch')
        if type(call['input_tokens']) is not int or call['input_tokens']<1 or type(call['output_tokens']) is not int or not 1<=call['output_tokens']<=ceiling:
            raise ValueError('actual need token accounting required')
        emitted=validate_decision(json.loads(call['output'].strip()),'request')
        if emitted!=record['request']:raise ValueError('stored need differs from actual emitted request')
        queries[qid]=emitted['need']
    if set(queries)!=set(expected):raise ValueError('complete prespecified fit/tune need coverage required')
    return queries,dict(path=str(Path(path).resolve()),sha256=hashlib.sha256(raw).hexdigest(),protocol=protocol,records=len(queries))
