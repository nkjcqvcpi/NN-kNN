import hashlib
import json
import os
from pathlib import Path
import sys
work=Path(__file__).resolve().parent
native=Path('C:/Users/Administrator/NN-KNN/nnknn-work/.venv/Library/bin')
os.environ['PATH']=str(native)+os.pathsep+os.environ.get('PATH','')
handle=os.add_dll_directory(str(native)) if sys.platform=='win32' else None
sys.path.insert(0,'C:/Users/Administrator/NN-KNN/nnknn-work')
sys.path.insert(0,str(work/'t3-format-libs'))
from model.t3.host_schema import decoder_schema
from lmformatenforcer import JsonSchemaParser
paths=[work.parent/'outputs'/name/'need_queries.json' for name in ('t3-train-need-queries-20261005','t3-train-need-queries-256-20261005')]
old,new=[json.loads(path.read_text()) for path in paths]
assert len(old['records'])==len(new['records'])==192
errors=[]
for index,(a,b) in enumerate(zip(old['records'],new['records']),1):
    assert a['question_id']==b['question_id'] and a['question']==b['question']
    if a['error'] is None:
        assert b['error'] is None and a['call']['output']==b['call']['output']
    else:
        assert b['error'] is not None and a['call']['output_tokens']==128 and b['call']['output_tokens']==256
        text=b['call']['output']
        reject=None
        parser=JsonSchemaParser(decoder_schema('request',request_profile='bounded'))
        for offset,char in enumerate(text):
            if char not in parser.get_allowed_characters():
                reject=offset;break
            parser=parser.add_character(char)
        assert reject is not None and text[reject]!='"'
        errors.append(dict(position=index,question_id=a['question_id'],output128=a['call']['output'],output256=text,
            bounded_schema_first_rejected_character=reject,emitted_need_prefix=text.split('"need": "',1)[1][:512]))
assert len(errors)==2
report=dict(questions=192,unchanged_valid_outputs=190,repeated_failures=errors,all_recorded_checks_passed=True,
    input_artifacts=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths],
    boundary='actual strings repeat until both token ceilings; bounded parser rejects further string extension at512; no fresh model output or guarantee of repaired generation')
(work.parent/'outputs/neural-cbr-t3-request-loop-diagnosis.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(unchanged_valid_outputs=190,failures=[{k:v for k,v in row.items() if k not in ('output128','output256','emitted_need_prefix')} for row in errors])))
