import hashlib
import json
from pathlib import Path
import sys

work=Path(__file__).resolve().parent
sys.path.insert(0,str(work/'t3-format-libs'))
sys.path.insert(0,r'C:\Users\Administrator\NN-KNN\nnknn-work')
from lmformatenforcer import JsonSchemaParser
from model.t3.host_schema import decision_schema,decoder_schema,validate_decision
def parse(schema,decision):
    text=json.dumps(decision)
    parser=JsonSchemaParser(schema)
    for character in text:
        assert character in parser.get_allowed_characters(),(character,text)
        parser=parser.add_character(character)
    assert parser.can_end()
request=dict(ready=False,need='Which city?',requested_types=['evidence'],observable_task_state='awaiting evidence')
answer=dict(ready=True,answer='UNKNOWN',supporting_facts=[])
try:parse(decision_schema('request'),request)
except Exception as error:observed=dict(type=type(error).__name__,message=str(error))
else:raise AssertionError('expected known boolean const incompatibility not reproduced')
for stage,decision in [('request',request),('answer',answer),('continuation',request),('continuation',answer)]:
    parse(decoder_schema(stage),decision)
    validate_decision(decision,stage)
relaxed=dict(request,ready=True)
parse(decoder_schema('request'),relaxed)
try:validate_decision(relaxed,'request')
except ValueError:pass
else:raise AssertionError('stage validation must reject parser-allowed wrong readiness')
report=dict(backend='lm-format-enforcer0.11.3',observed_boolean_const_failure=observed,valid_stage_checks=4,
    wrong_readiness_parses_but_runtime_rejects=True,all_checks_passed=True,scope='real backend character-parser tests,not host generations')
(work.parent/'outputs/neural-cbr-t3-format-parser-verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
