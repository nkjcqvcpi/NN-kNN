import hashlib
import json
from pathlib import Path
import sys
import zipfile

work=Path(__file__).resolve().parent
sys.path.insert(0,str(work/'t3-data-libs'))
import pyarrow.parquet as pq
root=work/'t3-public-data-20261005'
table=pq.read_table(root/'hotpot_dev_distractor_mirror.parquet').to_pylist()
converted=json.loads((root/'hotpot_dev_distractor_converted.json').read_text(encoding='utf-8'))
assert len(table)==len(converted)==7405
for before,after in zip(table,converted):
    assert len(before['context']['title'])==len(before['context']['sentences'])
    assert len(before['supporting_facts']['title'])==len(before['supporting_facts']['sent_id'])
    for field,source in [('_id','id'),('question','question'),('answer','answer'),('type','type'),('level','level')]:
        assert after[field]==before[source]
    assert after['context']==list(map(list,zip(before['context']['title'],before['context']['sentences'])))
    assert after['supporting_facts']==list(map(list,zip(before['supporting_facts']['title'],before['supporting_facts']['sent_id'])))
report=json.loads((work.parent/'outputs/t3-public-data-audit-20261005-preserved-gaps/verification.json').read_text())
freeze=json.loads((work/'t3_public_data_freeze.json').read_text())
assert report['source_fingerprint']==freeze['fingerprint']
assert hashlib.sha256(Path(freeze['zip']).read_bytes()).hexdigest()==freeze['zip_sha256']
with zipfile.ZipFile(freeze['zip']) as bundle:
    assert bundle.testzip() is None and len(bundle.namelist())==freeze['entries']==135
    assert all(bundle.read(name)==(Path(freeze['archive'])/name).read_bytes() for name in bundle.namelist())
hp=next(r for r in report['records'] if r['name'].startswith('hotpot'))
assert hp['unavailable_supporting_facts']=={'5ae61bfd5542992663a4f261':[['Jimmy Butler (basketball)',902]]}
output=dict(source=freeze,converted_rows=7405,all_original_fields_preserved=True,source_annotation_gaps=hp['unavailable_supporting_facts'],
    scorer_pairs=report['scorer_pairs'],all_checks_passed=True,scope='full columnar/JSON comparison and sourceZIP integrity; no host outcome')
(work.parent/'outputs/neural-cbr-t3-public-conversion-verification.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
print('conversion exact',len(table),'scorer pairs',report['scorer_pairs'])
