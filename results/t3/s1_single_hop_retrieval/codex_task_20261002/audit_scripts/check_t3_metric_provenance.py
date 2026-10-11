import json
from pathlib import Path
import sys
repo=Path(r'C:/Users/Administrator/NN-KNN/nnknn-work')
work=Path(__file__).resolve().parent
sys.path.insert(0,str(repo))
from model.t3.metric_provenance import metric_provenance
data=json.loads((work/'t3-public-data-20261005/manifest_mirror_complete.json').read_text())
train=next(row for row in data['files'] if row['name']=='squad_train_v11.json')
samples=json.loads((work.parent/'outputs/t3-public-data-audit-20261005-preserved-gaps/verification.json').read_text())['samples']['squad_train_v11.json']
records=[]
for name in ('t3-need-matching-20261005','t3-hard-negative-20261005'):
    for seed in (8,9,10):
        result=metric_provenance(work/name/f's{seed}/metric.pt')
        assert result['data_source']==train
        assert result['fit_ids']==samples['fit_ids'] and result['tune_ids']==samples['tune_ids']
        records.append(result)
(work.parent/'outputs/neural-cbr-t3-metric-provenance-verification.json').write_text(json.dumps(dict(runs=records,all_recorded_checks_passed=True),indent=2))
print([(row['seed'],row['selection'],row['selected_epoch']) for row in records])
