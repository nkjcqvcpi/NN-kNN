"""Recount saved ag1/ag2 regression gate decisions; no fitting or inference.

Fixed historical audit: seven datasets, seeds 40/41/42, six method variants.
Writes a generated summary under results/; leaves recorded runs unchanged.
"""
import json
import numpy as np
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from model.common import tracking
from model.t1 import records
tasks=('airfoil','car','yacht','energy_efficiency','student_performance','body_fat','diabets')
methods=('original','disagreement_gate','probability_gate','utility_gate','threshold_frozen','threshold_joint')
output=[]
for method in methods:
    totals=dict(n=0,applied=0,wrong=0,wrong_full=0,harm=0,help=0,harm_full=0,help_full=0,wrong_skipped=0,help_skipped=0)
    metrics=[]; provenance=[]
    for task in tasks:
        for seed in (40,41,42):
            is_static=method in ('original','disagreement_gate')
            is_threshold=method.startswith('threshold_')
            study,exp=('ag2','threshold_v1') if is_threshold else ('ag1','gate_v1')
            arm='controls' if is_static else method
            key=records.key(study,exp,task,arm,'s'+str(seed))
            unit=tracking.load_unit(key,config=None,arrays=False)
            assert unit is not None
            state=method+'_test' if is_static else 'test_predictions'
            a=unit.load_state(state)
            y,p,f,full=(a[k].numpy().astype(float).reshape(-1) for k in ('target','retrieval','final','full'))
            apply=a['apply'].numpy().astype(bool).reshape(-1)
            assert y.shape==p.shape==f.shape==full.shape==apply.shape
            # Float32 additive reconstruction may differ from the full branch by rounding.
            np.testing.assert_allclose(f,np.where(apply,full,p),atol=1e-6,rtol=1e-6)
            r,c,cf=y-p,f-p,full-p
            wrong=(abs(r)>1e-8)&(abs(c)>1e-8)&(np.sign(r)!=np.sign(c))
            wrong_full=(abs(r)>1e-8)&(abs(cf)>1e-8)&(np.sign(r)!=np.sign(cf))
            change=(y-f)**2-(y-p)**2; change_full=(y-full)**2-(y-p)**2
            rec=next(r for r in records.rows(unit,'rows') if r['method']==method)
            assert wrong.sum()==rec['test_wrong_direction']
            metrics.append(dict(task=task,seed=seed,rmse=np.sqrt(np.mean((y-f)**2)),full_rmse=np.sqrt(np.mean((y-full)**2))))
            np.testing.assert_allclose(metrics[-1]['rmse'],rec['test_error'],rtol=1e-6,atol=1e-8)
            sums=dict(n=len(y),applied=apply.sum(),wrong=wrong.sum(),wrong_full=wrong_full.sum(),harm=(change>1e-8).sum(),help=(change < -1e-8).sum(),
                      harm_full=(change_full>1e-8).sum(),help_full=(change_full < -1e-8).sum(),wrong_skipped=(wrong_full & ~apply).sum(),
                      help_skipped=((change_full < -1e-8)&~apply).sum())
            for k,v in sums.items(): totals[k]+=int(v)
            provenance.append(dict(unit=key.run_id,state=state,n=len(y),applied=int(apply.sum()),wrong=int(wrong.sum()),wrong_full=int(wrong_full.sum())))
    totals.update(wrong_pct_all=100*totals['wrong']/totals['n'],wrong_pct_applied=100*totals['wrong']/totals['applied'],
                  own_full_wrong_pct=100*totals['wrong_full']/totals['n'],apply_pct=100*totals['applied']/totals['n'],
                  applied_harm_pct=100*totals['harm']/totals['applied'],
                  mean_rmse=float(np.mean([np.mean([r['rmse'] for r in metrics if r['task']==t]) for t in tasks])),
                  own_full_mean_rmse=float(np.mean([np.mean([r['full_rmse'] for r in metrics if r['task']==t]) for t in tasks])))
    output.append(dict(method=method,**totals,per_run=metrics,sources=provenance))
    print(json.dumps({k:v for k,v in output[-1].items() if k not in ('per_run','sources')},sort_keys=True))
path=REPO / 'results/adaptation_improvement/ag2/gating_direction_denominators.json'
path.write_text(json.dumps(dict(scope='126 regression method/split prediction sets from recorded units; no model inference',
                 note='Denominator applied uses saved apply decisions, including any zero proposals; repeated observations across seeds',
                 results=output),indent=2)+'\n')
print('saved',path)
