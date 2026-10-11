import json
from pathlib import Path
import sys
import torch
work=Path(__file__).resolve().parent
sys.path.insert(0,str(work/'t2_critic_label_matching_source_archive'))
import model.nnknn_rl_workflow as w
torch.set_num_threads(1)
baseline={r['seed']:r for r in json.loads((work/'acrobot_startup_8192/manifest.json').read_text())['roles'] if r['configured_target_mode']=='hard'}
rows=[]
for package in ('critic_label_modes_8192','critic_matching_m025','critic_matching_m1'):
    for row in json.loads((work/package/'manifest.json').read_text())['roles']:
        base_row=baseline[row['seed']]
        mode=package+'/'+row['critic_label_mode']
        base=w.load_nnknn_rl_checkpoint(Path(base_row['run_dir'])/'checkpoint.pt',device='cpu')
        live=w.load_nnknn_rl_checkpoint(Path(row['run_dir'])/'checkpoint.pt',device='cpu')
        saved=torch.load(Path(base_row['run_dir'])/'checkpoint.pt',weights_only=True,map_location='cpu')
        cfg=w.NNKNNRLConfig(**saved['config'])
        base_events=[];events=[]
        kwargs=dict(episodes=2,seed=40000+2*row['seed'],global_step=8192,device='cpu',max_audit_queries=8)
        reference=w.evaluate_critic_holdout(row['task'],base['model'],base['value_model'],base['target_model'],cfg,
            audit_sink=base_events.append,**kwargs)
        actual=w.evaluate_critic_holdout(row['task'],base['model'],live['value_model'],base['target_model'],cfg,
            audit_sink=events.append,**kwargs)
        assert reference==base_row['independent_mc']
        for key in ('critic_holdout_target_mean','critic_holdout_samples','critic_holdout_environment_steps',
                    'critic_holdout_mean_episode_return','critic_holdout_truncated_bootstraps'):
            assert actual[key]==reference[key]
        assert len(base_events)==len(events)==8
        for left,right in zip(base_events,events):
            assert (left['query'],left['target'])==(right['query'],right['target'])
        rows.append(dict(mode=mode,role=row['role'],seed=row['seed'],baseline=reference,common_policy=actual,
            own_policy=row['independent_mc'],matching_reference_queries=8,all_checks_passed=True))
        print('common policy',len(rows),'/24',mode,row['role'],flush=True)
(work.parent/'outputs/neural-cbr-t2-label-fixed-policy-verification.json').write_text(json.dumps(dict(
    runs=rows,all_checks_passed=True,scope='fixed baseline hard/fixed-label actor and time-limit bootstrap; off-policy transfer diagnostic, not on-policy calibration'),indent=2),encoding='utf-8')
