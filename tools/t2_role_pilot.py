"""Bounded current-core four-role pilot, explicitly not a full-cycle gate."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch

from model.nnknn_rl_workflow import (
    NNKNNValueNetwork, make_nnknn_rl_config, train_nnknn_rl,
    evaluate_critic_holdout,
)
from model.t1.artifacts import source_fingerprint
from model.t2.audit import summarize_credit


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    ap.add_argument('--steps',type=int,default=512)
    ap.add_argument('--seeds',type=int,nargs='+',default=[8,9,10])
    ap.add_argument('--tasks',nargs='+',default=['cartpole'])
    args=ap.parse_args()
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parents[1]
    fingerprint=source_fingerprint(source)
    import gymnasium
    versions=dict(torch=torch.__version__,gymnasium=gymnasium.__version__)
    records=[]
    torch.set_num_threads(1)
    for task in args.tasks:
      for seed in args.seeds:
        for actor_type,critic_type in [('mlp','mlp'),('nnknn','mlp'),('mlp','nnknn'),('nnknn','nnknn')]:
            label=f'{task}/{actor_type}_{critic_type}/s{seed}'
            cfg=make_nnknn_rl_config('smoke',seed=seed,actor_type=actor_type,critic_type=critic_type,
                total_timesteps=args.steps,case_capacity=128,critic_case_capacity=128,
                eval_frequency=0,eval_episode_frequency=100000,eval_episodes=3,
                critic_holdout_episode_frequency=0,policy_update_episodes=2,
                early_stopping=False,success_threshold=None,top_k=8,
                min_case_entries=8,min_cases_per_action=2)
            start=time.perf_counter()
            state=train_nnknn_rl(task,cfg,output_dir=root/label,device='cpu',progress=False)
            train_seconds=time.perf_counter()-start
            events=[]
            holdout=evaluate_critic_holdout(task,state['model'],state['value_model'],state['target_model'],
                cfg,episodes=2,seed=40000+seed*2,global_step=args.steps,device='cpu',
                audit_sink=events.append,max_audit_queries=8)
            assert state['summary']['actual_timesteps']==args.steps
            if isinstance(state['value_model'],NNKNNValueNetwork):
                assert events
                for e in events:
                    assert abs(e['full_prediction']-e['unmasked_holdout_prediction'])<1e-5
            audit_state=dict(critic_state=state['value_model'].state_dict(),
                target_state=state['target_model'].state_dict() if state['target_model'] else None,
                config=asdict(cfg))
            run_dir=Path(state['run_dir'])
            torch.save(audit_state,run_dir/'role_audit_state.pt')
            (run_dir/'role_audit_events.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
            credit=summarize_credit(events)
            credit_json=[dict(role=k[0],stream=k[1],cases=v) for k,v in credit.items()]
            record=dict(role=label,task=task,seed=seed,actor_type=actor_type,critic_type=critic_type,
                source_fingerprint=fingerprint,versions=versions,run_dir=str(run_dir),
                training_seconds=train_seconds,summary=state['summary'],independent_mc=holdout,
                audit_events=len(events),interventions=sum(len(e['interventions']) for e in events),
                credit=credit_json,checkpoint_sha256=hashlib.sha256((run_dir/'checkpoint.pt').read_bytes()).hexdigest())
            records.append(record)
            (root/'manifest.json').write_text(json.dumps(dict(status='exploratory_current_core_not_stage_a_gate',
                roles=records,source_fingerprint=fingerprint,versions=versions),indent=2),encoding='utf-8')
            print(f'completed {len(records)}/{4*len(args.seeds)*len(args.tasks)} {label} events={len(events)}',flush=True)


if __name__=='__main__':
    main()
