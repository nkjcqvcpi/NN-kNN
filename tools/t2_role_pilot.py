"""Bounded current-core four-role pilot, explicitly not a full-cycle gate."""
import argparse
from dataclasses import asdict
import hashlib
import json
from itertools import product
from pathlib import Path
import time

import torch

from model.nnknn_rl_workflow import (
    NNKNNValueNetwork, make_nnknn_rl_config, train_nnknn_rl,
    evaluate_critic_holdout,
)
from model.common.artifacts import source_fingerprint
from model.t2.audit import summarize_credit


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    ap.add_argument('--steps',type=int,default=512)
    ap.add_argument('--seeds',type=int,nargs='+',default=[8,9,10])
    ap.add_argument('--tasks',nargs='+',default=['cartpole'])
    ap.add_argument('--roles',nargs='+',choices=['mlp_mlp','nnknn_mlp','mlp_nnknn','nnknn_nnknn'],
                    default=['mlp_mlp','nnknn_mlp','mlp_nnknn','nnknn_nnknn'])
    ap.add_argument('--critic-target-modes',nargs='+',choices=['ema','hard','none'],default=['ema'])
    ap.add_argument('--audit-queries-per-batch',type=int,default=0)
    ap.add_argument('--critic-target-sync-interval',type=int,default=4)
    ap.add_argument('--case-optimizer-maintenance',nargs='+',choices=['reset','preserve_by_id'],default=['reset'])
    ap.add_argument('--eval-frequency',type=int,default=0)
    ap.add_argument('--eval-episode-frequency',type=int,default=100000)
    ap.add_argument('--quality-tracking',action='store_true')
    ap.add_argument('--critic-holdout-episode-frequency',type=int,default=0)
    ap.add_argument('--critic-holdout-episodes',type=int,default=5)
    ap.add_argument('--retention-fraction',type=float,default=None)
    ap.add_argument('--retention-frequency',type=int,default=1000)
    ap.add_argument('--retention-queries',type=int,default=16)
    ap.add_argument('--retention-loss-budget',type=float,default=0.)
    ap.add_argument('--retention-query-budget',type=float,default=0.)
    ap.add_argument('--no-retention-query-guard',action='store_true')
    ap.add_argument('--retention-roles',choices=['actor','critic','both'],default='both')
    ap.add_argument('--case-maintenance-frequency',type=int,default=1000)
    ap.add_argument('--critic-label-modes',nargs='+',choices=['fixed','mutable','trainable','hybrid'],default=['fixed'])
    ap.add_argument('--critic-label-activation-threshold',type=float,default=0.)
    ap.add_argument('--critic-label-update-alpha',type=float,default=.25)
    args=ap.parse_args()
    if any(mode!='fixed' for mode in args.critic_label_modes) and any(role.endswith('_mlp') for role in args.roles):
        ap.error('nonfixed label variants require explicit NN-kNN critic roles')
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parents[1]
    fingerprint=source_fingerprint(source)
    import gymnasium
    versions=dict(torch=torch.__version__,gymnasium=gymnasium.__version__)
    records=[]
    torch.set_num_threads(1)
    conditions=product(args.tasks,args.seeds,args.roles,args.critic_target_modes,args.case_optimizer_maintenance,args.critic_label_modes)
    for task,seed,role,target_mode,optimizer_mode,label_mode in conditions:
        actor_type,critic_type=role.split('_')
        label=f'{task}/{role}__{target_mode}__{optimizer_mode}/s{seed}'
        if label_mode!='fixed':label=f'{task}/{role}__{target_mode}__{optimizer_mode}__labels_{label_mode}/s{seed}'
        cfg=make_nnknn_rl_config('smoke',seed=seed,actor_type=actor_type,critic_type=critic_type,
            total_timesteps=args.steps,case_capacity=128,critic_case_capacity=128,
            eval_frequency=args.eval_frequency,eval_episode_frequency=args.eval_episode_frequency,eval_episodes=3,
            critic_holdout_episode_frequency=args.critic_holdout_episode_frequency,
            critic_holdout_episodes=args.critic_holdout_episodes,policy_update_episodes=2,
            early_stopping=False,success_threshold=None,top_k=8,
            min_case_entries=8,min_cases_per_action=2,
            critic_target_value_mode=target_mode,
            critic_target_sync_interval=args.critic_target_sync_interval,
            critic_mutable_value_labels=label_mode in {'mutable','hybrid'},
            critic_trainable_value_labels=label_mode in {'trainable','hybrid'},
            critic_value_label_activation_threshold=args.critic_label_activation_threshold,
            critic_value_label_update_alpha=args.critic_label_update_alpha,
            case_optimizer_maintenance=optimizer_mode,
            case_quality_tracking=args.quality_tracking,
            case_maintenance_frequency=args.case_maintenance_frequency,
            case_retention_keep_fraction=args.retention_fraction,
            case_retention_frequency=args.retention_frequency,case_retention_queries=args.retention_queries,
            case_retention_loss_budget=args.retention_loss_budget,
            case_retention_query_budget=None if args.no_retention_query_guard else args.retention_query_budget,
            case_retention_roles=args.retention_roles,
            case_audit_queries_per_batch=args.audit_queries_per_batch)
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
            configured_target_mode=target_mode,audit_queries_per_batch=args.audit_queries_per_batch,
            critic_label_mode=label_mode,
            critic_label_activation_threshold=args.critic_label_activation_threshold,
            critic_label_update_alpha=args.critic_label_update_alpha,
            case_optimizer_maintenance=optimizer_mode,
            source_fingerprint=fingerprint,versions=versions,run_dir=str(run_dir),
            training_seconds=train_seconds,summary=state['summary'],independent_mc=holdout,
            audit_events=len(events),interventions=sum(len(e['interventions']) for e in events),
            credit=credit_json,checkpoint_sha256=hashlib.sha256((run_dir/'checkpoint.pt').read_bytes()).hexdigest())
        records.append(record)
        (root/'manifest.json').write_text(json.dumps(dict(status='exploratory_current_core_not_stage_a_gate',
            roles=records,source_fingerprint=fingerprint,versions=versions),indent=2),encoding='utf-8')
        total=len(args.roles)*len(args.critic_target_modes)*len(args.seeds)*len(args.tasks)*len(args.case_optimizer_maintenance)*len(args.critic_label_modes)
        print(f'completed {len(records)}/{total} {label} events={len(events)}',flush=True)


if __name__=='__main__':
    main()
