"""Bounded current-core four-role pilot, explicitly not a full-cycle gate.

Each condition is the run unit runs/t2/<study>/<task>/<role>__<target>__<optimizer>[__labels_<mode>]/s<seed>
(``--study`` is required): training, then the role audit in the same unit (named state role_audit_state, records
role_audit and role_audit/events). A complete unit is read back before any training and never trained or audited
again. The manifest is built from the units and goes to --output (default
<worktree>/results/t2/role_pilot_<UTC timestamp>/manifest.json).
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from itertools import product
from pathlib import Path
import sys
import time

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.nnknn_rl_workflow import (
    NNKNNValueNetwork, make_nnknn_rl_config, train_nnknn_rl,
    evaluate_critic_holdout,
)
from model.common.artifacts import source_fingerprint
from model.t2 import runs as t2runs
from model.t2.audit import summarize_credit


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default=None,help='Folder of the manifest (made; must not exist); default '
        '<worktree>/results/t2/role_pilot_<UTC timestamp>.')
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
    ap.add_argument('--study',required=True,help='Required: each condition is the run unit '
        'runs/t2/<study>/<task>/<role...>/s<seed> (training, role audit events and state); rerunning skips complete '
        'units before any training.')
    args=ap.parse_args()
    try:t2runs.check_study(args.study)
    except (FileNotFoundError,ValueError) as exc:ap.error(f'--study: {exc}')
    if any(mode!='fixed' for mode in args.critic_label_modes) and any(role.endswith('_mlp') for role in args.roles):
        ap.error('nonfixed label variants require explicit NN-kNN critic roles')
    root=Path(args.output) if args.output else ROOT_DIR/'results'/'t2'/f"role_pilot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
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
        def train_and_audit(run):
            start=time.perf_counter()
            state=train_nnknn_rl(task,cfg,device='cpu',progress=False,run=run,resumable=True)
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
            # the same unit holds the role audit: state role_audit_state, records role_audit and role_audit/events
            run.save_state('role_audit_state',audit_state)
            run.log_records('role_audit/events',events)
            run.log_record('role_audit',dict(independent_mc=holdout,training_seconds=train_seconds))
        files=t2runs.run_training(args.study,t2runs.unit_parts(label),
            t2runs.unit_config('nnknn_rl',task,cfg.to_dict(),'cpu',tool='t2_role_pilot'),train_and_audit,label='t2-role-pilot')
        audit=files.data.record('role_audit')
        # training time over all sessions of the unit (audit['training_seconds'] is the last session's)
        summary,holdout,train_seconds=files.json('summary.json'),audit['independent_mc'],files.train_seconds()
        events,run_dir=files.data.records('role_audit/events'),files.path
        checkpoint_sha256=files.data.states()['checkpoint']['file']['sha256']
        credit=summarize_credit(events)
        credit_json=[dict(role=k[0],stream=k[1],cases=v) for k,v in credit.items()]
        record=dict(role=label,task=task,seed=seed,actor_type=actor_type,critic_type=critic_type,
            configured_target_mode=target_mode,audit_queries_per_batch=args.audit_queries_per_batch,
            critic_label_mode=label_mode,
            critic_label_activation_threshold=args.critic_label_activation_threshold,
            critic_label_update_alpha=args.critic_label_update_alpha,
            case_optimizer_maintenance=optimizer_mode,
            source_fingerprint=fingerprint,versions=versions,run_dir=str(run_dir),
            training_seconds=train_seconds,summary=summary,independent_mc=holdout,
            audit_events=len(events),interventions=sum(len(e['interventions']) for e in events),
            credit=credit_json,checkpoint_sha256=checkpoint_sha256)
        records.append(record)
        (root/'manifest.json').write_text(json.dumps(dict(status='exploratory_current_core_not_stage_a_gate',
            roles=records,source_fingerprint=fingerprint,versions=versions),indent=2),encoding='utf-8')
        total=len(args.roles)*len(args.critic_target_modes)*len(args.seeds)*len(args.tasks)*len(args.case_optimizer_maintenance)*len(args.critic_label_modes)
        print(f'completed {len(records)}/{total} {label} events={len(events)}',flush=True)


if __name__=='__main__':
    main()
