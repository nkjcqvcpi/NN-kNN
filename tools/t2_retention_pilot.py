"""Post-checkpoint retention pilot; training-only selection, independent diagnostics.

Reads only run units: every ``run_dir`` of ``--runs`` (e.g. a t2_role_pilot manifest) must be a complete run unit of
NN-KNN/runs; its hash-checked weights are loaded, nothing is trained. Each condition is recorded as the run unit
runs/t2/<study>/retention/<checkpoint state sha256[:12]>/<condition> (``--study`` is required): named state ``states``
(before/after/reference, the file mode's states.pt), record ``result`` (its result.json) and named state ``rng_after``
(the global RNG state when the condition finished). A complete condition unit is read back before any computation
and the RNG is restored from ``rng_after``, so the conditions after it compute what an uninterrupted pilot computes.
The manifest is built from the units and goes to --output (default
<worktree>/results/t2/retention_pilot_<UTC timestamp>/manifest.json); its ``result`` entries are unit paths.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--runs',type=Path,required=True,help='JSON with a runs (or roles) list of run_dir entries, '
        'e.g. a t2_role_pilot manifest')
    parser.add_argument('--output',type=Path,default=None,help='Folder of the manifest (made; must not exist); default '
        '<worktree>/results/t2/retention_pilot_<UTC timestamp>.')
    parser.add_argument('--study',required=True,help='Required: each condition is the run unit '
        'runs/t2/<study>/retention/<checkpoint>/<condition>; a complete one is read back, not recomputed.')
    parser.add_argument('--reference-batches',type=int,default=1)
    parser.add_argument('--query-loss-budget',type=float,default=None)
    args=parser.parse_args()
    if args.reference_batches<1:parser.error('--reference-batches must be positive')
    if args.query_loss_budget is not None and (not math.isfinite(args.query_loss_budget) or args.query_loss_budget<0):
        parser.error('--query-loss-budget must be finite and nonnegative')
    sys.path.insert(0,str(args.source.resolve()))
    import torch
    import model.nnknn_rl_workflow as w
    from model.t2.retention import TrainingReference,constrained_retain
    from model.t2.audit import evaluation,policy_output
    from model.common.artifacts import source_fingerprint
    from model.common.tracking import RunKey,load_unit,open_run
    from model.t2.runs import RunFiles,check_study,require_unit_in_runs_root,unit_component
    try:check_study(args.study)
    except (FileNotFoundError,ValueError) as exc:parser.error(f'--study: {exc}')
    if args.output is None:
        args.output=ROOT_DIR/'results'/'t2'/f"retention_pilot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(source_fingerprint=source_fingerprint(args.source),runs=[],skips=[],
        selection='latest recorded training reference; fixed parameters, full-bank refill',reference_batches=args.reference_batches,
        query_loss_budget=args.query_loss_budget,
        scope='post-checkpoint one-role compaction; no retraining or live maintenance integration')

    def equal(a,b):
        if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b

    def snapshot(state):
        return dict(actor=copy.deepcopy(state['model'].state_dict()),
            critic=copy.deepcopy(state['value_model'].state_dict()),
            target=copy.deepcopy(state['target_model'].state_dict()),
            optimizers={k:w._optimizer_state_snapshot(v) for k,v in state['optimizers'].items()})

    def diagnostic(state,cp):
        cfg=w.NNKNNRLConfig(**cp['config'])
        return dict(greedy=w.evaluate_nnknn_rl(cp['task']['name'],state['model'],episodes=3,seed=cfg.eval_seed,device='cpu'),
            mc=w.evaluate_critic_holdout(cp['task']['name'],state['model'],state['value_model'],state['target_model'],cfg,
                episodes=2,seed=40000+2*cfg.seed,global_step=cp['selected_step'],device='cpu'))

    def condition_unit(run,checkpoint,cp,base_metrics,role,ref,budget,keep_capacity,condition,step,steps,selected):
        """One retention condition, computed and recorded in the open run of its unit."""
        state=w.load_nnknn_rl_checkpoint(checkpoint,device='cpu',restore_optimizers=True)
        model=state['value_model'] if role=='critic' else state['model']
        other=state['model'] if role=='critic' else state['value_model']
        old_ids=w._case_store_ids(model);before=snapshot(state)
        optim=state['optimizers'].get('critic') if role=='critic' else state['optimizers']['actor_or_joint']
        if optim is None:optim=state['optimizers']['actor_or_joint']
        other_state=copy.deepcopy(other.state_dict())
        private=w._nnknn_case_parameter_names(model)
        parameter_before={n:p.detach().clone() for n,p in model.named_parameters()}
        moment_before={n:copy.deepcopy(optim.state.get(p,{})) for n,p in model.named_parameters()}
        target_before=copy.deepcopy(state['target_model'].state_dict())
        target_ids=w._case_store_ids(state['target_model']).tolist()
        started=time.perf_counter()
        result=constrained_retain(model,ref,keep_capacity=keep_capacity,loss_budget=budget,
            max_query_loss_increase=args.query_loss_budget)
        elapsed=time.perf_counter()-started
        alignment=w._realign_optimizer_case_state(optim,model,old_ids)
        if role=='critic':w._align_nnknn_target_case_store(model,state['target_model'])
        assert equal(other.state_dict(),other_state)
        keep=[old_ids.tolist().index(i) for i in result['final_ids']]
        checked_moments=0
        for name,p in model.named_parameters():
            old=parameter_before[name]
            if name not in private:assert torch.equal(p,old)
            else:assert torch.equal(p[:len(keep)],old[keep])
            for key,value in moment_before[name].items():
                current=optim.state[p][key]
                if name in private and torch.is_tensor(value) and value.ndim>0 and value.shape==p.shape:
                    expected=torch.zeros_like(value);expected[:len(keep)]=value[keep]
                    assert torch.equal(current,expected);checked_moments+=1
                else:assert equal(current,value)
        if role=='critic':
            target=state['target_model'];tkeep=[target_ids.index(i) for i in result['final_ids']]
            for field in ('labels','biases','negative_weights','glocal_weights'):
                assert torch.equal(getattr(target.nnknn_model,field)[:len(tkeep)],target_before['nnknn_model.'+field][tkeep])
            assert w._case_store_ids(target).tolist()==result['final_ids']
            assert target.nnknn_model.cases.data_ptr()==model.nnknn_model.cases.data_ptr()
            assert target.nnknn_model.labels.data_ptr()!=model.nnknn_model.labels.data_ptr()
        else:assert equal(state['target_model'].state_dict(),target_before)
        # Independent core replay of every candidate and the feasible greedy choice.
        replay=w.load_nnknn_rl_checkpoint(checkpoint,device='cpu',restore_optimizers=True)
        original=replay['value_model'] if role=='critic' else replay['model'];core=original.nnknn_model
        mask=torch.ones(len(old_ids),dtype=torch.bool)
        def losses(m):
            output=core(ref.queries,case_mask=m,return_retrieval=True)
            if role=='critic':return (output[0].flatten()-ref.targets).square(),output[-1]
            probs=policy_output(output[0]);mix=(1-ref.behavior_epsilons[:,None])*probs+ref.behavior_epsilons[:,None]/original.action_dim
            return -ref.advantages*mix.gather(1,ref.actions[:,None]).flatten().clamp_min(ref.probability_floor).log(),output[-1]
        with evaluation(original):
            initial,retrieval=losses(mask);current=initial
            exposure=torch.zeros(len(old_ids),dtype=torch.long)
            exposure[retrieval['case_indices']]=(retrieval['weights']>0).sum(0)
            assert exposure.tolist()==result['exposure']
            grouped={}
            for trial in result['trials']:grouped.setdefault(trial['round'],[]).append(trial)
            labels=original.action_tensor().tolist() if role=='actor' else None
            for round_index,trials in grouped.items():
                eligible=[i for i in range(len(old_ids)) if mask[i] and exposure[i]>=1 and
                    (role=='critic' or sum(bool(mask[j]) and labels[j]==labels[i] for j in range(len(old_ids)))>original.min_cases_per_action)]
                assert [t['case_id'] for t in trials]==sorted(int(old_ids[i]) for i in eligible)
                feasible=[]
                for trial in trials:
                    slot=old_ids.tolist().index(trial['case_id']);candidate=mask.clone();candidate[slot]=False
                    values,_=losses(candidate);delta=values-current
                    c=float(delta.clamp_min(0).sum());h=float((-delta).clamp_min(0).sum());q=(c+1)/(c+h+2)
                    assert values.tolist()==trial['query_losses']
                    assert (c,h,q,float(values.mean()))==(trial['C'],trial['H'],trial['Q'],trial['mean_loss'])
                    feasible_flag=float(values.mean())<=float(initial.mean())+budget+1e-7
                    assert feasible_flag==trial['within_budget']
                    query_flag=args.query_loss_budget is None or bool((values<=initial+args.query_loss_budget+1e-7).all())
                    assert query_flag==trial['within_query_budget']
                    if feasible_flag and query_flag:feasible.append((q,float(values.mean()),trial['case_id'],slot,values))
                if round_index<len(result['accepted']):
                    best=min(feasible,key=lambda t:t[:3]);assert best[2]==result['accepted'][round_index]
                    mask[best[3]]=False;current=best[4]
                else:assert not feasible
            assert current.tolist()==result['final_query_losses']
            assert old_ids[mask].tolist()==result['final_ids']
        assert result['target_capacity']==keep_capacity
        run.save_state('states',dict(before=before,after=snapshot(state),reference=ref.__dict__))  # was states.pt
        metrics=diagnostic(state,cp)
        record=dict(condition=condition,checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            role=role,reference_step=step,reference_steps=steps,reference_events=selected,result=result,alignment=alignment,
            checked_moment_tensors=checked_moments,selection_seconds=elapsed,baseline=base_metrics,after=metrics,
            verified_candidate_trials=len(result['trials']),all_checks_passed=True)
        run.log_record('result',record)  # was result.json
        run.save_rng('rng_after')  # the global RNG state a later condition starts from
        print(f"{condition}: {len(old_ids)} -> {result['actual_capacity']}; replayed {len(result['trials'])} candidates",flush=True)

    runs_data=json.loads(args.runs.read_text())
    run_records=runs_data['runs'] if 'runs' in runs_data else runs_data.get('roles')  # a runs list, or a t2_role_pilot manifest's roles
    if run_records is None:parser.error(f"--runs {args.runs}: no 'runs' or 'roles' list")
    for run_record in run_records:
        run=Path(run_record['run_dir'])
        require_unit_in_runs_root(run,'--runs')  # only run units under NN-KNN/runs ($NNKNN_RUNS)
        # run unit: hash-checked weights/checkpoint.pt, events from the records, snapshots relative to the unit
        files=RunFiles(run)
        checkpoint=files.checkpoint_path()
        checkpoint_state_sha256=files.data.states()['checkpoint']['state_sha256']
        events=files.json('case_audit_events.json')['events']
        resolve=files.resolve
        cp=torch.load(checkpoint,weights_only=True,map_location='cpu')
        baseline=w.load_nnknn_rl_checkpoint(checkpoint,device='cpu',restore_optimizers=True)
        base_metrics=diagnostic(baseline,cp)
        for role,stream in [('critic','training_gae'),('actor','actor_policy_surrogate')]:
            role_events=[e for e in events if e['role']==role and e['stream']==stream and
                (e['global_step']<cp['selected_step'] if cp['selected_source']=='best_eval' else e['global_step']<=cp['selected_step'])]
            if not role_events:
                manifest['skips'].append(dict(run=str(run),role=role,reason='no actual ready-policy training reference',conditions=4))
                continue
            steps=sorted({e['global_step'] for e in role_events})[-args.reference_batches:]
            step=max(steps)
            selected=[e for e in role_events if e['global_step'] in steps]
            for e in selected:
                assert hashlib.sha256(Path(resolve(e['snapshot'])).read_bytes()).hexdigest()==e['snapshot_sha256']
                if role=='actor':assert e['executed_under_ready_policy']
            ref=TrainingReference(torch.tensor([e['query'] for e in selected]),stream,
                targets=torch.tensor([e['target'] for e in selected]) if role=='critic' else None,
                actions=torch.tensor([e['action'] for e in selected]) if role=='actor' else None,
                advantages=torch.tensor([e['advantage'] for e in selected]) if role=='actor' else None,
                behavior_epsilons=torch.tensor([e['behavior_epsilon'] for e in selected]) if role=='actor' else None,
                policy_ready=torch.tensor([e['executed_under_ready_policy'] for e in selected]) if role=='actor' else None,
                probability_floor=selected[0]['probability_floor'] if role=='actor' else 1e-8)
            n_cases=len(w._case_store_ids(baseline['value_model'] if role=='critic' else baseline['model']))
            for fraction in (.75,.9):
                for budget in (0.,.03):
                    keep_capacity=max(1,math.floor(n_cases*fraction))
                    condition=f"{cp['task']['name']}_s{cp['config']['seed']}_{role}_k{keep_capacity}_b{budget}"
                    unit_config=dict(kind='t2_retention_pilot',checkpoint_state_sha256=checkpoint_state_sha256,role=role,
                        stream=stream,keep_fraction=fraction,keep_capacity=keep_capacity,loss_budget=budget,
                        reference_batches=args.reference_batches,reference_steps=steps,
                        query_loss_budget=args.query_loss_budget)
                    key=RunKey('t2',args.study,('retention',checkpoint_state_sha256[:12],unit_component(condition)))
                    done=load_unit(key,config=unit_config,arrays=False)  # before any computation
                    if done is not None:  # recorded: nothing is recomputed; the RNG continues as it was left
                        done.restore_rng('rng_after')
                        unit_path=done.path
                        print(f"{condition}: complete, not recomputed",flush=True)
                    else:
                        with open_run(key,config=unit_config) as handle:
                            unit_path=Path(handle.path)
                            condition_unit(handle,checkpoint,cp,base_metrics,role,ref,budget,keep_capacity,condition,step,steps,selected)
                    manifest['runs'].append(dict(condition=condition,result=str(unit_path)))
                    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()
