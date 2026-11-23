"""Train and independently select a lexical NN-kNN metric on SQuAD sources.

Need-match supervision is provided passage identity,not downstream usefulness.
Public dev annotations are never loaded here. This is a sandbox experiment.

Without --study the tool writes protocol.json and summary.json to --output and,per seed,
s<seed>/{metric.pt,training_events.json,summary.json}. With --study S every seed is one run
unit runs/t3/S/s<seed>/ (TensorBoard event files: per-epoch fit/tune curves,the per-step
training events,the seed summary and the metric checkpoint as the named state 'metric');
the s<seed>/ folders are then not written and summary.json is built from the units.
Rerunning a study skips complete seeds and resumes an interrupted seed after its last epoch.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from model.nnknn_model import GlocalFeatureWeight
from model.common.artifacts import source_fingerprint
from model.common.tracking import NullRun,RunKey,load_run,open_run,resolve_unit,unit_dir
from model.t3.benchmarks import load_squad
from model.t3.lexical import artifact_text,bm25_rank,candidate_bank,core_bank,need_loss
from model.t3.need_queries import load_need_queries

EVALUATION=('mean_need_loss','source_recall_at1','source_recall_at2')
LEXICAL=('source_recall_at1','source_recall_at2')
GROUPS=('fit','tune')


def unit_key(study,seed):
    return RunKey('t3',study,(f's{seed}',))


def unit_config(args,seed,*,source,samples_sha256,query_binding):
    """Identity of one seed's unit: every setting that changes its numbers,no host paths."""
    return dict(tool='t3_need_matching',seed=seed,epochs=args.epochs,dimensions=args.dimensions,objective=args.objective,
        selection=args.selection,term_frequency=args.term_frequency,data_sha256=source['sha256'],samples_sha256=samples_sha256,
        need_queries_sha256=query_binding['sha256'] if query_binding else None)


def seed_result(data):
    """The per-seed summary (s<seed>/summary.json content,same key order) rebuilt from a complete unit."""
    final=data.final()
    curves={f'{group}/{name}':dict(data.series(f'{group}/{name}')) for group in GROUPS for name in EVALUATION}
    epochs=sorted(curves['fit/mean_need_loss'])
    point=lambda epoch:{group:{name:curves[f'{group}/{name}'][epoch] for name in EVALUATION} for group in GROUPS}
    return dict(seed=final['seed'],selected_epoch=final['selected_epoch'],initial=point(0),
        selected={group:{name:final[f'selected/{group}/{name}'] for name in EVALUATION} for group in GROUPS},
        bm25={group:{name:final[f'bm25/{group}/{name}'] for name in LEXICAL} for group in GROUPS},
        actual_training_steps=final['actual_training_steps'],nonzero_gradient_steps=final['nonzero_gradient_steps'],
        parameter_delta_l2=final['parameter_delta_l2'],trainable_parameters=final['trainable_parameters'],
        training_seconds=final['training_seconds'],history=[dict(epoch=epoch,**point(epoch)) for epoch in epochs])


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--samples',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=[8,9,10])
    ap.add_argument('--epochs',type=int,default=20)
    ap.add_argument('--dimensions',type=int,default=256)
    ap.add_argument('--objective',choices=['softmax','hard_negative'],default='softmax')
    ap.add_argument('--selection',choices=['tune_loss','tune_recall'],default='tune_loss')
    ap.add_argument('--need-queries',type=Path)
    ap.add_argument('--term-frequency',choices=('count','sublinear','binary'),default='count')
    ap.add_argument('--study',default=None,help='record every seed as a run unit under NN-KNN/runs/t3/<study>/s<seed>/ '
                    '(TensorBoard event files and weights); skips complete seeds and resumes interrupted ones')
    args=ap.parse_args(argv)
    if args.epochs<1:ap.error('positive fixed epoch budget required')
    if args.study is not None:
        if len(set(args.seeds))!=len(args.seeds):ap.error('--study: seeds must be unique')
        try:unit_dir(unit_key(args.study,args.seeds[0]))  # run root and study name,before any data is loaded
        except (FileNotFoundError,ValueError) as exc:ap.error(f'--study: {exc}')
    torch.set_num_threads(1)
    data=json.loads((args.data_root/'manifest_mirror_complete.json').read_text())
    source=next(r for r in data['files'] if r['name']=='squad_train_v11.json')
    tasks,_=load_squad(args.data_root/source['name'],expected_sha256=source['sha256'])
    samples=json.loads(args.samples.read_text())['samples'][source['name']]
    samples_sha256=hashlib.sha256(args.samples.read_bytes()).hexdigest()
    by_id={t.question_id:t for t in tasks}
    fit=[by_id[i] for i in samples['fit_ids']];tune=[by_id[i] for i in samples['tune_ids']]
    assert {t.group for t in fit}.isdisjoint({t.group for t in tune})
    queries={t.question_id:t.question for t in fit+tune};query_binding=None
    if args.need_queries is not None:
        queries,query_binding=load_need_queries(args.need_queries,fit+tune,data_sha256=source['sha256'],sample_sha256=samples_sha256)
    args.output.mkdir(parents=True,exist_ok=False)
    corpus=tuple({c.case_id:c for t in tasks for c in t.cases}.values())
    banks={t.question_id:candidate_bank(t,corpus,size=64) for t in fit+tune}
    metric=GlocalFeatureWeight(args.dimensions,1)
    models={t.question_id:core_bank(banks[t.question_id],dimensions=args.dimensions,metric=metric,term_frequency=args.term_frequency) for t in fit+tune}
    def evaluate(group):
        loss=top1=top2=0.
        with torch.no_grad():
            for task in group:
                value,result=need_loss(models[task.question_id],queries[task.question_id],[c.case_id for c in task.cases],dimensions=args.dimensions,objective=args.objective,term_frequency=args.term_frequency)
                loss+=float(value)
                ids=models[task.question_id].active_case_ids()
                order=torch.argsort(result['distances'][0],stable=True)
                positive={c.case_id for c in task.cases}
                top1+=int(int(ids[order[0]]) in positive)
                top2+=int(any(int(ids[i]) in positive for i in order[:2]))
        return dict(mean_need_loss=loss/len(group),source_recall_at1=top1/len(group),source_recall_at2=top2/len(group))
    protocol=dict(source_fingerprint=source_fingerprint(Path(__file__).resolve().parents[1]),data_source=source,
        fit_ids=samples['fit_ids'],tune_ids=samples['tune_ids'],candidate_ids={k:[c.case_id for c in v] for k,v in banks.items()},
        objective='provided-passage identity need matching; not answer or outcome utility',dimensions=args.dimensions,
        bank_size=64,seeds=args.seeds,epochs=args.epochs,learning_rate=.03,weight_decay=0.,trainable='global diagonal feature weights only',
        projection='clamp .05..20 then mean-normalize to1',selection=args.selection,need_objective=args.objective,
        selection_rules=dict(tune_loss='minimum tune loss,including epoch0',tune_recall='maximum tune recall@2,then@1,then minimum loss; including epoch0'),
        hard_negative_count=8,hard_negative_margin=.2,
        representation=f'fixed SHA256 lexical {args.term_frequency} L2; encoder and case biases untrained',term_frequency=args.term_frequency,
        public_dev_annotations_used=False,global_promotion=False,query_binding=query_binding,
        query_source='actual frozen host request' if query_binding else 'original public question',query_texts=queries)
    if args.study is not None:protocol['study']=args.study
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    lexical={}
    for name,group in [('fit',fit),('tune',tune)]:
        count1=count2=0
        for task in group:
            ranked=bm25_rank(queries[task.question_id],banks[task.question_id]);positive={c.case_id for c in task.cases}
            count1+=ranked[0]['case_id'] in positive
            count2+=any(r['case_id'] in positive for r in ranked[:2])
        lexical[name]=dict(source_recall_at1=count1/len(group),source_recall_at2=count2/len(group))
    def selection_key(row):
        tune=row['tune']
        return (tune['mean_need_loss'],) if args.selection=='tune_loss' else (-tune['source_recall_at2'],-tune['source_recall_at1'],tune['mean_need_loss'])
    def log_epoch(run,row,epoch_events=()):
        values={f'{group}/{name}':row[group][name] for group in GROUPS for name in EVALUATION}
        if epoch_events:
            values.update({'train/mean_loss':sum(e['loss'] for e in epoch_events)/len(epoch_events),
                           'train/mean_gradient_norm':sum(e['gradient_norm'] for e in epoch_events)/len(epoch_events)})
            run.log_records('training_events',epoch_events)
        run.log_step(row['epoch'],values)
    def train(seed,run=NullRun(),resumable=False):
        """One seed; with a run it logs curves and events at step=epoch and checkpoints after epochs (resumable)."""
        tracking=not isinstance(run,NullRun)
        if tracking:run.log_record('protocol',protocol)
        random_stream=random.Random(seed)
        metric.feature_weights.data.fill_(1)
        optimizer=torch.optim.Adam(metric.parameters(),lr=.03)
        payload=run.load_resume() if resumable else None
        if payload is None:
            initial=metric.feature_weights.detach().clone()
            baseline=dict(fit=evaluate(fit),tune=evaluate(tune))
            history=[dict(epoch=0,**baseline)]
            best=selection_key(baseline);best_epoch=0
            selected=copy.deepcopy(metric.state_dict());selected_optimizer=copy.deepcopy(optimizer.state_dict())
            events=[];start=time.perf_counter();first=1
            if tracking:log_epoch(run,history[0])
        else:
            state=payload.state
            metric.load_state_dict(state['metric']);optimizer.load_state_dict(state['optimizer'])
            initial=state['initial'];history=state['history'];events=state['events']
            baseline={group:history[0][group] for group in GROUPS}
            best=tuple(state['best']);best_epoch=state['best_epoch']
            selected=state['selected'];selected_optimizer=state['selected_optimizer']
            version,internal,gauss=state['random_stream'];random_stream.setstate((version,tuple(internal),gauss))
            start=time.perf_counter()-state['elapsed_seconds'];first=payload.step+1
        for epoch in range(first,args.epochs+1):
            order=list(fit);random_stream.shuffle(order);epoch_start=len(events)
            for task in order:
                optimizer.zero_grad()
                loss,result=need_loss(models[task.question_id],queries[task.question_id],[c.case_id for c in task.cases],dimensions=args.dimensions,objective=args.objective,term_frequency=args.term_frequency)
                loss.backward()
                grad=metric.feature_weights.grad
                if grad is None or not bool(torch.isfinite(grad).all()):raise ValueError('missing/nonfinite actual need gradient')
                events.append(dict(epoch=epoch,question_id=task.question_id,loss=float(loss.detach()),gradient_norm=float(grad.norm()),
                    positive_ids=[c.case_id for c in task.cases],actual_case_ids=models[task.question_id].active_case_ids().tolist()))
                optimizer.step()
                with torch.no_grad():
                    metric.feature_weights.clamp_(.05,20)
                    metric.feature_weights.div_(metric.feature_weights.mean())
            row=dict(epoch=epoch,fit=evaluate(fit),tune=evaluate(tune));history.append(row)
            if selection_key(row)<best:
                best=selection_key(row);best_epoch=epoch
                selected=copy.deepcopy(metric.state_dict());selected_optimizer=copy.deepcopy(optimizer.state_dict())
            if tracking:log_epoch(run,row,events[epoch_start:])
            if resumable and epoch<args.epochs and run.checkpoint_due(epoch):
                version,internal,gauss=random_stream.getstate()
                run.save_resume(dict(metric=metric.state_dict(),optimizer=optimizer.state_dict(),initial=initial,history=history,
                    events=events,best=list(best),best_epoch=best_epoch,selected=selected,selected_optimizer=selected_optimizer,
                    random_stream=[version,list(internal),gauss],elapsed_seconds=time.perf_counter()-start),step=epoch)
        final=copy.deepcopy(metric.state_dict());metric.load_state_dict(selected)
        result=dict(seed=seed,selected_epoch=best_epoch,initial=baseline,selected=dict(fit=evaluate(fit),tune=evaluate(tune)),
            bm25=lexical,actual_training_steps=len(events),nonzero_gradient_steps=sum(e['gradient_norm']>0 for e in events),
            parameter_delta_l2=float((metric.feature_weights-initial).detach().norm()),trainable_parameters=metric.feature_weights.numel(),
            training_seconds=time.perf_counter()-start,history=history)
        checkpoint=dict(selected_metric=selected,selected_optimizer=selected_optimizer,final_metric=final,
                        final_optimizer=optimizer.state_dict(),initial_metric=dict(feature_weights=initial))
        if not tracking:return result,checkpoint,events
        finals={key:result[key] for key in ('seed','selected_epoch','actual_training_steps','nonzero_gradient_steps',
                                            'parameter_delta_l2','trainable_parameters','training_seconds')}
        finals.update({f'selected/{group}/{name}':result['selected'][group][name] for group in GROUPS for name in EVALUATION})
        finals.update({f'bm25/{group}/{name}':lexical[group][name] for group in GROUPS for name in LEXICAL})
        run.log_final(finals)
        run.save_state('metric',checkpoint,step=args.epochs)
        return result,checkpoint,events
    runs=[]
    for seed in args.seeds:
        if args.study is None:
            result,checkpoint,events=train(seed)
            folder=args.output/f's{seed}';folder.mkdir()
            torch.save(checkpoint,folder/'metric.pt')
            (folder/'training_events.json').write_text(json.dumps(events),encoding='utf-8')
            (folder/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            result['checkpoint_sha256']=hashlib.sha256((folder/'metric.pt').read_bytes()).hexdigest()
        else:
            key=unit_key(args.study,seed)
            config=unit_config(args,seed,source=source,samples_sha256=samples_sha256,query_binding=query_binding)
            if resolve_unit(key,config=config) is None:
                with open_run(key,config=config) as run:train(seed,run,resumable=True)
            recorded=load_run(unit_dir(key),arrays=False)
            result=seed_result(recorded)
            result['checkpoint_sha256']=recorded.states()['metric']['file']['sha256']
        runs.append(result)
        print(seed,result['selected_epoch'],result['selected']['tune'],result['parameter_delta_l2'],flush=True)
    (args.output/'summary.json').write_text(json.dumps(dict(runs=runs,protocol=protocol),indent=2),encoding='utf-8')


if __name__=='__main__':main()
