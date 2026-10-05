"""Train and independently select a lexical NN-kNN metric on SQuAD sources.

Need-match supervision is provided passage identity,not downstream usefulness.
Public dev annotations are never loaded here. This is a sandbox experiment.
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
from model.t1.artifacts import source_fingerprint
from model.t3.benchmarks import load_squad
from model.t3.lexical import artifact_text,bm25_rank,candidate_bank,core_bank,need_loss


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--samples',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=[8,9,10])
    ap.add_argument('--epochs',type=int,default=20)
    ap.add_argument('--dimensions',type=int,default=256)
    ap.add_argument('--objective',choices=['softmax','hard_negative'],default='softmax')
    ap.add_argument('--selection',choices=['tune_loss','tune_recall'],default='tune_loss')
    args=ap.parse_args()
    if args.epochs<1:ap.error('positive fixed epoch budget required')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    data=json.loads((args.data_root/'manifest_mirror_complete.json').read_text())
    source=next(r for r in data['files'] if r['name']=='squad_train_v11.json')
    tasks,_=load_squad(args.data_root/source['name'],expected_sha256=source['sha256'])
    samples=json.loads(args.samples.read_text())['samples'][source['name']]
    by_id={t.question_id:t for t in tasks}
    fit=[by_id[i] for i in samples['fit_ids']];tune=[by_id[i] for i in samples['tune_ids']]
    assert {t.group for t in fit}.isdisjoint({t.group for t in tune})
    corpus=tuple({c.case_id:c for t in tasks for c in t.cases}.values())
    banks={t.question_id:candidate_bank(t,corpus,size=64) for t in fit+tune}
    metric=GlocalFeatureWeight(args.dimensions,1)
    models={t.question_id:core_bank(banks[t.question_id],dimensions=args.dimensions,metric=metric) for t in fit+tune}
    def evaluate(group):
        loss=top1=top2=0.
        with torch.no_grad():
            for task in group:
                value,result=need_loss(models[task.question_id],task.question,[c.case_id for c in task.cases],dimensions=args.dimensions,objective=args.objective)
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
        representation='fixed SHA256 lexical count L2; encoder and case biases untrained',
        public_dev_annotations_used=False,global_promotion=False)
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    lexical={}
    for name,group in [('fit',fit),('tune',tune)]:
        count1=count2=0
        for task in group:
            ranked=bm25_rank(task.question,banks[task.question_id]);positive={c.case_id for c in task.cases}
            count1+=ranked[0]['case_id'] in positive
            count2+=any(r['case_id'] in positive for r in ranked[:2])
        lexical[name]=dict(source_recall_at1=count1/len(group),source_recall_at2=count2/len(group))
    runs=[]
    def selection_key(row):
        tune=row['tune']
        return (tune['mean_need_loss'],) if args.selection=='tune_loss' else (-tune['source_recall_at2'],-tune['source_recall_at1'],tune['mean_need_loss'])
    for seed in args.seeds:
        random_stream=random.Random(seed)
        metric.feature_weights.data.fill_(1)
        optimizer=torch.optim.Adam(metric.parameters(),lr=.03)
        initial=metric.feature_weights.detach().clone()
        baseline=dict(fit=evaluate(fit),tune=evaluate(tune))
        history=[dict(epoch=0,**baseline)]
        best=selection_key(baseline);best_epoch=0
        selected=copy.deepcopy(metric.state_dict());selected_optimizer=copy.deepcopy(optimizer.state_dict())
        events=[];start=time.perf_counter()
        for epoch in range(1,args.epochs+1):
            order=list(fit);random_stream.shuffle(order)
            for task in order:
                optimizer.zero_grad()
                loss,result=need_loss(models[task.question_id],task.question,[c.case_id for c in task.cases],dimensions=args.dimensions,objective=args.objective)
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
        final=copy.deepcopy(metric.state_dict());metric.load_state_dict(selected)
        result=dict(seed=seed,selected_epoch=best_epoch,initial=baseline,selected=dict(fit=evaluate(fit),tune=evaluate(tune)),
            bm25=lexical,actual_training_steps=len(events),nonzero_gradient_steps=sum(e['gradient_norm']>0 for e in events),
            parameter_delta_l2=float((metric.feature_weights-initial).detach().norm()),trainable_parameters=metric.feature_weights.numel(),
            training_seconds=time.perf_counter()-start,history=history)
        folder=args.output/f's{seed}';folder.mkdir()
        torch.save(dict(selected_metric=selected,selected_optimizer=selected_optimizer,final_metric=final,
                        final_optimizer=optimizer.state_dict(),initial_metric=dict(feature_weights=initial)),folder/'metric.pt')
        (folder/'training_events.json').write_text(json.dumps(events),encoding='utf-8')
        (folder/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        result['checkpoint_sha256']=hashlib.sha256((folder/'metric.pt').read_bytes()).hexdigest()
        runs.append(result)
        print(seed,best_epoch,result['selected']['tune'],result['parameter_delta_l2'],flush=True)
    (args.output/'summary.json').write_text(json.dumps(dict(runs=runs,protocol=protocol),indent=2),encoding='utf-8')


if __name__=='__main__':main()
