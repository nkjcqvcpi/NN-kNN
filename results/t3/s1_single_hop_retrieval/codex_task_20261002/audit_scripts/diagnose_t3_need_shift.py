import hashlib
import json
from pathlib import Path
import sys

import torch
work=Path(__file__).resolve().parent
freeze=json.loads((work/'t3_format_enforced_freeze.json').read_text());sys.path.insert(0,freeze['archive'])
from model.t3.lexical import HashQuery,hash_vector,core_bank,bm25_rank
from model.t3.retrieval import Case,Retriever,Access,Request
out=work.parent/'outputs/t3-public-host-20261005-format-enforced'
protocol=json.loads((out/'protocol.json').read_text())
verification=json.loads((work.parent/'outputs/neural-cbr-t3-format-host-verification.json').read_text())
assert verification['all_recorded_retrieval_traces_verified'] and not verification['provisional_snapshot']
for name,bound in verification['input_artifacts'].items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==bound['sha256']
questions={d:{q['question_id']:q['question'] for q in qs} for d,qs in protocol['questions'].items()}
rows=[json.loads(line) for line in (out/'trials.jsonl').read_text().splitlines()]
metric=torch.load(work/'t3-need-matching-20261005/s8/metric.pt',weights_only=True)['selected_metric']['feature_weights']
records=[];manual=0;fresh_ids=[]
for row in rows:
    dataset,qid,condition=row['dataset'],row['question_id'],row['condition']
    events=row.get('result',{}).get('events',row.get('partial_events',[]))
    for old in events:
        audit=old['audit'];question=questions[dataset][qid];need=audit['request']['need']
        bank=[Case(**c) for c in protocol['case_pools'][dataset][qid]]
        vector_equal=torch.equal(hash_vector(need),hash_vector(question))
        request=Request(**dict(audit['request'],need=question))
        if condition=='bm25':
            ranking=bm25_rank(question,bank);selected=[r['case_id'] for r in ranking[:request.max_cases]]
            diagnostic=dict(query=question,candidates=ranking,selected_ids=selected,boundary='new explicitly direct-question lexical control,not a host-issued request')
        else:
            model=core_bank(bank)
            if condition in {'learned','iterative'}:model.glocal_weightor.feature_weights.data.copy_(metric)
            model.eval().requires_grad_(False)
            retriever=Retriever(model,bank,HashQuery(),model_version='fixed' if condition=='fixed' else f"need-s8-{protocol['metric_checkpoint_sha256'][:12]}",encoder_version='hash256-v1')
            fresh=retriever.retrieve(request,Access('public-benchmark','public-benchmark'),now=audit['timestamp'])
            assert fresh['audit']['model_state_sha256']==audit['model_state_sha256']
            assert fresh['audit']['bank_version']==audit['bank_version']
            fresh_ids.append(fresh['event_id']);selected=fresh['audit']['selected_ids']
            contrib=(model.cases-hash_vector(question)).square()*model.glocal_weightor.feature_weights[0]
            distances=contrib.sum(-1).sqrt()
            for candidate in fresh['audit']['candidates']:
                slot=next(i for i,c in enumerate(bank) if c.case_id==candidate['case_id'])
                torch.testing.assert_close(torch.tensor(candidate['distance']),distances[slot],rtol=1e-6,atol=1e-7)
                torch.testing.assert_close(torch.tensor(candidate['feature_distance_contributions']),contrib[slot],rtol=0,atol=0);manual+=1
            diagnostic=fresh
        records.append(dict(dataset=dataset,question_id=qid,condition=condition,source_event_id=old['event_id'],
            public_question=question,actual_host_need=need,same_hash_vector=vector_equal,
            vector_l2_difference=float((hash_vector(need)-hash_vector(question)).norm()),
            source_selected_ids=audit['selected_ids'],diagnostic_selected_ids=selected,selection_changed=selected!=audit['selected_ids'],
            diagnostic=diagnostic))
assert len(records)==64 and len(fresh_ids)==48 and len(set(fresh_ids))==48
counts=dict(comparisons=len(records),new_actual_nn_retrievals=48,bm25_query_controls=16,manual_candidates=manual,
    changed_hash_vectors=sum(not r['same_hash_vector'] for r in records),changed_selected_orders=sum(r['selection_changed'] for r in records),
    changed_selected_memberships=sum(set(r['source_selected_ids'])!=set(r['diagnostic_selected_ids']) for r in records),
    by_condition={c:dict(comparisons=len(group),changed_vectors=sum(not r['same_hash_vector'] for r in group),changed_orders=sum(r['selection_changed'] for r in group),
        changed_memberships=sum(set(r['source_selected_ids'])!=set(r['diagnostic_selected_ids']) for r in group))
        for c in ('bm25','fixed','learned','iterative') for group in [[r for r in records if r['condition']==c]]})
report=dict(source=freeze,counts=counts,records=records,all_recorded_checks_passed=True,
    boundary='new explicit direct-question NN/lexical counterfactuals; primary host-issued needs/results untouched; no new LLM generation,hidden planner,reconstructed old IDs or QA improvement claim')
(work.parent/'outputs/neural-cbr-t3-need-shift-diagnosis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(counts))
