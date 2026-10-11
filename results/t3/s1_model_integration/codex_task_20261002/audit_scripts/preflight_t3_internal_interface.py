from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile
work=Path(__file__).resolve().parent
native=Path('C:/Users/Administrator/NN-KNN/nnknn-work/.venv/Library/bin')
os.environ['PATH']=str(native)+os.pathsep+os.environ.get('PATH','')
handle=os.add_dll_directory(str(native)) if sys.platform=='win32' else None
repo=Path('C:/Users/Administrator/NN-KNN/nnknn-work');sys.path.insert(0,str(repo))
import torch
from transformers import AutoTokenizer
from model.t1.artifacts import source_fingerprint
from model.t3.internal import ArtifactLogitMixture
from model.t3.lexical import core_bank,HashQuery,hash_vector
from model.t3.retrieval import Case,Request,Access,Retriever
source=work.parent/'outputs/t3-public-host-20261005-format-enforced'
verified=json.loads((work.parent/'outputs/neural-cbr-t3-format-host-verification.json').read_text())
assert verified['all_recorded_retrieval_traces_verified'] and verified['counts']['trials']==80
for name,bound in verified['input_artifacts'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==bound['sha256']
protocol=json.loads((source/'protocol.json').read_text())
rows=[json.loads(line) for line in (source/'trials.jsonl').read_text().splitlines()]
rows=[r for r in rows if r['condition']=='learned'];assert len(rows)==16
metric_path=work/'t3-need-matching-20261005/s8/metric.pt'
assert hashlib.sha256(metric_path.read_bytes()).hexdigest()==protocol['metric_checkpoint_sha256']
metric=torch.load(metric_path,weights_only=True)['selected_metric']['feature_weights']
model_dir=work/'t3-models/qwen3-0.6b'
tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
vocab=json.loads((model_dir/'config.json').read_text())['vocab_size']
torch.set_num_threads(1);records=[];manual_candidates=0
for row in rows:
    events=row['partial_events'] if 'error' in row else row['result']['events'];assert len(events)==1
    old=events[0];bank=[Case(**c) for c in protocol['case_pools'][row['dataset']][row['question_id']]]
    core=core_bank(bank);core.glocal_weightor.feature_weights.data.copy_(metric);core.eval().requires_grad_(False)
    retriever=Retriever(core,bank,HashQuery(),model_version=old['audit']['model_version'],encoder_version='hash256-v1')
    fresh=retriever.retrieve(Request(**old['audit']['request']),Access('public-benchmark','public-benchmark'),now=old['audit']['timestamp'])
    assert fresh['evidence']==old['evidence']
    for key in ('selected_ids','bank_version','model_state_sha256','encoder_state_sha256'):assert fresh['audit'][key]==old['audit'][key]
    q=hash_vector(old['audit']['request']['need'])
    contrib=(core.cases-q).square()*metric[0]
    distance=contrib.sum(-1).sqrt()
    for candidate in fresh['audit']['candidates']:
        slot=next(i for i,c in enumerate(bank) if c.case_id==candidate['case_id'])
        torch.testing.assert_close(torch.tensor(candidate['distance']),distance[slot],rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(torch.tensor(candidate['feature_distance_contributions']),contrib[slot],rtol=0,atol=0)
        manual_candidates+=1
    weights={r['case_id']:r['weight'] for r in fresh['audit']['candidates'] if r['case_id'] in fresh['audit']['selected_ids']}
    mixture=ArtifactLogitMixture(fresh['evidence'],weights,tokenizer,vocabulary_size=vocab,alpha=.1,tokenizer_version='verified-qwen3-files')
    pointer=torch.zeros(vocab)
    for case in fresh['evidence']:
        obj=json.loads(case['content']);text=obj['title']+' '+(obj['passage'] if 'passage' in obj else ' '.join(obj['sentences']))
        ids=tokenizer.encode(text,add_special_tokens=False)
        counts=Counter(i for i in ids if i not in tokenizer.all_special_ids)
        for token,count in counts.items():pointer[token]+=weights[case['case_id']]/sum(weights.values())*count/sum(counts.values())
    logits=torch.zeros(1,vocab);input_ids=torch.tensor([[1,2]])
    result=mixture(input_ids,logits)
    expected_full=.9*logits.softmax(-1)[0]+.1*pointer
    print('probability_rounding',float((result.exp()[0]-expected_full).abs().max()),float((result.double().softmax(-1)[0]-expected_full).abs().max()),float(result.exp().sum()),flush=True)
    torch.testing.assert_close(result.double().softmax(-1)[0],expected_full.double(),rtol=1e-5,atol=1e-7)
    chosen=int(pointer.argmax());empty=next(i for i in range(vocab) if pointer[i]==0 and i not in tokenizer.all_special_ids)
    masked=torch.full((1,vocab),-torch.inf);masked[0,chosen]=masked[0,empty]=0
    output=mixture(input_ids,masked)
    expected=torch.zeros_like(masked);expected[0,chosen]=.55;expected[0,empty]=.45
    torch.testing.assert_close(output.softmax(-1),expected,rtol=1e-6,atol=1e-7)
    assert torch.isneginf(output[~torch.isfinite(masked)]).all()
    assert mixture.events[-1]['allowed_artifact_token_ids']==[chosen]
    records.append(dict(dataset=row['dataset'],question_id=row['question_id'],fresh_event=fresh['event_id'],
        selected_ids=fresh['audit']['selected_ids'],mixture=mixture.metadata,full_and_masked_probability_checks=True))
assert len(records)==16 and manual_candidates==592
report=dict(source_fingerprint=source_fingerprint(repo),records=records,actual_nn_retrievals=16,manual_candidates=manual_candidates,
    actual_tokenizer_histograms=32,independent_probability_checks=32,all_recorded_checks_passed=True,
    source_inputs=verified['input_artifacts'],scope='actual public cases/core geometry/Qwen tokenizer histograms with synthetic logits; no fresh LLM answers or measured internal quality')
(work.parent/'outputs/neural-cbr-t3-internal-preflight-verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ('records','source_inputs')}))
