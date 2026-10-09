"""Declared lexical controls and actual NN-kNN need-matching geometry.

These small lexical representations are baselines,not semantic encoders or
outcome-utility estimators. All distances use the maintained retrieval core.
"""
from collections import Counter
import contextlib
import hashlib
import io
import json
import math
import re

import torch

from model.common.core import CoreConfig, build_model
from .retrieval import Request


def tokens(text):
    return re.findall(r'[a-z0-9]+',text.lower())


def artifact_text(case):
    obj=json.loads(case.content)
    return obj['title']+' '+(obj['passage'] if 'passage' in obj else ' '.join(obj['sentences']))


def hash_vector(text, dimensions=256,*,term_frequency='count'):
    if type(dimensions) is not int or dimensions<2:
        raise ValueError('hash representation requires at least two dimensions')
    if term_frequency not in ('count','sublinear','binary'):
        raise ValueError('unsupported lexical term frequency')
    v=torch.zeros(dimensions)
    if term_frequency=='count':
        for token in tokens(text):
            v[int.from_bytes(hashlib.sha256(token.encode()).digest()[:8],'big')%dimensions]+=1
    else:
        for token,count in Counter(tokens(text)).items():
            value=1+math.log(count) if term_frequency=='sublinear' else 1.
            v[int.from_bytes(hashlib.sha256(token.encode()).digest()[:8],'big')%dimensions]+=value
    return v/v.norm().clamp_min(1e-9)


class HashQuery(torch.nn.Module):
    def __init__(self,dimensions=256,*,term_frequency='count'):
        super().__init__()
        # Configuration is part of the versioned module topology via extra_repr.
        if type(dimensions) is not int or dimensions<2:
            raise ValueError('hash representation requires at least two dimensions')
        if term_frequency not in ('count','sublinear','binary'):raise ValueError('unsupported lexical term frequency')
        self.dimensions=dimensions
        self.term_frequency=term_frequency

    def extra_repr(self):
        base=f'dimensions={self.dimensions}, tokenizer=ascii-alnum-lower, hash=sha256-64bit, norm=l2, fields=need'
        return base if self.term_frequency=='count' else base+f', term_frequency={self.term_frequency}'

    def forward(self,request: Request):
        return hash_vector(request.need,self.dimensions,term_frequency=self.term_frequency).unsqueeze(0)


def candidate_bank(task,corpus,*,size=64,seed=20261005):
    """SQuAD supplied passage plus ID-hashed negatives; no answer inspection.

    Explicitly a conditional candidate pool,not unrestricted corpus retrieval.
    Original ordering is hidden by stable ID sorting for both controls.
    """
    by_id={c.case_id:c for c in corpus}
    if len(by_id)!=len(corpus) or type(size) is not int or not len(task.cases)<=size<=len(corpus):
        raise ValueError('candidate pool requires unique corpus IDs and a valid size')
    if any(by_id.get(c.case_id)!=c for c in task.cases):
        raise ValueError('provided task passages must occur in declared corpus')
    mandatory={c.case_id for c in task.cases}
    negatives=sorted((c for c in corpus if c.case_id not in mandatory),
        key=lambda c:hashlib.sha256(f'{seed}\0{task.question_id}\0{c.case_id}'.encode()).digest())
    return tuple(sorted((*task.cases,*negatives[:size-len(task.cases)]),key=lambda c:c.case_id))


def bm25_rank(query,cases,*,k1=1.2,b=.75):
    if not cases or k1<=0 or not 0<=b<=1:
        raise ValueError('BM25 requires cases,k1>0 and b in [0,1]')
    docs=[Counter(tokens(artifact_text(c))) for c in cases]
    lengths=[sum(d.values()) for d in docs]
    average=sum(lengths)/len(lengths)
    df=Counter(token for d in docs for token in d)
    query_counts=Counter(tokens(query))
    scores=[]
    for case,doc,length in zip(cases,docs,lengths):
        score=0.
        for token,count in query_counts.items():
            freq=doc[token]
            if freq:
                idf=math.log1p((len(docs)-df[token]+.5)/(df[token]+.5))
                score+=count*idf*freq*(k1+1)/(freq+k1*(1-b+b*length/max(average,1e-12)))
        scores.append(dict(case_id=case.case_id,score=score))
    return sorted(scores,key=lambda row:(-row['score'],row['case_id']))


def core_bank(cases,*,dimensions=256,metric=None,term_frequency='count'):
    X=torch.stack([hash_vector(artifact_text(c),dimensions,term_frequency=term_frequency) for c in cases])
    with contextlib.redirect_stdout(io.StringIO()):
        model=build_model(X,torch.zeros(len(cases)),CoreConfig(task_type='regression',bias_init='manual',top_k=len(cases)),None)
    model.case_ids.copy_(torch.tensor([c.case_id for c in cases]))
    if term_frequency!='count':model.config['t3_term_frequency']=term_frequency
    if metric is not None:model.glocal_weightor=metric
    for parameter in model.parameters():parameter.requires_grad_(False)
    if metric is not None:
        for parameter in metric.parameters():parameter.requires_grad_(True)
    return model


def need_loss(model,question,positive_ids,*,dimensions=256,objective='softmax',hard_negatives=8,margin=.2,term_frequency='count'):
    if model.config.get('t3_term_frequency','count')!=term_frequency:
        raise ValueError('query/case term frequency mismatch')
    if (model.sampling_cases_flag or model.case_normalizer!='softmax' or
            not model.normalize_over_cases or model.config.get('case_score_mode')!='bias_minus_distance' or
            model.config.get('top_k',0)<model.case_count()):
        raise ValueError('need training requires exact full-bank softmax geometry')
    ids=model.active_case_ids()
    mask=torch.isin(ids,torch.tensor(tuple(positive_ids)))
    if not bool(mask.any()):raise ValueError('need training requires an actual positive source ID')
    result=model.retrieve(hash_vector(question,dimensions,term_frequency=term_frequency).unsqueeze(0),exclude_identical=False)
    # Actual core distances and biases define the probability. The full bank is
    # used for training; no top-k loss proxy,answer text or MC outcome enters it.
    logits=(model.biases-result['distances'][0])/model.tau
    if objective=='softmax':
        loss=torch.logsumexp(logits,0)-torch.logsumexp(logits[mask],0)
    elif objective=='hard_negative':
        if type(hard_negatives) is not int or hard_negatives<1 or not math.isfinite(margin) or margin<0 or bool(mask.all()):
            raise ValueError('hard-negative objective needs negatives,positive count and finite nonnegative margin')
        distances=result['distances'][0]
        closest_positive=distances[mask].min()
        closest_negative=distances[~mask].sort(stable=True).values[:hard_negatives]
        loss=torch.nn.functional.softplus((closest_positive-closest_negative+margin)/model.tau).mean()
    else:raise ValueError('unsupported declared need objective')
    return loss,result
