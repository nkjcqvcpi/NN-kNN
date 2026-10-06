"""Candidate frozen-host output interface, not semantic or utility training.

Mix next-token probabilities with an activation-weighted original-artifact token
histogram. Respect existing hard decoding masks; never authorize tools/actions.
The public QA adapter accepts title plus passage/sentence-list evidence only.
"""
from collections import Counter
import copy
import hashlib
import math
from types import SimpleNamespace

import torch

from .lexical import artifact_text


class ArtifactLogitMixture:
    def __init__(self,evidence,weights,tokenizer,*,vocabulary_size,alpha,tokenizer_version):
        if type(vocabulary_size) is not int or vocabulary_size<2:
            raise ValueError('explicit host output vocabulary required')
        if isinstance(alpha,bool) or not isinstance(alpha,(int,float)) or not math.isfinite(alpha) or not 0<=alpha<1:
            raise ValueError('fixed finite mixture alpha in[0,1) required')
        if not isinstance(tokenizer_version,str) or not tokenizer_version.strip():
            raise ValueError('explicit frozen tokenizer binding required')
        if not isinstance(evidence,(list,tuple)) or not evidence:
            raise ValueError('delivered original evidence required')
        ids=[record['case_id'] for record in evidence]
        if any(type(i) is not int or i<0 for i in ids) or len(set(ids))!=len(ids) or set(weights)!=set(ids):
            raise ValueError('unique stable IDs must align exactly with activation weights')
        if any(isinstance(w,bool) or not isinstance(w,(int,float)) or not math.isfinite(w) or not 0<=w<=1 for w in weights.values()) or sum(weights.values())<=0:
            raise ValueError('finite nonnegative actual activation weights required')
        excluded=set(tokenizer.all_special_ids)
        pointer=torch.zeros(vocabulary_size,dtype=torch.float32)
        cases=[]
        total=sum(weights.values())
        for record in evidence:
            if not isinstance(record['source'],str) or not record['source'].strip():
                raise ValueError('original evidence provenance required')
            text=artifact_text(SimpleNamespace(content=record['content']))
            encoded=tokenizer.encode(text,add_special_tokens=False)
            if any(type(token) is not int or not 0<=token<vocabulary_size for token in encoded):
                raise ValueError('artifact token outside host output vocabulary')
            counts=Counter(token for token in encoded if token not in excluded)
            if not counts:raise ValueError('artifact contains no non-control tokens')
            mass=weights[record['case_id']]/total
            size=sum(counts.values())
            for token,count in counts.items():pointer[token]+=mass*count/size
            cases.append(dict(case_id=record['case_id'],source=record['source'],
                content_sha256=hashlib.sha256(record['content'].encode()).hexdigest(),
                activation_weight=weights[record['case_id']],normalized_activation=mass,
                token_counts=sorted(counts.items()),token_count=size,
                excluded_special_tokens=len(encoded)-size))
        self._pointer=pointer;self._alpha=float(alpha);self._events=[]
        self._metadata=dict(interface='frozen activation-weighted artifact token probability mixture',
            alpha=float(alpha),vocabulary_size=vocabulary_size,tokenizer_version=tokenizer_version,
            cases=cases,excluded_special_ids=sorted(excluded),trainable_parameters=0,
            boundary='lexical output interface candidate; activation is need match,not utility; grammar remains authoritative; no tool authorization')

    @property
    def metadata(self):return copy.deepcopy(self._metadata)

    @property
    def events(self):return copy.deepcopy(self._events)

    def __call__(self,input_ids,scores):
        if input_ids.ndim!=2 or input_ids.shape[0]!=1 or scores.ndim!=2 or scores.shape!=(1,len(self._pointer)):
            raise ValueError('one task/one host batch and exact output vocabulary required')
        if torch.isnan(scores).any() or torch.isposinf(scores).any() or not torch.isfinite(scores).any():
            raise ValueError('finite allowed host logits required')
        allowed=torch.isfinite(scores)
        pointer=self._pointer.to(scores.device).unsqueeze(0)*allowed
        mass=pointer.sum()
        host=scores.float().log_softmax(-1)
        active=self._alpha>0 and bool(mass>0)
        if active:
            pointer=pointer/mass
            result=torch.logaddexp(host+math.log1p(-self._alpha),pointer.log()+math.log(self._alpha))
            result=result.masked_fill(~allowed,-torch.inf)
        else:result=scores
        old=int(scores.argmax(-1)[0]);new=int(result.argmax(-1)[0])
        if not bool(torch.isneginf(result[~allowed]).all()):
            raise ValueError('internal interface changed a forbidden host token')
        self._events.append(dict(step=len(self._events),input_length=input_ids.shape[1],
            allowed_tokens=int(allowed.sum()),allowed_artifact_mass=float(mass),active=active,
            allowed_artifact_token_ids=torch.nonzero(pointer[0]>0).flatten().cpu().tolist(),
            forbidden_tokens_preserved=True,
            host_argmax=old,mixture_argmax=new,argmax_changed=old!=new,
            artifact_probability_at_argmax=float(pointer[0,new]) if active else 0.,
            host_probability_at_argmax=float(host[0,new].exp()),
            mixture_probability_at_argmax=float(result[0,new].float().exp()) if active else float(host[0,new].exp())))
        return result
