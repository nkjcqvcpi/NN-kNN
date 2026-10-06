import json
import math
import pytest
import torch
from model.t3.internal import ArtifactLogitMixture


class Tokenizer:
    all_special_ids=[3]
    def encode(self,text,*,add_special_tokens):
        assert not add_special_tokens
        return [dict(A=0,B=1,C=2,SPECIAL=3)[token] for token in text.split()]


def case(case_id,text):
    return dict(case_id=case_id,source='public source',content=json.dumps(dict(title='A',passage=text)))


def test_exact_activation_weighted_probability_mixture_and_case_lineage():
    evidence=[case(10,'A B'),case(20,'B')]
    mixture=ArtifactLogitMixture(evidence,{10:.75,20:.25},Tokenizer(),vocabulary_size=4,alpha=.2,tokenizer_version='frozen-test')
    # title is part of original text: first histogram A2/B1,second A1/B1.
    pointer=torch.tensor([.625,.375,0.,0.])
    host=torch.tensor([[.6,.3,.1,0.]])
    logits=host.log();before=logits.clone()
    output=mixture(torch.tensor([[0,1]]),logits)
    torch.testing.assert_close(output.softmax(-1),.8*host+.2*pointer,rtol=1e-6,atol=1e-7)
    assert torch.equal(logits,before)
    assert [row['case_id'] for row in mixture.metadata['cases']]==[10,20]
    assert mixture.metadata['trainable_parameters']==0
    assert mixture.events[0]['allowed_tokens']==3
    evidence[0]['content']='edited external payload'
    assert mixture.metadata['cases'][0]['token_count']==3


def test_hard_grammar_masks_remain_forbidden_and_allowed_mass_renormalizes():
    mixture=ArtifactLogitMixture([case(7,'B')],{7:1.},Tokenizer(),vocabulary_size=4,alpha=.5,tokenizer_version='frozen-test')
    logits=torch.tensor([[0.,-torch.inf,0.,-torch.inf]])
    result=mixture(torch.tensor([[0]]),logits)
    assert torch.isneginf(result[0,1]) and torch.isneginf(result[0,3])
    torch.testing.assert_close(result.softmax(-1),torch.tensor([[.75,0.,.25,0.]]))
    assert mixture.events[0]['allowed_artifact_mass']==.5


def test_no_allowed_artifact_mass_or_zero_gate_returns_exact_host_logits():
    for alpha in (0.,.3):
        mixture=ArtifactLogitMixture([case(8,'B SPECIAL')],{8:1.},Tokenizer(),vocabulary_size=4,alpha=alpha,tokenizer_version='frozen-test')
        logits=torch.tensor([[-torch.inf,-torch.inf,2.,-torch.inf]])
        assert mixture(torch.tensor([[0]]),logits) is logits
        assert mixture.events[0]['active'] is False
        assert mixture.metadata['cases'][0]['excluded_special_tokens']==1


def test_bad_activation_alignment_and_cross_task_batch_are_rejected():
    for weights in ({9:1.},{8:-1.},{8:float('nan')},{8:True},{8:0.}):
        with pytest.raises(ValueError):ArtifactLogitMixture([case(8,'B')],weights,Tokenizer(),vocabulary_size=4,alpha=.1,tokenizer_version='frozen-test')
    mixture=ArtifactLogitMixture([case(8,'B')],{8:1.},Tokenizer(),vocabulary_size=4,alpha=.1,tokenizer_version='frozen-test')
    with pytest.raises(ValueError,match='one task'):mixture(torch.ones(2,3,dtype=torch.long),torch.zeros(2,4))
    with pytest.raises(ValueError,match='finite allowed'):mixture(torch.ones(1,3,dtype=torch.long),torch.full((1,4),-torch.inf))


def test_large_host_vocabulary_trace_matches_composed_probability():
    vocab=151936
    mixture=ArtifactLogitMixture([case(8,'A B')],{8:1.},Tokenizer(),vocabulary_size=vocab,alpha=.1,tokenizer_version='large-host-test')
    output=mixture(torch.tensor([[1,2]]),torch.zeros(1,vocab))
    expected=.9/vocab+.1*2/3
    assert mixture.events[0]['mixture_argmax']==0
    assert mixture.events[0]['mixture_probability_at_argmax']==pytest.approx(expected,rel=1e-6,abs=1e-7)
    assert float(output[0,0].exp())==pytest.approx(expected,rel=1e-6,abs=1e-7)
