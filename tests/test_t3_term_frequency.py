from collections import Counter
import hashlib
import json
import math
import pytest
import torch
from model.t3.lexical import HashQuery,hash_vector,tokens,core_bank,need_loss
from model.t3.retrieval import Case,Request,Retriever,Access


def independent(text,mode,dimensions=256):
    values=[0.]*dimensions
    for token,count in Counter(tokens(text)).items():
        index=int.from_bytes(hashlib.sha256(token.encode()).digest()[:8],'big')%dimensions
        values[index]+=count if mode=='count' else 1+math.log(count) if mode=='sublinear' else 1
    norm=math.sqrt(sum(value*value for value in values))
    return torch.tensor([value/max(norm,1e-9) for value in values])


@pytest.mark.parametrize('mode',['count','sublinear','binary'])
def test_manual_word_frequency_before_hash_collision_and_l2(mode):
    text='finance aid year year year year unrelated'
    torch.testing.assert_close(hash_vector(text,term_frequency=mode),independent(text,mode),rtol=1e-6,atol=1e-7)
    # Dimension2 makes collisions common; compression is per original word.
    torch.testing.assert_close(hash_vector(text,2,term_frequency=mode),independent(text,mode,2),rtol=1e-6,atol=1e-7)


def test_default_count_is_bit_exact_and_binary_repetition_invariant():
    text='finance aid year '+('year '*70)
    legacy=torch.zeros(256)
    for token in tokens(text):legacy[int.from_bytes(hashlib.sha256(token.encode()).digest()[:8],'big')%256]+=1
    legacy/=legacy.norm().clamp_min(1e-9)
    assert torch.equal(hash_vector(text),legacy)
    assert torch.equal(hash_vector(text,term_frequency='binary'),hash_vector('finance aid year',term_frequency='binary'))
    assert HashQuery().extra_repr()=='dimensions=256, tokenizer=ascii-alnum-lower, hash=sha256-64bit, norm=l2, fields=need'


def test_training_retrieval_and_audit_use_same_frequency_and_reject_mismatch():
    cases=[Case(i,json.dumps(dict(title='title',passage=text)),'evidence','test','global',validated=True) for i,text in [(1,'finance aid year year'),(2,'year year year history')]]
    model=core_bank(cases,term_frequency='sublinear')
    loss,result=need_loss(model,'finance year year',[1],term_frequency='sublinear')
    retriever=Retriever(model,cases,HashQuery(term_frequency='sublinear'),model_version='test',encoder_version='sublinear')
    event=retriever.retrieve(Request('finance year year',('evidence',),max_cases=2),Access('s','u'),now=1)
    for candidate in event['audit']['candidates']:
        slot=next(i for i,c in enumerate(cases) if c.case_id==candidate['case_id'])
        assert candidate['distance']==pytest.approx(float(result['distances'][0,slot]))
    with pytest.raises(ValueError,match='frequency mismatch'):need_loss(model,'finance',[1])
    with pytest.raises(ValueError,match='frequency mismatch'):Retriever(model,cases,HashQuery(),model_version='wrong',encoder_version='wrong')
    with pytest.raises(ValueError,match='frequency mismatch'):Retriever(core_bank(cases),cases,HashQuery(term_frequency='binary'),model_version='wrong',encoder_version='wrong')
