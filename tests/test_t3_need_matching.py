import json

import pytest
import torch

from model.nnknn_model import GlocalFeatureWeight
from model.t3.benchmarks import PublicTask
from model.t3.lexical import HashQuery,bm25_rank,candidate_bank,core_bank,need_loss
from model.t3.retrieval import Case,Request,Retriever,Access


def case(i,text):return Case(i,json.dumps(dict(title=str(i),passage=text)),'evidence','fixture','global',validated=True)


def test_pool_is_id_based_and_bm25_uses_actual_artifact_text():
    cases=[case(1,'green badge'),case(2,'red city'),case(3,'blue city')]
    task=PublicTask('q','Which badge?',(cases[0],),'g')
    bank=candidate_bank(task,cases,size=2)
    assert cases[0] in bank and bank==candidate_bank(task,list(reversed(cases)),size=2)
    assert bm25_rank('badge',bank)[0]['case_id']==1
    with pytest.raises(ValueError):candidate_bank(task,cases,size=4)


def test_need_loss_has_actual_core_gradient_and_shared_audit_geometry():
    cases=[case(10,'green badge'),case(20,'red city')]
    metric=GlocalFeatureWeight(32,1)
    model=core_bank(cases,dimensions=32,metric=metric)
    loss,result=need_loss(model,'badge',[10],dimensions=32)
    expected=-torch.log(result['weights'][0,0])
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert metric.feature_weights.grad is not None and bool((metric.feature_weights.grad!=0).any())
    assert all(p.grad is None for name,p in model.named_parameters() if name!='glocal_weightor.feature_weights')
    r=Retriever(model,cases,HashQuery(32),model_version='frozen-after-gradient',encoder_version='hash32')
    audit=r.retrieve(Request('badge',('evidence',),max_cases=2),Access('s','u'),now=1)['audit']
    assert audit['encoder_verification']=='module_state'
    for col,row in enumerate(audit['candidates']):
        assert row['distance']==pytest.approx(float(result['distances'][0,col].detach()))
        assert sum(row['feature_distance_contributions'])==pytest.approx(row['distance']**2)
    with pytest.raises(ValueError):need_loss(model,'badge',[30],dimensions=32)
