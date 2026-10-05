import json

import pytest
import torch

from model.t1.core import CoreConfig, build_model
from model.t3.retrieval import Access, Case, Request, Retriever, conditional_credit
from model.t3.orchestrator import Budget, run_loop


def fixture_bank():
    model = build_model(torch.tensor([[0.], [.1], [.2], [.3], [.4], [.5]]),
        torch.zeros(6), CoreConfig(task_type="regression", bias_init="manual"), None)
    cases = [Case(0, "global original", "evidence", "fixture", "global", validated=True),
        Case(1, "private other user", "evidence", "fixture", "user", "other", explicit_retention=True),
        Case(2, "session original", "relation", "fixture", "session", "s", expires_at=10),
        Case(3, "quarantined original", "evidence", "fixture", "global", validated=True, quarantined=True),
        Case(4, "domain original", "tool", "fixture", "domain", "d", validated=True),
        Case(5, "user original", "evidence", "fixture", "user", "u", explicit_retention=True)]
    return model, cases


def retriever():
    model, cases = fixture_bank()
    return Retriever(model, cases, lambda request: torch.tensor([[0.]]),
        model_version="fixed-0", encoder_version="fixture-vector-0")


def test_actual_masked_geometry_and_same_event_channels():
    r = retriever()
    req = Request("public need", ("evidence", "relation", "tool"), max_cases=3)
    result = r.retrieve(req, Access("s", "u", frozenset({"d"})), now=9)
    assert result["audit"]["eligible_ids"] == [0, 2, 4, 5]
    assert result["audit"]["selected_ids"] == [0, 2, 4]
    assert [c["case_id"] for c in result["evidence"]] == [0, 2, 4]
    assert result["event_id"] == result["audit"]["event_id"]
    assert "private other user" not in json.dumps(result)
    assert "quarantined original" not in json.dumps(result)
    candidates = result["audit"]["candidates"]
    expected = torch.softmax(torch.tensor([0., -.2, -.4]), dim=0).tolist()
    assert [c["weight"] for c in candidates[:3]] == pytest.approx(expected)
    assert sum(c["weight"] for c in candidates) == pytest.approx(1)
    assert all(c["reliability"] == "uncertain" for c in result["evidence"])
    assert all("weight" not in c and "Q" not in c for c in result["evidence"])
    for c in candidates:
        assert sum(c["feature_distance_contributions"]) == pytest.approx(c["distance"]**2)


def test_expiry_type_novelty_and_membership_gates_before_normalization():
    r = retriever()
    result = r.retrieve(Request("need", ("relation", "tool")), Access("s", "u"), now=10)
    assert result["evidence"] == [] and result["audit"]["candidates"] == []
    result = r.retrieve(Request("need", ("evidence",), already_retrieved_ids=(0,)), Access("s", "u"), now=1)
    assert result["audit"]["selected_ids"] == [5]
    assert result["audit"]["candidates"][0]["weight"] == pytest.approx(1)


@pytest.mark.parametrize("kwargs", [dict(scope="session", owner="s"),
    dict(scope="user", owner="u"), dict(scope="domain", owner="d"), dict(scope="global")])
def test_admission_requires_scope_lifecycle_intent_or_validation(kwargs):
    with pytest.raises(ValueError):
        Case(0, "original", "evidence", "fixture", **kwargs)


def test_loop_requires_host_query_stops_duplicate_and_preserves_originals():
    def host(**kwargs):
        return dict(ready=False, need="lookup public fact", requested_types=["evidence"])
    result = run_loop("task", host, retriever(), Access("s", "u"), Budget(), now=1)
    assert result["stop_reason"] == "duplicate_need"
    assert result["retrieval_rounds"] == 1
    assert [c["content"] for c in result["evidence"]] == ["global original", "user original"]
    assert result["events"][0]["delivered_to_host"]


def test_context_budget_and_host_ready_do_not_imply_delivery():
    def host(**kwargs):
        return dict(ready=True, answer="public answer") if kwargs["evidence"] else dict(
            ready=False, need="lookup public fact", requested_types=["evidence"])
    small = run_loop("task", host, retriever(), Access("s", "u"), Budget(max_evidence_chars=1), now=1)
    assert small["stop_reason"] == "context_budget" and not small["evidence"]
    assert small["events"][0]["delivered_to_host"] is False
    full = run_loop("task", host, retriever(), Access("s", "u"), Budget(), now=1)
    assert full["answer"] == "public answer" and full["stop_reason"] == "host_ready"


def test_objective_conditional_credit_has_no_same_label_rule():
    assert conditional_credit(.5, .2, .8) == pytest.approx(dict(utility=.6, C=.3, H=0, Q=1.3/2.3))
    harmful = conditional_credit(.5, .8, .2)
    assert harmful["H"] == pytest.approx(.3) and harmful["Q"] < .5
    with pytest.raises(ValueError):
        conditional_credit(.5, float("nan"), .2)


def test_event_ids_do_not_collide_across_reconstructed_retrievers():
    request, access = Request("need", ("evidence",)), Access("s", "u")
    assert retriever().retrieve(request, access, now=1)["event_id"] != retriever().retrieve(request, access, now=1)["event_id"]


def test_missing_readiness_is_a_host_schema_failure_not_invented_need():
    with pytest.raises(ValueError, match="boolean ready"):
        run_loop("task", lambda **kwargs: dict(need="guess", requested_types=["evidence"]),
            retriever(), Access("s", "u"), Budget(), now=1)


@pytest.mark.parametrize('field,value', [('validated','false'),('explicit_retention',1),
    ('quarantined',0),('case_id',-1),('content',[]),('source',' '),('expires_at',True)])
def test_metadata_rejects_truthy_admission_and_invalid_identifiers(field, value):
    kwargs = dict(case_id=0,content='original',artifact_type='evidence',source='fixture',scope='global',validated=True)
    kwargs[field] = value
    with pytest.raises(ValueError):
        Case(**kwargs)


def test_snapshot_copies_metadata_and_request_access_collections():
    model,cases = fixture_bank()
    r = Retriever(model,cases,lambda request:torch.tensor([[0.]]),model_version='v1',encoder_version='v1')
    cases[0] = Case(0,'replacement','evidence','other','global',validated=True)
    assert r.cases[0].content == 'global original'
    with pytest.raises(TypeError):
        r.cases[0] = cases[0]
    with pytest.raises(AttributeError):
        r.cases = {}
    types,seen,domains = ['evidence'],[0],{'d'}
    req,access = Request('need',types,already_retrieved_ids=seen),Access('s','u',domains)
    types.append('tool'); seen.clear(); domains.clear()
    assert req.requested_types == ('evidence',) and req.already_retrieved_ids == (0,)
    assert access.domain_ids == frozenset({'d'})


@pytest.mark.parametrize('change', ['bias','raw_case','metric','config','tau','count'])
def test_actual_model_drift_rejected_even_when_declared_version_unchanged(change):
    r = retriever()
    with torch.no_grad():
        if change == 'bias': r.model.biases[0].add_(1)
        elif change == 'raw_case': r.model.cases[0].add_(1)
        elif change == 'metric': next(r.model.glocal_weightor.parameters()).add_(1)
        elif change == 'config': r.model.config['tau'] = 2
        elif change == 'tau': r.model.tau = 2
        elif change == 'count': r.model.set_active_case_count(5)
    with pytest.raises(ValueError,match='model changed'):
        r.retrieve(Request('need',('evidence',)),Access('s','u'),now=1)


def test_only_active_slots_require_metadata_and_are_audited():
    model,cases = fixture_bank()
    model.set_active_case_count(2)
    model.case_ids[2:] = -1
    r = Retriever(model,cases[:2],lambda request:torch.tensor([[0.]]),model_version='v1',encoder_version='v1')
    result = r.retrieve(Request('need',('evidence',)),Access('s','u'),now=1)
    assert result['audit']['eligible_ids'] == [0] and result['audit']['gated_count'] == 1
    assert result['audit']['encoder_verification'] == 'declared_only'
    assert result['audit']['encoder_state_sha256'] is None


def test_drift_during_encoding_never_delivers_evidence():
    model,cases = fixture_bank()
    def encoder(request):
        model.biases[0].add_(1)
        return torch.tensor([[0.]])
    r = Retriever(model,cases,encoder,model_version='v1',encoder_version='v1')
    with pytest.raises(ValueError,match='model changed'):
        r.retrieve(Request('need',('evidence',)),Access('s','u'),now=1)


def test_module_encoder_weights_are_bound_and_mixed_modes_restored():
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1,1)
            self.dropout = torch.nn.Dropout(.9)
        def forward(self, request):
            assert not self.training and not self.dropout.training
            return self.dropout(self.linear(torch.zeros(1,1)))
    model,cases = fixture_bank()
    model.train(); model.glocal_weightor.eval()
    encoder = Encoder(); encoder.linear.eval()
    r = Retriever(model,cases,encoder,model_version='v1',encoder_version='v1')
    before = [(m,m.training) for root in (model,encoder) for m in root.modules()]
    result = r.retrieve(Request('need',('evidence',)),Access('s','u'),now=1)
    assert all(m.training == mode for m,mode in before)
    assert result['audit']['encoder_verification'] == 'module_state'
    with torch.no_grad(): encoder.linear.weight.add_(1)
    with pytest.raises(ValueError,match='encoder changed'):
        r.retrieve(Request('need',('evidence',)),Access('s','u'),now=1)


def test_feature_cache_is_not_a_versioned_source_and_is_restored():
    model = build_model(torch.tensor([[0.],[.1],[.2]]),torch.zeros(3),
        CoreConfig(task_type='regression',representation='mlp',bias_init='manual'),None)
    cases = [Case(i,str(i),'evidence','fixture','global',validated=True) for i in range(3)]
    model.cached_features = torch.full((3,model.feature_dim),1e6)
    old_cache = model.cached_features
    r = Retriever(model,cases,lambda request:torch.tensor([[0.]]),model_version='v1',encoder_version='v1')
    result = r.retrieve(Request('need',('evidence',)),Access('s','u'),now=1)
    assert model.cached_features is old_cache
    expected = model.feature_extractor(model.cases).detach().tolist()
    assert result['audit']['case_features'] == expected


def test_iterative_round_budget_delivers_one_then_novel_one():
    def host(**kwargs):
        if len(kwargs['evidence'])==2:return dict(ready=True,answer='done')
        return dict(ready=False,need='global evidence' if not kwargs['evidence'] else 'user original information',requested_types=['evidence'])
    result=run_loop('task',host,retriever(),Access('s','u'),Budget(max_rounds=2,max_cases=2,max_cases_per_round=1),now=1)
    assert result['stop_reason']=='host_ready' and result['retrieval_rounds']==2
    assert [len(e['evidence']) for e in result['events']]==[1,1]
    assert len({c['case_id'] for c in result['evidence']})==2


def test_journal_preserves_completed_retrieval_when_next_host_fails():
    events=[]
    def host(**kwargs):
        if kwargs['evidence']:raise ValueError('malformed next host decision')
        return dict(ready=False,need='global need',requested_types=['evidence'])
    with pytest.raises(ValueError,match='malformed'):
        run_loop('task',host,retriever(),Access('s','u'),Budget(),now=1,event_sink=events.append)
    assert len(events)==1 and events[0]['delivered_to_host'] is True
    assert events[0]['event_id']==events[0]['audit']['event_id']
    events.clear()
    result=run_loop('task',host,retriever(),Access('s','u'),Budget(max_evidence_chars=1),now=1,event_sink=events.append)
    assert events==result['events'] and events[0]['delivered_to_host'] is False
