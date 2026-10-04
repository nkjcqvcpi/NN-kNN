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
        return dict(need="lookup public fact", requested_types=["evidence"])
    result = run_loop("task", host, retriever(), Access("s", "u"), Budget(), now=1)
    assert result["stop_reason"] == "duplicate_need"
    assert result["retrieval_rounds"] == 1
    assert [c["content"] for c in result["evidence"]] == ["global original", "user original"]
    assert result["events"][0]["delivered_to_host"]


def test_context_budget_and_host_ready_do_not_imply_delivery():
    def host(**kwargs):
        return dict(ready=True, answer="public answer") if kwargs["evidence"] else dict(
            need="lookup public fact", requested_types=["evidence"])
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
