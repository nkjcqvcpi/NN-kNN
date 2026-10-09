"""Actual UI session actions, checkpoint replay and localhost request boundary."""
import copy
import json
import threading
from http.server import HTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest
import torch

from model.common.core import CoreConfig, build_model
from model.common.outcomes import final_prediction, frozen_evaluation
from model.t1.reviewer_ui import ReviewerSession
from tools.t1_reviewer_ui import make_handler


@pytest.fixture
def session(tmp_path):
    return ReviewerSession(tmp_path / "session", epochs=2)


def request(session, op, **kwargs):
    return {"version": session.version, "operation": op, "case_id": 4, "query_id": 4,
            "actor": "engineering test", "reason": "blue circle does not satisfy red-circle rule",
            "expected_effect": "improve", "confidence": 90, **kwargs}


def test_displayed_credit_and_predictions_share_actual_event(session):
    view = session.snapshot
    A, C, H = {}, {}, {}
    for query in view["queries"]:
        assert sum(query["activations"]) == pytest.approx(1)
        assert query["prediction"] == max(range(2), key=lambda c: query["probabilities"][c])
        correct = query["prediction"] == int(session.query_targets[query["id"]])
        for cid, activation in zip(query["case_ids"], query["activations"]):
            A[cid] = A.get(cid, 0) + activation
            dest = C if correct else H
            dest[cid] = dest.get(cid, 0) + activation
    for case in view["cases"]:
        cid = case["id"]
        assert case["A"] == pytest.approx(A.get(cid, 0), abs=1e-6)
        assert case["C"] == pytest.approx(C.get(cid, 0), abs=1e-6)
        assert case["H"] == pytest.approx(H.get(cid, 0), abs=1e-6)


def test_revision_is_no_gradient_and_undo_restores_target_and_parameters(session):
    original = copy.deepcopy(session.model.state_dict())
    optimizer = copy.deepcopy(session.optimizer.state_dict())
    result = session.act(request(session, "relabel", label=0))
    assert session.targets[4] == 0
    assert all(torch.equal(v, session.model.state_dict()[k]) for k,v in original.items() if k != "labels")
    for idx, state in optimizer["state"].items():
        for key, value in state.items():
            assert torch.equal(value, session.optimizer.state_dict()["state"][idx][key])
    assert result["actions"][-1]["predeclared_target_event"]["retrieval_event_id"] == "ui-v0-q4"
    session.act(request(session, "undo", intervention_id=session.log.records[-1]["intervention_id"]))
    assert torch.equal(session.targets, session.m0_targets)
    assert all(torch.equal(v, session.model.state_dict()[k]) for k,v in original.items())
    assert len(session.actions) == 2


def test_invalid_or_stale_actions_do_not_mutate_session(session):
    before = copy.deepcopy(session.snapshot)
    states = copy.deepcopy(session.model.state_dict())
    for invalid in (request(session, "relabel", label=7), request(session, "relabel", case_id=True, label=0),
                    request(session, "adjust_bias", delta=float("inf")),
                    request(session, "quarantine", version=-1), request(session, "quarantine", reason="")):
        with pytest.raises(ValueError):
            session.act(invalid)
        assert session.snapshot == before and session.version == 0
        assert all(torch.equal(v, session.model.state_dict()[k]) for k,v in states.items())


def test_quarantine_protection_archive_restore_and_force(session):
    session.act(request(session, "protect"))
    with pytest.raises(ValueError, match="Unprotect"):
        session.act(request(session, "archive_remove"))
    session.act(request(session, "unprotect"))
    original = copy.deepcopy(session.model.state_dict())
    session.act(request(session, "quarantine"))
    assert all(4 not in q["case_ids"] for q in session.snapshot["queries"])
    session.act(request(session, "archive_remove"))
    assert session.model.case_count() == 7
    session.act(request(session, "restore"))
    assert session.model.case_count() == 8 and 4 in session.log.quarantined
    slot = session.model.active_case_ids().tolist().index(4)
    assert torch.equal(session.model.labels[slot], original["labels"][4])
    session.act(request(session, "release"))
    model_before = copy.deepcopy(session.model.state_dict())
    session.act(request(session, "force", case_ids=[4], activation_floor=.3))
    event = session.controlled_result["event"]
    assert event["activations"][event["case_ids"].index(4)] >= .3 - 1e-6
    assert all(torch.equal(v, session.model.state_dict()[k]) for k,v in model_before.items())


def test_M2_MC_fixed_budget_and_saved_checkpoints_replay(session):
    session.act(request(session, "relabel", label=0))
    session.act(request(session, "retrain", epochs=2))
    assert len(session.retraining["M2_history"]) == len(session.retraining["MC_history"]) == 2
    checkpoint = torch.load(session.output / "v0002-M2/checkpoint.pt", weights_only=True)
    model = build_model(checkpoint["X_train"], checkpoint["training_targets"], CoreConfig(**checkpoint["core"]), 2)
    model.load_state_dict(checkpoint["model_state"])
    model.set_active_case_count(checkpoint["active_case_count"])
    with frozen_evaluation(model):
        r = model.retrieve(checkpoint["X_queries"], exclude_identical=False)
        _, post, _ = final_prediction(model, checkpoint["X_queries"], r)
    assert post.argmax(1).tolist() == [q["prediction"] for q in session.snapshot["queries"]]
    assert (session.output / "MC.pt").exists()
    with pytest.raises(ValueError, match="比较已完成"):
        session.act(request(session, "relabel", label=1))


def test_loopback_http_requires_session_token_origin_and_valid_payload(session):
    server = HTTPServer(("127.0.0.1", 0), make_handler(session, "test-session-token"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(url + "/api/state") as response:
            assert json.load(response)["version"] == 0
        payload = json.dumps(request(session, "relabel", label=0)).encode()
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url + "/api/action", payload, method="POST"))
        assert error.value.code == 403 and session.version == 0
        for origin in (url, "https://unrelated.invalid"):
            req = Request(url + "/api/action", payload, method="POST",
                          headers={"X-Reviewer-Token": "test-session-token", "Origin": origin})
            if origin == url:
                with urlopen(req) as response:
                    assert json.load(response)["version"] == 1
            else:
                with pytest.raises(HTTPError) as error:
                    urlopen(req)
                assert error.value.code == 403
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)
