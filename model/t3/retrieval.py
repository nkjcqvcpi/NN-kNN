"""Scoped original-artifact retrieval using the maintained NN-kNN geometry.

This first implementation is the declared single-shared-metric ablation. It
does not imply cross-head score calibration, utility training, or host grounding.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import threading
import uuid
from typing import Callable

import torch

TYPES = frozenset({"evidence", "tool", "procedure", "experience", "correction", "relation"})
SCOPES = frozenset({"session", "user", "domain", "global"})


@dataclass(frozen=True)
class Case:
    case_id: int
    content: str
    artifact_type: str
    source: str
    scope: str
    owner: str | None = None
    expires_at: float | None = None
    explicit_retention: bool = False
    validated: bool = False
    quarantined: bool = False
    reliability: str = "uncertain"

    def __post_init__(self):
        if self.artifact_type not in TYPES or self.scope not in SCOPES:
            raise ValueError("unsupported artifact type or scope")
        if isinstance(self.case_id, bool) or not isinstance(self.case_id, int):
            raise ValueError("case ID must be an integer")
        if not self.content or not self.source:
            raise ValueError("original content and provenance are required")
        if self.reliability not in {"uncertain", "approved"}:
            raise ValueError("reliability is explicit, not inferred from match")
        if self.scope != "global" and not self.owner:
            raise ValueError("local scopes require an owner")
        if self.scope == "session" and self.expires_at is None:
            raise ValueError("temporary memory requires an explicit expiry")
        if self.expires_at is not None and not math.isfinite(self.expires_at):
            raise ValueError("expiry must be finite")
        if self.scope == "user" and not self.explicit_retention:
            raise ValueError("persistent user memory requires explicit retention intent")
        if self.scope in {"domain", "global"} and not self.validated:
            raise ValueError("shared memory requires validated admission")


@dataclass(frozen=True)
class Access:
    session_id: str
    user_id: str
    domain_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Request:
    need: str
    requested_types: tuple[str, ...]
    observable_task_state: str = ""
    already_retrieved_ids: tuple[int, ...] = ()
    max_cases: int = 3

    def __post_init__(self):
        if not self.need.strip() or len(self.need) > 4096 or len(self.observable_task_state) > 8192:
            raise ValueError("host request needs bounded explicit observable text")
        if not self.requested_types or not set(self.requested_types) <= TYPES:
            raise ValueError("host must request supported artifact types")
        if isinstance(self.max_cases, bool) or not isinstance(self.max_cases, int) or not 1 <= self.max_cases <= 32:
            raise ValueError("retrieval case budget must be 1..32")


class Retriever:
    def __init__(self, model, cases: list[Case], encoder: Callable[[Request], torch.Tensor], *,
                 model_version: str, encoder_version: str):
        if model.nn_cdh is not None or getattr(model, "classification_adapter", None) is not None:
            raise ValueError("T3 returns original artifacts without a case-adaptation network")
        if model.sampling_cases_flag or not model.normalize_over_cases or model.case_normalizer != "softmax":
            raise ValueError("audited T3 path requires exact normalized softmax retrieval")
        if model.config.get("case_score_mode") != "bias_minus_distance":
            raise ValueError("this contract exposes the declared bias-minus-distance geometry")
        ids = [int(i) for i in model.case_ids.tolist()]
        self.cases = {c.case_id: c for c in cases}
        if len(self.cases) != len(cases) or set(ids) != set(self.cases) or len(ids) != len(cases):
            raise ValueError("stable artifact IDs must exactly match active NN-kNN rows")
        if not model_version or not encoder_version:
            raise ValueError("retrieval and representation versions are required")
        self.model, self.encoder = model, encoder
        self.model_version, self.encoder_version = model_version, encoder_version
        self.sequence = 0
        self.event_namespace = uuid.uuid4().hex
        self.lock = threading.Lock()
        self.bank_version = hashlib.sha256(json.dumps([asdict(self.cases[i]) for i in sorted(ids)],
            sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _eligible(case: Case, access: Access, now: float) -> bool:
        if case.quarantined or (case.expires_at is not None and now >= case.expires_at):
            return False
        return (case.scope == "global" or
            case.scope == "session" and case.owner == access.session_id or
            case.scope == "user" and case.owner == access.user_id or
            case.scope == "domain" and case.owner in access.domain_ids)

    def retrieve(self, request: Request, access: Access, *, now: float) -> dict:
        if not math.isfinite(now):
            raise ValueError("lifecycle evaluation requires a finite explicit timestamp")
        with self.lock, torch.no_grad():
            model = self.model
            ids = [int(i) for i in model.case_ids.tolist()]
            if set(ids) != set(self.cases) or len(ids) != len(self.cases):
                raise ValueError("memory changed; construct a new versioned retriever")
            eligible = [self._eligible(self.cases[i], access, now) and
                self.cases[i].artifact_type in request.requested_types and
                i not in request.already_retrieved_ids for i in ids]
            self.sequence += 1
            event_id = f"{self.event_namespace}:{self.sequence}"
            audit = dict(event_id=event_id, request=asdict(request), model_version=self.model_version,
                encoder_version=self.encoder_version, bank_version=self.bank_version,
                routing="single_shared_metric", timestamp=now, eligible_ids=[i for i, ok in zip(ids, eligible) if ok],
                gated_count=len(ids)-sum(eligible), candidates=[], selected_ids=[])
            evidence = []
            if any(eligible):
                query = self.encoder(request).to(model.biases.device)
                if query.ndim != 2 or query.shape[0] != 1 or not bool(torch.isfinite(query).all()):
                    raise ValueError("encoder must supply one finite declared query vector")
                mask = torch.tensor(eligible, device=query.device)
                old_k, old_mask, training = model.config.get("top_k"), model.config.get("pre_topk_mask"), model.training
                try:
                    model.eval()
                    model.config.update(top_k=request.max_cases, pre_topk_mask=True)
                    result = model.retrieve(query, exclude_identical=False, case_mask=mask)
                finally:
                    model.config.update(top_k=old_k, pre_topk_mask=old_mask)
                    model.train(training)
                rows = result["case_indices"].tolist()
                for col, row in enumerate(rows):
                    if not eligible[row]:
                        continue  # private/ineligible artifacts are not disclosed in the audit
                    case = self.cases[ids[row]]
                    distance = float(result["distances"][0, col])
                    weight = float(result["weights"][0, col])
                    bias = float(model.biases[row])
                    audit["candidates"].append(dict(case_id=case.case_id, distance=distance,
                        bias=bias, activation=bias-distance, weight=weight, scope=case.scope,
                        expires_at=case.expires_at, source=case.source, reliability=case.reliability,
                        utility_status="unobserved", feature_distance_contributions=
                            result["feature_distance_contributions"][0, col].cpu().tolist()))
                    if weight > 0:
                        evidence.append(dict(case_id=case.case_id, content=case.content,
                            artifact_type=case.artifact_type, source=case.source, reliability=case.reliability))
                selected = sorted(audit["candidates"], key=lambda c: (-c["weight"], c["case_id"]))
                audit["selected_ids"] = [c["case_id"] for c in selected if c["weight"] > 0]
                by_id = {c["case_id"]: c for c in evidence}
                evidence = [by_id[i] for i in audit["selected_ids"]]
                audit["query_features"] = result["query_features"].cpu().tolist()
                audit["case_features"] = [result["case_features"][col].cpu().tolist()
                    for col, row in enumerate(rows) if eligible[row]]
            return dict(event_id=event_id, evidence=evidence, audit=audit)


def conditional_credit(weight: float, full_loss: float, removed_loss: float, *, smoothing: float = 1.0) -> dict:
    """One observed objective counterfactual; no label equality or automatic update."""
    if not all(math.isfinite(v) for v in (weight, full_loss, removed_loss, smoothing)) or not 0 <= weight <= 1 or smoothing <= 0:
        raise ValueError("finite observed losses, activation and positive smoothing are required")
    utility = removed_loss - full_loss
    c, h = weight * max(utility, 0), weight * max(-utility, 0)
    return dict(utility=utility, C=c, H=h, Q=(c+smoothing)/(c+h+2*smoothing))
