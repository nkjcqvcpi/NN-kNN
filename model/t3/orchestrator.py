"""Bounded host-directed prompt retrieval, without a hidden query planner."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import time

from .retrieval import Access, Request


@dataclass(frozen=True)
class Budget:
    max_rounds: int = 3
    max_cases: int = 6
    max_evidence_chars: int = 8192
    max_seconds: float = 60.0
    max_cases_per_round: int = 3

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
               for v in (self.max_rounds, self.max_cases, self.max_evidence_chars, self.max_cases_per_round)):
            raise ValueError("positive integer budgets are required")
        if not 0 < self.max_seconds < float("inf"):
            raise ValueError("a finite positive wall-clock budget is required")


def run_loop(task: str, host, retriever, access: Access, budget: Budget, *, now: float, clock=time.monotonic, event_sink=None):
    """Host returns ready/answer or need/requested_types/observable_task_state.

    The callback must enforce its own generation/token timeout. Wall-clock
    limits here are checked at call boundaries, not by interrupting a provider.
    Evidence transport is measured as serialized Unicode characters, not tokens.
    An optional sink records each retrieval before the next host call,including
    withheld events. Host failures propagate after completed events are saved.
    """
    start = clock()
    events, evidence, needs, decisions = [], [], [], []
    used_chars = 0
    reason, answer = "round_budget", None
    for _ in range(budget.max_rounds + 1):
        if clock()-start >= budget.max_seconds:
            reason = "wall_clock_budget"
            break
        decision = host(task=task, evidence=copy.deepcopy(evidence))
        if not isinstance(decision, dict):
            raise ValueError("host must emit an explicit structured decision")
        if type(decision.get("ready")) is not bool:
            raise ValueError("host must explicitly emit boolean ready")
        if clock()-start >= budget.max_seconds:
            reason = "wall_clock_budget"
            break
        if decision.get("ready") is True:
            answer = decision.get("answer")
            if not isinstance(answer, str):
                raise ValueError("answer-ready requires a public answer")
            decisions.append(dict(ready=True, answer=answer))
            reason = "host_ready"
            break
        if len(events) >= budget.max_rounds:
            reason = "round_budget"
            break
        remaining = budget.max_cases-len(evidence)
        if remaining <= 0:
            reason = "case_budget"
            break
        request = Request(need=decision["need"], requested_types=tuple(decision["requested_types"]),
            observable_task_state=decision.get("observable_task_state", ""),
            already_retrieved_ids=tuple(c["case_id"] for c in evidence), max_cases=min(remaining, budget.max_cases_per_round))
        decisions.append(dict(ready=False, request=request.__dict__))
        words = frozenset(request.need.lower().split())
        if any(len(words & old)/max(1, len(words | old)) >= .9 for old in needs):
            reason = "duplicate_need"
            break
        needs.append(words)
        event = retriever.retrieve(request, access, now=now+(clock()-start))
        events.append(event)
        payload = event["evidence"]
        if not payload:
            event["delivered_to_host"] = False
            if event_sink is not None:event_sink(copy.deepcopy(event))
            reason = "no_novel_eligible_case"
            break
        # Preserve originals. An oversized set is withheld rather than silently truncated.
        payload_chars = len(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        if used_chars+payload_chars > budget.max_evidence_chars:
            reason = "context_budget"
            event["delivered_to_host"] = False
            if event_sink is not None:event_sink(copy.deepcopy(event))
            break
        event["delivered_to_host"] = True
        if event_sink is not None:event_sink(copy.deepcopy(event))
        used_chars += payload_chars
        evidence.extend(payload)
    return dict(answer=answer, stop_reason=reason, evidence=evidence, events=events,
        host_decisions=decisions, retrieval_rounds=len(events), evidence_chars=used_chars,
        elapsed_seconds=clock()-start, budgets=budget.__dict__,
        timing_enforcement="between_calls; host callback must enforce generation limits")
