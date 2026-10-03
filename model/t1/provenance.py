"""Stable-ID outcome evidence (PI September 20 design).

C/H share each final query outcome by normalized activation; stored-label
agreement and retrieval-only removal remain historical diagnostic baselines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np
import torch


@dataclass
class CaseStats:
    case_id: int
    case_role: str = "classification"
    cohort: str | None = None
    initial_bias: float | None = None
    current_bias: float | None = None
    retrieval_count: float = 0.0  # R_i
    activation_mass: float = 0.0  # A_i
    positive_support: float = 0.0  # C_i
    harmful_support: float = 0.0  # H_i
    cf_pos: float = 0.0  # positive mean final-loss influence (optional)
    cf_neg: float = 0.0
    error_support: float = 0.0  # activation on unsuccessful final query outcomes
    provenance_semantics: str = "final_query_outcome_activation_weighted/v2"
    audits: int = 0
    protected: bool = False
    protected_reason: str | None = None
    insertion_step: int = 0
    last_retrieved_step: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def smoothed_quality(C: float, H: float, s: float) -> float:
    if s <= 0:
        raise ValueError("Smoothing s must be > 0 (plan section 8.2).")
    return (C + s) / (C + H + 2.0 * s)


def within_cohort_percentile(values: np.ndarray, cohorts: list[Any]) -> np.ndarray:
    """Mid-rank percentile of ``values`` inside each cohort, mapped to (0, 1).

    Ties receive the same (average) rank, so a cohort whose biases are all equal
    maps to 0.5 everywhere: equal biases carry no evidence either way.
    A singleton cohort maps to 0.5.
    """
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, 0.5, dtype=np.float64)
    cohort_arr = np.asarray([str(c) for c in cohorts])
    for c in np.unique(cohort_arr):
        idx = np.nonzero(cohort_arr == c)[0]
        n = idx.size
        if n <= 1:
            continue
        v = values[idx]
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        # average ranks over ties
        uniq, inv = np.unique(v, return_inverse=True)
        sums = np.bincount(inv, weights=ranks)
        cnts = np.bincount(inv)
        mid = sums[inv] / cnts[inv]
        out[idx] = (mid - 0.5) / n
    return out


def median_mad_score(values: np.ndarray, cohorts: list[Any]) -> np.ndarray:
    """Ablation bias normalization: logistic of the within-cohort robust z-score."""
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, 0.5, dtype=np.float64)
    cohort_arr = np.asarray([str(c) for c in cohorts])
    for c in np.unique(cohort_arr):
        idx = np.nonzero(cohort_arr == c)[0]
        v = values[idx]
        med = np.median(v)
        mad = np.median(np.abs(v - med)) * 1.4826
        if mad <= 1e-12:
            continue
        out[idx] = 1.0 / (1.0 + np.exp(-(v - med) / mad))
    return out


class CaseStatisticsStore:
    """Stable-ID keyed ledger of per-case provenance."""

    def __init__(self, case_role: str = "classification") -> None:
        self.case_role = case_role
        self._stats: dict[int, CaseStats] = {}

    def __contains__(self, case_id: int) -> bool:
        return int(case_id) in self._stats

    def __len__(self) -> int:
        return len(self._stats)

    def get(self, case_id: int) -> CaseStats:
        return self._stats[int(case_id)]

    def ensure(self, case_id: int, **init: Any) -> CaseStats:
        cid = int(case_id)
        st = self._stats.get(cid)
        if st is None:
            st = CaseStats(case_id=cid, case_role=self.case_role, **init)
            self._stats[cid] = st
        return st

    def items(self) -> Iterable[tuple[int, CaseStats]]:
        return self._stats.items()

    def reset_counts(self, case_ids: Iterable[int] | None = None) -> None:
        ids = list(self._stats) if case_ids is None else [int(c) for c in case_ids]
        for cid in ids:
            st = self._stats[cid]
            st.retrieval_count = st.activation_mass = 0.0
            st.positive_support = st.harmful_support = 0.0
            st.cf_pos = st.cf_neg = st.error_support = 0.0

    def state_dict(self) -> dict[str, Any]:
        return {"case_role": self.case_role, "stats": {cid: st.to_dict() for cid, st in self._stats.items()}}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "CaseStatisticsStore":
        store = cls(state.get("case_role", "classification"))
        for cid, d in state["stats"].items():
            store._stats[int(cid)] = CaseStats(**d)
        return store


# ---------------------------------------------------------------------------
# Standalone Q and B snapshots and exposure evidence
# ---------------------------------------------------------------------------


@dataclass
class ScoreConfig:
    smoothing: float  # s
    min_retrieval_count: float
    min_activation_mass: float
    bias_normalization: str = "within_cohort_percentile"  # | "median_mad"
    bias_source: str = "current"  # | "delta_from_init"
    eps: float = 1e-6


@dataclass
class CaseScores:
    case_ids: np.ndarray
    cohorts: list[str]
    R: np.ndarray
    A: np.ndarray
    C: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    B: np.ndarray
    evidenced: np.ndarray
    bias: np.ndarray
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    def row(self, i: int) -> dict[str, Any]:
        d = {
            "case_id": int(self.case_ids[i]),
            "cohort": self.cohorts[i],
            "R": float(self.R[i]),
            "A": float(self.A[i]),
            "C": float(self.C[i]),
            "H": float(self.H[i]),
            "Q": float(self.Q[i]),
            "B": float(self.B[i]),
            "evidenced": bool(self.evidenced[i]),
            "bias": float(self.bias[i]),
        }
        for k, v in self.extra.items():
            d[k] = v[i].item() if hasattr(v[i], "item") else v[i]
        return d


def score_cases(
    case_ids: np.ndarray,
    cohorts: list[str],
    biases: np.ndarray,
    store: CaseStatisticsStore,
    cfg: ScoreConfig,
) -> CaseScores:
    if cfg.smoothing <= 0:
        raise ValueError("Smoothing must be positive")
    n = len(case_ids)
    R = np.zeros(n)
    A = np.zeros(n)
    C = np.zeros(n)
    H = np.zeros(n)
    init_b = np.zeros(n)
    for j, cid in enumerate(case_ids):
        st = store.ensure(int(cid), cohort=cohorts[j], initial_bias=float(biases[j]), current_bias=float(biases[j]))
        st.current_bias = float(biases[j])
        R[j], A[j], C[j], H[j] = st.retrieval_count, st.activation_mass, st.positive_support, st.harmful_support
        init_b[j] = st.initial_bias if st.initial_bias is not None else float(biases[j])
    Q = (C + cfg.smoothing) / (C + H + 2.0 * cfg.smoothing)
    b_src = biases - init_b if cfg.bias_source == "delta_from_init" else biases
    if cfg.bias_normalization == "within_cohort_percentile":
        B = within_cohort_percentile(b_src, cohorts)
    elif cfg.bias_normalization == "median_mad":
        B = median_mad_score(b_src, cohorts)
    else:
        raise ValueError(cfg.bias_normalization)
    evidenced = (R >= cfg.min_retrieval_count) & (A >= cfg.min_activation_mass)
    return CaseScores(np.asarray(case_ids), list(cohorts), R, A, C, H, Q, B, evidenced, np.asarray(biases, dtype=float))


# ---------------------------------------------------------------------------
# Audits: accumulate activation-weighted final query outcomes
# ---------------------------------------------------------------------------


def _cohort_of_label(label_row: torch.Tensor, task_type: str, reg_bins: np.ndarray | None) -> str:
    if task_type == "classification":
        return f"class_{int(torch.argmax(label_row).item())}"
    v = float(label_row.view(-1)[0].item())
    if reg_bins is None:
        return "all"
    return f"bin_{int(np.searchsorted(reg_bins, v, side='right'))}"


def active_cohorts(model, reg_bins: np.ndarray | None = None) -> list[str]:
    n = model.case_count()
    return [_cohort_of_label(model.labels[i], model.task_type, reg_bins) for i in range(n)]


@torch.no_grad()
def audit_provenance(
    model, X_audit, y_audit, store, *, exclude_identical,
    retrieval_eps=1e-6, counterfactual=True, batch_size=256, step=0,
    reg_bins=None, query_case_ids=None, adapter=None,
    regression_success_tolerance=None,
):
    """Evidence from final outcomes, with pre-adaptation loss kept for diagnosis.

    Identity-aware LOO should be supplied for case queries. A regression
    success criterion is required explicitly; counterfactual influence stays
    separate from C/H. No model parameters, gradient flags, or RNG are changed.
    """
    from .outcomes import final_prediction, frozen_evaluation, prediction_loss, query_success
    from .candidates import MaintenanceReference, removal_influence

    device, task = model.cases.device, model.task_type
    n_active = model.case_count()
    ids = model.active_case_ids().detach().cpu().numpy()
    cohorts = active_cohorts(model, reg_bins)
    biases = model.biases[:n_active].detach().cpu().numpy()
    if len(y_audit) == 0:
        raise ValueError("Audit reference must be nonempty")
    loss_pre = loss_final = 0.0
    with frozen_evaluation(model, adapter):
        for j, cid in enumerate(ids):
            st = store.ensure(int(cid), cohort=cohorts[j], initial_bias=float(biases[j]), current_bias=float(biases[j]))
            st.cohort, st.current_bias = cohorts[j], float(biases[j])
        for start in range(0, len(y_audit), batch_size):
            xb = X_audit[start:start + batch_size].to(device)
            yb = y_audit[start:start + batch_size].to(device)
            qids = None if query_case_ids is None else query_case_ids[start:start + batch_size].to(device)
            excl = exclude_identical if qids is None else False
            r = model.retrieve(xb, exclude_identical=excl, query_case_ids=qids)
            w = r["weights"]
            if not torch.equal(r["case_indices"], torch.arange(n_active, device=device)):
                raise ValueError("Audit requires full active-case retrieval, without case sampling")
            pre, final, kind = final_prediction(model, xb, r, adapter=adapter, query_case_ids=qids, exclude_identical=excl)
            success = query_success(final, yb, task, regression_success_tolerance)
            C = (w * success[:, None]).sum(0)
            H = (w * (~success)[:, None]).sum(0)
            R, A = (w > retrieval_eps).sum(0), w.sum(0)
            loss_pre += float(prediction_loss(pre, yb, "nll" if task == "classification" else "squared_error").sum())
            loss_final += float(prediction_loss(final, yb, kind).sum())
            for j, cid in enumerate(ids):
                st = store.get(int(cid))
                st.retrieval_count += float(R[j])
                st.activation_mass += float(A[j])
                st.positive_support += float(C[j])
                st.harmful_support += float(H[j])
                st.error_support += float(H[j])
                if R[j] > 0:
                    st.last_retrieved_step = step
        if counterfactual:
            if query_case_ids is not None or not exclude_identical:
                ref = MaintenanceReference(X_audit, y_audit, query_case_ids, regression_success_tolerance, adapter,
                                           "train_loo" if query_case_ids is not None else "maintenance_split")
                cf = removal_influence(model, ref, batch_size=batch_size)
                # Signed mean final-loss influence, not C/H outcome credit.
                for j, cid in enumerate(ids):
                    st = store.get(int(cid))
                    st.cf_pos += max(float(cf["full"][j]), 0.0)
                    st.cf_neg += max(-float(cf["full"][j]), 0.0)
            else:
                # Compatibility for older direct callers without IDs. Retrieve-only
                # closed form cannot audit adapted paths safely.
                from .outcomes import active_adapter
                if active_adapter(model, adapter) is not None or model.nn_cdh is not None:
                    raise ValueError("Adapted counterfactual audit requires stable query IDs")
                for start in range(0, len(y_audit), batch_size):
                    xb = X_audit[start:start + batch_size].to(device)
                    yb = y_audit[start:start + batch_size].to(device)
                    r = model.retrieve(xb, exclude_identical=True)
                    delta = _counterfactual_delta(model, r, yb, model.labels[:n_active].float())
                    for j, cid in enumerate(ids):
                        st = store.get(int(cid))
                        st.cf_pos += float(delta[:, j].clamp_min(0).sum()) / len(y_audit)
                        st.cf_neg += float((-delta[:, j]).clamp_min(0).sum()) / len(y_audit)
        for cid in ids:
            store.get(int(cid)).audits += 1
    return {"n_queries": len(y_audit), "mean_loss_pre": loss_pre / len(y_audit),
            "mean_loss_final": loss_final / len(y_audit), "n_active": n_active,
            "provenance_semantics": "final_query_outcome_activation_weighted/v2",
            "regression_success_tolerance": regression_success_tolerance}


def _counterfactual_delta(model, r: dict[str, torch.Tensor], yb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Loss(f_without_i) - Loss(f_with_i) per query (rows) and active case (cols).

    Exact closed form for ``case_normalizer='softmax'`` with score modes
    ``bias_minus_distance`` / ``neg_distance``, with or without the
    pre-normalization top-k mask: removing a case in the top-k set lets the
    best excluded-from-set, non-excluded case enter, and the softmax is
    renormalized over the new set. Cases outside the set have delta 0.
    """
    w = r["weights"]
    B, N = w.shape
    mode = model.config.get("case_score_mode", "bias_minus_distance")
    if model.case_normalizer != "softmax" or mode not in {"bias_minus_distance", "neg_distance"}:
        raise NotImplementedError("Counterfactual audit is implemented for softmax activations only.")
    distances = r["distances"]
    z = -distances if mode == "neg_distance" else model.biases[r["case_indices"]].unsqueeze(0) - distances
    z = z / float(model.tau)
    excluded = r.get("excluded")
    zz = z.masked_fill(excluded, float("-inf")) if excluded is not None else z
    in_set = w > 0
    zmax = zz.masked_fill(~in_set, float("-inf")).amax(1, keepdim=True)
    zmax = torch.where(torch.isfinite(zmax), zmax, torch.zeros_like(zmax))
    e = torch.exp(zz - zmax).masked_fill(~in_set, 0.0)  # [B, N]
    denom = e.sum(1)  # [B]
    topk = bool(model.config.get("pre_topk_mask", False)) and int(model.config.get("top_k", N)) < N
    if topk:
        zz_out = zz.masked_fill(in_set, float("-inf"))
        next_val, next_idx = zz_out.max(1)
        e_next = torch.where(torch.isfinite(next_val), torch.exp(next_val - zmax.squeeze(1)), torch.zeros_like(next_val))
    else:
        next_idx = torch.zeros(B, dtype=torch.long, device=w.device)
        e_next = torch.zeros(B, device=w.device)
    Y = labels.view(N, -1).float()  # [N, C] one-hot (classification) or [N, 1] values
    num_base = (w @ Y) * denom.unsqueeze(1)  # [B, C] = sum_set e_j y_j
    y_next = Y[next_idx]  # [B, C]
    num = num_base.unsqueeze(1) - e.unsqueeze(2) * Y.unsqueeze(0) + (e_next.unsqueeze(1) * y_next).unsqueeze(1)  # [B, N, C]
    den = (denom.unsqueeze(1) - e + e_next.unsqueeze(1)).clamp_min(1e-30)  # [B, N]
    f_wo = num / den.unsqueeze(2)
    f_w = (w @ Y)
    if model.task_type == "classification":
        y_idx = yb.long().view(-1)
        p_wo = f_wo.gather(2, y_idx.view(B, 1, 1).expand(B, N, 1)).squeeze(2)
        p_w = f_w.gather(1, y_idx.view(B, 1))
        delta = -torch.log(p_wo.clamp_min(1e-8)) + torch.log(p_w.clamp_min(1e-8))
    else:
        t = yb.float().view(B, 1)
        delta = (f_wo[:, :, 0] - t) ** 2 - (f_w[:, 0:1] - t) ** 2
    # a case that is the only member of the set cannot be removed meaningfully without a replacement
    return torch.where(in_set, delta, torch.zeros_like(delta))
