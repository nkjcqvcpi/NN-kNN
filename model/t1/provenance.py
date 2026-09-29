"""Case provenance statistics and trustworthiness (T1 plan sections 8-9).

Everything here is keyed by the stable ``case_id`` stored on ``NN_KNN_Model``,
never by tensor slot, so statistics survive compaction, archive and restore.

Definitions (plan section 8):

    R_i = sum_x 1[i is retrieved for x]
    A_i = sum_x a_i(x)
    C_i = sum_x a_i(x) * 1[c_i = y_x]          (classification)
    H_i = sum_x a_i(x) * 1[c_i != y_x]         (classification)
    Q_i = (C_i + s) / (C_i + H_i + 2 s)
    B_i = within-cohort percentile of the trained case bias, in [0, 1]
    T_i = Q_i^alpha * B_i^(1 - alpha)          (computed in log space)

For regression/value cases (section 9.1), C_i / H_i come from counterfactual
removal of case i with activation renormalization on the pre-adaptation output:

    Delta_i(x) = Loss(f_without_i(x), y_x) - Loss(f_with_i(x), y_x)
    C_i = sum_x max(Delta_i(x), 0);  H_i = sum_x max(-Delta_i(x), 0)

The counterfactual delta is also available for classification (NLL of the
retrieval-only class mass) and stored separately (``cf_pos`` / ``cf_neg``).
"""

from __future__ import annotations

import math
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
    cf_pos: float = 0.0  # classification-only counterfactual support (optional)
    cf_neg: float = 0.0
    error_support: float = 0.0  # activation on queries the model gets wrong while backing the wrong class
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


def trustworthiness(Q: float, B: float, alpha: float, mode: str = "geometric", eps: float = 1e-6) -> float:
    """T_i. Geometric (primary) is computed in log space; arithmetic is an ablation."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    if mode == "geometric":
        q = min(max(Q, eps), 1.0)
        b = min(max(B, eps), 1.0)
        return float(math.exp(alpha * math.log(q) + (1.0 - alpha) * math.log(b)))
    if mode == "arithmetic":
        return float(alpha * Q + (1.0 - alpha) * B)
    raise ValueError(f"Unknown trust mode {mode!r}")


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
# Scoring snapshot: Q, B, T and evidence for the active case set
# ---------------------------------------------------------------------------


@dataclass
class ScoreConfig:
    smoothing: float  # s
    alpha: float  # trust weight
    min_retrieval_count: float
    min_activation_mass: float
    bias_normalization: str = "within_cohort_percentile"  # | "median_mad"
    bias_source: str = "current"  # | "delta_from_init"
    trust_mode: str = "geometric"  # | "arithmetic"
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
    T: np.ndarray
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
            "T": float(self.T[i]),
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
    T = np.array([trustworthiness(Q[j], B[j], cfg.alpha, cfg.trust_mode, cfg.eps) for j in range(n)])
    evidenced = (R >= cfg.min_retrieval_count) & (A >= cfg.min_activation_mass)
    return CaseScores(np.asarray(case_ids), list(cohorts), R, A, C, H, Q, B, T, evidenced, np.asarray(biases, dtype=float))


# ---------------------------------------------------------------------------
# Audits: accumulate statistics from pre-adaptation retrieval
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
    model,
    X_audit: torch.Tensor,
    y_audit: torch.Tensor,
    store: CaseStatisticsStore,
    *,
    exclude_identical: bool,
    retrieval_eps: float = 1e-6,
    counterfactual: bool = True,
    batch_size: int = 256,
    step: int = 0,
    reg_bins: np.ndarray | None = None,
) -> dict[str, Any]:
    """One audit pass over a designated audit set with the *pre-adaptation* retrieval.

    ``exclude_identical`` must be True when the audit queries are the training
    cases themselves (leave-one-out); otherwise every case retrieves itself at
    distance zero and C_i is inflated.

    Counterfactual deltas use the closed form for softmax/top-k normalization:
    removing case i and renormalizing the remaining activations. With a
    pre-normalization top-k mask the (k+1)-th case enters; this is handled
    exactly by recomputing the normalizer over the shifted top-k.
    """
    was_training = model.training
    model.eval()
    device = model.cases.device
    task = model.task_type
    n_active = model.case_count()
    ids = model.active_case_ids().detach().cpu().numpy()
    cohorts = active_cohorts(model, reg_bins)
    biases = model.biases[:n_active].detach().cpu().numpy()
    for j, cid in enumerate(ids):
        st = store.ensure(int(cid), cohort=cohorts[j], initial_bias=float(biases[j]), current_bias=float(biases[j]))
        st.cohort = cohorts[j]
        st.current_bias = float(biases[j])

    labels = model.labels[:n_active].float()
    if task == "classification":
        case_cls = labels.argmax(dim=1)
    n_queries = 0
    loss_sum = 0.0
    for start in range(0, X_audit.size(0), batch_size):
        xb = X_audit[start : start + batch_size].to(device)
        yb = y_audit[start : start + batch_size].to(device)
        r = model.retrieve(xb, exclude_identical=exclude_identical)
        w = r["weights"]  # [B, N_sel]
        idx = r["case_indices"]
        if idx.numel() != n_active or not torch.equal(idx, torch.arange(n_active, device=idx.device)):
            raise RuntimeError("Provenance audit requires full retrieval over active cases (disable case sampling).")
        retrieved = (w > retrieval_eps).float()
        R_add = retrieved.sum(0)
        A_add = w.sum(0)
        if task == "classification":
            y_idx = yb.long().view(-1)
            match = (case_cls.unsqueeze(0) == y_idx.unsqueeze(1)).float()  # [B, N]
            C_add = (w * match).sum(0)
            H_add = (w * (1.0 - match)).sum(0)
            p0 = w @ labels
            pred = p0.argmax(1)
            wrong = (pred != y_idx).float().unsqueeze(1)
            backs_pred = (case_cls.unsqueeze(0) == pred.unsqueeze(1)).float()
            err_add = (w * wrong * backs_pred).sum(0)
            loss_sum += float(-torch.log(p0.gather(1, y_idx.view(-1, 1)).clamp_min(1e-8)).sum().item())
        else:
            err_add = torch.zeros(n_active, device=device)
            y_case = labels.view(n_active, -1)[:, 0]
            f = w @ y_case
            loss_sum += float(((f - yb.float().view(-1)) ** 2).sum().item())
        cf_pos = torch.zeros(n_active, device=device)
        cf_neg = torch.zeros(n_active, device=device)
        if counterfactual or task != "classification":
            delta = _counterfactual_delta(model, r, yb, labels)  # [B, N] loss_without - loss_with
            cf_pos = delta.clamp_min(0).sum(0)
            cf_neg = (-delta).clamp_min(0).sum(0)
        if task != "classification":
            C_add, H_add = cf_pos, cf_neg
        R_add, A_add, C_add, H_add = (t.cpu().numpy() for t in (R_add, A_add, C_add, H_add))
        cf_pos_np, cf_neg_np, err_np = cf_pos.cpu().numpy(), cf_neg.cpu().numpy(), err_add.cpu().numpy()
        for j, cid in enumerate(ids):
            st = store.get(int(cid))
            st.retrieval_count += float(R_add[j])
            st.activation_mass += float(A_add[j])
            st.positive_support += float(C_add[j])
            st.harmful_support += float(H_add[j])
            st.cf_pos += float(cf_pos_np[j])
            st.cf_neg += float(cf_neg_np[j])
            st.error_support += float(err_np[j])
            if R_add[j] > 0:
                st.last_retrieved_step = step
        n_queries += int(xb.size(0))
    for cid in ids:
        store.get(int(cid)).audits += 1
    if was_training:
        model.train()
    return {"n_queries": n_queries, "mean_loss_pre": loss_sum / max(n_queries, 1), "n_active": n_active}


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
