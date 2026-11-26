"""Established case-maintenance methods and data-valuation baselines as fixed-K retention policies.

Handoff 2026-11-26 pack p2. Exploratory, our operational choices, not PI decisions. Opt-in: nothing here runs unless a
config names one of ``retention.BASELINE_POLICIES``. Each policy turns a published selection or valuation procedure into
a keep priority over the active cases (``scores.extra[policy]``); ``retention.select_active_set`` then keeps the
protected cases and the cohort floors first and fills the budget K by that priority, exactly as for the other policies.

Shared relations, all under the frozen trained core (CBR_CASE_MAINTENANCE_SURVEY_2026-09-17.md sections 3, 5, 12):

* retrieved(c, q): normalized activation a_c(q) > ``retention.activation_threshold`` (the plan's solver threshold of
  the coverage/reachability candidate; with the pre-top-k mask only the top-k cases have a_c(q) > 0);
* agree(c, q): classification y_c == y_q; regression |y_c - y_q| <= ``retention.solve_tolerance`` (standardized target);
* solves(c, q) = retrieved(c, q) and agree(c, q); Reach(q) = {c : solves(c, q)};
* correct(q | S): the core's prediction for q from the case set S succeeds (argmax == y_q; regression
  |prediction - y_q| <= ``solve_tolerance``). A query with no eligible case in S is not correct.

Reference problems D are the maintenance reference queries (train_loo: every training case with its own stable ID
excluded). Case problems are the active cases themselves as queries, own ID excluded (equal to D under train_loo).

Policies (higher priority = kept earlier):

* ``rc_cnn`` (Smyth and McKenna 1999): RC(c) = sum over q in D with solves(c, q) of 1/|Reach(q)| (original definition,
  survey section 3). CNN in decreasing RC order: a case is added when the condensed set does not answer its own problem
  correctly; passes repeat until one adds nothing. Priority: added cases in addition order, then the others by RC.
* ``crr`` (Delany and Cunningham 2004): Cov(c) = {case problems q : retrieved, agree, correct(q | all)}. Cases in
  increasing |Cov| order; the next remaining case is retained and the cases it covers leave the candidate list.
  Priority: retained cases in retention order, then the others by increasing |Cov|.
* ``bbnr`` (Delany and Cunningham 2004): Liability(c) = {case problems q : retrieved, not agree, not correct(q | all)}.
  Cases with liability > 0 in decreasing liability order are deleted when every q in Cov(c) is still correct without
  them (deletions accumulate). Priority: survivors first, deleted after; inside each group the ``fill_policy`` order.
* ``enn`` (Wilson 1972 editing): a case is edited when its own problem is not correct under all cases (one simultaneous
  pass). Survivors first, edited after; inside each group the ``fill_policy`` order (an existing policy's ranking).
* ``energy_compress`` (Badra et al. 2025, modified): sequential deletion of the eligible case with the lowest removal
  competence L_D(S without c) - L_D(S), recomputed after every deletion with the fixed reference set D, until K cases
  remain (protected cases and cohort floors are never deleted). Classification loss: hinge on the energy 1 - P with
  margin ``energy_margin``; regression (outside the paper): squared error of the prediction. The paper's final
  accuracy-based choice among visited case bases is replaced by the fixed budget K.
* ``knn_shapley`` (Jia et al. 2019, closed form): exact Shapley values of the unweighted K-NN utility
  U(S) = (1/K) sum over the K highest-scoring cases of S of agree(c, q), K = the core's top_k, cases ordered by the
  learned retrieval score, averaged over D. Lowest values are removed first.
* ``loo_value``: leave-one-out value L_D(S without c) - L_D(S) with the final prediction loss (nll / squared error,
  the loss of ``candidates.removal_influence``), one pass on the full memory; lowest values are removed first.

All scores come from one activation engine: the learned scores z = (b - d)/tau of every query against every active
case, computed once through ``model.retrieve``; removing cases masks columns and recomputes the top-k softmax, which is
what ``retrieve(case_mask=...)`` does (checked against ``model.retrieve`` at build time). Only the normalized softmax
path with ``bias_minus_distance``/``neg_distance`` scores, no case sampling and no adapter is supported.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import torch

from .outcomes import active_adapter, frozen_evaluation
from .retention import BASELINE_POLICIES, candidate_rank, protection_requirements, tie_keys

ENGINE_TOLERANCE = 1e-4  # largest allowed |engine prediction - model prediction| on the full memory
TOLERANCE_POLICIES = frozenset({"rc_cnn", "crr", "bbnr", "enn", "knn_shapley"})  # regression needs solve_tolerance
PROBLEM_POLICIES = frozenset({"rc_cnn", "crr", "bbnr", "enn"})  # need the case problems


# --------------------------------------------------------------------------- activation engine


def _check_supported(model, adapter=None):
    mode = model.config.get("case_score_mode", "bias_minus_distance")
    if (not model.normalize_over_cases or model.case_normalizer != "softmax" or model.sampling_cases_flag
            or mode not in {"bias_minus_distance", "neg_distance"}):
        raise ValueError("Maintenance baselines need unsampled normalized softmax retrieval with learned scores")
    if model.nn_cdh is not None or active_adapter(model, adapter) is not None:
        raise ValueError("Maintenance baselines score the retrieval prediction only; the core has an adapter")


class ActivationEngine:
    """Learned retrieval scores of fixed queries against the active cases, with exact case-mask recomputation."""

    def __init__(self, model, X, y, query_case_ids=None, *, batch_size=256):
        _check_supported(model)
        n = model.case_count()
        mode = model.config.get("case_score_mode", "bias_minus_distance")
        tau = max(float(model.tau), 1e-8)
        device = model.cases.device
        active_ids = model.active_case_ids()
        labels = model.labels[:n].float().view(n, -1)
        rows, model_pred = [], []
        with frozen_evaluation(model):
            for start in range(0, len(y), batch_size):
                xb = X[start:start + batch_size].to(device)
                qb = None if query_case_ids is None else query_case_ids[start:start + batch_size].to(device)
                r = model.retrieve(xb, exclude_identical=False, query_case_ids=qb)
                if not torch.equal(r["case_indices"], torch.arange(n, device=device)):
                    raise ValueError("The engine needs full active-case retrieval")
                d = r["distances"].double()
                z = -d if mode == "neg_distance" else model.biases[:n].detach().double().unsqueeze(0) - d
                if qb is not None:
                    z = z.masked_fill(qb.view(-1, 1) == active_ids.view(1, -1), float("-inf"))
                rows.append((z / tau).cpu().numpy())
                model_pred.append((r["weights"] @ labels).double().cpu().numpy())
        self.Z = np.concatenate(rows) if rows else np.zeros((0, n))
        self.n, self.nq = n, self.Z.shape[0]
        self.task = model.task_type
        self.Y = labels.double().cpu().numpy()
        if self.task == "classification":
            self.case_label = self.Y.argmax(1)
            self.y = np.asarray(y.detach().cpu().numpy()).astype(np.int64).reshape(-1)
        else:
            self.case_label = self.Y[:, 0]
            self.y = np.asarray(y.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        self.top_k = int(model.config.get("top_k", 20))
        self.topk = min(self.top_k, n) if model.config.get("pre_topk_mask", False) else None
        full = np.ones(n, dtype=bool)
        P = self.predict(full)
        M = np.concatenate(model_pred) if model_pred else np.zeros_like(P)
        if self.task == "classification":
            M = M / np.clip(M.sum(1, keepdims=True), 1e-12, None)
        self.check_max_abs_diff = float(np.max(np.abs(P - M))) if len(P) else 0.0
        if self.check_max_abs_diff > ENGINE_TOLERANCE:
            raise RuntimeError(f"Activation engine disagrees with model.retrieve by {self.check_max_abs_diff:.3g}")

    @classmethod
    def from_arrays(cls, Z, case_labels, y, *, task_type="classification", top_k=None, num_classes=None):
        """An engine over given scores ``Z`` [queries, cases] (already divided by tau; -inf = excluded), for tests
        and hand-checkable examples. ``top_k`` None means no pre-top-k mask."""
        eng = cls.__new__(cls)
        eng.Z = np.asarray(Z, dtype=float)
        eng.nq, eng.n = eng.Z.shape
        eng.task = task_type
        lab = np.asarray(case_labels)
        if task_type == "classification":
            C = int(num_classes or max(lab.max(), np.max(y)) + 1)
            eng.Y = np.eye(C)[lab.astype(int)]
            eng.case_label = lab.astype(int)
            eng.y = np.asarray(y).astype(np.int64)
        else:
            eng.Y = lab.astype(float).reshape(-1, 1)
            eng.case_label = lab.astype(float)
            eng.y = np.asarray(y, dtype=float)
        eng.top_k = eng.n if top_k is None else int(top_k)
        eng.topk = None if top_k is None else min(int(top_k), eng.n)
        eng.check_max_abs_diff = 0.0
        return eng

    # ----------------------------------------------------------------- neighborhoods

    def _chunk(self, kk):
        return max(1, int(4_000_000 // max(1, kk * self.Y.shape[1])))

    def neighborhoods(self, mask, rows):
        """For ``rows``: members [r, k] (score order, -1 where no eligible case), their scores, next case [r] and its
        score (the case that enters when one member is removed; -1 when none or without the pre-top-k mask)."""
        Z = self.Z[rows]
        if not mask.all():
            Z = np.where(mask[None, :], Z, -np.inf)
        n, r = self.n, len(rows)
        if self.topk is None:
            order = np.lexsort((np.broadcast_to(np.arange(n), Z.shape), -Z), axis=1)
            members = order
            nxt = np.full(r, -1)
        else:
            m = min(self.topk + 1, n)
            part = np.argpartition(-Z, m - 1, axis=1)[:, :m] if m < n else np.broadcast_to(np.arange(n), (r, n)).copy()
            zp = np.take_along_axis(Z, part, 1)
            o = np.lexsort((part, -zp), axis=1)
            part = np.take_along_axis(part, o, 1)
            members = part[:, :self.topk]
            nxt = part[:, self.topk] if m > self.topk else np.full(r, -1)
        zm = np.take_along_axis(Z, members, 1)
        members = np.where(np.isfinite(zm), members, -1)
        znx = np.where(nxt >= 0, Z[np.arange(r), np.maximum(nxt, 0)], -np.inf)
        nxt = np.where(np.isfinite(znx), nxt, -1)
        return members, zm, nxt, znx

    @staticmethod
    def _exp(zm, znx):
        top = np.max(np.where(np.isfinite(zm), zm, -np.inf), axis=1)
        top = np.where(np.isfinite(top), top, 0.0)
        e = np.where(np.isfinite(zm), np.exp(zm - top[:, None]), 0.0)
        e_next = np.where(np.isfinite(znx), np.exp(znx - top), 0.0)
        return e, e_next

    def _empty_prediction(self, shape):
        C = self.Y.shape[1]
        return np.full(shape + (C,), 1.0 / C) if self.task == "classification" else np.zeros(shape + (1,))

    def weights(self, mask, rows):
        """Members and their normalized activations for ``rows`` (zeros where no eligible case)."""
        members, zm, _, _ = self.neighborhoods(mask, rows)
        e, _ = self._exp(zm, np.full(len(rows), -np.inf))
        den = e.sum(1, keepdims=True)
        return members, np.divide(e, den, out=np.zeros_like(e), where=den > 0)

    def predict(self, mask, rows=None):
        rows = np.arange(self.nq) if rows is None else np.asarray(rows)
        out = []
        kk = self.n if self.topk is None else self.topk
        step = self._chunk(kk)
        for s in range(0, len(rows), step):
            members, w = self.weights(mask, rows[s:s + step])
            P = np.einsum("rk,rkc->rc", w, self.Y[np.maximum(members, 0)])
            empty = w.sum(1) <= 0
            if empty.any():
                P[empty] = self._empty_prediction((int(empty.sum()),))
            out.append(P)
        return np.concatenate(out) if out else np.zeros((0, self.Y.shape[1]))

    def success(self, P, rows, tolerance):
        y = self.y[rows]
        if self.task == "classification":
            return P.argmax(-1) == y
        if tolerance is None:
            raise ValueError("Regression baselines need retention.solve_tolerance")
        return np.abs(P[..., 0] - y) <= tolerance

    def agree(self, cases, rows, tolerance):
        """agree(c, q) for case indices ``cases`` [r, k] (or [r]) and query ``rows`` [r]."""
        y = self.y[rows].reshape((-1,) + (1,) * (np.ndim(cases) - 1))
        lab = self.case_label[np.maximum(cases, 0)]
        if self.task == "classification":
            return lab == y
        if tolerance is None:
            raise ValueError("Regression baselines need retention.solve_tolerance")
        return np.abs(lab - y) <= tolerance

    def correct_row(self, row, mask, tolerance):
        rows = np.array([row])
        members, w = self.weights(mask, rows)
        if w.sum() <= 0:
            return False
        P = np.einsum("rk,rkc->rc", w, self.Y[np.maximum(members, 0)])
        return bool(self.success(P, rows, tolerance)[0])

    # ----------------------------------------------------------------- removal losses

    def loss(self, P, rows, kind, margin=None):
        """Per-query loss of predictions ``P`` [r, ..., C] for ``rows``."""
        y = self.y[rows]
        if kind == "squared_error":
            return (P[..., 0] - y.reshape((-1,) + (1,) * (P.ndim - 2))) ** 2
        idx = y.reshape((-1,) + (1,) * (P.ndim - 1))
        true = np.take_along_axis(P, np.broadcast_to(idx, P.shape[:-1] + (1,)), -1)[..., 0]
        if kind == "nll":
            return -np.log(np.clip(true, 1e-8, None))
        if kind == "hinge":
            wrong = P.copy()
            np.put_along_axis(wrong, np.broadcast_to(idx, P.shape[:-1] + (1,)), -np.inf, -1)
            return np.maximum(0.0, float(margin) + wrong.max(-1) - true)
        raise ValueError(kind)

    def removal_rows(self, mask, rows, kind, margin=None):
        """For ``rows``: members [r, k], their removal deltas loss(S without member) - loss(S) [r, k] (0 for absent
        members), next case [r] and the current loss [r]."""
        members, zm, nxt, znx = self.neighborhoods(mask, rows)
        e, e_next = self._exp(zm, znx)
        Ym = self.Y[np.maximum(members, 0)]
        num = np.einsum("rk,rkc->rc", e, Ym)
        den = e.sum(1)
        P = np.divide(num, den[:, None], out=self._empty_prediction((len(rows),)).copy(), where=den[:, None] > 0)
        base = self.loss(P, rows, kind, margin)
        Yn = self.Y[np.maximum(nxt, 0)]
        num_wo = num[:, None, :] - e[:, :, None] * Ym + (e_next[:, None] * Yn)[:, None, :]
        den_wo = den[:, None] - e + e_next[:, None]
        empty = self._empty_prediction(members.shape)
        P_wo = np.divide(num_wo, den_wo[:, :, None], out=empty, where=den_wo[:, :, None] > 1e-300)
        delta = self.loss(P_wo, rows, kind, margin) - base[:, None]
        delta = np.where(members >= 0, delta, 0.0)
        return members, delta, nxt, base

    def removal_values(self, mask, kind, margin=None):
        """Mean over all queries of loss(S without c) - loss(S) for every case (0 for masked cases), and the mean
        current loss."""
        total, base_sum = np.zeros(self.n), 0.0
        kk = self.n if self.topk is None else self.topk
        step = max(1, self._chunk(kk) // max(1, kk))
        for s in range(0, self.nq, step):
            rows = np.arange(s, min(s + step, self.nq))
            members, delta, _, base = self.removal_rows(mask, rows, kind, margin)
            valid = members >= 0
            np.add.at(total, members[valid], delta[valid])
            base_sum += float(base.sum())
        return total / max(self.nq, 1), base_sum / max(self.nq, 1)


# --------------------------------------------------------------------------- relations and scores


def _solver_entries(engine, mask, threshold, tolerance):
    """Rows, cases, activation, agree, correct(row) for every entry with activation > ``threshold``."""
    out_r, out_c, out_w, out_a = [], [], [], []
    correct = np.zeros(engine.nq, dtype=bool)
    kk = engine.n if engine.topk is None else engine.topk
    step = engine._chunk(kk)
    for s in range(0, engine.nq, step):
        rows = np.arange(s, min(s + step, engine.nq))
        members, w = engine.weights(mask, rows)
        P = np.einsum("rk,rkc->rc", w, engine.Y[np.maximum(members, 0)])
        has = w.sum(1) > 0
        correct[rows] = has & engine.success(P, rows, tolerance)
        sel = (members >= 0) & (w > threshold)
        rr = np.broadcast_to(rows[:, None], members.shape)[sel]
        cc = members[sel]
        out_r.append(rr)
        out_c.append(cc)
        out_w.append(w[sel])
        out_a.append(engine.agree(cc, rr, tolerance))
    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)  # noqa: E731
    return cat(out_r, np.int64), cat(out_c, np.int64), cat(out_w, float), cat(out_a, bool), correct


def relative_coverage(engine, threshold, tolerance=None, mask=None):
    """RC(c) = sum over queries q that c solves of 1/|Reach(q)| (Smyth and McKenna 1999, Definition 8)."""
    mask = np.ones(engine.n, dtype=bool) if mask is None else mask
    r, c, _, a, _ = _solver_entries(engine, mask, threshold, tolerance)
    r, c = r[a], c[a]
    reach = np.bincount(r, minlength=engine.nq).astype(float)
    rc = np.zeros(engine.n)
    np.add.at(rc, c, 1.0 / reach[r])
    return rc, reach


def coverage_liability(engine, threshold, tolerance=None):
    """Cov(c) and Liability(c) over the case problems (Delany and Cunningham 2004), as lists of row indices, and
    correct(q | all cases)."""
    mask = np.ones(engine.n, dtype=bool)
    r, c, _, a, correct = _solver_entries(engine, mask, threshold, tolerance)
    cov = [[] for _ in range(engine.n)]
    liab = [[] for _ in range(engine.n)]
    good, bad = a & correct[r], ~a & ~correct[r]
    for q, j in zip(r[good].tolist(), c[good].tolist()):
        cov[j].append(q)
    for q, j in zip(r[bad].tolist(), c[bad].tolist()):
        liab[j].append(q)
    return cov, liab, correct


def knn_shapley(engine, tolerance=None, K=None):
    """Mean exact KNN-Shapley value of every case over the engine's queries (Jia et al. 2019, Theorem 1).

    Per query, cases are ordered by learned score (ties by slot); excluded cases (the query's own ID, masked cases)
    take no part. With alpha_1..alpha_N that order and m_i = agree(alpha_i, q):
    s_N = m_N / N and s_i = s_{i+1} + (m_i - m_{i+1}) / K * min(K, i) / i.
    """
    K = engine.top_k if K is None else int(K)
    n = engine.n
    values = np.zeros(n)
    step = max(1, int(2_000_000 // max(1, n)))
    pos = np.arange(1, n + 1, dtype=float)
    coef = np.minimum(K, pos) / (K * pos)
    for s in range(0, engine.nq, step):
        rows = np.arange(s, min(s + step, engine.nq))
        Z = engine.Z[rows]
        order = np.lexsort((np.broadcast_to(np.arange(n), Z.shape), -Z), axis=1)
        valid = np.isfinite(np.take_along_axis(Z, order, 1))
        N = valid.sum(1)
        M = (engine.agree(order, rows, tolerance) & valid).astype(float)
        t = np.zeros_like(M)
        t[:, :-1] = (M[:, :-1] - M[:, 1:]) * coef[None, :-1]
        t[np.arange(n)[None, :] >= (N[:, None] - 1)] = 0.0
        tail = np.cumsum(t[:, ::-1], axis=1)[:, ::-1]
        last = M[np.arange(len(rows)), np.maximum(N - 1, 0)] / np.maximum(N, 1)
        sv = np.where(valid, last[:, None] + tail, 0.0)
        np.add.at(values, order[valid], sv[valid])
    return values / max(engine.nq, 1)


# --------------------------------------------------------------------------- policies


def _order(values, tie, descending):
    v = -np.asarray(values, dtype=float) if descending else np.asarray(values, dtype=float)
    return np.lexsort((tie, v))


def _priority_from_sequence(first, rest, n):
    """Priority array: the cases ``first`` in order above every case of ``rest`` in order."""
    pr = np.zeros(n)
    for i, j in enumerate(first):
        pr[j] = 2 * n + (n - i)
    for i, j in enumerate(rest):
        pr[j] = n - i
    return pr


def _fill_order(scores, cfg, tie):
    rank = candidate_rank(scores, dataclasses.replace(cfg, policy=cfg.fill_policy))
    return _order(rank, tie, descending=True)


def _grouped(keep, scores, cfg, tie):
    order = _fill_order(scores, cfg, tie)
    return _priority_from_sequence([j for j in order if keep[j]], [j for j in order if not keep[j]], len(keep))


def _problem_engine(model, reference, ref_engine):
    n = model.case_count()
    ids = model.active_case_ids().detach().cpu()
    same = (reference.query_case_ids is not None and reference.query_case_ids.numel() == n
            and torch.equal(reference.query_case_ids.cpu().long(), ids.long())
            and torch.equal(reference.X.cpu(), model.cases[:n].detach().cpu()))
    if same:
        return ref_engine
    labels = model.labels[:n]
    y = labels.argmax(1) if model.task_type == "classification" else labels.view(-1)
    return ActivationEngine(model, model.cases[:n].detach(), y.detach(), ids)


def baseline_priority(model, reference, cfg, scores, K, dist=None):
    """``(extra, summary)``: ``extra[cfg.policy]`` is the keep priority over the active slots (higher = kept earlier),
    plus per-case diagnostics; ``summary`` holds scalars for the maintenance summary."""
    if cfg.policy not in BASELINE_POLICIES:
        raise ValueError(cfg.policy)
    if reference is None:
        raise ValueError(f"{cfg.policy} needs designated maintenance reference queries")
    reference.validate()
    _check_supported(model, reference.adapter)
    t0 = time.perf_counter()
    if model.task_type != "classification" and cfg.policy in TOLERANCE_POLICIES and cfg.solve_tolerance is None:
        raise ValueError(f"Regression {cfg.policy} needs retention.solve_tolerance (standardized target units)")
    if model.task_type == "classification" and cfg.policy == "energy_compress" and cfg.energy_margin is None:
        raise ValueError("Classification energy_compress needs retention.energy_margin")
    ref = ActivationEngine(model, reference.X, reference.y, reference.query_case_ids)
    prob = _problem_engine(model, reference, ref) if cfg.policy in PROBLEM_POLICIES else None
    extra, summary = policy_priority(cfg, scores, K, ref, prob, dist)
    summary.update(engine_check_max_abs_diff=ref.check_max_abs_diff, n_reference_queries=ref.nq,
                   baseline_seconds=time.perf_counter() - t0)
    if prob is not None:
        summary["problem_engine_shared"] = int(prob is ref)
    return extra, summary


def policy_priority(cfg, scores, K, ref, prob=None, dist=None):
    """The policy's keep priority from engines: ``ref`` over the reference problems D, ``prob`` over the case
    problems (row j = active slot j; needed by rc_cnn, crr, bbnr, enn)."""
    n = ref.n
    tie = tie_keys(scores.case_ids, cfg)
    tol = cfg.solve_tolerance
    regression = ref.task != "classification"
    extra, summary = {}, {}
    full = np.ones(n, dtype=bool)
    if cfg.policy == "rc_cnn":
        rc, _ = relative_coverage(ref, cfg.activation_threshold, tol)
        order = _order(rc, tie, descending=True)
        S = np.zeros(n, dtype=bool)
        added, passes, changed = [], 0, True
        while changed:
            changed, passes = False, passes + 1
            for c in order:
                if S[c]:
                    continue
                if not S.any() or not prob.correct_row(int(c), S, tol):
                    S[c] = True
                    added.append(int(c))
                    changed = True
        rest = [int(c) for c in order if not S[c]]
        index = {c: i for i, c in enumerate(added)}
        extra["relative_coverage"] = rc
        extra["cnn_addition_index"] = np.array([index.get(j, -1) for j in range(n)], dtype=float)
        extra[cfg.policy] = _priority_from_sequence(added, rest, n)
        summary.update(baseline_set_size=len(added), cnn_passes=passes)
    elif cfg.policy == "enn":
        P = prob.predict(full)
        has = prob.weights(full, np.arange(prob.nq))[1].sum(1) > 0
        keep = has & prob.success(P, np.arange(prob.nq), tol)
        extra["enn_edited"] = (~keep).astype(float)
        extra[cfg.policy] = _grouped(keep, scores, cfg, tie)
        summary.update(baseline_set_size=int(keep.sum()), enn_edited=int((~keep).sum()))
    elif cfg.policy in {"crr", "bbnr"}:
        cov, liab, _ = coverage_liability(prob, cfg.activation_threshold, tol)
        csize = np.array([len(x) for x in cov], dtype=float)
        lsize = np.array([len(x) for x in liab], dtype=float)
        extra["coverage_size"], extra["liability_size"] = csize, lsize
        if cfg.policy == "crr":
            order = _order(csize, tie, descending=False)
            gone = np.zeros(n, dtype=bool)
            retained = []
            for c in order:
                if gone[c]:
                    continue
                retained.append(int(c))
                gone[c] = True
                gone[cov[c]] = True
            kept = set(retained)
            extra[cfg.policy] = _priority_from_sequence(retained, [int(c) for c in order if c not in kept], n)
            summary.update(baseline_set_size=len(retained))
        else:
            cand = [int(c) for c in _order(lsize, tie, descending=True) if lsize[c] > 0]
            S = full.copy()
            deleted = []
            for c in cand:
                S[c] = False
                if all(prob.correct_row(q, S, tol) for q in cov[c]):
                    deleted.append(c)
                else:
                    S[c] = True
            extra["bbnr_deleted"] = (~S).astype(float)
            extra[cfg.policy] = _grouped(S, scores, cfg, tie)
            summary.update(baseline_set_size=int(S.sum()), bbnr_candidates=len(cand), bbnr_deleted=len(deleted))
    elif cfg.policy == "knn_shapley":
        v = knn_shapley(ref, tol)
        extra["knn_shapley_value"] = v
        extra[cfg.policy] = v.copy()
        summary.update(knn_shapley_K=ref.top_k)
    elif cfg.policy == "loo_value":
        v, base = ref.removal_values(full, "nll" if not regression else "squared_error")
        extra["loo_value"] = v
        extra[cfg.policy] = v.copy()
        summary.update(reference_loss=base)
    elif cfg.policy == "energy_compress":
        extra_e, summ = _energy_compress(ref, cfg, scores, K, dist, tie, regression)
        extra.update(extra_e)
        summary.update(summ)
    else:
        raise ValueError(cfg.policy)
    return extra, summary


def _energy_compress(ref, cfg, scores, K, dist, tie, regression):
    if regression:
        kind, margin = "squared_error", None
    else:
        if cfg.energy_margin is None:
            raise ValueError("Classification energy_compress needs retention.energy_margin")
        kind, margin = "hinge", cfg.energy_margin
    n = ref.n
    protected, floors, _ = protection_requirements(scores, cfg, dist)
    cohorts = np.asarray(scores.cohorts)
    names = sorted(floors)
    coh = np.array([names.index(c) for c in cohorts])
    floor = np.array([floors[c] for c in names])
    count = np.bincount(coh, minlength=len(names))
    active = np.ones(n, dtype=bool)
    rows = np.arange(ref.nq)
    members, delta, nxt, base = ref.removal_rows(active, rows, kind, margin)
    total = np.zeros(n)
    valid = members >= 0
    np.add.at(total, members[valid], delta[valid])
    initial_loss = float(base.mean()) if ref.nq else 0.0
    step_of = np.full(n, -1.0)
    score_at = np.full(n, np.nan)
    removed = []
    stop = "capacity_reached"
    target = min(int(K), n)
    while active.sum() > target:
        eligible = active & ~protected & (count[coh] > floor[coh])
        if not eligible.any():
            stop = "protection_limit"
            break
        comp = total / max(ref.nq, 1)
        pool = np.flatnonzero(eligible)
        j = int(pool[np.lexsort((tie[pool], comp[pool]))[0]])
        step_of[j], score_at[j] = len(removed), comp[j]
        removed.append(j)
        active[j] = False
        count[coh[j]] -= 1
        if ref.topk is None:
            aff = rows
        else:
            aff = np.flatnonzero((members == j).any(1) | (nxt == j))
        if len(aff):
            old_m, old_d = members[aff], delta[aff]
            ok = old_m >= 0
            np.add.at(total, old_m[ok], -old_d[ok])
            m2, d2, n2, b2 = ref.removal_rows(active, aff, kind, margin)
            members[aff], delta[aff], nxt[aff], base[aff] = m2, d2, n2, b2
            ok = m2 >= 0
            np.add.at(total, m2[ok], d2[ok])
        total[j] = 0.0
    comp = total / max(ref.nq, 1)
    survivors = [int(j) for j in np.lexsort((tie, -comp)) if active[j]]
    pr = _priority_from_sequence(survivors, removed[::-1], n)
    extra = {"energy_removal_step": step_of, "energy_competence_at_removal": np.where(np.isnan(score_at), comp, score_at),
             cfg.policy: pr}
    summary = {"energy_steps": len(removed), "energy_initial_reference_loss": initial_loss,
               "energy_final_reference_loss": float(base.mean()) if ref.nq else 0.0, "energy_stop_reason": stop,
               "energy_loss": kind, "baseline_set_size": int(active.sum())}
    if margin is not None:
        summary["energy_margin"] = float(margin)
    return extra, summary
