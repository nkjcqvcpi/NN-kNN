"""MCB stability measurements (plan section 3.5 / Phase 5).

MCB itself lives in the core: ``NN_KNN_Model.momentum_encoder`` (no-grad EMA of
the online encoder) represents stored cases; ``CoreConfig.mcb_enabled`` switches
it. This tracker measures benefit *and* over-stabilization, matched by case_id:

* representation drift  - mean L2 change of stored-case embeddings between checkpoints
* neighborhood churn     - 1 - mean Jaccard of top-k retrieved case_id sets on fixed probes
* selection churn        - 1 - Jaccard of kept case_id sets between maintenance events
"""

from __future__ import annotations

from typing import Any

import torch

from .geometry import active_case_features


class StabilityTracker:
    def __init__(self, probes: torch.Tensor, k: int = 5) -> None:
        self.probes = probes
        self.k = k
        self._prev_emb: dict[int, torch.Tensor] | None = None
        self._prev_nbrs: list[set[int]] | None = None
        self._prev_keep: set[int] | None = None
        self.records: list[dict[str, Any]] = []

    @torch.no_grad()
    def snapshot(self, model, step: int, kept_case_ids: list[int] | None = None) -> dict[str, Any]:
        model.eval()
        ids = model.active_case_ids().cpu().tolist()
        emb = active_case_features(model).cpu()
        cur = {cid: emb[i] for i, cid in enumerate(ids)}
        r = model.retrieve(self.probes.to(model.cases.device), exclude_identical=False)
        k = min(self.k, r["weights"].shape[1])
        top = torch.topk(r["weights"], k, dim=1).indices.cpu()
        id_t = torch.tensor(ids)
        nbrs = [set(id_t[row].tolist()) for row in top]
        rec: dict[str, Any] = {"step": step, "n_cases": len(ids)}
        if self._prev_emb is not None:
            common = [c for c in cur if c in self._prev_emb]
            if common:
                a = torch.stack([cur[c] for c in common])
                b = torch.stack([self._prev_emb[c] for c in common])
                rec["representation_drift"] = float((a - b).norm(dim=1).mean())
        if self._prev_nbrs is not None:
            jac = [len(x & y) / max(len(x | y), 1) for x, y in zip(nbrs, self._prev_nbrs)]
            rec["neighborhood_churn"] = 1.0 - sum(jac) / max(len(jac), 1)
        if kept_case_ids is not None:
            keep = set(int(c) for c in kept_case_ids)
            if self._prev_keep is not None:
                rec["selection_churn"] = 1.0 - len(keep & self._prev_keep) / max(len(keep | self._prev_keep), 1)
            self._prev_keep = keep
        self._prev_emb, self._prev_nbrs = cur, nbrs
        self.records.append(rec)
        return rec
