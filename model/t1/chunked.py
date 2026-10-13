"""Query-chunked retrieval for large evaluation streams (T1 only).

``model.retrieve`` materializes [queries, cases, features] tensors. On the harder
tasks a whole validation or test stream exceeds RAM, or an accelerator's
single-allocation limit (about 4 GiB on the Arc A770). ``retrieve_chunked`` calls
``model.retrieve`` unchanged while that size fits ``RETRIEVE_BUDGET`` elements, so
every stream under the budget is computed exactly as before. Above it, queries
are retrieved in chunks and the per-query outputs concatenated: they do not
depend on other queries (an encoder may differ in the last bit with batch size).
Chunked results omit the [queries, cases, features] contribution tensor, which no
T1 consumer reads.
"""

from __future__ import annotations

import os

import torch

RETRIEVE_BUDGET = int(os.environ.get("NNKNN_T1_RETRIEVE_BUDGET", 2 ** 27))  # 512 MiB in float32


def chunk_rows(model, n_queries: int, budget: int | None = None) -> int | None:
    """Queries per chunk, or None when one ``model.retrieve`` call fits the budget."""
    budget = RETRIEVE_BUDGET if budget is None else budget
    dim = getattr(model, "feature_dim", None) or model.cases[0].numel()
    per_query = max(1, model.case_count() * int(dim))
    if n_queries * per_query <= budget:
        return None
    return max(1, budget // per_query)


def retrieve_chunked(model, X: torch.Tensor, *, query_case_ids=None, budget: int | None = None, **kwargs) -> dict:
    rows = chunk_rows(model, len(X), budget)
    if rows is None:
        return model.retrieve(X, query_case_ids=query_case_ids, **kwargs)
    parts = [model.retrieve(X[s:s + rows], query_case_ids=None if query_case_ids is None else query_case_ids[s:s + rows],
                            **kwargs) for s in range(0, len(X), rows)]
    first = parts[0]
    return {"case_indices": first["case_indices"], "case_features": first["case_features"],
            "query_features": torch.cat([p["query_features"] for p in parts]),
            "distances": torch.cat([p["distances"] for p in parts]),
            "weights": torch.cat([p["weights"] for p in parts]),
            "excluded": None if first["excluded"] is None else torch.cat([p["excluded"] for p in parts]),
            "feature_distance_contributions": None, "override": first.get("override"), "chunk_rows": rows}
