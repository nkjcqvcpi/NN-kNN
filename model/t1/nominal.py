"""Train-fitted nominal groups and stable-ID bindings for aggregate NN-CDH.

Covered fields are declared individually and excluded from the side channel.
The adapter receives only one-hot query-minus-neighborhood differences, never
the original category values or raw query/neighborhood embeddings.
"""
from __future__ import annotations

import json
import math
import copy
from typing import Any

import numpy as np
import torch

from model.nn_cdh import ClassificationNNCDHAdapter


def category_token(value: Any) -> str | None:
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        value = value.item()
    if value is None or isinstance(value, (float, np.floating)) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        kind = "bool"
    elif isinstance(value, int):
        kind = "int"
    elif isinstance(value, float) and math.isfinite(value):
        kind = "float"
    elif isinstance(value, str):
        kind = "str"
    else:
        raise ValueError("nominal categories must be finite scalar strings/numbers/bools or missing")
    return json.dumps([kind, value], ensure_ascii=False, separators=(",", ":"))


class NominalSchema:
    def __init__(self, fields):
        self.fields = [dict(field) for field in fields]
        names = []
        for field in self.fields:
            if set(field) != {"name", "covered_by_representation"}:
                raise ValueError("each nominal field must explicitly declare name and representation coverage")
            if not isinstance(field["name"], str) or not field["name"] or type(field["covered_by_representation"]) is not bool:
                raise ValueError("invalid nominal field name/coverage")
            names.append(field["name"])
        if len(names) != len(set(names)):
            raise ValueError("nominal field names must be unique")
        self.groups = []
        self.dimension = 0
        self.fitted = False

    def fit(self, training_values):
        if self.fitted:
            raise ValueError("a fitted nominal vocabulary cannot be refit on validation/test data")
        if not isinstance(training_values, dict) or any(isinstance(v, (str, bytes)) for v in training_values.values()):
            raise ValueError("nominal training values must be columns of row values")
        if set(training_values) != {f["name"] for f in self.fields}:
            raise ValueError("nominal values and field-by-field coverage declarations must match exactly")
        counts = {len(v) for v in training_values.values()}
        if len(counts) != 1 or not counts or next(iter(counts)) < 1:
            raise ValueError("training nominal columns must be nonempty and aligned")
        self.groups, self.dimension = [], 0
        for field in self.fields:
            if field["covered_by_representation"]:
                continue
            categories = sorted({t for v in training_values[field["name"]] if (t := category_token(v)) is not None})
            group = dict(name=field["name"], categories=categories, offset=self.dimension,
                         missing_index=0, unknown_index=1, width=2 + len(categories))
            self.groups.append(group)
            self.dimension += group["width"]
        self.fitted = True
        return self

    def encode(self, values, *, rows=None):
        if not self.fitted:
            raise ValueError("nominal vocabularies must be fit on training values before encoding")
        if not isinstance(values, dict) or any(isinstance(v, (str, bytes)) for v in values.values()):
            raise ValueError("nominal query values must be columns of row values")
        if set(values) != {f["name"] for f in self.fields}:
            raise ValueError("nominal query fields must match the fitted declaration")
        counts = {len(v) for v in values.values()}
        if len(counts) != 1 or rows is not None and counts != {rows}:
            raise ValueError("nominal query columns must align with queries")
        n = next(iter(counts), rows or 0)
        out = torch.zeros(n, self.dimension)
        for group in self.groups:
            lookup = {value: i + 2 for i, value in enumerate(group["categories"])}
            for row, value in enumerate(values[group["name"]]):
                token = category_token(value)
                index = 0 if token is None else lookup.get(token, 1)
                out[row, group["offset"] + index] = 1.
        return out

    def state_dict(self):
        if not self.fitted:
            raise ValueError("cannot serialize an unfitted nominal vocabulary")
        return copy.deepcopy(dict(version=1, fit_stream="train_only", fields=self.fields,
                    groups=self.groups, dimension=self.dimension))

    @classmethod
    def from_state_dict(cls, state):
        if state.get("version") != 1 or state.get("fit_stream") != "train_only":
            raise ValueError("unsupported nominal vocabulary manifest")
        schema = cls(state["fields"])
        schema.groups = copy.deepcopy(state["groups"])
        offset = 0
        expected = [f["name"] for f in schema.fields if not f["covered_by_representation"]]
        if [g["name"] for g in schema.groups] != expected:
            raise ValueError("nominal group order must match declared uncovered fields")
        for g in schema.groups:
            if g["offset"] != offset or g["width"] != len(g["categories"]) + 2 or g["missing_index"] != 0 or g["unknown_index"] != 1:
                raise ValueError("invalid nominal group layout")
            if len(set(g["categories"])) != len(g["categories"]):
                raise ValueError("duplicate nominal vocabulary categories")
            offset += g["width"]
        if offset != state["dimension"]:
            raise ValueError("nominal dimension differs from preserved field groups")
        schema.dimension, schema.fitted = offset, True
        return schema


class NominalClassificationAdapter(ClassificationNNCDHAdapter):
    """Frozen original-case nominal values bound by stable IDs, not live slots."""
    def __init__(self, *, schema, case_ids, case_values, **kwargs):
        super().__init__(nominal_dim=schema.dimension, **kwargs)
        self.nominal_schema = schema
        ids = torch.as_tensor(case_ids, dtype=torch.long).view(-1)
        values = torch.as_tensor(case_values, dtype=torch.float32)
        if not len(ids) or (ids < 0).any() or ids.unique().numel() != len(ids) or values.shape != (len(ids), schema.dimension) or not torch.isfinite(values).all():
            raise ValueError("nominal case bindings require unique stable IDs and aligned groups")
        for group in schema.groups:
            v = values[:, group["offset"]:group["offset"] + group["width"]]
            if not ((v == 0) | (v == 1)).all() or not (v.sum(1) == 1).all():
                raise ValueError("nominal cases must retain each separate one-hot group")
        order = ids.argsort()
        self.register_buffer("nominal_case_ids", ids[order].clone())
        self.register_buffer("nominal_case_values", values[order].clone())

    def values_for_ids(self, ids):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.nominal_case_ids.device)
        positions = torch.searchsorted(self.nominal_case_ids, ids)
        positions = positions.clamp(max=len(self.nominal_case_ids) - 1)
        if not (self.nominal_case_ids[positions] == ids).all():
            raise ValueError("an active case has no nominal stable-ID binding")
        return self.nominal_case_values[positions]

    def nominal_delta(self, model, retrieval, query_values):
        if query_values is None:
            raise ValueError("uncovered nominal fields require explicit encoded query values")
        w = retrieval["weights"]
        query = torch.as_tensor(query_values, dtype=w.dtype, device=w.device).detach()
        if query.shape != (w.shape[0], self.nominal_dim) or not torch.isfinite(query).all():
            raise ValueError("nominal queries must align with this actual retrieval event")
        for group in self.nominal_schema.groups:
            q = query[:, group["offset"]:group["offset"] + group["width"]]
            if not ((q == 0) | (q == 1)).all() or not (q.sum(1) == 1).all():
                raise ValueError("each nominal field must retain its own one-hot group")
        selected = model.active_case_ids()[retrieval["case_indices"]]
        case = self.values_for_ids(selected).to(w)
        return query - w @ case


def nominal_difference(adapter, model, retrieval, query_values):
    if getattr(adapter, "nominal_dim", 0):
        if not hasattr(adapter, "nominal_delta"):
            raise ValueError("explicit nominal reuse requires fitted stable-ID bindings")
        return adapter.nominal_delta(model, retrieval, query_values)
    if query_values is not None and torch.as_tensor(query_values).shape[-1] != 0:
        raise ValueError("covered nominal fields cannot be duplicated as adapter input")
    return None


def encoded_queries(adapter, data, stream):
    schema = getattr(adapter, "nominal_schema", None)
    if schema is None:
        return None
    values = getattr(data, f"nominal_{stream}", None)
    if values is None:
        raise ValueError(f"nominal adapter needs explicit {stream} values")
    return schema.encode(values, rows=len(getattr(data, f"y_{stream}")))


def fit_nominal_schema(data, fields):
    declared = {field["name"]: field["covered_by_representation"] for field in fields}
    known_coverage = getattr(data, "meta", {}).get("nominal_coverage")
    if known_coverage is not None and declared != known_coverage:
        raise ValueError("field coverage differs from the explicitly documented data representation")
    return NominalSchema(fields).fit(getattr(data, "nominal_train", None))
