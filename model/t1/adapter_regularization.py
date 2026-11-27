"""Dropout for the active T1 residual MLP, with unchanged parameter keys."""
from __future__ import annotations

import math
import torch
from torch import nn


def validate_dropout_placement(placement, design):
    if placement not in (None, "all_hidden"):
        raise ValueError("dropout_placement must be None (legacy first hidden) or 'all_hidden'")
    if placement is not None and design not in {"residual", "query_aware", "query_aware_no_prediction"}:
        raise ValueError("all-hidden dropout is supported for the three residual adapters")


def add_hidden_dropout(adapter, p: float, *, all_hidden: bool = True):
    """Wrap hidden ReLUs; no parameter/RNG change, and p=0 is an exact no-op.

    Construct classification adapters with dropout=0 before applying this helper.
    Wrapping the parameterless activation retains every Linear/LayerNorm state
    key, so rates share identical parameter shapes and can load matched weights.
    The retained regression pair network is never modified.
    """
    if not math.isfinite(p) or not 0 <= p < 1:
        raise ValueError("dropout must be finite and in [0, 1)")
    if p == 0:
        return adapter
    if getattr(adapter, "adapter_design", "residual") not in {"residual", "query_aware", "query_aware_no_prediction"}:
        raise ValueError("hidden dropout expects a residual adapter")
    network = adapter.net if hasattr(adapter, "net") else adapter.adapt_net_final.model
    if any(isinstance(layer, nn.Dropout) for layer in network.modules()):
        raise ValueError("construct the base adapter without dropout before adding hidden dropout")
    indices = [i for i, layer in enumerate(network) if isinstance(layer, nn.ReLU)]
    if len(indices) != 2:
        raise ValueError("expected the two-hidden-layer residual MLP")
    for index in indices if all_hidden else indices[:1]:
        activation = network[index]
        wrapped = nn.Sequential(activation, nn.Dropout(p))
        wrapped.train(activation.training)
        network[index] = wrapped
    return adapter


def dropout_description(adapter):
    return [dict(module=name, p=float(layer.p)) for name, layer in adapter.named_modules()
            if isinstance(layer, nn.Dropout)]
