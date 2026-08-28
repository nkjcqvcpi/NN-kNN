"""Convolutional encoders for image-observation RL tasks.

This module hosts the single Nature-DQN convolutional encoder shared by every
RL workflow in the repo (DQN, NEC, PPO and NN-kNN-RL). Keeping one
implementation here means the ALE image path uses the *same* trunk everywhere,
so cross-method comparisons on `ale_pong` / `ale_breakout` differ only in the
learning rule, not in the encoder.

Architecture (Mnih et al. 2015, "Human-level control through deep
reinforcement learning", Nature):

    Conv2d(C, 32, kernel 8, stride 4) + ReLU
    Conv2d(32, 64, kernel 4, stride 2) + ReLU
    Conv2d(64, 64, kernel 3, stride 1) + ReLU
    Flatten
    Linear(3136, 512) + ReLU

For the canonical 4x84x84 stacked-grayscale input the flattened convolutional
output is 3136, so the encoder emits a 512-dimensional feature vector. The
trailing ReLU is part of the encoder: every consumer attaches its own *linear*
head (Q-values, policy logits, value scalar, NEC embedding) on top of it, which
matches the Nature architecture where the 512-unit hidden layer is the last
shared representation.

Input convention
----------------
`forward` accepts either a single observation shaped like `obs_shape` or a
batch shaped `(N, *obs_shape)`, in any float or integer dtype. Inputs are cast
to float32 and, when `scale_input` is true (the default), divided by 255 inside
the encoder. That keeps raw `uint8` frames valid everywhere in the repo:
replay buffers can store `uint8` (4x memory saving), and the NN-kNN case store
can hold raw `float32` frames on the 0-255 scale without a separate
normalization wrapper. No workflow applies observation normalization wrappers
on top of this.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn


NATURE_CNN_FEATURE_DIM = 512


def orthogonal_layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Orthogonal weight init used by the PPO trunk (CleanRL convention)."""

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureCNNEncoder(nn.Module):
    """Nature-DQN convolutional encoder producing a `feature_dim` vector."""

    def __init__(
        self,
        obs_shape: Sequence[int],
        *,
        feature_dim: int = NATURE_CNN_FEATURE_DIM,
        scale_input: bool = True,
        layer_init: Callable[[nn.Module], nn.Module] | None = None,
    ):
        super().__init__()
        shape = tuple(int(value) for value in obs_shape)
        if len(shape) != 3:
            raise ValueError(
                f"NatureCNNEncoder expects a (channels, height, width) observation shape, got {shape}."
            )
        channels, height, width = shape
        if height < 36 or width < 36:
            raise ValueError(
                "NatureCNNEncoder needs at least 36x36 spatial dimensions for its 8/4/3 kernel "
                f"stack, got {height}x{width}."
            )
        init = layer_init if layer_init is not None else (lambda layer: layer)
        self.obs_shape = shape
        self.scale_input = bool(scale_input)
        self.conv = nn.Sequential(
            init(nn.Conv2d(channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            conv_output_dim = int(self.conv(torch.zeros(1, channels, height, width)).shape[1])
        self.conv_output_dim = conv_output_dim
        self.head = nn.Sequential(
            init(nn.Linear(conv_output_dim, int(feature_dim))),
            nn.ReLU(),
        )
        # `feature_dim` is the attribute name `model/nnknn_model.py` reads when a
        # feature extractor is attached to an NN-kNN case store.
        self.feature_dim = int(feature_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations
        if x.dim() == len(self.obs_shape) and tuple(x.shape) == self.obs_shape:
            x = x.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError(
                f"NatureCNNEncoder expects (N, C, H, W) or {self.obs_shape} input, got shape {tuple(x.shape)}."
            )
        x = x.float()
        if self.scale_input:
            x = x / 255.0
        return self.head(self.conv(x))


def build_nature_cnn_encoder(
    obs_shape: Sequence[int],
    *,
    feature_dim: int = NATURE_CNN_FEATURE_DIM,
    orthogonal_init: bool = False,
    scale_input: bool = True,
) -> NatureCNNEncoder:
    """Convenience constructor selecting between default and orthogonal init."""

    return NatureCNNEncoder(
        obs_shape,
        feature_dim=feature_dim,
        scale_input=scale_input,
        layer_init=orthogonal_layer_init if orthogonal_init else None,
    )
