"""
Variance-Gated Normalization (VGN) module.

Implements the core VGN layer with a custom autograd function for
backpropagation through ensemble mean and variance statistics.

Reference: "Variance-Gated Ensembles: An Epistemic-Aware Framework for
Uncertainty Estimation"
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _VarianceGatedNormalizerFn(torch.autograd.Function):
    """
    Custom autograd function for variance-gated normalization.

    Implements the forward pass:
        G = 1 - exp(-mu / (k * s))  # Signal-to-noise gate
        Q = (P * G) / Z             # Gated normalization

    And the backward pass with gradients through mu, s, and k.
    """

    @staticmethod
    def forward(ctx: Any, P: Tensor, k: Tensor) -> Tensor:
        """
        Forward pass of variance-gated normalization.

        Args:
            P: Ensemble probability tensor (B, M, C)
            k: Per-class gate parameter (C,)

        Returns:
            Q: Gated normalized probabilities (B, M, C)
        """
        B, _, C = P.shape

        # Ensemble statistics (mu, sigma)
        mu = P.mean(dim=1)  # (B, C)
        sigma = P.std(dim=1, unbiased=False)  # (B, C)
        s = sigma + 1.0e-8  # (B, C); numerical stability

        # k safety: Avoid division in Gamma and its partials
        k_eff = torch.clamp(k, min=1e-3)  # (C,)
        k_clamped = k <= 1e-3  # (C,)
        k_safe = k_eff.view(1, C).expand(B, C)  # (B, C)

        # Gate (Gamma): 1 - exp(-mu / (k s))
        G = 1.0 - torch.exp(-mu / (k_safe * s))  # (B, C)

        # Clamp Gamma to be strictly positive
        G_clamped = G <= 1.0e-8  # (B, C)
        G_safe = torch.clamp(G, min=1.0e-8)  # (B, C)

        # Compute Q = (P * Gamma) / Z per member
        PG_safe = P * G_safe.unsqueeze(dim=1)  # (B, M, C)
        Z = PG_safe.sum(dim=-1, keepdim=True)  # (B, M, 1)
        Z_clamped = Z <= 1.0e-8  # (B, M, 1)
        Z_safe = torch.where(Z_clamped, torch.ones_like(Z), Z)  # (B, M, 1)

        Q = PG_safe / Z_safe  # (B, M, C)

        # Fallback: If Z is tiny, return identity (Q <- P)
        Q_safe = torch.where(Z_clamped.expand_as(P), P, Q)  # (B, M, C)

        # Save tensors for backward
        ctx.save_for_backward(
            P, Q_safe, G_safe, G_clamped, mu, s, k_eff, k_clamped, Z_clamped
        )

        return Q_safe

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor, Tensor]:
        """
        Backward pass computing gradients w.r.t. P and k.
        """
        (U,) = grad_outputs  # dL/dQ (B, M, C)
        P, Q, G, G_clamped, mu, s, k_eff, k_clamped, Z_clamped = ctx.saved_tensors

        _, M, C = P.shape
        k = k_eff.view(1, C).expand_as(mu)  # (B, C)

        # Recompute safe Z for backward
        Z = (
            (P * G.unsqueeze(dim=1)).sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        )  # (B, M, 1)

        # A = (q_m^T U) - scalar per (B, M, 1) member
        A = (Q * U).sum(dim=-1, keepdim=True)  # (B, M, 1)

        # dL/dG per member (VJP)
        dL_dG_per_member = (P * U - P * A) / Z  # (B, M, C)
        dL_dG_per_member = torch.where(
            Z_clamped, torch.zeros_like(dL_dG_per_member), dL_dG_per_member
        )
        dL_dG = dL_dG_per_member.sum(dim=1)  # (B, C)

        # Zero gradient where Gamma was clamped
        dL_dG = torch.where(G_clamped, torch.zeros_like(dL_dG), dL_dG)  # (B, C)

        # Gate partials
        dG_dmu = (1.0 - G) / (k * s)  # (B, C)
        dG_ds = -(1.0 - G) * (mu / (k * s * s))  # (B, C)
        dG_dk = -(1.0 - G) * (mu / (k * k * s))  # (B, C)

        # Chain into mu and s paths
        dL_dmu = dL_dG * dG_dmu  # (B, C)
        dL_ds = dL_dG * dG_ds  # (B, C)

        # Direct normalization path (with G fixed)
        via_Z = (G.unsqueeze(dim=1) * U - G.unsqueeze(dim=1) * A) / Z  # (B, M, C)

        # Mean path
        via_mu = dL_dmu.unsqueeze(dim=1) / M  # (B, M, C)

        # Variance path
        ds_dP_per_member = (P - mu.unsqueeze(dim=1)) / (M * s.unsqueeze(dim=1))
        via_s = dL_ds.unsqueeze(dim=1) * ds_dP_per_member  # (B, M, C)

        # Total gradient wrt. P
        dP = via_Z + via_mu + via_s  # (B, M, C)

        # Identity fallback where Z was clamped
        dP = torch.where(Z_clamped.expand_as(P), U, dP)

        # Gate parameter gradient (per-class), then batch-reduce
        dL_dk = (dL_dG * dG_dk).sum(dim=0)  # (C,)

        # Zero dL/dk where k was clamped
        dL_dk = torch.where(k_clamped, torch.zeros_like(dL_dk), dL_dk)  # (C,)

        return dP, dL_dk


class VarianceGatedNormalizer(nn.Module):
    """
    Variance-Gated Normalizer module.

    Applies variance-gated normalization to ensemble probability distributions,
    using a signal-to-noise gate that suppresses high-variance (uncertain) predictions.

    The gate is computed as:
        Gamma = 1 - exp(-mu / (k * s))

    where:
        - mu: ensemble mean probability
        - s: ensemble standard deviation
        - k: learnable per-class gate parameter

    Args:
        num_classes: Number of output classes
        init_log_k: Initial value for log(k) parameter (default: 0.0)
        learn_k: Whether k is learnable (default: True)
    """

    def __init__(
        self,
        num_classes: int,
        init_log_k: float = 0.0,
        learn_k: bool = True,
    ) -> None:
        super().__init__()

        self.learn_k = learn_k
        self._init_log_k_scalar = init_log_k

        k_init_vec = torch.full((num_classes,), self._init_log_k_scalar)

        if self.learn_k:
            # Learn l so k = softplus(l) > 0
            self.log_k = nn.Parameter(k_init_vec)  # (C,)
        else:
            # Fixed k = softplus(init_log_k)
            self.k_fixed: Tensor
            self.register_buffer("k_fixed", F.softplus(k_init_vec))

    def k_value(self) -> Tensor:
        """Get current k values (C,)."""
        return (F.softplus(self.log_k) + 1.0e-8) if self.learn_k else self.k_fixed

    def forward(self, P: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply variance-gated normalization.

        Args:
            P: Ensemble probabilities (B, M, C)

        Returns:
            Q: Gated probabilities (B, M, C)
            k: Gate parameter (C,)
        """
        k = self.k_value()  # (C,)

        Q = _VarianceGatedNormalizerFn.apply(P, k.to(P))
        assert isinstance(Q, Tensor)

        return Q, k
