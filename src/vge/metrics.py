"""
Uncertainty metrics for Variance-Gated Ensembles.

Implements uncertainty decomposition (TU, AU, EU), pairwise divergence
metrics (EPKL, EPJS), and the novel VGMU (Variance-Gated Margin Uncertainty).

Reference: "Variance-Gated Ensembles: An Epistemic-Aware Framework for
Uncertainty Estimation"
"""

from __future__ import annotations

import torch
from torch import Tensor


def safe_probabilities(
    p: Tensor,
    eps: float = 1e-8,
    dim: int = -1,
) -> Tensor:
    """
    Clip and renormalize probabilities to remove zeros and fp drift.

    Args:
        p: Probability tensor (arbitrary shape, sums to 1 along *dim*)
        eps: Floor value to prevent log(0)
        dim: Dimension along which probabilities sum to 1

    Returns:
        Cleaned probability tensor, same shape as input
    """
    # First pass to remove zeros and prevent log(0)
    p = p.clamp(min=eps, max=1.0)
    p = p / p.sum(dim=dim, keepdim=True).clamp(min=eps)

    # Second pass to remove drift from the first normalization
    p = p.clamp(min=eps, max=1.0)
    p = p / p.sum(dim=dim, keepdim=True).clamp(min=eps)
    return p


def safe_decomposition(
    probs: Tensor,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute uncertainty decomposition using raw ensemble probabilities.

    Args:
        probs: Ensemble probabilities (B, M, C)
        eps: Small constant for numerical stability

    Returns:
        TU: Total uncertainty (B,)
        AU: Aleatoric uncertainty (B,)
        EU: Epistemic uncertainty (B,)
    """
    probs = safe_probabilities(probs, eps)  # (B, M, C)
    mean_probs = safe_probabilities(probs.mean(dim=1), eps)  # (B, C)

    # Number of classes for normalization
    C = probs.shape[2]
    log2_C = torch.log2(
        torch.tensor(
            C,
            dtype=probs.dtype,
            device=probs.device,
        )
    )

    # Total Uncertainty (normalized to [0, 1])
    TU = -torch.sum(mean_probs * torch.log2(mean_probs), dim=-1) / log2_C  # (B,)

    # Per-member entropy (normalized to [0, 1])
    H_members = -torch.sum(probs * torch.log2(probs), dim=-1) / log2_C  # (B, M)

    # Aleatoric Uncertainty
    AU = H_members.mean(dim=-1)  # (B,)

    # Epistemic Uncertainty
    EU = TU - AU  # (B,)

    return TU, AU, EU


def batch_pairwise_kl(
    probs: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """
    Compute pairwise KL divergence between ensemble members.

    Args:
        probs: Ensemble probabilities (B, M, C)
        eps: Small constant for numerical stability

    Returns:
        KL divergence matrix (B, M, M)
    """
    probs = safe_probabilities(probs, eps)  # (B, M, C)

    # Expand for pairwise computation
    P_i = probs.unsqueeze(2)  # (B, M, 1, C)
    P_j = probs.unsqueeze(1)  # (B, 1, M, C)

    # KL(P_i || P_j)
    kl = torch.sum(
        P_i * (torch.log2(P_i) - torch.log2(P_j)),
        dim=-1,
    )

    return kl  # (B, M, M)


def batch_pairwise_jsd(
    probs: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """
    Compute pairwise Jensen-Shannon divergence between ensemble members.

    Args:
        probs: Ensemble probabilities (B, M, C)
        eps: Small constant for numerical stability

    Returns:
        JSD matrix (B, M, M)
    """
    probs = safe_probabilities(probs, eps)  # (B, M, C)

    P_i = probs.unsqueeze(2)  # (B, M, 1, C)
    P_j = probs.unsqueeze(1)  # (B, 1, M, C)

    # Mixture (already safe since P_i, P_j >= eps)
    M_ij = 0.5 * (P_i + P_j)  # (B, M, M, C)

    # JSD = 0.5 * KL(P_i || M) + 0.5 * KL(P_j || M)
    kl_i = torch.sum(
        P_i * (torch.log2(P_i) - torch.log2(M_ij)),
        dim=-1,
    )
    kl_j = torch.sum(
        P_j * (torch.log2(P_j) - torch.log2(M_ij)),
        dim=-1,
    )

    jsd = 0.5 * (kl_i + kl_j)

    return jsd  # (B, M, M)


def compute_epkl(
    probs: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """
    Compute Expected Pairwise KL divergence.

    Args:
        probs: Ensemble probabilities (B, M, C)
        eps: Small constant for numerical stability

    Returns:
        EPKL values (B,)
    """
    kl_matrix = batch_pairwise_kl(probs, eps)  # (B, M, M)

    M = probs.shape[1]
    mask = ~torch.eye(M, dtype=torch.bool, device=probs.device)  # (M, M)
    epkl = kl_matrix[:, mask].sum(dim=1) / (M * (M - 1))

    return epkl  # (B,)


def compute_epjs(
    probs: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """
    Compute Expected Pairwise Jensen-Shannon divergence.

    Args:
        probs: Ensemble probabilities (B, M, C)
        eps: Small constant for numerical stability

    Returns:
        EPJS values (B,)
    """
    jsd_matrix = batch_pairwise_jsd(probs, eps)  # (B, M, M)

    M = probs.shape[1]
    mask = ~torch.eye(M, dtype=torch.bool, device=probs.device)  # (M, M)
    epjs = jsd_matrix[:, mask].sum(dim=1) / (M * (M - 1))

    return epjs  # (B,)


def compute_vgmu(
    probs: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """
    Compute Variance-Gated Margin Uncertainty (VGMU).

    VGMU is a gated margin score based on the signal-to-noise ratio
    of the top-2 predicted classes:

        margin = μ_top1 - μ_top2
        snr    = margin / (σ_top1 + σ_top2)
        VGMS   = (1 - exp(-snr)) * μ_top1

    Higher VGMU indicates higher confidence (larger gated margin).

    Args:
        probs: Ensemble probabilities (B, M, C)
        eps: Small constant for numerical stability

    Returns:
        VGMU uncertainty scores (B,)
    """
    probs = safe_probabilities(probs, eps)  # (B, M, C)

    mu = probs.mean(dim=1)  # (B, C)
    s = probs.std(dim=1) + eps  # (B, C)

    # Get top-2 mean values and their indices
    top2_values, top2_indices = torch.topk(mu, k=2, dim=-1)  # (B, 2)

    # Gather the corresponding standard deviations
    std1 = s.gather(1, top2_indices[:, 0:1]).squeeze(1)  # (B,)
    std2 = s.gather(1, top2_indices[:, 1:2]).squeeze(1)  # (B,)

    # Margin between top-2 means
    margin = top2_values[:, 0] - top2_values[:, 1]  # (B,)

    # Signal-to-noise ratio
    snr = margin / (std1 + std2)

    vgms = (1.0 - torch.exp(-snr)) * top2_values[:, 0]
    vgmu = 1.0 - vgms  # return uncertainty

    return vgmu
