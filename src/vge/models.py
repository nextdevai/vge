"""
Neural network models for Variance-Gated Ensembles.

Contains:
- BasicBlock: Residual block for WideResNet
- WideResNet: Wide Residual Network with multihead support
- DeepEnsemble: Ensemble of independently initialized models
- LastLayerEnsemble: Ensemble via multiple classification heads
- VGN: Wrapper combining ensemble with variance-gated normalization

Reference: "Variance-Gated Ensembles: An Epistemic-Aware Framework for
Uncertainty Estimation"
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .vgn import VarianceGatedNormalizer


class BasicBlock(nn.Module):
    """
    Basic residual block for WideResNet.

    Architecture: BN -> ReLU -> Dropout -> Conv -> BN -> ReLU -> Dropout -> Conv
    with a skip connection (identity or 1x1 conv for dimension matching).
    """

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int,
        p: float = 0.0,
    ) -> None:
        super().__init__()

        self.skip_connection: nn.Module = (
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
            if stride != 1 or in_planes != out_planes
            else nn.Identity()
        )

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p),
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p),
            nn.Conv2d(
                out_planes,
                out_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(x) + self.skip_connection(x)


class WideResNet(nn.Module):
    """
    Wide Residual Network (WRN) for image classification.

    Default configuration is WRN-28-10 (depth=28, widen_factor=10).
    Supports multiple classification heads via n_tasks parameter for
    Last-Layer Ensemble (LLE) training.

    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        out_channels: Number of output classes
        depth: Network depth, must satisfy (depth-4) % 6 == 0
        widen_factor: Width multiplier for channels
        p: Dropout probability
        n_tasks: Number of classification heads (for LLE)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 10,
        depth: int = 28,
        widen_factor: int = 10,
        p: float = 0.0,
        n_tasks: int = 1,
    ) -> None:
        super().__init__()

        self.n_tasks = n_tasks

        assert (depth - 4) % 6 == 0, "Depth should be 6n + 4"
        n = (depth - 4) // 6
        k = widen_factor

        widths = [16, 16 * k, 32 * k, 64 * k]

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                widths[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        )

        self.layers = nn.Sequential(
            self._make_layer(widths[0], widths[1], blocks=n, stride=1, p=p),
            self._make_layer(widths[1], widths[2], blocks=n, stride=2, p=p),
            self._make_layer(widths[2], widths[3], blocks=n, stride=2, p=p),
        )

        self.tail = nn.Sequential(
            nn.BatchNorm2d(widths[3]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(widths[3], widths[3]),
            nn.BatchNorm1d(widths[3]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p, inplace=False),
        )

        self.classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(widths[3], widths[3]),
                    nn.BatchNorm1d(widths[3]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=p, inplace=False),
                    nn.Linear(in_features=widths[3], out_features=out_channels),
                )
                for _ in range(self.n_tasks)
            ]
        )

    def _make_layer(
        self,
        in_planes: int,
        out_planes: int,
        blocks: int,
        stride: int,
        p: float,
    ) -> nn.Sequential:
        layers = [BasicBlock(in_planes, out_planes, stride, p)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_planes, out_planes, stride=1, p=p))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor | list[Tensor]:
        x = self.stem(x)
        x = self.layers(x)
        x = self.tail(x)
        x = self.mlp(x)

        logits = [head(x) for head in self.classifiers]  # list of (B, C)

        if self.n_tasks == 1:
            return logits[0]  # (B, C) tensor

        return logits


class DeepEnsemble(nn.Module):
    """
    Deep Ensemble of independently initialized models.

    Creates M copies of a base model with different random initializations.
    Each ensemble member is trained independently, and predictions are
    combined by stacking their softmax outputs.

    Args:
        model: Base model to copy (e.g., WideResNet with n_tasks=1)
        M: Number of ensemble members (default: 10)
        deterministic: Whether to use deterministic initialization
    """

    def __init__(
        self,
        model: nn.Module,
        M: int = 10,
        deterministic: bool = False,
    ) -> None:
        super().__init__()

        self.M = M
        self.deterministic = deterministic
        self.members = nn.ModuleList([copy.deepcopy(model) for _ in range(M)])

        # Reinitialize each member with different seed
        base_seed = 0
        for i, m in enumerate(self.members):
            seed = base_seed + i
            self._reinit_model(m, seed, self.deterministic)

    @staticmethod
    def _reinit_model(model: nn.Module, seed: int, deterministic: bool) -> None:
        """Reinitialize all parameters in the model with a specific seed."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        for module in model.modules():
            reset = getattr(module, "reset_parameters", None)
            if reset is not None:
                reset()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through all ensemble members.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            P: Stacked softmax probabilities (B, M, C)
        """
        P = torch.stack([member(x).softmax(dim=-1) for member in self.members], dim=1)
        return P  # (B, M, C)


class LastLayerEnsemble(nn.Module):
    """
    Last-Layer Ensemble (LLE).

    Wraps a multihead model (e.g., WideResNet with n_tasks > 1) to produce
    ensemble probability distributions. Each head acts as an ensemble member,
    sharing the feature backbone but with independent classification layers.

    Args:
        model: Multi-head base model with n_tasks > 1
        n_tasks: Number of ensemble members (must match model.n_tasks)
    """

    def __init__(self, model: nn.Module, n_tasks: int) -> None:
        super().__init__()

        self.model = model
        self.n_tasks = n_tasks

        assert self.n_tasks > 1, "Last layer ensemble requires more than one head."

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass returning stacked ensemble probabilities.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            P: Ensemble probabilities (B, M, C) where M = n_tasks
        """
        logits = self.model(x)
        P = torch.stack([F.softmax(logit, dim=-1) for logit in logits], dim=1)
        return P  # (B, M, C)


class VGN(nn.Module):
    """
    Variance-Gated Network.

    Combines an ensemble model with variance-gated normalization.
    The ensemble produces probability distributions that are then
    processed by the VGN layer to suppress uncertain predictions.

    Args:
        ensemble: Ensemble model (e.g., LastLayerEnsemble)
        num_classes: Number of output classes
        init_log_k: Initial value for log(k) (default: 0.0)
        learn_k: Whether k is learnable (default: True)
    """

    def __init__(
        self,
        ensemble: nn.Module,
        num_classes: int,
        init_log_k: float = 0.0,
        learn_k: bool = True,
    ) -> None:
        super().__init__()
        self.ensemble = ensemble

        self.vgn = VarianceGatedNormalizer(
            num_classes=num_classes,
            init_log_k=init_log_k,
            learn_k=learn_k,
        )

        for p in self.vgn.parameters():
            p.requires_grad_(learn_k)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass through ensemble and VGN layer.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Q: Gated probabilities (B, M, C)
            k: Gate parameter k (C,)
        """
        P = self.ensemble(x)  # (B, M, C)
        Q, k = self.vgn(P)
        return Q, k
