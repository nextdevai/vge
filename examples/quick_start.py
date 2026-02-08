#!/usr/bin/env python3
"""
Quick Start Example for Variance-Gated Ensembles (VGE).

This script demonstrates the core functionality of the VGE framework:
1. Last-Layer Ensemble (LLE) with optional VGN
2. Deep Ensemble (DE) with optional VGN
3. Monte Carlo Dropout (MCD) inference
4. Uncertainty metrics computation

Run from the project root:
    python examples/quick_start.py

Reference: "Variance-Gated Ensembles: An Epistemic-Aware Framework for
Uncertainty Estimation"
"""

import torch
import torch.nn.functional as F

from vge import (
    VGN,
    DeepEnsemble,
    LastLayerEnsemble,
    WideResNet,
    set_seeds,
)
from vge.metrics import (
    compute_epjs,
    compute_epkl,
    compute_vgmu,
    safe_decomposition,
)


def main():
    # Set seeds for reproducibility
    set_seeds(42)

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create dummy input (batch of 4)
    x = torch.randn(4, 3, 32, 32, device=device)
    num_classes = 10

    print("=" * 60)
    print("1. Last-Layer Ensemble (LLE) with VGN")
    print("=" * 60)

    # Build LLE: shared backbone with M=5 classification heads
    backbone = WideResNet(
        in_channels=3,
        out_channels=num_classes,
        depth=16,  # Smaller for demonstration (use 28 for full experiments)
        widen_factor=4,  # Smaller for demonstration (use 10 for full experiments)
        p=0.3,
        n_tasks=5,  # 5 ensemble heads
    )
    lle = LastLayerEnsemble(model=backbone, n_tasks=5)

    # Wrap with Variance-Gated Normalization
    lle_vgn = VGN(
        ensemble=lle,
        num_classes=num_classes,
        init_log_k=0.0,
        learn_k=True,  # k is learnable per class
    ).to(device)

    print(f"LLE parameters: {sum(p.numel() for p in lle_vgn.parameters()):,}")

    # Forward pass
    lle_vgn.eval()
    with torch.no_grad():
        Q, k = lle_vgn(x)

    print(f"Q (gated probs):   {Q.shape}  # (batch, members, classes)")
    print(f"k (learned param): {k.shape}  # (classes,)")

    # Compute uncertainty metrics
    TU, AU, EU = safe_decomposition(Q)
    VGMU = compute_vgmu(Q)

    print("\nUncertainty (sample 0):")
    print(f"  Total (TU):     {TU[0].item():.3f}")
    print(f"  Aleatoric (AU): {AU[0].item():.3f}")
    print(f"  Epistemic (EU): {EU[0].item():.3f}")
    print(f"  VGMU:           {VGMU[0].item():.3f}")

    # Predictions from gated ensemble mean
    Q_bar = Q.mean(dim=1)  # (batch, classes)
    preds = Q_bar.argmax(dim=-1)
    confidence = Q_bar.max(dim=-1).values
    print(f"\nPredictions: {preds.tolist()}")
    print(f"Confidence:  {confidence.round(decimals=3)}")

    print("\n" + "=" * 60)
    print("2. Deep Ensemble (DE)")
    print("=" * 60)

    # Build DE: M=3 independent models (smaller for quick start)
    base_model = WideResNet(
        in_channels=3,
        out_channels=num_classes,
        depth=16,
        widen_factor=4,
        p=0.3,
        n_tasks=1,  # Single head per member
    )
    de = DeepEnsemble(model=base_model, M=3).to(device)

    print(f"DE parameters: {sum(p.numel() for p in de.parameters()):,}")
    print(f"Members: {de.M}")

    # Forward pass
    de.eval()
    with torch.no_grad():
        P_de = de(x)  # (batch, M, classes)

    print(f"P (ensemble probs): {P_de.shape}")

    # Uncertainty from raw ensemble
    TU_de, AU_de, EU_de = safe_decomposition(P_de)
    EPKL = compute_epkl(P_de)

    print("\nUncertainty (sample 0):")
    print(f"  Total (TU):     {TU_de[0].item():.4f}")
    print(f"  Aleatoric (AU): {AU_de[0].item():.4f}")
    print(f"  Epistemic (EU): {EU_de[0].item():.4f}")
    print(f"  EPKL:           {EPKL[0].item():.4f}")

    # Predictions from ensemble mean
    P_bar = P_de.mean(dim=1)
    preds_de = P_bar.argmax(dim=-1)
    print(f"\nPredictions: {preds_de.tolist()}")

    print("\n" + "=" * 60)
    print("3. Deep Ensemble with VGN")
    print("=" * 60)

    # Wrap DE with VGN
    de_vgn = VGN(
        ensemble=de,
        num_classes=num_classes,
        learn_k=True,
    ).to(device)

    de_vgn.eval()
    with torch.no_grad():
        Q_v, k_v = de_vgn(x)

    print(f"Q (gated probs):  {Q_v.shape}")
    print(f"Learned k: {k_v[:5].round(decimals=3)}")  # First 5 classes

    print("\n" + "=" * 60)
    print("4. Monte Carlo Dropout (MCD) - Demonstration")
    print("=" * 60)

    # For MCD, we use dropout at inference time
    # This is a simplified Demonstration- see evaluate.py --method mcd for full implementation

    mcd_model = WideResNet(
        in_channels=3,
        out_channels=num_classes,
        depth=16,
        widen_factor=4,
        p=0.3,  # Dropout probability
        n_tasks=1,
    ).to(device)

    # Enable dropout at inference (train mode but no gradient)
    mcd_model.train()
    n_samples = 10

    with torch.no_grad():
        # Collect multiple stochastic forward passes
        samples = torch.stack(
            [F.softmax(mcd_model(x), dim=-1) for _ in range(n_samples)],
            dim=-1,
        )  # (batch, n_samples, classes)

    print(f"MCD samples: {samples.shape}")

    # Uncertainty from MC samples
    P_mean = samples.mean(dim=-1)
    EPJS = compute_epjs(samples)

    # Predictive entropy (total uncertainty)
    TU_mcd = -(P_mean * torch.log(P_mean + 1e-8)).sum(dim=-1)

    # Expected entropy (aleatoric)
    sample_entropy = -(samples * torch.log(samples + 1e-8)).sum(dim=-1)
    AU_mcd = sample_entropy.mean(dim=1)

    # Mutual information (epistemic)
    EU_mcd = TU_mcd - AU_mcd

    print("\nMCD Uncertainty (sample 0):")
    print(f"  Total (TU):     {TU_mcd[0].item():.4f}")
    print(f"  Aleatoric (AU): {AU_mcd[0].item():.4f}")
    print(f"  Epistemic (EU): {EU_mcd[0].item():.4f}")
    print(f"  EPJS:           {EPJS[0].item():.4f}")

    print("\n" + "=" * 60)
    print("5. Comparing Ensemble Types")
    print("=" * 60)

    print("\n| Method | Parameters | Members | VGN Support |")
    print("|--------|------------|---------|-------------|")
    print(
        f"| LLE    | {sum(p.numel() for p in lle.parameters()):>10,} | 5       | Yes         |"
    )
    print(
        f"| DE     | {sum(p.numel() for p in de.parameters()):>10,} | 3       | Yes         |"
    )
    print(
        f"| MCD    | {sum(p.numel() for p in mcd_model.parameters()):>10,} | N/A     | No          |"
    )

    print("\nKey differences:")
    print("  - LLE: Shared backbone, multiple heads (efficient)")
    print("  - DE:  Independent models (more diverse, higher cost)")
    print("  - MCD: Single model, stochastic inference (simplest)")

    print("\n" + "=" * 60)
    print("Quick Start Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Train LLE:     python train.py --method lle --n-heads 10")
    print("  - Train LLE+VGN: python train.py --method lle --n-heads 10 --use-vgn")
    print("  - Train DE:      python train.py --method de --n-members 5")
    print("  - Train DE+VGN:  python train.py --method de --n-members 5 --use-vgn")
    print("  - Train for MCD: python train.py --method baseline")
    print(
        "  - Evaluate:      python evaluate.py --method lle --checkpoint ./logs/cifar10_lle_model.pth"
    )


if __name__ == "__main__":
    main()
