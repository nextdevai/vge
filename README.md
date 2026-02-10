# Variance-Gated Ensembles (VGE)

Official demonstration code for the manuscript:

> **Variance-Gated Ensembles: An Epistemic-Aware Framework for Uncertainty Estimation**

## Overview

Variance-Gated Ensembles (VGE) is a framework for epistemic-aware uncertainty estimation in deep learning.

### Key Components

**Variance-Gated Margin Uncertainty (VGMU)**:
A gated margin score based on the signal-to-noise ratio of the top-2 predicted classes:

```
margin = μ_top1 - μ_top2
snr    = margin / (σ_top1 + σ_top2)

𝛾 = 1 - exp[-snr]

VGMS = 𝛾·μ_top1 (probability score)
VGMU = 1 - VGMS (uncertainty score)
```

**Variance-Gated Normalization (VGN)**:
A signal-to-noise gate that suppresses uncertain predictions (during training):

```
SNR = -μ / (k·σ)

Γ = 1 - exp[-SNR]

q = p·Γ / pᵀΓ (normalized probability score)
```

where μ is the ensemble mean, σ is the ensemble standard deviation, and k is a learnable per-class parameter.

## Installation

```bash
# Clone the repository
git clone https://github.com/nextdevai/vge.git
cd vge
```

### From Source

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install as a package
pip install -e .
```

### From Wheel

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

pip install dist/vge-1.0.0-py3-none-any.whl
```

To build the wheel yourself:

```bash
pip install build
python -m build --wheel
```

This produces `dist/vge-1.0.0-py3-none-any.whl`. The wheel includes the `vge` package only. The training script (`train.py`), evaluation script (`evaluate.py`), and examples are not included — clone the repository for those.

With the wheel installed, all `vge` functionality is available as a library:

```python
from vge import (
    WideResNet, DeepEnsemble, LastLayerEnsemble, VGN,
    Trainer, EnsembleTrainer, VGNTrainer,
    predict, ensemble_predict, vgn_predict, mcd_predict,
    compute_vgmu,
)

# Build a model
backbone = WideResNet(in_channels=3, out_channels=10, depth=28, widen_factor=10, p=0.3, n_tasks=5)
ensemble = LastLayerEnsemble(model=backbone, n_tasks=5)
model = VGN(ensemble=ensemble, num_classes=10)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = VGNTrainer(model=model, optimizer=optimizer, device="cuda")
trainer.fit(num_epochs=100, train_batches=train_loader, valid_batches=val_loader)

# Predict and evaluate
vgn_predict(model, dataloader=test_loader, device="cuda", fname="predictions.npz")
```

See [Programmatic Usage](#programmatic-usage) and the sections below for full API details.

## Quick Start

Run the quick start example to see all methods in action:

```bash
python examples/quick_start.py
```

This demonstrates:

- Last-Layer Ensemble (LLE) with optional VGN
- Deep Ensemble (DE) with optional VGN
- Monte Carlo Dropout (MCD) inference
- Uncertainty metrics (TU, AU, EU, VGMU, EPKL)

## Project Structure

```
vge/
├── src/
│   └── vge/
│       ├── __init__.py       # Package exports
│       ├── vgn.py            # VarianceGatedNormalizer (custom autograd)
│       ├── models.py         # WideResNet, DeepEnsemble, LastLayerEnsemble, VGN
│       ├── trainers.py       # Trainer, EnsembleTrainer, VGNTrainer
│       ├── predictors.py     # predict, ensemble_predict, vgn_predict, mcd_predict
│       ├── metrics.py        # Uncertainty metrics (TU, AU, EU, EPKL, EPJS, VGMU)
│       └── utils.py          # Device selection, seeds, early stopping
├── train.py           # Unified training script (baseline, DE, LLE, with optional VGN)
├── evaluate.py        # Evaluate saved .npz predictions (accuracy, F1, uncertainty)
├── examples/
│   └── quick_start.py # Quick start demo of all methods
├── pyproject.toml     # Package configuration
└── README.md          # This file
```

## Usage

### Training

All training uses a single `train.py` script with a `--method` flag:

```bash
# Last-Layer Ensemble
python train.py --method lle --epochs 100 --n-heads 5

# Last-Layer Ensemble with VGN
python train.py --method lle --epochs 100 --n-heads 5 --use-vgn

# Deep Ensemble
python train.py --method de --epochs 100 --n-members 5

# Deep Ensemble with VGN
python train.py --method de --epochs 100 --n-members 5 --use-vgn

# Baseline (for MC Dropout experiments)
python train.py --method baseline --epochs 100 --dropout 0.3
```

After training completes, `train.py` automatically runs the appropriate predictor on the test set and saves the results to `<log-dir>/cifar10_<suffix>_predictions.npz`.

Full list of training arguments:

```bash
python train.py --help
```

| Argument                     | Default   | Description                                    |
| ---------------------------- | --------- | ---------------------------------------------- |
| `--method`                   | —         | `baseline`, `de`, or `lle` (required)          |
| `--data-dir`                 | ./data    | Directory for dataset storage                  |
| `--depth`                    | 28        | WideResNet depth                               |
| `--widen-factor`             | 10        | WideResNet width multiplier                    |
| `--dropout`                  | 0.3       | Dropout probability                            |
| `--n-members`                | 5         | Number of ensemble members (de only)           |
| `--n-heads`                  | 5         | Number of ensemble heads (lle only)            |
| `--use-vgn`                  | False     | Wrap ensemble with VGN (de/lle only)           |
| `--init-log-k`               | 0.0       | Initial value for softplus(k) in VGN layer     |
| `--learn-k` / `--no-learn-k` | True      | Whether k is learnable (VGN only)              |
| `--epochs`                   | 100       | Training epochs                                |
| `--batch-size`               | 128       | Batch size                                     |
| `--lr`                       | 1e-3      | Learning rate                                  |
| `--weight-decay`             | 0.0       | Weight decay (L2 regularization)               |
| `--betas`                    | 0.9 0.999 | Adam beta parameters                           |
| `--num-workers`              | 4         | Number of data loading workers                 |
| `--early-stopping`           | False     | Enable early stopping                          |
| `--patience`                 | 5         | Early stopping patience                        |
| `--min-delta`                | 1e-4      | Minimum improvement for early stopping         |
| `--seed`                     | 42        | Random seed                                    |
| `--deterministic`            | False     | Use deterministic algorithms (slower)          |
| `--log-dir`                  | ./logs    | Output directory                               |
| `--device`                   | auto      | Device to use (auto-detected if not specified) |

### Evaluation

The `evaluate.py` script computes metrics from saved `.npz` prediction files produced by the predictor functions (`predict`, `ensemble_predict`, `vgn_predict`, `mcd_predict`):

```bash
# Evaluate ensemble or MCD predictions (P has shape (N, M, C))
python evaluate.py predictions.npz

# Evaluate VGN predictions (Q has shape (N, M, C), includes k values)
python evaluate.py vgn_predictions.npz

# Evaluate baseline predictions (P has shape (N, C), no uncertainty metrics)
python evaluate.py baseline_predictions.npz

# Save detailed per-sample results to CSV
python evaluate.py predictions.npz --output results.csv
```

The evaluation script outputs:

- Accuracy and F1 score (macro)
- Uncertainty decomposition (TU, AU, EU) — for ensemble/MCD predictions
- Pairwise divergences (EPKL, EPJS) — for ensemble/MCD predictions
- Variance-Gated Margin Uncertainty (VGMU) — for ensemble/MCD predictions
- Learned k values per class — for VGN predictions

| Argument      | Description                                |
| ------------- | ------------------------------------------ |
| `predictions` | Path to `.npz` predictions file (required) |
| `--output`    | Save per-sample results to CSV (optional)  |

### Programmatic Usage

```python
import torch
from vge import (
    VarianceGatedNormalizer,
    WideResNet,
    DeepEnsemble,
    LastLayerEnsemble,
    VGN,
    Trainer,
    EnsembleTrainer,
    VGNTrainer,
    predict,
    ensemble_predict,
    vgn_predict,
    mcd_predict,
    compute_vgmu,
)
```

---

## Ensemble Methods

This framework supports multiple ensemble architectures, each with optional Variance-Gated Normalization (VGN).

### Deep Ensembles

Deep Ensembles train M independent model copies with different random initializations. This provides diversity through independent training trajectories.

#### Deep Ensemble without VGN

```python
import torch
from vge import WideResNet, DeepEnsemble, EnsembleTrainer, ensemble_predict

# Create base model (single head)
base_model = WideResNet(
    in_channels=3,
    out_channels=10,  # CIFAR-10
    depth=28,
    widen_factor=10,
    p=0.3,
    n_tasks=1,  # Single head for deep ensemble
)

# Create deep ensemble with M=5 members
ensemble = DeepEnsemble(model=base_model, M=5, deterministic=False)

# Each member is reinitialized with a different seed
# ensemble.members[i] has independent parameters

# Training
optimizer = torch.optim.Adam(ensemble.parameters(), lr=1e-3)
trainer = EnsembleTrainer(
    model=ensemble,
    optimizer=optimizer,
    device="cuda",
    log_dir="./logs/deep_ensemble",
)

# trainer.fit(num_epochs=100, train_batches=train_loader, valid_batches=val_loader)

# Inference
x = torch.randn(32, 3, 32, 32)
P = ensemble(x)  # (B, M, C) - stacked softmax probabilities

# Predictions from mean
P_bar = P.mean(dim=1)  # (B, C)
preds = P_bar.argmax(dim=-1)  # (B,)

# For VGMU, use ensemble probabilities
from vge import compute_vgmu
P = model.ensemble(x)  # (B, M, C) ensemble output
VGMU = compute_vgmu(P)
```

#### Deep Ensemble with VGN

```python
from vge import WideResNet, DeepEnsemble, VGN, VGNTrainer, vgn_predict

# Create deep ensemble
base_model = WideResNet(in_channels=3, out_channels=10, n_tasks=1)
ensemble = DeepEnsemble(model=base_model, M=5)

# Wrap with Variance-Gated Normalization
model = VGN(
    ensemble=ensemble,
    num_classes=10,
    init_log_k=0.0,  # softplus(0.0) = 0.693; starting point for k
    learn_k=True,    # k is learnable per class
)

# Training with VGN
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = VGNTrainer(
    model=model,
    optimizer=optimizer,
    device="cuda",
)

# trainer.fit(num_epochs=100, train_batches=train_loader, valid_batches=val_loader)

# Inference with VGN outputs
x = torch.randn(32, 3, 32, 32)
Q, k = model(x)

# Q: Gated probabilities (B, M, C) - suppressed by uncertainty
# k: Learned per-class gate parameter (C,)

# Compute uncertainty metrics on gated probabilities
from vge.metrics import safe_decomposition
TU, AU, EU = safe_decomposition(Q)

# For VGMU, use ensemble probabilities
from vge import compute_vgmu
Q = model.ensemble(x)  # (B, M, C) ensemble output
VGMU = compute_vgmu(Q)
```

### Last-Layer Ensembles

Last-Layer Ensembles (LLE) share a backbone network and use multiple classification heads. This is more efficient than Deep Ensembles while still capturing epistemic uncertainty.

#### Last-Layer Ensemble without VGN

```python
from vge import WideResNet, LastLayerEnsemble, EnsembleTrainer, ensemble_predict

# Create backbone with multiple heads
backbone = WideResNet(
    in_channels=3,
    out_channels=10,
    depth=28,
    widen_factor=10,
    p=0.3,
    n_tasks=10,  # 10 classification heads
)

# Wrap as ensemble
ensemble = LastLayerEnsemble(model=backbone, n_tasks=10)

# Training
optimizer = torch.optim.Adam(ensemble.parameters(), lr=1e-3)
trainer = EnsembleTrainer(
    model=ensemble,
    optimizer=optimizer,
    device="cuda",
    log_dir="./logs/lle",
)

# trainer.fit(num_epochs=100, train_batches=train_loader, valid_batches=val_loader)

# Inference
x = torch.randn(32, 3, 32, 32)
P = ensemble(x)  # (B, M, C) where M=10 heads

# Each head provides a probability distribution
# Diversity comes from independent head parameters
P_bar = P.mean(dim=1)  # (B, C) mixture prediction
preds = P_bar.argmax(dim=-1)
```

#### Last-Layer Ensemble with VGN

```python
from vge import WideResNet, LastLayerEnsemble, VGN, VGNTrainer

# Create LLE backbone
backbone = WideResNet(
    in_channels=3,
    out_channels=10,
    depth=28,
    widen_factor=10,
    p=0.3,
    n_tasks=10,
)
ensemble = LastLayerEnsemble(model=backbone, n_tasks=10)

# Add Variance-Gated Normalization
model = VGN(
    ensemble=ensemble,
    num_classes=10,
    learn_k=True,
)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = VGNTrainer(model=model, optimizer=optimizer, device="cuda")

# trainer.fit(num_epochs=100, train_batches=train_loader, valid_batches=val_loader)

# Inference
Q, k = model(x)

# Gated predictions suppress uncertain classes
Q_bar = Q.mean(dim=1)  # (B, C)
preds = Q_bar.argmax(dim=-1)
```

---

## Monte Carlo Dropout

Monte Carlo Dropout (MCD) estimates uncertainty by performing multiple stochastic forward passes with dropout enabled at inference time. This framework supports two modes:

| Mode           | Description          | Dropout Location              |
| -------------- | -------------------- | ----------------------------- |
| **Standard**   | Classic MC Dropout   | All dropout layers active     |
| **Last-Layer** | Multihead MC Dropout | Only MLP and classifier heads |

### Standard Monte Carlo Dropout

Standard MCD applies dropout throughout the network during inference, requiring multiple forward passes through the entire model.

```python
from vge import WideResNet, mcd_predict

# Train a baseline model first
model = WideResNet(
    in_channels=3,
    out_channels=10,
    depth=28,
    widen_factor=10,
    p=0.3,       # Dropout probability used during training
    n_tasks=1,   # Single head
)

# ... train and save checkpoint ...

mcd_predict(
    model=model,
    checkpoint_path="./logs/model.pth",
    dataloader=test_loader,
    device="cuda",
    dropout_prob=0.1,      # Dropout prob at inference (can differ from training)
    num_heads=1,           # Single head for standard MCD
    num_samples=50,        # Number of stochastic forward passes
    seed=42,
    last_layer=False,      # Standard mode: dropout everywhere
    fname="mcd_predictions.npz",
)

# Load saved predictions
import numpy as np
data = np.load("mcd_predictions.npz")
P = torch.from_numpy(data["P"])  # (N, 50, C) - 50 samples per input
L = torch.from_numpy(data["L"])  # (N,) - true labels
```

### Last-Layer Monte Carlo Dropout

Last-layer MCD keeps the backbone deterministic (no dropout) and only applies dropout in the MLP and classifier heads. This is faster because backbone features are computed once per input.

#### Single Head, Multiple Samples

```python
mcd_predict(
    model=model,
    checkpoint_path="./logs/model.pth",
    dataloader=test_loader,
    device="cuda",
    dropout_prob=0.1,
    num_heads=1,           # Single head
    num_samples=100,       # 100 stochastic passes through the head
    seed=42,
    last_layer=True,       # Last-layer mode: dropout only in heads
    fname="ll_mcd_predictions.npz",
)

data = np.load("ll_mcd_predictions.npz")
P = torch.from_numpy(data["P"])  # (N, 100, C)
```

#### Multiple Heads, Multiple Samples

Combining multiple heads with multiple samples per head provides richer uncertainty estimates. The total number of predictions is `num_heads × num_samples`.

```python
# Model with multiple classifier heads
model = WideResNet(
    in_channels=3,
    out_channels=10,
    depth=28,
    widen_factor=10,
    p=0.3,
    n_tasks=10,  # 10 classifier heads
)

# ... train and save checkpoint ...

# Last-layer MCD with 10 heads × 10 samples = 100 predictions
mcd_predict(
    model=model,
    checkpoint_path="./logs/model.pth",
    dataloader=test_loader,
    device="cuda",
    dropout_prob=0.1,
    num_heads=10,          # 10 classifier heads
    num_samples=10,        # 10 samples per head
    seed=42,
    last_layer=True,
    fname="multihead_mcd_predictions.npz",
)

data = np.load("multihead_mcd_predictions.npz")
P = torch.from_numpy(data["P"])  # (N, 100, C) - 10 heads × 10 samples
```

### Comparing MCD Configurations

```python
import numpy as np
import torch
from vge import WideResNet, mcd_predict

model = WideResNet(in_channels=3, out_channels=10, n_tasks=10, p=0.3)
checkpoint = "./logs/model.pth"

# Configuration 1: Standard MCD (50 full forward passes)
mcd_predict(
    model=model, checkpoint_path=checkpoint, dataloader=test_loader,
    device="cuda", dropout_prob=0.1, num_heads=1, num_samples=50,
    last_layer=False, fname="standard_mcd.npz",
)

# Configuration 2: Last-layer MCD (1 head × 50 samples)
mcd_predict(
    model=model, checkpoint_path=checkpoint, dataloader=test_loader,
    device="cuda", dropout_prob=0.1, num_heads=1, num_samples=50,
    last_layer=True, fname="ll_single_mcd.npz",
)

# Configuration 3: Multihead last-layer MCD (10 heads × 5 samples)
mcd_predict(
    model=model, checkpoint_path=checkpoint, dataloader=test_loader,
    device="cuda", dropout_prob=0.1, num_heads=10, num_samples=5,
    last_layer=True, fname="ll_multi_mcd.npz",
)

# All produce .npz files with P of shape (N, 50, C), with different uncertainty characteristics
# Load and compare:
P1 = torch.from_numpy(np.load("standard_mcd.npz")["P"])
P2 = torch.from_numpy(np.load("ll_single_mcd.npz")["P"])
P3 = torch.from_numpy(np.load("ll_multi_mcd.npz")["P"])
```

### MCD Uncertainty Metrics

```python
import numpy as np
import torch

# After running MCD inference, load saved predictions
data = np.load("mcd_predictions.npz")
P = torch.from_numpy(data["P"])  # (N, S, C) where S = num_samples * num_heads

# Mean prediction
P_mean = P.mean(dim=1)  # (N, C)

# Predictive entropy (total uncertainty)
TU = -(P_mean * torch.log2(P_mean + 1e-8)).sum(dim=-1)  # (N,)

# Expected entropy (aleatoric uncertainty)
sample_entropy = -(P * torch.log2(P + 1e-8)).sum(dim=-1)  # (N, S)
AU = sample_entropy.mean(dim=1)  # (N,)

# Mutual information (epistemic uncertainty)
EU = TU - AU  # (N,)
```

---

## Trainers Summary

| Trainer           | Model Type                      | Loss Function                  |
| ----------------- | ------------------------------- | ------------------------------ |
| `Trainer`         | Baseline (single head)          | Cross-entropy on softmax       |
| `EnsembleTrainer` | DeepEnsemble, LastLayerEnsemble | Cross-entropy on mixture mean  |
| `VGNTrainer`      | VGN-wrapped ensembles           | Cross-entropy on gated mixture |

```python
from vge import Trainer, EnsembleTrainer, VGNTrainer

# Baseline model
trainer = Trainer(model=baseline_model, optimizer=opt, device="cuda")

# Ensemble (Deep or Last-Layer) without VGN
trainer = EnsembleTrainer(model=ensemble, optimizer=opt, device="cuda")

# Any ensemble with VGN
trainer = VGNTrainer(model=vgn_model, optimizer=opt, device="cuda")
```

---

## Predictors Summary

All predictor functions save results to `.npz` files and return `None`. Use `numpy.load()` to read the saved predictions.

| Function           | Model Type           | Saved Keys                 |
| ------------------ | -------------------- | -------------------------- |
| `predict`          | Baseline             | `L (N,)`, `P (N,C)`        |
| `ensemble_predict` | DeepEnsemble, LLE    | `L (N,)`, `P (N,M,C)`      |
| `vgn_predict`      | VGN-wrapped          | `L (N,)`, `Q (N,M,C)`, `k` |
| `mcd_predict`      | Any with classifiers | `L (N,)`, `P (N,S,C)`      |

```python
from vge import predict, ensemble_predict, vgn_predict, mcd_predict

# Baseline inference (saves to .npz)
predict(model, dataloader, device="cuda", fname="baseline.npz")

# Ensemble inference (saves to .npz)
ensemble_predict(ensemble, dataloader, device="cuda", fname="ensemble.npz")

# VGN inference (saves to .npz)
vgn_predict(vgn_model, dataloader, device="cuda", fname="vgn.npz")

# MC Dropout inference (saves to .npz)
mcd_predict(model, checkpoint, dataloader, device="cuda", fname="mcd.npz")
```

## Supported Dataset

| Dataset  | Classes | Train Size | Test Size |
| -------- | ------- | ---------- | --------- |
| CIFAR-10 | 10      | 50,000     | 10,000    |

## Uncertainty Metrics

The framework computes several uncertainty metrics:

| Metric   | Description                                     |
| -------- | ----------------------------------------------- |
| **TU**   | Total Uncertainty (entropy of mixture)          |
| **AU**   | Aleatoric Uncertainty (expected member entropy) |
| **EU**   | Epistemic Uncertainty (mutual information)      |
| **EPKL** | Expected Pairwise KL Divergence                 |
| **EPJS** | Expected Pairwise Jensen-Shannon Divergence     |
| **VGMU** | Variance-Gated Margin Uncertainty               |

## Architecture Details

### WideResNet

The default architecture is WideResNet-28-10:

- Depth: 28 layers
- Width multiplier: 10
- Total parameters: ~36M (with 10 heads)

### Last-Layer Ensemble

Instead of training M independent models, we use a single backbone with M classification heads. This provides:

- Shared feature learning
- Reduced memory footprint
- Faster training and inference
- Ensemble diversity through head-specific parameters

## Citation

If you use this code, please cite:

```bibtex
@article{Gillis2026VGE,
  author = {Gillis, H. Martin  and Xu, Isaac and Trappenberg, Thomas}
  title={Variance-Gated Ensembles: An Epistemic-Aware Framework for Uncertainty Estimation},
  year={2026},
  url={https://arxiv.org/abs/2602.08142},
}
```

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.
