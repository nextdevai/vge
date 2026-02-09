#!/usr/bin/env python3
"""
Evaluate saved predictions and compute uncertainty metrics.

Loads .npz files produced by the predictor functions in src/vge/predictors.py
(predict, ensemble_predict, vgn_predict, mcd_predict) and computes:
accuracy, F1, TU, AU, EU, EPKL, EPJS, VGMU.

Usage:
    # Evaluate ensemble/MCD predictions (P has shape (N, M, C))
    python evaluate.py predictions.npz

    # Evaluate VGN predictions (Q has shape (N, M, C), includes k values)
    python evaluate.py vgn_predictions.npz

    # Evaluate baseline predictions (P has shape (N, C), no uncertainty metrics)
    python evaluate.py baseline_predictions.npz

    # Save detailed per-sample results to CSV
    python evaluate.py predictions.npz --output results.csv

Reference: "Variance-Gated Ensembles: An Epistemic-Aware Framework for
Uncertainty Estimation"
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from vge.metrics import (
    compute_epjs,
    compute_epkl,
    compute_vgmu,
    safe_decomposition,
)


def evaluate(path: str, output: str | None = None) -> None:
    """Load predictions and compute metrics."""
    data = np.load(path)

    labels = torch.from_numpy(data["L"])  # (N,)

    if "Q" in data:
        probs = torch.from_numpy(data["Q"])
    elif "P" in data:
        probs = torch.from_numpy(data["P"])
    else:
        print(
            "Error: prediction file must contain 'P' or 'Q'",
            file=sys.stderr,
        )
        sys.exit(1)

    # Derive predictions from probabilities
    if probs.ndim == 3:
        preds = probs.mean(dim=1).argmax(dim=-1)
    else:
        preds = probs.argmax(dim=-1)

    # Classification metrics
    accuracy = accuracy_score(labels.numpy(), preds.numpy())
    f1 = f1_score(labels.numpy(), preds.numpy(), average="macro")

    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 (macro): {f1:.3f}")

    # Uncertainty metrics require ensemble dimension (N, M, C)
    if probs.ndim == 3:
        TU, AU, EU = safe_decomposition(probs)
        EPKL = compute_epkl(probs)
        EPJS = compute_epjs(probs)
        VGMU = compute_vgmu(probs)

        print(f"TU (mean): {TU.mean().item():.3f}")
        print(f"AU (mean): {AU.mean().item():.3f}")
        print(f"EU (mean): {EU.mean().item():.3f}")
        print(f"EPKL (mean): {EPKL.mean().item():.3f}")
        print(f"EPJS (mean): {EPJS.mean().item():.3f}")
        print(f"VGMU (mean): {VGMU.mean().item():.3f}")

        if output:
            df = pd.DataFrame(
                {
                    "labels": labels.numpy(),
                    "predictions": preds.numpy(),
                    "TU": TU.numpy(),
                    "AU": AU.numpy(),
                    "EU": EU.numpy(),
                    "EPKL": EPKL.numpy(),
                    "EPJS": EPJS.numpy(),
                    "VGMU": VGMU.numpy(),
                }
            )
            df.to_csv(output, index=False)
            print(f"Results saved to {output}")
    else:
        print("(No ensemble dimension — uncertainty metrics skipped)")
        if output:
            df = pd.DataFrame(
                {
                    "labels": labels.numpy(),
                    "predictions": preds.numpy(),
                }
            )
            df.to_csv(output, index=False)
            print(f"Results saved to {output}")

    # VGN k values
    if "k" in data:
        k = data["k"]
        print(f"k: {k}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved predictions and compute uncertainty metrics.",
    )
    parser.add_argument("predictions", help="Path to .npz predictions file")
    parser.add_argument("--output", help="Save per-sample results to CSV")
    args = parser.parse_args()

    evaluate(args.predictions, args.output)


if __name__ == "__main__":
    main()
