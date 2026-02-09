#!/usr/bin/env python3
"""
Unified training script for all model types.

Supports baseline (for MC Dropout), Deep Ensemble, and Last-Layer Ensemble
training, with optional Variance-Gated Normalization (VGN) wrapping.

Example usage:
    # Baseline (for MC Dropout experiments)
    python train.py --method baseline --epochs 100 --dropout 0.3

    # Deep Ensemble
    python train.py --method de --epochs 100 --n-members 5

    # Deep Ensemble with VGN
    python train.py --method de --epochs 100 --n-members 5 --use-vgn

    # Last-Layer Ensemble
    python train.py --method lle --epochs 100 --n-heads 5

    # Last-Layer Ensemble with VGN
    python train.py --method lle --epochs 100 --n-heads 5 --use-vgn

Reference: "Variance-Gated Ensembles: An Epistemic-Aware Framework for
Uncertainty Estimation"
"""

import argparse
import logging
import os

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vge import (
    VGN,
    DeepEnsemble,
    EarlyStopping,
    EnsembleTrainer,
    LastLayerEnsemble,
    Trainer,
    VGNTrainer,
    WideResNet,
    ensemble_predict,
    predict,
    select_device,
    set_seeds,
    vgn_predict,
)

# For deterministic behavior (for reproducibility)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_dataset(data_dir="./data"):
    """
    Load CIFAR-10 dataset with appropriate transforms.

    Args:
        data_dir: Directory for dataset storage

    Returns:
        Tuple of (train_dataset, test_dataset, num_classes)
    """
    transform_train = transforms.Compose(
        [
            # transforms.Pad(4, padding_mode="reflect"),
            # transforms.RandomCrop(32),
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
            transforms.RandomErasing(0.5),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    num_classes = 10

    return train_dataset, test_dataset, num_classes


def build_model(args, num_classes, device):
    """
    Build model based on the selected method.

    Returns:
        Model moved to the specified device.
    """
    if args.method == "baseline":
        logging.info(f"Building WideResNet-{args.depth}-{args.widen_factor}")
        logging.info(f"Single head (baseline), Dropout: {args.dropout}")

        model = WideResNet(
            in_channels=3,
            out_channels=num_classes,
            depth=args.depth,
            widen_factor=args.widen_factor,
            p=args.dropout,
            n_tasks=1,
        )

    elif args.method == "de":
        logging.info(f"Building WideResNet-{args.depth}-{args.widen_factor}")
        logging.info(
            f"Deep Ensemble with M={args.n_members} members, Dropout: {args.dropout}"
        )

        base_model = WideResNet(
            in_channels=3,
            out_channels=num_classes,
            depth=args.depth,
            widen_factor=args.widen_factor,
            p=args.dropout,
            n_tasks=1,
        )
        ensemble = DeepEnsemble(
            model=base_model,
            M=args.n_members,
            deterministic=args.deterministic,
        )

        if args.use_vgn:
            logging.info(f"Wrapping with VGN (learn_k={args.learn_k})")
            model = VGN(
                ensemble=ensemble,
                num_classes=num_classes,
                init_log_k=args.init_log_k,
                learn_k=args.learn_k,
            )
        else:
            model = ensemble

    elif args.method == "lle":
        logging.info(f"Building WideResNet-{args.depth}-{args.widen_factor}")
        logging.info(f"Ensemble heads: {args.n_heads}, Dropout: {args.dropout}")

        backbone = WideResNet(
            in_channels=3,
            out_channels=num_classes,
            depth=args.depth,
            widen_factor=args.widen_factor,
            p=args.dropout,
            n_tasks=args.n_heads,
        )
        ensemble = LastLayerEnsemble(backbone, n_tasks=args.n_heads)

        if args.use_vgn:
            logging.info(f"Wrapping with VGN (learn_k={args.learn_k})")
            model = VGN(
                ensemble=ensemble,
                num_classes=num_classes,
                init_log_k=args.init_log_k,
                learn_k=args.learn_k,
            )
        else:
            model = ensemble
    else:
        raise ValueError(f"Unknown method: {args.method}")

    return model.to(device)


def build_trainer(args, model, optimizer, early_stopping, device):
    """
    Build the appropriate trainer for the selected method.
    """
    kwargs = dict(
        model=model,
        optimizer=optimizer,
        early_stopping=early_stopping,
        log_dir=args.log_dir,
        device=device,
    )

    if args.method == "baseline":
        return Trainer(**kwargs)

    if args.use_vgn:
        return VGNTrainer(**kwargs)

    return EnsembleTrainer(**kwargs)


def get_suffix(args):
    """Return a file-naming suffix based on method and VGN flag."""
    if args.method == "baseline":
        return "baseline"
    if args.method == "de":
        return "deep_ensemble_vgn" if args.use_vgn else "deep_ensemble"
    if args.method == "lle":
        return "lle_vgn" if args.use_vgn else "lle"


def main():
    parser = argparse.ArgumentParser(
        description="Train models for uncertainty estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Method selection
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["baseline", "de", "lle"],
        help="Training method: baseline (for MCD), de (Deep Ensemble), lle (Last-Layer Ensemble)",
    )

    # Dataset arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for dataset storage",
    )

    # Model arguments
    parser.add_argument(
        "--depth",
        type=int,
        default=28,
        help="WideResNet depth (must satisfy (depth-4) %% 6 == 0)",
    )
    parser.add_argument(
        "--widen-factor",
        type=int,
        default=10,
        help="WideResNet width multiplier",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability",
    )

    # Deep Ensemble arguments
    parser.add_argument(
        "--n-members",
        type=int,
        default=5,
        help="Number of ensemble members for Deep Ensemble (--method de)",
    )

    # Last-Layer Ensemble arguments
    parser.add_argument(
        "--n-heads",
        type=int,
        default=5,
        help="Number of ensemble heads for Last-Layer Ensemble (--method lle)",
    )

    # VGN arguments
    parser.add_argument(
        "--use-vgn",
        action="store_true",
        default=False,
        help="Wrap ensemble with Variance-Gated Normalization (de/lle only)",
    )
    parser.add_argument(
        "--init-log-k",  # k is parameterized as softplus(k) for stability
        type=float,
        default=0.0,  # softplus(0.0) = 0.693; starting point for k
        help="Initial value for k in VGN layer",
    )
    parser.add_argument(
        "--learn-k",
        action="store_true",
        default=True,
        help="Make k learnable",
    )
    parser.add_argument(
        "--no-learn-k",
        action="store_false",
        dest="learn_k",
        help="Fix k (not learnable)",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs=2,
        default=[0.9, 0.999],
        help="Adam beta parameters",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )

    # Early stopping
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        default=False,
        help="Enable early stopping",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum improvement for early stopping",
    )

    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Use deterministic algorithms (slower)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for logs and checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (auto-detected if not specified)",
    )

    args = parser.parse_args()

    # Setup logging
    suffix = get_suffix(args)

    os.makedirs(args.log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(args.log_dir, f"training_{suffix}.log"),
                mode="w",
            ),
            logging.StreamHandler(),
        ],
    )

    method_labels = {
        "baseline": "Baseline Model Training (for MC Dropout)",
        "de": "Deep Ensemble Training",
        "lle": "Last-Layer Ensemble Training",
    }

    logging.info("=" * 60)
    logging.info(method_labels[args.method])
    logging.info("=" * 60)

    # Set seeds
    set_seeds(args.seed, args.deterministic)
    logging.info(f"Random seed: {args.seed}")

    # Device selection
    device = args.device if args.device else select_device()
    logging.info(f"Using device: {device}")

    # Load dataset
    logging.info("Loading dataset: cifar10")
    train_dataset, test_dataset, num_classes = get_dataset(args.data_dir)
    logging.info(
        f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}"
    )
    logging.info(f"Number of classes: {num_classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Build model
    model = build_model(args, num_classes, device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
    )

    # Early stopping
    early_stopping = None
    if args.early_stopping:
        early_stopping = EarlyStopping(
            min_delta=args.min_delta,
            patience=args.patience,
        )
        logging.info(f"Early stopping enabled (patience={args.patience})")

    # Trainer
    trainer = build_trainer(args, model, optimizer, early_stopping, device)

    # Train
    logging.info(f"Starting training for {args.epochs} epochs")
    logging.info(f"Batch size: {args.batch_size}, LR: {args.lr}")

    train_losses, valid_losses, train_acc, valid_acc = trainer.fit(
        num_epochs=args.epochs,
        train_batches=train_loader,
        valid_batches=test_loader,
        fname=f"cifar10_{suffix}_training_results.csv",
        model_name=f"cifar10_{suffix}_model.pth",
    )

    # Final results
    logging.info("=" * 60)
    logging.info("Training Complete")
    logging.info("=" * 60)
    logging.info(
        f"Final Train Loss: {train_losses[-1]:.4f}, Accuracy: {train_acc[-1]:.4f}"
    )
    logging.info(
        f"Final Valid Loss: {valid_losses[-1]:.4f}, Accuracy: {valid_acc[-1]:.4f}"
    )

    model_path = os.path.join(args.log_dir, f"cifar10_{suffix}_model.pth")
    logging.info(f"Model saved to: {model_path}")

    if args.method == "baseline":
        logging.info("")
        logging.info("To evaluate with MC Dropout, run:")
        logging.info(f"  python evaluate.py --method mcd --checkpoint {model_path}")

    # Run prediction on test set
    logging.info("")
    logging.info("Running prediction on test set...")
    pred_fname = os.path.join(args.log_dir, f"cifar10_{suffix}_predictions.npz")

    if args.method == "baseline":
        predict(model, test_loader, device, fname=pred_fname)
    elif args.method in ("de", "lle") and args.use_vgn:
        assert isinstance(model, VGN)
        vgn_predict(model, test_loader, device, fname=pred_fname)
    elif args.method in ("de", "lle"):
        ensemble_predict(model, test_loader, device, fname=pred_fname)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    logging.info(f"Predictions saved to: {pred_fname}")


if __name__ == "__main__":
    main()
