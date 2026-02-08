"""
Training utilities for Variance-Gated Ensembles.

Contains:
- Trainer: For training baseline non-ensemble models
- EnsembleTrainer: For training DeepEnsemble/LastLayerEnsemble models
- VGNTrainer: For training VGN-wrapped models with variance gating

Reference: "Variance-Gated Ensembles: An Epistemic-Aware Framework for
Uncertainty Estimation"
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import pandas as pd
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader

    from .utils import EarlyStopping


class Trainer:
    """
    Trainer for baseline non-ensemble models.

    Trains single-head models using standard cross-entropy loss.

    Args:
        model: Model to train (single output head)
        optimizer: PyTorch optimizer
        early_stopping: Optional EarlyStopping instance
        log_dir: Directory for saving logs and checkpoints
        device: Device to train on (e.g. 'cuda', 'mps', 'cpu', or torch.device)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        early_stopping: EarlyStopping | None = None,
        log_dir: str = "./logs",
        device: str | torch.device = "cuda",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.log_dir = log_dir
        self.device = device

    def fit(
        self,
        num_epochs: int,
        train_batches: DataLoader,
        valid_batches: DataLoader,
        num_eval_batches: int | None = None,
        eps: float = 1.0e-8,
        fname: str = "training_results.csv",
        model_name: str = "model.pth",
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        """
        Train the model.

        Args:
            num_epochs: Maximum number of training epochs
            train_batches: Training DataLoader
            valid_batches: Validation DataLoader
            num_eval_batches: Optional limit on batches for evaluation
            eps: Small constant for numerical stability
            fname: Filename for training results CSV
            model_name: Filename for model checkpoint

        Returns:
            Tuple of (train_losses, valid_losses, train_accuracy, valid_accuracy)
        """
        self.model.train()

        train_losses, valid_losses = [], []
        train_accuracy, valid_accuracy = [], []

        for epoch in range(num_epochs):
            for i, batch in enumerate(train_batches):
                values = [*batch]
                inputs, labels = values[0].to(self.device), values[1].to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(inputs)
                log_probs = F.log_softmax(logits, dim=-1)

                loss = F.nll_loss(log_probs, labels)
                loss.backward()

                self.optimizer.step()

            train_loss, train_acc = self.evaluate(train_batches, num_eval_batches)
            valid_loss, valid_acc = self.evaluate(valid_batches, num_eval_batches)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            train_accuracy.append(train_acc)
            valid_accuracy.append(valid_acc)

            if not self.early_stopping:
                logging.info(
                    f"Epoch {epoch + 1}/{num_epochs} --> "
                    f"Train Loss (Accuracy): {train_loss:.4f} ({train_acc:.4f}), "
                    f"Valid Loss (Accuracy): {valid_loss:.4f} ({valid_acc:.4f})"
                )
            else:
                stop = self.early_stopping.step(valid_loss)
                logging.info(
                    f"Epoch {epoch + 1}/{num_epochs} --> "
                    f"Train Loss (Accuracy): {train_loss:.4f} ({train_acc:.4f}), "
                    f"Valid Loss (Accuracy): {valid_loss:.4f} ({valid_acc:.4f})"
                    f"\n[ES] Current={valid_loss:.4f} Best={self.early_stopping.best:.4f} "
                    f"Wait={self.early_stopping.wait}/{self.early_stopping.patience}"
                )
                if stop:
                    logging.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        # Save results
        df = pd.DataFrame(
            {
                "Epoch": range(1, len(train_losses) + 1),
                "Train Loss": train_losses,
                "Valid Loss": valid_losses,
                "Train Accuracy": train_accuracy,
                "Valid Accuracy": valid_accuracy,
            }
        )
        df.to_csv(os.path.join(self.log_dir, fname), index=False)

        # Save checkpoint
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            os.path.join(self.log_dir, model_name),
        )

        return train_losses, valid_losses, train_accuracy, valid_accuracy

    def evaluate(
        self,
        dataloader: DataLoader,
        num_eval_batches: int | None = None,
    ) -> tuple[float, float]:
        """
        Evaluate the model on a dataset.

        Args:
            dataloader: DataLoader for evaluation
            num_eval_batches: Optional limit on batches to evaluate
            eps: Small constant for numerical stability

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()

        total_samples = 0
        total_nll = 0.0
        total_correct = 0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                values = [*batch]
                inputs, labels = values[0].to(self.device), values[1].to(self.device)

                logits = self.model(inputs)
                log_probs = F.log_softmax(logits, dim=-1)

                total_nll += F.nll_loss(log_probs, labels, reduction="sum").item()

                preds = log_probs.argmax(dim=-1)
                total_correct += (preds == labels).sum().item()

                total_samples += inputs.size(0)

                if (num_eval_batches is not None) and (i + 1 >= num_eval_batches):
                    break

        loss = total_nll / total_samples
        accuracy = total_correct / total_samples

        self.model.train()

        return loss, accuracy


class EnsembleTrainer:
    """
    Trainer for ensemble models (DeepEnsemble or LastLayerEnsemble).

    Trains ensemble models using cross-entropy loss on the mixture
    distribution P.mean(dim=1), which averages predictions across
    ensemble members.

    Args:
        model: Ensemble model (DeepEnsemble or LastLayerEnsemble)
        optimizer: PyTorch optimizer
        early_stopping: Optional EarlyStopping instance
        log_dir: Directory for saving logs and checkpoints
        device: Device to train on (e.g. 'cuda', 'mps', 'cpu', or torch.device)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        early_stopping: EarlyStopping | None = None,
        log_dir: str = "./logs",
        device: str | torch.device = "cuda",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.log_dir = log_dir
        self.device = device

    def fit(
        self,
        num_epochs: int,
        train_batches: DataLoader,
        valid_batches: DataLoader,
        num_eval_batches: int | None = None,
        eps: float = 1.0e-8,
        fname: str = "training_results.csv",
        model_name: str = "model.pth",
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        """
        Train the ensemble model.

        Args:
            num_epochs: Maximum number of training epochs
            train_batches: Training DataLoader
            valid_batches: Validation DataLoader
            num_eval_batches: Optional limit on batches for evaluation
            eps: Small constant for numerical stability
            fname: Filename for training results CSV
            model_name: Filename for model checkpoint

        Returns:
            Tuple of (train_losses, valid_losses, train_accuracy, valid_accuracy)
        """
        self.model.train()

        train_losses, valid_losses = [], []
        train_accuracy, valid_accuracy = [], []

        for epoch in range(num_epochs):
            for i, batch in enumerate(train_batches):
                values = [*batch]
                inputs, labels = values[0].to(self.device), values[1].to(self.device)

                self.optimizer.zero_grad()

                # Per-member probabilities (B, M, C)
                P = self.model(inputs)

                mixture = P.mean(dim=1)  # (B, C)
                log_probs = torch.log(mixture.clamp_min(eps))

                loss = F.nll_loss(log_probs, labels)
                loss.backward()

                self.optimizer.step()

            train_loss, train_acc = self.evaluate(train_batches, num_eval_batches)
            valid_loss, valid_acc = self.evaluate(valid_batches, num_eval_batches)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            train_accuracy.append(train_acc)
            valid_accuracy.append(valid_acc)

            if not self.early_stopping:
                logging.info(
                    f"Epoch {epoch + 1}/{num_epochs} --> "
                    f"Train Loss (Accuracy): {train_loss:.4f} ({train_acc:.4f}), "
                    f"Valid Loss (Accuracy): {valid_loss:.4f} ({valid_acc:.4f})"
                )
            else:
                stop = self.early_stopping.step(valid_loss)
                logging.info(
                    f"Epoch {epoch + 1}/{num_epochs} --> "
                    f"Train Loss (Accuracy): {train_loss:.4f} ({train_acc:.4f}), "
                    f"Valid Loss (Accuracy): {valid_loss:.4f} ({valid_acc:.4f})"
                    f"\n[ES] Current={valid_loss:.4f} Best={self.early_stopping.best:.4f} "
                    f"Wait={self.early_stopping.wait}/{self.early_stopping.patience}"
                )
                if stop:
                    logging.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        # Save results
        df = pd.DataFrame(
            {
                "Epoch": range(1, len(train_losses) + 1),
                "Train Loss": train_losses,
                "Valid Loss": valid_losses,
                "Train Accuracy": train_accuracy,
                "Valid Accuracy": valid_accuracy,
            }
        )
        df.to_csv(os.path.join(self.log_dir, fname), index=False)

        # Save checkpoint
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            os.path.join(self.log_dir, model_name),
        )

        return train_losses, valid_losses, train_accuracy, valid_accuracy

    def evaluate(
        self,
        dataloader: DataLoader,
        num_eval_batches: int | None = None,
        eps: float = 1.0e-8,
    ) -> tuple[float, float]:
        """
        Evaluate the ensemble model on a dataset.

        Args:
            dataloader: DataLoader for evaluation
            num_eval_batches: Optional limit on batches to evaluate
            eps: Small constant for numerical stability

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()

        total_samples = 0
        total_nll = 0.0
        total_correct = 0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                values = [*batch]
                inputs, labels = values[0].to(self.device), values[1].to(self.device)

                # Per-member probabilities (B, M, C)
                P = self.model(inputs)

                mixture = P.mean(dim=1)  # (B, C)
                log_probs = torch.log(mixture.clamp_min(eps))

                total_nll += F.nll_loss(log_probs, labels, reduction="sum").item()

                preds = log_probs.argmax(dim=-1)
                total_correct += (preds == labels).sum().item()

                total_samples += inputs.size(0)

                if (num_eval_batches is not None) and (i + 1 >= num_eval_batches):
                    break

        loss = total_nll / total_samples
        accuracy = total_correct / total_samples

        self.model.train()

        return loss, accuracy


class VGNTrainer:
    """
    Trainer for Variance-Gated Network models.

    Trains the model using cross-entropy loss on the gated mixture
    distribution Q.mean(dim=1), which averages predictions across
    ensemble members after variance-gated normalization.

    Args:
        model: VGN model to train
        optimizer: PyTorch optimizer
        early_stopping: Optional EarlyStopping instance
        log_dir: Directory for saving logs and checkpoints
        device: Device to train on (e.g. 'cuda', 'mps', 'cpu', or torch.device)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        early_stopping: EarlyStopping | None = None,
        log_dir: str = "./logs",
        device: str | torch.device = "cuda",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.log_dir = log_dir
        self.device = device

    def fit(
        self,
        num_epochs: int,
        train_batches: DataLoader,
        valid_batches: DataLoader,
        num_eval_batches: int | None = None,
        eps: float = 1.0e-8,
        fname: str = "training_results.csv",
        model_name: str = "model.pth",
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        """
        Train the model.

        Args:
            num_epochs: Maximum number of training epochs
            train_batches: Training DataLoader
            valid_batches: Validation DataLoader
            num_eval_batches: Optional limit on batches for evaluation
            eps: Small constant for numerical stability
            fname: Filename for training results CSV
            model_name: Filename for model checkpoint

        Returns:
            Tuple of (train_losses, valid_losses, train_accuracy, valid_accuracy)
        """
        self.model.train()

        train_losses, valid_losses = [], []
        train_accuracy, valid_accuracy = [], []

        for epoch in range(num_epochs):
            for i, batch in enumerate(train_batches):
                values = [*batch]
                inputs, labels = values[0].to(self.device), values[1].to(self.device)

                self.optimizer.zero_grad()

                # Forward pass through VGN
                Q, _ = self.model(inputs)

                # Mixture over members (already normalized)
                mixture = Q.mean(dim=1)  # (B, C)
                log_probs = torch.log(mixture.clamp_min(eps))

                loss = F.nll_loss(log_probs, labels)
                loss.backward()

                self.optimizer.step()

            train_loss, train_acc = self.evaluate(train_batches, num_eval_batches)
            valid_loss, valid_acc = self.evaluate(valid_batches, num_eval_batches)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            train_accuracy.append(train_acc)
            valid_accuracy.append(valid_acc)

            if not self.early_stopping:
                logging.info(
                    f"Epoch {epoch + 1}/{num_epochs} --> "
                    f"Train Loss (Accuracy): {train_loss:.4f} ({train_acc:.4f}), "
                    f"Valid Loss (Accuracy): {valid_loss:.4f} ({valid_acc:.4f})"
                )
            else:
                stop = self.early_stopping.step(valid_loss)
                logging.info(
                    f"Epoch {epoch + 1}/{num_epochs} --> "
                    f"Train Loss (Accuracy): {train_loss:.4f} ({train_acc:.4f}), "
                    f"Valid Loss (Accuracy): {valid_loss:.4f} ({valid_acc:.4f})"
                    f"\n[ES] Current={valid_loss:.4f} Best={self.early_stopping.best:.4f} "
                    f"Wait={self.early_stopping.wait}/{self.early_stopping.patience}"
                )
                if stop:
                    logging.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        # Save results
        df = pd.DataFrame(
            {
                "Epoch": range(1, len(train_losses) + 1),
                "Train Loss": train_losses,
                "Valid Loss": valid_losses,
                "Train Accuracy": train_accuracy,
                "Valid Accuracy": valid_accuracy,
            }
        )
        df.to_csv(os.path.join(self.log_dir, fname), index=False)

        # Save checkpoint
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            os.path.join(self.log_dir, model_name),
        )

        return train_losses, valid_losses, train_accuracy, valid_accuracy

    def evaluate(
        self,
        dataloader: DataLoader,
        num_eval_batches: int | None = None,
        eps: float = 1.0e-8,
    ) -> tuple[float, float]:
        """
        Evaluate the model on a dataset.

        Args:
            dataloader: DataLoader for evaluation
            num_eval_batches: Optional limit on batches to evaluate
            eps: Small constant for numerical stability

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()

        total_samples = 0
        total_nll = 0.0
        total_correct = 0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                values = [*batch]
                inputs, labels = values[0].to(self.device), values[1].to(self.device)

                # Forward pass
                Q, _ = self.model(inputs)

                # Mixture over members
                mixture = Q.mean(dim=1)  # (B, C)
                log_probs = torch.log(mixture.clamp_min(eps))

                # Accumulate loss
                total_nll += F.nll_loss(log_probs, labels, reduction="sum").item()

                # Calculate accuracy
                preds = log_probs.argmax(dim=-1)
                total_correct += (preds == labels).sum().item()

                total_samples += inputs.size(0)

                if (num_eval_batches is not None) and (i + 1 >= num_eval_batches):
                    break

        loss = total_nll / total_samples
        accuracy = total_correct / total_samples

        self.model.train()

        return loss, accuracy
