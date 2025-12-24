"""Training infrastructure for the Universal Macro Emulator.

This module provides a Trainer class that handles:
- Training loop with epoch management
- Validation and early stopping
- Checkpoint saving/loading (best and periodic)
- Learning rate scheduling (cosine annealing with warmup)
- Logging to console and optional wandb
- Config-driven training via dataclass or dict
- Support for both baseline models and UniversalEmulator (regime-aware)
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader

# Optional wandb integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training.

    This dataclass defines all hyperparameters and settings for training.
    Can be loaded from YAML or dict.
    """

    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    epochs: int = 100
    grad_clip: float = 1.0

    # Learning rate schedule
    warmup_epochs: int = 5
    scheduler: str = "cosine"  # "cosine" or "none"

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4  # Minimum improvement to count as progress

    # Checkpointing
    checkpoint_dir: str = "runs/default"
    save_every_n_epochs: int = 10
    save_best: bool = True

    # Logging
    log_every_n_steps: int = 50
    use_wandb: bool = False
    wandb_project: str = "macro-emulator"
    wandb_run_name: str | None = None

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Loss function
    loss_fn: str = "mse"  # "mse" or "mae"
    horizon_weights: str = "uniform"  # "uniform", "exponential", or "impact"
    horizon_decay_tau: float = 20.0  # For exponential weighting
    use_advanced_loss: bool = False  # Use MultiHorizonLoss/CombinedLoss from losses.py

    # Multi-task (if using trajectory loss)
    lambda_traj: float = 0.0  # Weight for trajectory loss (0.0 = IRF only)
    lambda_reg: float = 0.0  # Regularization weight
    lambda_smooth: float = 0.01  # Smoothness penalty coefficient

    # Information regime (for UniversalEmulator)
    regime: str | None = None  # "A", "B1", "C" or None (auto-detect)

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)


class Trainer:
    """Trainer for IRF prediction models.

    Handles training loop, validation, checkpointing, and logging.
    Supports config-driven training with deterministic seeding.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig | dict[str, Any],
    ):
        """Initialize trainer.

        Args:
            model: PyTorch model to train
                - Baseline: forward(theta, world_id) -> IRF
                - Universal: forward(world_id, shock_idx, theta, ...) -> IRF
            train_loader: DataLoader for training set
            val_loader: DataLoader for validation set
            config: Training configuration (TrainingConfig or dict)
        """
        # Convert dict to TrainingConfig if needed
        if isinstance(config, dict):
            config = TrainingConfig(**config)

        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Detect model type (baseline vs universal)
        self.is_universal = self._is_universal_model(model)

        # Set regime for universal model
        if self.is_universal:
            if config.regime is None:
                # Default to Regime A if not specified
                self.regime = "A"
                logger.warning("Universal model detected but regime not specified. Defaulting to Regime A.")
            else:
                self.regime = config.regime
        else:
            self.regime = None

        # Set device
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Set random seed for reproducibility
        self._set_seed(config.seed)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Loss function
        self.loss_fn = self._create_loss_fn()

        # Horizon weights for multi-horizon loss (legacy, kept for compatibility)
        self.horizon_weights = self._create_horizon_weights() if not config.use_advanced_loss else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self._setup_logging()

        # History tracking
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "epoch": [],
        }

    def _is_universal_model(self, model: nn.Module) -> bool:
        """Detect if model is a UniversalEmulator.

        Args:
            model: PyTorch model

        Returns:
            True if model is UniversalEmulator, False otherwise
        """
        # Check if model has the characteristic attributes of UniversalEmulator
        return (
            hasattr(model, "world_embedding") and
            hasattr(model, "param_encoder") and
            hasattr(model, "shock_encoder")
        )

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler | None:
        """Create learning rate scheduler."""
        if self.config.scheduler == "none":
            return None

        if self.config.scheduler == "cosine":
            # Total steps for cosine annealing
            total_steps = self.config.epochs * len(self.train_loader)
            warmup_steps = self.config.warmup_epochs * len(self.train_loader)

            # Warmup schedule
            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                return 1.0

            warmup_scheduler = LambdaLR(self.optimizer, lr_lambda)

            # Cosine annealing after warmup
            # Ensure T_max is at least 1 to avoid division by zero
            t_max = max(1, total_steps - warmup_steps)
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=self.config.lr * 0.01,
            )

            # Combined scheduler
            # For simplicity, we'll use cosine with warmup handled in step()
            return cosine_scheduler

        raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

    def _create_loss_fn(self) -> Callable | nn.Module:
        """Create loss function.

        Returns either a simple loss (MSE/MAE) or advanced loss (MultiHorizonLoss/CombinedLoss).
        """
        if self.config.use_advanced_loss:
            # Use advanced loss functions from losses.py
            try:
                from emulator.training.losses import MultiHorizonLoss, CombinedLoss
            except ImportError:
                logger.warning("Advanced losses not available, falling back to basic MSE")
                return nn.MSELoss(reduction="none")

            # Determine horizon length (assume H=40 by default)
            H = 40  # Can be made configurable

            if self.config.lambda_smooth > 0:
                # Use CombinedLoss with smoothness
                return CombinedLoss(
                    H=H,
                    weight_scheme=self.config.horizon_weights,
                    tau=self.config.horizon_decay_tau,
                    lambda_smooth=self.config.lambda_smooth,
                )
            else:
                # Use MultiHorizonLoss only
                return MultiHorizonLoss(
                    H=H,
                    weight_scheme=self.config.horizon_weights,
                    tau=self.config.horizon_decay_tau,
                )
        else:
            # Legacy simple loss
            if self.config.loss_fn == "mse":
                return nn.MSELoss(reduction="none")  # We'll do custom reduction
            elif self.config.loss_fn == "mae":
                return nn.L1Loss(reduction="none")
            else:
                raise ValueError(f"Unknown loss function: {self.config.loss_fn}")

    def _create_horizon_weights(self) -> torch.Tensor | None:
        """Create horizon weights for multi-horizon loss.

        Returns:
            Tensor of shape (H+1,) with weights, or None for uniform
        """
        if self.config.horizon_weights == "uniform":
            return None  # Will use equal weights

        # Assume H=40 for now (can be made configurable)
        H = 40

        if self.config.horizon_weights == "exponential":
            # Exponential decay: w[h] = exp(-h/tau) / Z
            tau = self.config.horizon_decay_tau
            h = torch.arange(H + 1, dtype=torch.float32)
            weights = torch.exp(-h / tau)
            weights = weights / weights.sum()  # Normalize
            return weights

        elif self.config.horizon_weights == "impact":
            # Impact-weighted: w[0] = 0.3, w[h>0] = 0.7/H
            weights = torch.ones(H + 1, dtype=torch.float32) * (0.7 / H)
            weights[0] = 0.3
            return weights

        else:
            raise ValueError(f"Unknown horizon weighting: {self.config.horizon_weights}")

    def _setup_logging(self) -> None:
        """Setup logging (console + wandb)."""
        # Console logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Wandb logging
        if self.config.use_wandb:
            if not WANDB_AVAILABLE:
                logger.warning("wandb requested but not installed. Skipping wandb logging.")
                self.config.use_wandb = False
            else:
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name,
                    config=asdict(self.config),
                )
                logger.info(f"Initialized wandb run: {wandb.run.name}")

    def _forward_pass(self, batch: dict[str, Any]) -> torch.Tensor:
        """Regime-aware forward pass through the model.

        Args:
            batch: Batch dict from dataloader

        Returns:
            Model predictions
        """
        if self.is_universal:
            # Universal model: regime-aware forward pass
            world_ids = batch["world_ids"]
            # Move world_ids to device if it's a tensor
            if isinstance(world_ids, torch.Tensor):
                world_ids = world_ids.to(self.device)
            shock_idx = batch["shock_idx"].to(self.device)
            theta = batch["theta"].to(self.device) if "theta" in batch else None
            theta_mask = batch.get("theta_mask", None)
            if theta_mask is not None:
                theta_mask = theta_mask.to(self.device)

            # Regime-specific inputs
            kwargs = {
                "world_id": world_ids,
                "shock_idx": shock_idx,
                "regime": self.regime,
            }

            if self.regime == "A":
                # Regime A: theta, eps_sequence, shock_token
                kwargs["theta"] = theta
                kwargs["theta_mask"] = theta_mask
                if "eps_sequence" in batch:
                    kwargs["eps_sequence"] = batch["eps_sequence"].to(self.device)
                # History is optional in Regime A
                if "history" in batch:
                    kwargs["history"] = batch["history"].to(self.device)
                    if "history_mask" in batch:
                        kwargs["history_mask"] = batch["history_mask"].to(self.device)

            elif self.regime == "B1":
                # Regime B1: world_id, shock_token, history (no theta, no eps)
                if "history" not in batch:
                    raise ValueError("Regime B1 requires history in batch")
                kwargs["history"] = batch["history"].to(self.device)
                if "history_mask" in batch:
                    kwargs["history_mask"] = batch["history_mask"].to(self.device)

            elif self.regime == "C":
                # Regime C: theta, shock_token, history (no eps)
                kwargs["theta"] = theta
                kwargs["theta_mask"] = theta_mask
                if "history" not in batch:
                    raise ValueError("Regime C requires history in batch")
                kwargs["history"] = batch["history"].to(self.device)
                if "history_mask" in batch:
                    kwargs["history_mask"] = batch["history_mask"].to(self.device)

            predictions = self.model(**kwargs)

        else:
            # Baseline model: traditional forward pass
            theta = batch["theta"].to(self.device)
            world_ids = batch["world_ids"]

            # Handle single world vs mixed world
            if len(set(world_ids)) == 1:
                predictions = self.model(theta, world_ids[0])
            else:
                predictions = self.model(theta, world_ids)

        return predictions

    def compute_loss(self, batch: dict[str, Any], predictions: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute loss for a batch.

        Args:
            batch: Batch dict from dataloader (must have "irf" key)
            predictions: Model predictions, shape (batch_size, n_shocks, H+1, 3)
                        or (batch_size, H+1, 3) for single-shock models

        Returns:
            Scalar loss (or dict with loss components if using CombinedLoss)
        """
        if self.config.use_advanced_loss:
            # Use advanced loss functions
            targets = batch["irf"].to(self.device)

            # For universal model with single shock prediction
            if self.is_universal and predictions.dim() == 3:
                # Extract the IRF for the specific shock we predicted
                shock_idx = batch["shock_idx"]  # (batch,)
                # Select the target IRF for each sample's shock_idx
                batch_size = predictions.shape[0]
                targets_selected = torch.zeros_like(predictions)  # (batch, H+1, 3)
                for i in range(batch_size):
                    targets_selected[i] = targets[i, shock_idx[i], :, :]
                targets = targets_selected
                # No mask needed for single-shock predictions
                mask = None
            elif predictions.dim() == 3:
                # Single-shock baseline model
                targets = targets[:, 0, :, :]  # Take first shock
                mask = None
            else:
                # Multi-shock model - use mask
                mask = batch.get("irf_mask", None)
                if mask is not None:
                    mask = mask.to(self.device)

            loss_result = self.loss_fn(predictions, targets, mask=mask)

            # CombinedLoss returns dict, MultiHorizonLoss returns scalar
            if isinstance(loss_result, dict):
                return loss_result["loss"]  # Return total loss for backward
            else:
                return loss_result

        else:
            # Legacy loss computation
            targets = batch["irf"].to(self.device)  # (batch_size, n_shocks, H+1, 3)

            # Handle single-shock models (need to unsqueeze)
            if predictions.dim() == 3:  # (batch_size, H+1, 3)
                predictions = predictions.unsqueeze(1)  # (batch_size, 1, H+1, 3)
                targets = targets[:, :1, :, :]  # Take only first shock

            # Compute element-wise loss
            loss_elements = self.loss_fn(predictions, targets)  # (batch, n_shocks, H+1, 3)

            # Apply horizon weights if specified
            if self.horizon_weights is not None:
                weights = self.horizon_weights.to(self.device)
                weights = weights.view(1, 1, -1, 1)  # (1, 1, H+1, 1)
                loss_elements = loss_elements * weights

            # Apply mask for padded shocks (if present)
            if "irf_mask" in batch:
                mask = batch["irf_mask"].to(self.device)  # (batch, n_shocks)
                mask = mask.view(mask.shape[0], mask.shape[1], 1, 1)  # (batch, n_shocks, 1, 1)
                loss_elements = loss_elements * mask

            # Mean over all dimensions
            loss = loss_elements.mean()

            # Optional regularization
            if self.config.lambda_reg > 0 or self.config.lambda_smooth > 0:
                # Smoothness penalty: penalize high-frequency oscillations
                # Compute second differences along horizon dimension
                if self.config.lambda_smooth > 0:
                    # Δ²y[h] = y[h+1] - 2*y[h] + y[h-1]
                    second_diff = predictions[:, :, 2:, :] - 2 * predictions[:, :, 1:-1, :] + predictions[:, :, :-2, :]
                    smoothness_loss = (second_diff ** 2).mean()
                    loss = loss + self.config.lambda_smooth * smoothness_loss

            return loss

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Forward pass (regime-aware for universal model)
            predictions = self._forward_pass(batch)

            # Compute loss
            loss = self.compute_loss(batch, predictions)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.optimizer.step()

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Log periodically
            if batch_idx % self.config.log_every_n_steps == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {self.current_epoch} | Batch {batch_idx}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.6f} | LR: {current_lr:.2e}"
                )

                if self.config.use_wandb:
                    wandb.log({
                        "train/loss_step": loss.item(),
                        "train/lr": current_lr,
                        "train/global_step": self.global_step,
                    })

        avg_loss = total_loss / num_batches
        epoch_time = time.time() - start_time

        metrics = {
            "loss": avg_loss,
            "time": epoch_time,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        return metrics

    def validate(self) -> dict[str, float]:
        """Run validation.

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Forward pass (regime-aware for universal model)
                predictions = self._forward_pass(batch)

                # Compute loss
                loss = self.compute_loss(batch, predictions)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches

        metrics = {
            "loss": avg_loss,
        }

        return metrics

    def fit(self, n_epochs: int | None = None) -> dict[str, list]:
        """Full training loop.

        Args:
            n_epochs: Number of epochs to train (uses config.epochs if None)

        Returns:
            Training history dictionary
        """
        if n_epochs is None:
            n_epochs = self.config.epochs

        logger.info(f"Starting training for {n_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Checkpoint dir: {self.checkpoint_dir}")

        for epoch in range(n_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            logger.info(
                f"Epoch {epoch}/{n_epochs} | Train Loss: {train_metrics['loss']:.6f} | "
                f"Time: {train_metrics['time']:.1f}s"
            )

            # Validate
            val_metrics = self.validate()
            logger.info(f"Epoch {epoch}/{n_epochs} | Val Loss: {val_metrics['loss']:.6f}")

            # Record history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["lr"].append(train_metrics["lr"])
            self.history["epoch"].append(epoch)

            # Log to wandb
            if self.config.use_wandb:
                wandb.log({
                    "train/loss_epoch": train_metrics["loss"],
                    "val/loss": val_metrics["loss"],
                    "epoch": epoch,
                })

            # Check for improvement
            val_loss = val_metrics["loss"]
            if val_loss < self.best_val_loss - self.config.min_delta:
                logger.info(f"Validation loss improved: {self.best_val_loss:.6f} -> {val_loss:.6f}")
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0

                # Save best checkpoint
                if self.config.save_best:
                    self.save_checkpoint("best.pt", is_best=True)
            else:
                self.epochs_without_improvement += 1
                logger.info(
                    f"No improvement for {self.epochs_without_improvement}/{self.config.patience} epochs"
                )

            # Early stopping
            if self.epochs_without_improvement >= self.config.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

            # Periodic checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

        # Final checkpoint
        self.save_checkpoint("final.pt")

        # Save training history
        self._save_history()

        logger.info("Training complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.6f}")

        return self.history

    def save_checkpoint(self, filename: str, is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename (within checkpoint_dir)
            is_best: Whether this is the best checkpoint
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.config),
            "history": self.history,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        if is_best:
            logger.info(f"  (Best model so far)")

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint.get("history", self.history)

        logger.info(f"Resumed from epoch {self.current_epoch}")
        logger.info(f"Best validation loss: {self.best_val_loss:.6f}")

    def _save_history(self) -> None:
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Saved training history: {history_path}")

        # Also save config
        config_path = self.checkpoint_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        logger.info(f"Saved config: {config_path}")
