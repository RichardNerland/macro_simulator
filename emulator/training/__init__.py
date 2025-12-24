"""Training infrastructure for the Universal Macro Emulator.

This module provides:
- Trainer: Main training loop with early stopping, checkpointing, and logging
- TrainingConfig: Configuration dataclass for reproducible training (legacy)
- UniversalTrainingConfig: Regime-aware configuration for universal emulator
- BaselineTrainingConfig: Configuration for baseline models
- IRFDataset: Dataset wrapper for Zarr-backed IRF data
- Loss functions: MultiHorizonLoss, SmoothnessRegularization, CombinedLoss
"""

from emulator.training.config import (
    BaselineModelConfig,
    BaselineTrainingConfig,
    UniversalModelConfig,
    UniversalTrainingConfig,
)
from emulator.training.dataset import (
    IRFDataset,
    collate_mixed_worlds,
    collate_single_world,
)
from emulator.training.losses import (
    CombinedLoss,
    MultiHorizonLoss,
    SmoothnessRegularization,
)
from emulator.training.trainer import Trainer, TrainingConfig

__all__ = [
    # Datasets
    "IRFDataset",
    "collate_mixed_worlds",
    "collate_single_world",
    # Training
    "Trainer",
    "TrainingConfig",  # Legacy
    # Configs
    "UniversalTrainingConfig",
    "BaselineTrainingConfig",
    "UniversalModelConfig",
    "BaselineModelConfig",
    # Losses
    "MultiHorizonLoss",
    "SmoothnessRegularization",
    "CombinedLoss",
]
