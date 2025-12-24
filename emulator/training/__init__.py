"""Training infrastructure for the Universal Macro Emulator."""

from emulator.training.dataset import (
    IRFDataset,
    collate_mixed_worlds,
    collate_single_world,
)
from emulator.training.trainer import Trainer, TrainingConfig

__all__ = [
    "IRFDataset",
    "collate_mixed_worlds",
    "collate_single_world",
    "Trainer",
    "TrainingConfig",
]
