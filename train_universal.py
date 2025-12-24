"""Training script for Universal Macro Emulator.

Usage:
    python train_universal.py --config configs/smoke_test_regime_A.yaml
    python train_universal.py --config configs/universal_regime_A.yaml --epochs 50
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from emulator.models.universal import UniversalEmulator
from emulator.training.dataset import IRFDataset, collate_mixed_worlds
from emulator.training.trainer import Trainer, TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_yaml_config(config_path: str | Path) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary with config
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def create_model_from_config(model_config: dict, world_ids: list[str], dataset: IRFDataset) -> UniversalEmulator:
    """Create UniversalEmulator from config dict.

    Args:
        model_config: Model configuration dict
        world_ids: List of world IDs in the dataset
        dataset: IRFDataset to extract metadata

    Returns:
        UniversalEmulator instance
    """
    # Get param_dims for each world
    param_dims = {}
    max_shocks = 0
    for world_id in world_ids:
        info = dataset.get_world_info(world_id)
        param_dims[world_id] = info["n_params"]
        max_shocks = max(max_shocks, info["n_shocks"])

    # Create model
    model = UniversalEmulator(
        world_ids=world_ids,
        param_dims=param_dims,
        world_embed_dim=model_config.get("world_embed_dim", 32),
        theta_embed_dim=model_config.get("theta_embed_dim", 64),
        shock_embed_dim=model_config.get("shock_embed_dim", 16),
        history_embed_dim=model_config.get("eps_embed_dim", 32),  # Use eps_embed_dim for history
        trunk_dim=model_config.get("trunk_dim", 256),
        trunk_layers=model_config.get("trunk_layers", 4),
        trunk_architecture=model_config.get("encoder_type", "mlp"),  # Map encoder_type to trunk_architecture
        dropout=model_config.get("dropout", 0.1),
        H=model_config.get("H", 40),
        n_obs=model_config.get("n_obs", 3),
        max_shocks=max_shocks,
    )

    return model


def create_trainer_config(config: dict) -> TrainingConfig:
    """Create TrainingConfig from YAML dict.

    Args:
        config: Full YAML config dict with nested sections

    Returns:
        TrainingConfig instance
    """
    training = config.get("training", {})
    loss = config.get("loss", {})

    # Map YAML config to TrainingConfig
    trainer_config = TrainingConfig(
        # Training
        regime=training.get("regime", "A"),
        lr=float(training.get("lr", 1e-4)),
        weight_decay=training.get("weight_decay", 0.01),
        batch_size=training.get("batch_size", 64),
        epochs=training.get("epochs", 100),
        grad_clip=training.get("grad_clip", 1.0),

        # Schedule
        warmup_epochs=training.get("warmup_epochs", 5),
        scheduler=training.get("scheduler", "cosine"),

        # Early stopping
        patience=training.get("patience", 10),
        min_delta=training.get("min_delta", 1e-4),

        # Checkpointing
        checkpoint_dir=training.get("checkpoint_dir", "runs/default"),
        save_every_n_epochs=training.get("save_every_n_epochs", 10),
        save_best=training.get("save_best", True),

        # Logging
        log_every_n_steps=training.get("log_every_n_steps", 50),
        use_wandb=training.get("use_wandb", False),
        wandb_project=training.get("wandb_project", "macro-emulator"),
        wandb_run_name=training.get("wandb_run_name", None),

        # Loss
        use_advanced_loss=loss.get("type") == "multi_horizon" or loss.get("type") == "combined",
        horizon_weights=loss.get("weight_scheme", "uniform"),
        horizon_decay_tau=loss.get("tau", 20.0),
        lambda_smooth=loss.get("smoothness_lambda", 0.01),

        # Reproducibility
        seed=config.get("seed", 42),
        device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )

    return trainer_config


def main():
    parser = argparse.ArgumentParser(description="Train Universal Macro Emulator")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda)")
    args = parser.parse_args()

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_yaml_config(args.config)

    # Override epochs if specified
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
        logger.info(f"Overriding epochs to {args.epochs}")

    # Override device if specified
    if args.device is not None:
        config["device"] = args.device
        logger.info(f"Overriding device to {args.device}")

    # Extract config sections
    data_config = config.get("data", {})
    dataset_path = data_config.get("dataset_path", "datasets/v1.0-dev/")
    world_ids = data_config.get("worlds", ["lss", "var", "nk"])
    train_split = data_config.get("train_split", "train")
    val_split = data_config.get("val_split", "val")
    num_workers = data_config.get("num_workers", 0)
    pin_memory = data_config.get("pin_memory", False)
    shuffle_train = data_config.get("shuffle_train", True)

    # Create datasets
    logger.info(f"Loading dataset from {dataset_path}")
    logger.info(f"Worlds: {world_ids}")

    train_dataset = IRFDataset(
        zarr_root=dataset_path,
        world_ids=world_ids,
        split=train_split,
    )

    val_dataset = IRFDataset(
        zarr_root=dataset_path,
        world_ids=world_ids,
        split=val_split,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_mixed_worlds,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_mixed_worlds,
    )

    # Create model
    logger.info("Creating UniversalEmulator")
    model = create_model_from_config(config["model"], world_ids, train_dataset)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Create trainer config
    trainer_config = create_trainer_config(config)

    # Create trainer
    logger.info("Creating Trainer")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
    )

    # Train
    logger.info("Starting training")
    logger.info(f"Regime: {trainer_config.regime}")
    logger.info(f"Device: {trainer_config.device}")
    logger.info(f"Checkpoint dir: {trainer_config.checkpoint_dir}")

    history = trainer.fit()

    logger.info("Training complete!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.6f}")

    # Print final metrics
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Regime: {trainer_config.regime}")
    print(f"Epochs trained: {len(history['train_loss'])}")
    print(f"Best val loss: {trainer.best_val_loss:.6f}")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    print(f"Checkpoint dir: {trainer_config.checkpoint_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
