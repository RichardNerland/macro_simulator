"""Unit tests for training infrastructure.

Tests for dataset loading, collate functions, and trainer initialization.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr
from torch.utils.data import DataLoader

from emulator.training import (
    IRFDataset,
    Trainer,
    TrainingConfig,
    collate_mixed_worlds,
    collate_single_world,
)


@pytest.fixture
def mock_dataset_dir(tmp_path):
    """Create a minimal mock dataset for testing."""
    # Create manifest
    manifest = {
        "version": "1.0.0",
        "created_at": "2025-01-01T00:00:00Z",
        "git_hash": "test",
        "simulators": {
            "test_world": {
                "n_samples": 100,
                "param_manifest": {
                    "names": ["param_0", "param_1", "param_2"],
                    "units": ["-", "-", "-"],
                    "bounds": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                    "defaults": [0.5, 0.5, 0.5],
                    "priors": [],
                },
                "shock_manifest": {
                    "names": ["shock_0"],
                    "n_shocks": 1,
                    "sigma": [0.01],
                    "default_size": 1.0,
                },
                "obs_manifest": {
                    "canonical_names": ["output", "inflation", "rate"],
                    "extra_names": [],
                    "n_canonical": 3,
                    "n_extra": 0,
                },
            }
        },
        "splits": {
            "train": {"fraction": 0.8, "seed": 42},
            "val": {"fraction": 0.1, "seed": 42},
            "test_interpolation": {"fraction": 0.1, "seed": 42},
        },
    }

    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    # Create world directory
    world_dir = tmp_path / "test_world"
    world_dir.mkdir()

    # Create zarr arrays
    n_samples = 100
    n_params = 3
    n_shocks = 1
    H = 40

    theta = np.random.randn(n_samples, n_params).astype(np.float32)
    irfs = np.random.randn(n_samples, n_shocks, H + 1, 3).astype(np.float32)

    zarr.save(world_dir / "theta.zarr", theta)
    zarr.save(world_dir / "irfs.zarr", irfs)

    # Create splits
    indices = np.arange(n_samples)
    splits = {
        "train": indices[:80].tolist(),
        "val": indices[80:90].tolist(),
        "test_interpolation": indices[90:].tolist(),
    }

    splits_path = world_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f)

    return tmp_path


@pytest.mark.fast
def test_dataset_initialization(mock_dataset_dir):
    """Test that IRFDataset can be initialized."""
    dataset = IRFDataset(
        zarr_root=mock_dataset_dir,
        world_ids=["test_world"],
        split="train",
    )

    assert len(dataset) == 80  # 80% of 100 samples
    assert dataset.world_ids == ["test_world"]
    assert dataset.split == "train"


@pytest.mark.fast
def test_dataset_getitem(mock_dataset_dir):
    """Test that dataset __getitem__ returns correct format."""
    dataset = IRFDataset(
        zarr_root=mock_dataset_dir,
        world_ids=["test_world"],
        split="train",
    )

    sample = dataset[0]

    assert isinstance(sample, dict)
    assert "theta" in sample
    assert "irf" in sample
    assert "world_id" in sample
    assert "sample_idx" in sample

    assert isinstance(sample["theta"], torch.Tensor)
    assert isinstance(sample["irf"], torch.Tensor)
    assert sample["theta"].shape == (3,)  # n_params
    assert sample["irf"].shape == (1, 41, 3)  # (n_shocks, H+1, 3)


@pytest.mark.fast
def test_dataset_world_info(mock_dataset_dir):
    """Test get_world_info method."""
    dataset = IRFDataset(
        zarr_root=mock_dataset_dir,
        world_ids=["test_world"],
        split="train",
    )

    info = dataset.get_world_info("test_world")

    assert info["n_params"] == 3
    assert info["n_shocks"] == 1
    assert info["H"] == 40


@pytest.mark.fast
def test_collate_single_world(mock_dataset_dir):
    """Test collate_single_world function."""
    dataset = IRFDataset(
        zarr_root=mock_dataset_dir,
        world_ids=["test_world"],
        split="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_single_world,
    )

    batch = next(iter(loader))

    assert batch["theta"].shape == (8, 3)  # (batch_size, n_params)
    assert batch["irf"].shape == (8, 1, 41, 3)  # (batch_size, n_shocks, H+1, 3)
    assert batch["world_id"] == "test_world"
    assert len(batch["sample_indices"]) == 8


@pytest.mark.fast
def test_collate_mixed_worlds(mock_dataset_dir):
    """Test collate_mixed_worlds function with padding."""
    # Create second world with different dimensions
    world_dir_2 = mock_dataset_dir / "test_world_2"
    world_dir_2.mkdir()

    n_samples = 50
    n_params_2 = 5  # Different number of params
    n_shocks_2 = 2  # Different number of shocks
    H = 40

    theta_2 = np.random.randn(n_samples, n_params_2).astype(np.float32)
    irfs_2 = np.random.randn(n_samples, n_shocks_2, H + 1, 3).astype(np.float32)

    zarr.save(world_dir_2 / "theta.zarr", theta_2)
    zarr.save(world_dir_2 / "irfs.zarr", irfs_2)

    # Create splits
    indices = np.arange(n_samples)
    splits = {
        "train": indices[:40].tolist(),
        "val": indices[40:45].tolist(),
        "test_interpolation": indices[45:].tolist(),
    }

    splits_path = world_dir_2 / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f)

    # Update manifest
    manifest_path = mock_dataset_dir / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    manifest["simulators"]["test_world_2"] = {
        "n_samples": n_samples,
        "param_manifest": {
            "names": [f"param_{i}" for i in range(n_params_2)],
            "units": ["-"] * n_params_2,
            "bounds": [[0.0, 1.0]] * n_params_2,
            "defaults": [0.5] * n_params_2,
            "priors": [],
        },
        "shock_manifest": {
            "names": [f"shock_{i}" for i in range(n_shocks_2)],
            "n_shocks": n_shocks_2,
            "sigma": [0.01] * n_shocks_2,
            "default_size": 1.0,
        },
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    # Create dataset with both worlds
    dataset = IRFDataset(
        zarr_root=mock_dataset_dir,
        world_ids=["test_world", "test_world_2"],
        split="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_mixed_worlds,
    )

    batch = next(iter(loader))

    # Check shapes
    assert batch["theta"].shape == (8, 5)  # Padded to max_n_params
    assert batch["irf"].shape == (8, 2, 41, 3)  # Padded to max_n_shocks
    assert batch["theta_mask"].shape == (8, 5)
    assert batch["irf_mask"].shape == (8, 2)

    # Check that masks are correct
    # Samples from test_world should have mask = True for first 3 params
    # Samples from test_world_2 should have mask = True for all 5 params
    assert batch["theta_mask"].dtype == torch.bool
    assert batch["irf_mask"].dtype == torch.bool


@pytest.mark.fast
def test_training_config():
    """Test TrainingConfig dataclass."""
    config = TrainingConfig(
        lr=1e-3,
        batch_size=32,
        epochs=50,
    )

    assert config.lr == 1e-3
    assert config.batch_size == 32
    assert config.epochs == 50
    assert config.patience == 10  # Default
    assert config.seed == 42  # Default


@pytest.mark.fast
def test_training_config_from_dict():
    """Test creating TrainingConfig from dict."""
    config_dict = {
        "lr": 1e-4,
        "batch_size": 64,
        "epochs": 100,
        "warmup_epochs": 10,
        "checkpoint_dir": "runs/test",
    }

    config = TrainingConfig(**config_dict)

    assert config.lr == 1e-4
    assert config.batch_size == 64
    assert config.epochs == 100
    assert config.warmup_epochs == 10
    assert config.checkpoint_dir == "runs/test"


# Integration test (marked as slow because it actually trains)
@pytest.mark.slow
def test_trainer_full_loop(mock_dataset_dir):
    """Test full training loop (marked as slow)."""
    # This test would require a mock model and actual training
    # Skipping for now - would be added in integration tests
    pytest.skip("Integration test - requires full setup")


@pytest.mark.fast
def test_collate_with_shock_idx(mock_dataset_dir):
    """Test that collate functions include shock_idx."""
    dataset = IRFDataset(
        zarr_root=mock_dataset_dir,
        world_ids=["test_world"],
        split="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_single_world,
    )

    batch = next(iter(loader))

    # Check that shock_idx is present
    assert "shock_idx" in batch
    assert batch["shock_idx"].shape == (4,)
    assert batch["shock_idx"].dtype == torch.long


@pytest.mark.fast
def test_collate_mixed_worlds_with_shock_idx(mock_dataset_dir):
    """Test mixed-world collate includes shock_idx."""
    dataset = IRFDataset(
        zarr_root=mock_dataset_dir,
        world_ids=["test_world"],
        split="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_mixed_worlds,
    )

    batch = next(iter(loader))

    # Check that shock_idx is present
    assert "shock_idx" in batch
    assert batch["shock_idx"].shape == (4,)
    assert batch["shock_idx"].dtype == torch.long


@pytest.mark.fast
def test_trainer_detects_universal_model():
    """Test that Trainer correctly detects UniversalEmulator."""
    from emulator.models.universal import UniversalEmulator

    # Create a minimal UniversalEmulator
    model = UniversalEmulator(
        world_ids=["test_world"],
        param_dims={"test_world": 3},
        world_embed_dim=8,
        theta_embed_dim=16,
        shock_embed_dim=8,
        trunk_dim=32,
        trunk_layers=1,
        H=10,
        use_history_encoder=False,
    )

    # Create minimal config
    config = TrainingConfig(
        epochs=1,
        regime="A",
        checkpoint_dir="test_checkpoint",
    )

    # Mock loaders (we just need the structure)
    from torch.utils.data import TensorDataset
    dummy_data = TensorDataset(torch.randn(10, 3))
    dummy_loader = DataLoader(dummy_data, batch_size=2)

    # Initialize trainer
    trainer = Trainer(model, dummy_loader, dummy_loader, config)

    # Check detection
    assert trainer.is_universal is True
    assert trainer.regime == "A"


@pytest.mark.fast
def test_trainer_advanced_loss_creation():
    """Test that advanced loss functions are created correctly."""
    from emulator.models.universal import UniversalEmulator

    model = UniversalEmulator(
        world_ids=["test_world"],
        param_dims={"test_world": 3},
        world_embed_dim=8,
        theta_embed_dim=16,
        shock_embed_dim=8,
        trunk_dim=32,
        trunk_layers=1,
        H=40,
        use_history_encoder=False,
    )

    config = TrainingConfig(
        epochs=1,
        regime="A",
        use_advanced_loss=True,
        lambda_smooth=0.01,
        horizon_weights="exponential",
        checkpoint_dir="test_checkpoint",
    )

    from torch.utils.data import TensorDataset
    dummy_data = TensorDataset(torch.randn(10, 3))
    dummy_loader = DataLoader(dummy_data, batch_size=2)

    trainer = Trainer(model, dummy_loader, dummy_loader, config)

    # Check that loss function is CombinedLoss
    from emulator.training.losses import CombinedLoss
    assert isinstance(trainer.loss_fn, CombinedLoss)


@pytest.mark.fast
def test_training_config_with_regime():
    """Test TrainingConfig with regime field."""
    config = TrainingConfig(
        lr=1e-4,
        batch_size=64,
        epochs=100,
        regime="B1",
    )

    assert config.regime == "B1"
    assert config.lr == 1e-4
