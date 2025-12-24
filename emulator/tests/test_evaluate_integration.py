"""Tests for evaluation integration with UniversalEmulator.

This module tests the end-to-end evaluation pipeline:
- Loading UniversalEmulator from checkpoint
- Computing predictions on dataset splits
- Metric computation and aggregation
"""

import json

import numpy as np
import pytest
import torch

from emulator.eval.evaluate import (
    compute_all_metrics,
    compute_predictions,
    evaluate_on_split,
    load_universal_model,
)
from emulator.models.universal import UniversalEmulator
from emulator.training.config import UniversalModelConfig, UniversalTrainingConfig


@pytest.mark.fast
def test_compute_all_metrics():
    """Test metric computation on toy arrays."""
    # Create toy IRFs
    H = 10
    n_samples = 5
    n_obs = 3

    rng = np.random.default_rng(42)
    y_true = rng.normal(0, 1, (n_samples, H + 1, n_obs))
    y_pred = y_true + rng.normal(0, 0.1, (n_samples, H + 1, n_obs))  # Small error

    # Compute metrics
    metrics = compute_all_metrics(y_pred, y_true, weight_scheme="uniform")

    # Check structure
    assert "nrmse" in metrics
    assert "iae" in metrics
    assert "sign_at_impact" in metrics
    assert "hf_ratio" in metrics
    assert "overshoot_ratio" in metrics
    assert "sign_flip_count" in metrics

    # Check ranges
    assert metrics["nrmse"] >= 0.0
    assert metrics["iae"] >= 0.0
    assert 0.0 <= metrics["sign_at_impact"] <= 1.0
    assert metrics["hf_ratio"] >= 0.0

    # With small error, NRMSE should be small
    assert metrics["nrmse"] < 0.5


@pytest.mark.fast
def test_compute_predictions_shape():
    """Test that compute_predictions returns correct shape."""
    # Create minimal model
    world_ids = ["lss"]
    param_dims = {"lss": 36}

    model = UniversalEmulator(
        world_ids=world_ids,
        param_dims=param_dims,
        world_embed_dim=8,
        theta_embed_dim=16,
        shock_embed_dim=8,
        trunk_dim=32,
        trunk_layers=1,
        H=10,
        n_obs=3,
        use_history_encoder=False,
        dropout=0.0,
    )
    model.eval()

    # Create toy parameters
    n_samples = 5
    n_params = 36
    theta = np.random.randn(n_samples, n_params).astype(np.float32)

    # Compute predictions
    with torch.no_grad():
        y_pred = compute_predictions(
            model=model,
            theta=theta,
            world_id="lss",
            shock_idx=0,
            H=10,
            batch_size=2,
            regime="A",
        )

    # Check shape
    assert y_pred.shape == (n_samples, 11, 3)  # (n_samples, H+1, n_obs)
    assert isinstance(y_pred, np.ndarray)


@pytest.mark.fast
def test_load_universal_model_missing_checkpoint(tmp_path):
    """Test that load_universal_model raises error for missing checkpoint."""
    checkpoint_path = tmp_path / "nonexistent.pt"

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_universal_model(checkpoint_path)


@pytest.mark.fast
def test_load_universal_model_missing_keys(tmp_path):
    """Test that load_universal_model raises error for invalid checkpoint."""
    checkpoint_path = tmp_path / "invalid.pt"

    # Save invalid checkpoint (missing required keys)
    torch.save({"invalid": "data"}, checkpoint_path)

    with pytest.raises(KeyError, match="Checkpoint missing"):
        load_universal_model(checkpoint_path)


@pytest.mark.fast
def test_load_universal_model_valid_checkpoint(tmp_path):
    """Test loading UniversalEmulator from valid checkpoint."""
    checkpoint_path = tmp_path / "valid.pt"

    # Create minimal model
    world_ids = ["lss", "var"]
    param_dims = {"lss": 36, "var": 15}

    config = UniversalTrainingConfig(
        model=UniversalModelConfig(
            world_embed_dim=8,
            theta_embed_dim=16,
            shock_embed_dim=8,
            history_embed_dim=32,
            trunk_dim=32,
            trunk_layers=1,
            max_horizon=10,
            n_observables=3,
        ),
        worlds=world_ids,
        regime="A",
    )

    model = UniversalEmulator(
        world_ids=world_ids,
        param_dims=param_dims,
        world_embed_dim=config.model.world_embed_dim,
        theta_embed_dim=config.model.theta_embed_dim,
        shock_embed_dim=config.model.shock_embed_dim,
        history_embed_dim=config.model.history_embed_dim,
        trunk_dim=config.model.trunk_dim,
        trunk_layers=config.model.trunk_layers,
        H=config.model.max_horizon,
        n_obs=config.model.n_observables,
        use_history_encoder=False,
        dropout=0.0,
    )

    # Save checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "param_dims": param_dims,
        "epoch": 10,
        "val_loss": 0.1,
    }
    torch.save(checkpoint, checkpoint_path)

    # Load model
    loaded_model = load_universal_model(checkpoint_path)

    # Check model loaded correctly
    assert isinstance(loaded_model, UniversalEmulator)
    assert loaded_model.world_ids == world_ids
    assert loaded_model.H == 10
    assert not loaded_model.training  # Should be in eval mode

    # Check parameters match
    for p1, p2 in zip(model.parameters(), loaded_model.parameters(), strict=False):
        assert torch.allclose(p1, p2)


@pytest.mark.slow
@pytest.mark.integration
def test_evaluate_on_split_oracle_mode(tmp_path):
    """Test evaluate_on_split in oracle mode (no model).

    This creates a minimal synthetic dataset and runs evaluation.
    """
    # Create synthetic dataset
    dataset_path = tmp_path / "dataset"
    world_id = "lss"
    world_dir = dataset_path / world_id
    world_dir.mkdir(parents=True)

    # Create manifest
    manifest = {
        "version": "1.0.0",
        "simulators": {world_id: {"n_samples": 10}},
        "splits": {
            world_id: {
                "test_interpolation": list(range(10)),
            }
        },
    }
    with open(dataset_path / "manifest.json", "w") as f:
        json.dump(manifest, f)

    # Create synthetic data
    import zarr

    n_samples = 10
    n_shocks = 3
    H = 10
    n_obs = 3
    n_params = 36

    # IRFs
    irfs = np.random.randn(n_samples, n_shocks, H + 1, n_obs).astype(np.float32)
    irfs_store = zarr.open(str(world_dir / "irfs.zarr"), mode="w", shape=irfs.shape, dtype=irfs.dtype)
    irfs_store[:] = irfs

    # Parameters
    theta = np.random.randn(n_samples, n_params).astype(np.float32)
    theta_store = zarr.open(str(world_dir / "theta.zarr"), mode="w", shape=theta.shape, dtype=theta.dtype)
    theta_store[:] = theta

    # Shocks (dummy)
    shocks = np.random.randn(n_samples, 100, n_shocks).astype(np.float32)
    shocks_store = zarr.open(str(world_dir / "shocks.zarr"), mode="w", shape=shocks.shape, dtype=shocks.dtype)
    shocks_store[:] = shocks

    # Run evaluation in oracle mode
    metrics = evaluate_on_split(
        model=None,  # Oracle mode
        dataset_path=dataset_path,
        world_id=world_id,
        split_name="test_interpolation",
        shock_idx=0,
        weight_scheme="uniform",
    )

    # Check structure
    assert "nrmse" in metrics
    assert "iae" in metrics
    assert "sign_at_impact" in metrics
    assert "n_samples" in metrics
    assert metrics["n_samples"] == 10
    assert metrics["world_id"] == world_id

    # In oracle mode (y_pred == y_true), NRMSE should be 0
    assert metrics["nrmse"] == 0.0
    assert metrics["iae"] == 0.0


@pytest.mark.fast
def test_metric_aggregation():
    """Test that metrics aggregate correctly across samples."""
    # Create IRFs with known properties
    H = 10
    n_samples = 10
    n_obs = 3

    # True: simple exponential decay
    h_grid = np.arange(H + 1)
    y_true = np.zeros((n_samples, H + 1, n_obs))
    for i in range(n_samples):
        for j in range(n_obs):
            y_true[i, :, j] = np.exp(-h_grid / 5)

    # Pred: same as true (perfect prediction)
    y_pred = y_true.copy()

    metrics = compute_all_metrics(y_pred, y_true, weight_scheme="uniform")

    # Perfect prediction should give zero error
    assert metrics["nrmse"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["iae"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["sign_at_impact"] == 1.0  # All signs correct


@pytest.mark.fast
def test_different_weight_schemes():
    """Test that different weight schemes produce different NRMSE values."""
    H = 20
    n_samples = 5
    n_obs = 3

    rng = np.random.default_rng(42)
    y_true = rng.normal(0, 1, (n_samples, H + 1, n_obs))

    # Add more error at later horizons
    errors = rng.normal(0, 0.1, (n_samples, H + 1, n_obs))
    for h in range(H + 1):
        errors[:, h, :] *= (1 + h / H)  # Increasing error with horizon

    y_pred = y_true + errors

    metrics_uniform = compute_all_metrics(y_pred, y_true, weight_scheme="uniform")
    metrics_exponential = compute_all_metrics(y_pred, y_true, weight_scheme="exponential")

    # Both should be non-zero
    assert metrics_uniform["nrmse"] > 0
    assert metrics_exponential["nrmse"] > 0

    # Exponential weighting should give lower NRMSE (downweights later horizons)
    assert metrics_exponential["nrmse"] < metrics_uniform["nrmse"]
