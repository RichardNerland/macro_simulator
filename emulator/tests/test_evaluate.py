"""
Unit tests for evaluation harness.

Tests the evaluation CLI and metric computation pipeline.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from emulator.eval.evaluate import (
    aggregate_metrics,
    compute_all_metrics,
    save_results,
)


@pytest.mark.fast
def test_compute_all_metrics_shape():
    """Test that compute_all_metrics returns expected keys."""
    # Create toy IRF data
    n_samples = 10
    H = 40
    n_obs = 3

    y_pred = np.random.randn(n_samples, H + 1, n_obs)
    y_true = np.random.randn(n_samples, H + 1, n_obs)

    # Compute metrics
    metrics = compute_all_metrics(y_pred, y_true, weight_scheme="uniform")

    # Check keys
    expected_keys = {
        "nrmse",
        "nrmse_weighted",
        "iae",
        "sign_at_impact",
        "hf_ratio",
        "overshoot_ratio",
        "sign_flip_count",
    }

    assert set(metrics.keys()) == expected_keys

    # Check all are floats
    for key, value in metrics.items():
        assert isinstance(value, float)


@pytest.mark.fast
def test_compute_all_metrics_perfect_prediction():
    """Test that perfect prediction gives zero error metrics."""
    n_samples = 5
    H = 20
    n_obs = 3

    y_true = np.random.randn(n_samples, H + 1, n_obs)
    y_pred = y_true.copy()

    metrics = compute_all_metrics(y_pred, y_true, weight_scheme="uniform")

    # Error metrics should be near zero
    assert metrics["nrmse"] < 1e-6
    assert metrics["iae"] < 1e-6

    # Sign accuracy should be 1.0 (ignoring zeros)
    # Note: sign(0) == 0, so we may not get exactly 1.0
    assert metrics["sign_at_impact"] >= 0.9


@pytest.mark.fast
def test_compute_all_metrics_multishock():
    """Test that multi-shock IRFs are handled correctly."""
    n_samples = 8
    n_shocks = 4
    H = 30
    n_obs = 3

    y_pred = np.random.randn(n_samples, n_shocks, H + 1, n_obs)
    y_true = np.random.randn(n_samples, n_shocks, H + 1, n_obs)

    # Should average over shocks
    metrics = compute_all_metrics(y_pred, y_true, weight_scheme="uniform")

    assert isinstance(metrics["nrmse"], float)
    assert metrics["nrmse"] >= 0.0


@pytest.mark.fast
def test_aggregate_metrics_empty():
    """Test aggregation with empty list."""
    result = aggregate_metrics([])
    assert result == {}


@pytest.mark.fast
def test_aggregate_metrics_single():
    """Test aggregation with single metric dict."""
    metrics = {
        "nrmse": 0.15,
        "iae": 2.3,
        "sign_at_impact": 0.95,
        "hf_ratio": 0.12,
        "overshoot_ratio": 1.5,
        "sign_flip_count": 3.2,
        "n_samples": 100,
    }

    result = aggregate_metrics([metrics])

    # Should return same values
    assert result["nrmse"] == 0.15
    assert result["iae"] == 2.3
    assert result["n_samples"] == 100


@pytest.mark.fast
def test_aggregate_metrics_weighted():
    """Test weighted aggregation."""
    metrics_1 = {
        "nrmse": 0.10,
        "iae": 2.0,
        "sign_at_impact": 0.90,
        "hf_ratio": 0.10,
        "overshoot_ratio": 1.2,
        "sign_flip_count": 2.0,
        "n_samples": 100,
    }

    metrics_2 = {
        "nrmse": 0.20,
        "iae": 4.0,
        "sign_at_impact": 0.80,
        "hf_ratio": 0.20,
        "overshoot_ratio": 1.8,
        "sign_flip_count": 4.0,
        "n_samples": 100,
    }

    result = aggregate_metrics([metrics_1, metrics_2])

    # Should be average (equal weights)
    assert result["nrmse"] == pytest.approx(0.15)
    assert result["iae"] == pytest.approx(3.0)
    assert result["sign_at_impact"] == pytest.approx(0.85)
    assert result["n_samples"] == 200


@pytest.mark.fast
def test_aggregate_metrics_unequal_weights():
    """Test weighted aggregation with unequal sample sizes."""
    metrics_1 = {
        "nrmse": 0.10,
        "iae": 2.0,
        "sign_at_impact": 0.90,
        "hf_ratio": 0.10,
        "overshoot_ratio": 1.2,
        "sign_flip_count": 2.0,
        "n_samples": 300,  # 3x weight
    }

    metrics_2 = {
        "nrmse": 0.40,
        "iae": 8.0,
        "sign_at_impact": 0.60,
        "hf_ratio": 0.40,
        "overshoot_ratio": 2.4,
        "sign_flip_count": 8.0,
        "n_samples": 100,  # 1x weight
    }

    result = aggregate_metrics([metrics_1, metrics_2])

    # Weighted average: (0.10 * 300 + 0.40 * 100) / 400 = 0.175
    assert result["nrmse"] == pytest.approx(0.175)
    assert result["n_samples"] == 400


@pytest.mark.fast
def test_save_results():
    """Test saving results to JSON."""
    results = {
        "aggregate": {"nrmse": 0.15, "iae": 2.3},
        "per_world": {"lss": {"nrmse": 0.12}, "var": {"nrmse": 0.18}},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "results.json"

        save_results(results, output_path)

        assert output_path.exists()

        # Load and verify
        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["aggregate"]["nrmse"] == 0.15
        assert loaded["per_world"]["lss"]["nrmse"] == 0.12


@pytest.mark.fast
def test_save_results_creates_directory():
    """Test that save_results creates parent directories."""
    results = {"aggregate": {"nrmse": 0.15}}

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "nested" / "dir" / "results.json"

        save_results(results, output_path)

        assert output_path.exists()
        assert output_path.parent.exists()


@pytest.mark.fast
def test_compute_all_metrics_exponential_weights():
    """Test metrics with exponential weighting."""
    n_samples = 5
    H = 20
    n_obs = 3

    y_pred = np.random.randn(n_samples, H + 1, n_obs)
    y_true = np.random.randn(n_samples, H + 1, n_obs)

    metrics_uniform = compute_all_metrics(y_pred, y_true, weight_scheme="uniform")
    metrics_exp = compute_all_metrics(y_pred, y_true, weight_scheme="exponential")

    # Both should return valid metrics
    assert metrics_uniform["nrmse"] > 0
    assert metrics_exp["nrmse"] > 0

    # Exponential weights emphasize short horizon, so may differ
    # (but not guaranteed to be higher or lower in general)
    assert metrics_exp["nrmse"] >= 0.0
