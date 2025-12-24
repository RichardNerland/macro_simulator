"""Tests for figures.py visualization functions.

Tests that all figure generation functions work correctly and produce
valid matplotlib figure objects.
"""

import numpy as np
import pytest

from emulator.eval.figures import (
    plot_irf_panel,
    plot_nrmse_bar_chart,
    plot_regime_comparison,
    plot_shape_comparison,
)


@pytest.mark.fast
def test_plot_nrmse_bar_chart(tmp_path):
    """Test NRMSE bar chart generation."""
    # Create mock results
    results = {
        "per_world": {
            "lss": {"nrmse": 0.15, "iae": 1.2},
            "var": {"nrmse": 0.12, "iae": 1.0},
            "nk": {"nrmse": 0.18, "iae": 1.5},
        }
    }

    baseline_results = {
        "MLP": {
            "per_world": {
                "lss": {"nrmse": 0.20},
                "var": {"nrmse": 0.18},
                "nk": {"nrmse": 0.22},
            }
        }
    }

    output_path = tmp_path / "nrmse_bar_chart.png"

    # Generate figure
    fig = plot_nrmse_bar_chart(results, output_path, baseline_results)

    # Check figure was created
    assert fig is not None
    assert output_path.exists()

    # Clean up
    import matplotlib.pyplot as plt
    plt.close(fig)


@pytest.mark.fast
def test_plot_nrmse_bar_chart_no_baselines(tmp_path):
    """Test NRMSE bar chart without baselines."""
    results = {
        "per_world": {
            "lss": {"nrmse": 0.15},
            "var": {"nrmse": 0.12},
        }
    }

    output_path = tmp_path / "nrmse_no_baselines.png"
    fig = plot_nrmse_bar_chart(results, output_path)

    assert fig is not None
    assert output_path.exists()

    import matplotlib.pyplot as plt
    plt.close(fig)


@pytest.mark.fast
def test_plot_shape_comparison(tmp_path):
    """Test shape metric comparison plot."""
    results = {
        "per_world": {
            "lss": {
                "hf_ratio": 1.2,
                "overshoot_ratio": 0.3,
                "sign_flip_count": 1.5,
            },
            "var": {
                "hf_ratio": 1.1,
                "overshoot_ratio": 0.2,
                "sign_flip_count": 1.0,
            },
        }
    }

    output_path = tmp_path / "shape_comparison.png"
    fig = plot_shape_comparison(results, output_path)

    assert fig is not None
    assert output_path.exists()

    import matplotlib.pyplot as plt
    plt.close(fig)


@pytest.mark.fast
def test_plot_regime_comparison(tmp_path):
    """Test regime comparison plot."""
    results_A = {
        "per_world": {
            "lss": {"nrmse": 0.10},
            "var": {"nrmse": 0.12},
        }
    }

    results_B1 = {
        "per_world": {
            "lss": {"nrmse": 0.15},
            "var": {"nrmse": 0.18},
        }
    }

    results_C = {
        "per_world": {
            "lss": {"nrmse": 0.12},
            "var": {"nrmse": 0.14},
        }
    }

    output_path = tmp_path / "regime_comparison.png"
    fig = plot_regime_comparison(results_A, results_B1, results_C, output_path)

    assert fig is not None
    assert output_path.exists()

    import matplotlib.pyplot as plt
    plt.close(fig)


@pytest.mark.fast
def test_plot_regime_comparison_single_regime(tmp_path):
    """Test regime comparison with only Regime A."""
    results_A = {
        "per_world": {
            "lss": {"nrmse": 0.10},
            "var": {"nrmse": 0.12},
        }
    }

    output_path = tmp_path / "regime_A_only.png"
    fig = plot_regime_comparison(results_A, output_path=output_path)

    assert fig is not None
    assert output_path.exists()

    import matplotlib.pyplot as plt
    plt.close(fig)


@pytest.mark.fast
def test_plot_irf_panel_single_sample(tmp_path):
    """Test IRF panel plot with single sample."""
    H = 20
    n_obs = 3

    # Create toy IRFs
    rng = np.random.default_rng(42)
    y_true = rng.normal(0, 1, (H + 1, n_obs)).astype(np.float32)
    y_pred = y_true + rng.normal(0, 0.1, (H + 1, n_obs)).astype(np.float32)

    output_path = tmp_path / "irf_panel_single.png"
    fig = plot_irf_panel(
        y_pred=y_pred,
        y_true=y_true,
        world_id="lss",
        output_path=output_path,
    )

    assert fig is not None
    assert output_path.exists()

    import matplotlib.pyplot as plt
    plt.close(fig)


@pytest.mark.fast
def test_plot_irf_panel_batch(tmp_path):
    """Test IRF panel plot with batch of samples."""
    n_samples = 10
    H = 20
    n_obs = 3

    # Create toy IRFs
    rng = np.random.default_rng(42)
    y_true = rng.normal(0, 1, (n_samples, H + 1, n_obs)).astype(np.float32)
    y_pred = y_true + rng.normal(0, 0.1, (n_samples, H + 1, n_obs)).astype(np.float32)

    output_path = tmp_path / "irf_panel_batch.png"
    fig = plot_irf_panel(
        y_pred=y_pred,
        y_true=y_true,
        world_id="nk",
        output_path=output_path,
        shock_idx=1,
    )

    assert fig is not None
    assert output_path.exists()

    # Should have computed mean and confidence bands
    import matplotlib.pyplot as plt
    plt.close(fig)


@pytest.mark.fast
def test_plot_irf_panel_with_confidence_bands(tmp_path):
    """Test IRF panel plot with custom confidence bands."""
    H = 20
    n_obs = 3

    rng = np.random.default_rng(42)
    y_true = rng.normal(0, 1, (H + 1, n_obs)).astype(np.float32)
    y_pred = y_true + rng.normal(0, 0.1, (H + 1, n_obs)).astype(np.float32)

    # Create confidence bands
    lower = y_pred - 0.2
    upper = y_pred + 0.2
    confidence_bands = (lower, upper)

    output_path = tmp_path / "irf_panel_bands.png"
    fig = plot_irf_panel(
        y_pred=y_pred,
        y_true=y_true,
        world_id="var",
        output_path=output_path,
        confidence_bands=confidence_bands,
    )

    assert fig is not None
    assert output_path.exists()

    import matplotlib.pyplot as plt
    plt.close(fig)


@pytest.mark.fast
def test_plot_nrmse_bar_chart_missing_per_world(tmp_path):
    """Test that NRMSE bar chart raises error if per_world missing."""
    results = {"aggregate": {"nrmse": 0.15}}  # Missing per_world

    output_path = tmp_path / "invalid.png"

    with pytest.raises(ValueError, match="per_world"):
        plot_nrmse_bar_chart(results, output_path)


@pytest.mark.fast
def test_plot_shape_comparison_missing_per_world(tmp_path):
    """Test that shape comparison raises error if per_world missing."""
    results = {"aggregate": {"hf_ratio": 1.2}}  # Missing per_world

    output_path = tmp_path / "invalid.png"

    with pytest.raises(ValueError, match="per_world"):
        plot_shape_comparison(results, output_path)


@pytest.mark.fast
def test_plot_regime_comparison_missing_per_world(tmp_path):
    """Test that regime comparison raises error if per_world missing."""
    results_A = {"aggregate": {"nrmse": 0.15}}  # Missing per_world

    output_path = tmp_path / "invalid.png"

    with pytest.raises(ValueError, match="per_world"):
        plot_regime_comparison(results_A, output_path=output_path)
