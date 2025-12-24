"""
Unit tests for IRF visualization functions.

Tests cover plot generation, styling, and save functionality.
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from emulator.eval.plots import (
    plot_irf_comparison,
    plot_irf_panel,
    save_figure,
    setup_plot_style,
)


@pytest.mark.fast
def test_setup_plot_style():
    """Test that plot style configuration is applied."""
    setup_plot_style()

    # Check that some key style parameters are set
    assert plt.rcParams["figure.dpi"] == 150
    assert plt.rcParams["savefig.dpi"] == 300
    assert plt.rcParams["axes.grid"] is True


@pytest.mark.fast
def test_plot_irf_panel_2d_single_shock():
    """Test IRF panel plot with 2D IRFs (single shock, multiple simulators)."""
    # Create toy IRF data
    H = 20
    n_obs = 3

    # Simple decaying IRFs
    h = np.arange(H + 1)
    irf1 = np.column_stack([
        np.exp(-0.1 * h),  # output
        0.5 * np.exp(-0.2 * h),  # inflation
        -0.3 * np.exp(-0.15 * h),  # rate
    ])

    irf2 = np.column_stack([
        1.2 * np.exp(-0.12 * h),
        0.4 * np.exp(-0.18 * h),
        -0.25 * np.exp(-0.14 * h),
    ])

    irfs = {
        "Simulator A": irf1,
        "Simulator B": irf2,
    }

    # Generate plot
    fig = plot_irf_panel(irfs, title="Test IRF Panel")

    # Check that figure was created
    assert isinstance(fig, plt.Figure)

    # Check subplot structure: 3 rows (observables) x 2 cols (simulators)
    axes = fig.axes
    assert len(axes) == 6  # 3 rows x 2 cols

    # Clean up
    plt.close(fig)


@pytest.mark.fast
def test_plot_irf_panel_3d_multiple_shocks():
    """Test IRF panel plot with 3D IRFs (multiple shocks, multiple simulators)."""
    H = 15
    n_shocks = 2
    n_obs = 3

    # Create toy data: (n_shocks, H+1, n_obs)
    h = np.arange(H + 1)
    irf1 = np.zeros((n_shocks, H + 1, n_obs))
    irf2 = np.zeros((n_shocks, H + 1, n_obs))

    for shock_idx in range(n_shocks):
        scale = 1.0 + 0.3 * shock_idx
        irf1[shock_idx] = np.column_stack([
            scale * np.exp(-0.1 * h),
            0.5 * scale * np.exp(-0.2 * h),
            -0.3 * scale * np.exp(-0.15 * h),
        ])

        irf2[shock_idx] = np.column_stack([
            0.8 * scale * np.exp(-0.12 * h),
            0.4 * scale * np.exp(-0.18 * h),
            -0.25 * scale * np.exp(-0.14 * h),
        ])

    irfs = {
        "Model A": irf1,
        "Model B": irf2,
    }

    # Test obs_rows layout
    fig = plot_irf_panel(irfs, layout="obs_rows", title="Test 3D IRF Panel")
    assert isinstance(fig, plt.Figure)
    # 3 obs rows x 2 shock cols = 6 subplots
    assert len(fig.axes) == 6
    plt.close(fig)

    # Test shock_rows layout
    fig = plot_irf_panel(irfs, layout="shock_rows", title="Test 3D IRF Panel (Shock Rows)")
    assert isinstance(fig, plt.Figure)
    # 2 shock rows x 3 obs cols = 6 subplots
    assert len(fig.axes) == 6
    plt.close(fig)


@pytest.mark.fast
def test_plot_irf_panel_horizon_truncation():
    """Test that horizon parameter correctly truncates IRF plots."""
    H_full = 40
    H_plot = 20
    n_obs = 3

    h = np.arange(H_full + 1)
    irf = np.column_stack([
        np.exp(-0.1 * h),
        0.5 * np.exp(-0.2 * h),
        -0.3 * np.exp(-0.15 * h),
    ])

    irfs = {"Test": irf}

    fig = plot_irf_panel(irfs, horizon=H_plot)

    # Check that plot was created
    assert isinstance(fig, plt.Figure)

    # Get first axis and check x-data length
    ax = fig.axes[0]
    lines = ax.get_lines()
    assert len(lines) > 0

    # First line should have H_plot + 1 points
    x_data = lines[0].get_xdata()
    assert len(x_data) == H_plot + 1
    assert np.max(x_data) == H_plot

    plt.close(fig)


@pytest.mark.fast
def test_plot_irf_panel_custom_labels():
    """Test IRF panel with custom observable and shock labels."""
    H = 10
    n_shocks = 2
    n_obs = 3

    h = np.arange(H + 1)
    irf = np.zeros((n_shocks, H + 1, n_obs))

    for shock_idx in range(n_shocks):
        irf[shock_idx] = np.column_stack([
            np.exp(-0.1 * h),
            0.5 * np.exp(-0.2 * h),
            -0.3 * np.exp(-0.15 * h),
        ])

    irfs = {"Test": irf}

    obs_names = ["GDP", "CPI", "FFR"]
    shock_labels = ["Monetary", "Technology"]

    fig = plot_irf_panel(
        irfs,
        obs_names=obs_names,
        shock_labels=shock_labels,
        layout="obs_rows",
    )

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


@pytest.mark.fast
def test_plot_irf_panel_shape_validation():
    """Test that inconsistent IRF shapes raise errors."""
    irf1 = np.random.randn(21, 3)  # H=20, n_obs=3
    irf2 = np.random.randn(31, 3)  # H=30, n_obs=3 (different!)

    irfs = {
        "Model A": irf1,
        "Model B": irf2,
    }

    with pytest.raises(ValueError, match="All IRFs must have the same shape"):
        plot_irf_panel(irfs)


@pytest.mark.fast
def test_plot_irf_panel_invalid_dimensions():
    """Test that invalid IRF dimensions raise errors."""
    irf = np.random.randn(21, 3, 5, 2)  # 4D array (invalid)

    irfs = {"Test": irf}

    with pytest.raises(ValueError, match="IRF arrays must be 2D or 3D"):
        plot_irf_panel(irfs)


@pytest.mark.fast
def test_plot_irf_comparison():
    """Test IRF comparison plot (ground truth vs prediction)."""
    H = 20
    n_obs = 3

    h = np.arange(H + 1)
    irf_true = np.column_stack([
        np.exp(-0.1 * h),
        0.5 * np.exp(-0.2 * h),
        -0.3 * np.exp(-0.15 * h),
    ])

    # Add some noise to create "prediction"
    irf_pred = irf_true + 0.1 * np.random.randn(H + 1, n_obs)

    fig = plot_irf_comparison(irf_true, irf_pred, title="Test Comparison")

    # Check figure created
    assert isinstance(fig, plt.Figure)

    # Should have 3 subplots (one per observable)
    assert len(fig.axes) == 3

    # Each subplot should have 2 lines (true + pred)
    for ax in fig.axes:
        lines = ax.get_lines()
        # Filter out zero line
        data_lines = [line for line in lines if len(line.get_xdata()) > 2]
        assert len(data_lines) >= 2  # At least true and pred

    plt.close(fig)


@pytest.mark.fast
def test_plot_irf_comparison_custom_labels():
    """Test IRF comparison with custom observable labels."""
    H = 10
    n_obs = 2

    irf_true = np.random.randn(H + 1, n_obs)
    irf_pred = np.random.randn(H + 1, n_obs)

    obs_names = ["Variable 1", "Variable 2"]

    fig = plot_irf_comparison(irf_true, irf_pred, obs_names=obs_names)

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2
    plt.close(fig)


@pytest.mark.fast
def test_save_figure_png():
    """Test saving figure to PNG format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_plot.png"

        # Create simple figure
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 0])

        # Save figure
        save_figure(fig, output_path, close=True)

        # Check file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0


@pytest.mark.fast
def test_save_figure_multiple_formats():
    """Test saving figure in multiple formats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_base = Path(tmpdir) / "test_plot"

        # Create simple figure
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 0])

        # Save in multiple formats
        save_figure(fig, output_base, formats=["png", "pdf"], close=True)

        # Check both files created
        png_path = output_base.with_suffix(".png")
        pdf_path = output_base.with_suffix(".pdf")

        assert png_path.exists()
        assert pdf_path.exists()
        assert png_path.stat().st_size > 0
        assert pdf_path.stat().st_size > 0


@pytest.mark.fast
def test_save_figure_creates_directory():
    """Test that save_figure creates output directory if needed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use nested directory that doesn't exist
        output_path = Path(tmpdir) / "subdir1" / "subdir2" / "plot.png"

        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 0])

        # Should create directories automatically
        save_figure(fig, output_path, close=True)

        assert output_path.exists()


@pytest.mark.fast
def test_save_figure_infer_format_from_extension():
    """Test that format is inferred from file extension."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.pdf"

        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 0])

        # Don't specify formats - should infer PDF from extension
        save_figure(fig, output_path, formats=None, close=True)

        assert output_path.exists()
        # PDF files start with %PDF
        with open(output_path, "rb") as f:
            header = f.read(4)
            assert header == b"%PDF"


@pytest.mark.fast
def test_plot_irf_panel_empty_dict():
    """Test that empty IRF dictionary raises appropriate error."""
    irfs = {}

    with pytest.raises((ValueError, IndexError, KeyError)):
        plot_irf_panel(irfs)


@pytest.mark.fast
def test_plot_irf_panel_single_observable():
    """Test IRF panel with single observable (edge case)."""
    H = 10
    n_obs = 1

    irf = np.random.randn(H + 1, n_obs)
    irfs = {"Test": irf}

    fig = plot_irf_panel(irfs)

    assert isinstance(fig, plt.Figure)
    # Should have 1 row (1 observable) x 1 col (1 simulator) = 1 subplot
    assert len(fig.axes) == 1

    plt.close(fig)


@pytest.mark.fast
def test_zero_horizon():
    """Test IRF panel with zero horizon (impact effect only)."""
    H = 0
    n_obs = 3

    # Only impact effect (single point)
    irf = np.random.randn(1, n_obs)
    irfs = {"Test": irf}

    fig = plot_irf_panel(irfs, horizon=0)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)
