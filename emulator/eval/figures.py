"""Publication-quality figures for evaluation results.

This module provides high-level visualization functions for creating
standardized figures from evaluation results:
- NRMSE bar charts (universal vs baselines)
- Shape metric comparisons (HF-ratio, overshoot, sign-flips)
- Regime comparisons (A vs B1 vs C)
- IRF prediction panels (ground truth vs model)

All figures follow consistent styling for papers and blog posts.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from emulator.eval.plots import COLOR_PALETTE, setup_plot_style


def plot_nrmse_bar_chart(
    results: dict[str, Any],
    output_path: str | Path,
    baseline_results: dict[str, dict[str, Any]] | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot per-world NRMSE comparison bar chart.

    Creates a grouped bar chart comparing the universal emulator's NRMSE
    against baseline models (if provided) across all worlds.

    Args:
        results: Evaluation results dictionary with "per_world" key
        output_path: Path to save figure
        baseline_results: Optional dict mapping baseline_name -> results dict
        dpi: Resolution for saved figure

    Returns:
        Matplotlib figure object

    Example:
        >>> results = {"per_world": {"lss": {"nrmse": 0.15}, "var": {"nrmse": 0.12}}}
        >>> baselines = {"MLP": {"per_world": {"lss": {"nrmse": 0.20}, "var": {"nrmse": 0.18}}}}
        >>> fig = plot_nrmse_bar_chart(results, "nrmse_comparison.png", baselines)
    """
    setup_plot_style()

    per_world = results.get("per_world", {})
    if not per_world:
        raise ValueError("Results must contain 'per_world' key with metrics")

    # Get world IDs (sorted for consistency)
    world_ids = sorted(per_world.keys())
    n_worlds = len(world_ids)

    # Prepare data
    model_names = ["Universal"]
    nrmse_data = {
        "Universal": [per_world[wid].get("nrmse", 0.0) for wid in world_ids]
    }

    # Add baselines if provided
    if baseline_results:
        for baseline_name, baseline_result in baseline_results.items():
            model_names.append(baseline_name)
            baseline_per_world = baseline_result.get("per_world", {})
            nrmse_data[baseline_name] = [
                baseline_per_world.get(wid, {}).get("nrmse", 0.0) for wid in world_ids
            ]

    # Create figure
    n_models = len(model_names)
    fig, ax = plt.subplots(figsize=(max(10, n_worlds * 1.5), 6))

    # Bar positions
    x = np.arange(n_worlds)
    width = 0.8 / n_models  # Total width of 0.8 divided by number of models

    # Plot bars
    for i, model_name in enumerate(model_names):
        offset = (i - n_models / 2 + 0.5) * width
        values = nrmse_data[model_name]

        color = COLOR_PALETTE.get(model_name.lower(), f"C{i}")
        ax.bar(
            x + offset,
            values,
            width,
            label=model_name,
            color=color,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

        # Add value labels on top of bars
        for j, v in enumerate(values):
            ax.text(
                x[j] + offset,
                v + 0.01,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Styling
    ax.set_xlabel("World", fontweight="bold")
    ax.set_ylabel("NRMSE (Normalized RMSE)", fontweight="bold")
    ax.set_title("Per-World NRMSE Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([wid.upper() for wid in world_ids])
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved NRMSE bar chart to: {output_path}")

    return fig


def plot_shape_comparison(
    results: dict[str, Any],
    output_path: str | Path,
    baseline_results: dict[str, dict[str, Any]] | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot shape metric comparison (HF-ratio, overshoot, sign-flips).

    Creates a 3-panel figure showing:
    1. HF-ratio distribution across worlds
    2. Overshoot ratio distribution
    3. Sign-flip count histogram

    Args:
        results: Evaluation results dictionary with "per_world" key
        output_path: Path to save figure
        baseline_results: Optional dict mapping baseline_name -> results dict
        dpi: Resolution for saved figure

    Returns:
        Matplotlib figure object
    """
    setup_plot_style()

    per_world = results.get("per_world", {})
    if not per_world:
        raise ValueError("Results must contain 'per_world' key with metrics")

    world_ids = sorted(per_world.keys())
    n_worlds = len(world_ids)

    # Extract metrics
    def get_metric_values(data: dict[str, Any], metric: str) -> list[float]:
        return [data.get(wid, {}).get(metric, 0.0) for wid in world_ids]

    universal_hf = get_metric_values(per_world, "hf_ratio")
    universal_overshoot = get_metric_values(per_world, "overshoot_ratio")
    universal_sign_flips = get_metric_values(per_world, "sign_flip_count")

    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Shape Metric Comparison", fontsize=14, fontweight="bold")

    x = np.arange(n_worlds)
    width = 0.35

    # Panel 1: HF-ratio
    ax = axes[0]
    ax.bar(x, universal_hf, width, label="Universal", color=COLOR_PALETTE.get("prediction", "C0"), alpha=0.8)

    if baseline_results:
        for i, (baseline_name, baseline_result) in enumerate(baseline_results.items(), start=1):
            baseline_hf = get_metric_values(baseline_result.get("per_world", {}), "hf_ratio")
            ax.bar(
                x + i * width,
                baseline_hf,
                width,
                label=baseline_name,
                color=f"C{i}",
                alpha=0.8,
            )

    ax.set_xlabel("World")
    ax.set_ylabel("HF-ratio")
    ax.set_title("High-Frequency Ratio\n(Lower is better, measures oscillations)")
    ax.set_xticks(x + width * (len(baseline_results) if baseline_results else 0) / 2)
    ax.set_xticklabels([wid.upper() for wid in world_ids])
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Baseline=1.0")

    # Panel 2: Overshoot ratio
    ax = axes[1]
    ax.bar(x, universal_overshoot, width, label="Universal", color=COLOR_PALETTE.get("prediction", "C0"), alpha=0.8)

    if baseline_results:
        for i, (baseline_name, baseline_result) in enumerate(baseline_results.items(), start=1):
            baseline_overshoot = get_metric_values(baseline_result.get("per_world", {}), "overshoot_ratio")
            ax.bar(
                x + i * width,
                baseline_overshoot,
                width,
                label=baseline_name,
                color=f"C{i}",
                alpha=0.8,
            )

    ax.set_xlabel("World")
    ax.set_ylabel("Overshoot Ratio")
    ax.set_title("Overshoot Ratio\n(Fraction of IRFs with overshoot)")
    ax.set_xticks(x + width * (len(baseline_results) if baseline_results else 0) / 2)
    ax.set_xticklabels([wid.upper() for wid in world_ids])
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim([0, 1])

    # Panel 3: Sign-flip count
    ax = axes[2]
    ax.bar(x, universal_sign_flips, width, label="Universal", color=COLOR_PALETTE.get("prediction", "C0"), alpha=0.8)

    if baseline_results:
        for i, (baseline_name, baseline_result) in enumerate(baseline_results.items(), start=1):
            baseline_sign_flips = get_metric_values(baseline_result.get("per_world", {}), "sign_flip_count")
            ax.bar(
                x + i * width,
                baseline_sign_flips,
                width,
                label=baseline_name,
                color=f"C{i}",
                alpha=0.8,
            )

    ax.set_xlabel("World")
    ax.set_ylabel("Sign Flips (average)")
    ax.set_title("Sign Flip Count\n(Lower is better, measures spurious reversals)")
    ax.set_xticks(x + width * (len(baseline_results) if baseline_results else 0) / 2)
    ax.set_xticklabels([wid.upper() for wid in world_ids])
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved shape comparison to: {output_path}")

    return fig


def plot_regime_comparison(
    results_A: dict[str, Any],
    results_B1: dict[str, Any] | None = None,
    results_C: dict[str, Any] | None = None,
    output_path: str | Path = "regime_comparison.png",
    dpi: int = 300,
) -> plt.Figure:
    """Plot regime comparison (Regime A vs B1 vs C).

    Creates a grouped bar chart comparing NRMSE across information regimes.

    Args:
        results_A: Evaluation results for Regime A (full structural)
        results_B1: Evaluation results for Regime B1 (observables only)
        results_C: Evaluation results for Regime C (partial)
        output_path: Path to save figure
        dpi: Resolution for saved figure

    Returns:
        Matplotlib figure object
    """
    setup_plot_style()

    per_world_A = results_A.get("per_world", {})
    if not per_world_A:
        raise ValueError("results_A must contain 'per_world' key")

    world_ids = sorted(per_world_A.keys())
    n_worlds = len(world_ids)

    # Prepare data
    regime_names = ["Regime A (Full)"]
    nrmse_data = {
        "Regime A (Full)": [per_world_A[wid].get("nrmse", 0.0) for wid in world_ids]
    }

    if results_B1:
        regime_names.append("Regime B1 (Observable)")
        per_world_B1 = results_B1.get("per_world", {})
        nrmse_data["Regime B1 (Observable)"] = [
            per_world_B1.get(wid, {}).get("nrmse", 0.0) for wid in world_ids
        ]

    if results_C:
        regime_names.append("Regime C (Partial)")
        per_world_C = results_C.get("per_world", {})
        nrmse_data["Regime C (Partial)"] = [
            per_world_C.get(wid, {}).get("nrmse", 0.0) for wid in world_ids
        ]

    # Create figure
    n_regimes = len(regime_names)
    fig, ax = plt.subplots(figsize=(max(10, n_worlds * 1.5), 6))

    # Bar positions
    x = np.arange(n_worlds)
    width = 0.8 / n_regimes

    # Plot bars
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green
    for i, regime_name in enumerate(regime_names):
        offset = (i - n_regimes / 2 + 0.5) * width
        values = nrmse_data[regime_name]

        ax.bar(
            x + offset,
            values,
            width,
            label=regime_name,
            color=colors[i % len(colors)],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

        # Add value labels
        for j, v in enumerate(values):
            ax.text(
                x[j] + offset,
                v + 0.01,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=0,
            )

    # Styling
    ax.set_xlabel("World", fontweight="bold")
    ax.set_ylabel("NRMSE (Normalized RMSE)", fontweight="bold")
    ax.set_title("Information Regime Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([wid.upper() for wid in world_ids])
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved regime comparison to: {output_path}")

    return fig


def plot_irf_panel(
    y_pred: npt.NDArray,
    y_true: npt.NDArray,
    world_id: str,
    output_path: str | Path,
    obs_names: list[str] | None = None,
    shock_idx: int = 0,
    confidence_bands: tuple[npt.NDArray, npt.NDArray] | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot IRF prediction vs ground truth panel.

    Creates a 3-row panel (output, inflation, rate) showing model predictions
    overlaid with ground truth.

    Args:
        y_pred: Predicted IRFs, shape (n_samples, H+1, 3) or (H+1, 3)
        y_true: Ground truth IRFs, same shape as y_pred
        world_id: Simulator identifier for title
        output_path: Path to save figure
        obs_names: Observable names (default: ["Output", "Inflation", "Rate"])
        shock_idx: Shock index for title
        confidence_bands: Optional tuple of (lower, upper) bounds, shape (H+1, 3)
        dpi: Resolution for saved figure

    Returns:
        Matplotlib figure object
    """
    setup_plot_style()

    if obs_names is None:
        obs_names = ["Output", "Inflation", "Rate"]

    # Handle batch dimension
    if y_pred.ndim == 3:
        # Average over samples
        y_pred_mean = np.mean(y_pred, axis=0)  # (H+1, 3)
        y_true_mean = np.mean(y_true, axis=0)

        # Compute confidence bands from data if not provided
        if confidence_bands is None:
            lower = np.percentile(y_pred, 5, axis=0)
            upper = np.percentile(y_pred, 95, axis=0)
            confidence_bands = (lower, upper)
    else:
        y_pred_mean = y_pred
        y_true_mean = y_true

    H = y_pred_mean.shape[0] - 1
    h_grid = np.arange(H + 1)

    # Create 3-row panel
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(
        f"IRF Prediction vs Ground Truth: {world_id.upper()} (Shock {shock_idx})",
        fontsize=14,
        fontweight="bold",
    )

    for i, (ax, obs_name) in enumerate(zip(axes, obs_names, strict=False)):
        # Plot ground truth
        ax.plot(
            h_grid,
            y_true_mean[:, i],
            color=COLOR_PALETTE["ground_truth"],
            linewidth=2.5,
            label="Ground Truth",
            marker="o",
            markersize=4,
            markevery=5,
        )

        # Plot prediction
        ax.plot(
            h_grid,
            y_pred_mean[:, i],
            color=COLOR_PALETTE["prediction"],
            linewidth=2,
            label="Prediction",
            linestyle="--",
            marker="x",
            markersize=4,
            markevery=5,
        )

        # Add confidence bands if available
        if confidence_bands is not None:
            lower, upper = confidence_bands
            ax.fill_between(
                h_grid,
                lower[:, i],
                upper[:, i],
                color=COLOR_PALETTE["prediction"],
                alpha=0.2,
                label="5-95% CI",
            )

        # Zero line
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        # Styling
        ax.set_ylabel(obs_name, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", framealpha=0.9)

        # Only show x-label on bottom panel
        if i == 2:
            ax.set_xlabel("Horizon", fontweight="bold")

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved IRF panel to: {output_path}")

    return fig
