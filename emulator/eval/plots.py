"""
Visualization tools for IRF evaluation.

This module provides publication-quality plotting functions for impulse response functions
and related diagnostics. All plots follow consistent styling conventions and support
saving to various formats.
"""

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# Publication-quality plot styling
PLOT_STYLE = {
    "figure.figsize": (12, 8),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

# Canonical observable names (always in this order)
CANONICAL_OBS = ["Output", "Inflation", "Rate"]

# Color palette for different simulators/models
COLOR_PALETTE = {
    "lss": "#1f77b4",  # Blue
    "var": "#ff7f0e",  # Orange
    "nk": "#2ca02c",   # Green
    "rbc": "#d62728",  # Red
    "switching": "#9467bd",  # Purple
    "zlb": "#8c564b",  # Brown
    "ground_truth": "#000000",  # Black
    "prediction": "#e377c2",   # Pink
}


def setup_plot_style() -> None:
    """Apply consistent matplotlib styling for publication-quality plots."""
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update(PLOT_STYLE)


def plot_irf_panel(
    irfs: dict[str, npt.NDArray[np.float64]],
    horizon: int | None = None,
    obs_names: Sequence[str] | None = None,
    title: str = "Impulse Response Functions",
    shock_labels: list[str] | None = None,
    layout: str = "obs_rows",
) -> plt.Figure:
    """Plot IRF panel with multiple simulators or shocks.

    Creates a grid of subplots showing IRF curves. The layout can be organized
    either by observables (rows) or by shocks/simulators (columns).

    Args:
        irfs: Dictionary mapping labels to IRF arrays.
              Each IRF has shape (H+1, n_obs) or (n_shocks, H+1, n_obs).
              - If 2D: single IRF per label (e.g., different simulators, same shock)
              - If 3D: multiple IRFs per label (e.g., different shocks, same simulator)
        horizon: Horizon length to plot (if None, use full IRF length)
        obs_names: Observable names (default: ["Output", "Inflation", "Rate"])
        title: Figure title
        shock_labels: Labels for shocks if using 3D IRFs (default: Shock 0, 1, ...)
        layout: Layout mode:
                - "obs_rows": rows = observables, columns = simulators/shocks
                - "shock_rows": rows = shocks, columns = observables (only for 3D IRFs)

    Returns:
        Matplotlib figure object

    Example:
        >>> from simulators import LSSSimulator
        >>> import numpy as np
        >>> lss = LSSSimulator()
        >>> rng = np.random.default_rng(42)
        >>> theta = lss.sample_parameters(rng)
        >>> irf_shock0 = lss.compute_irf(theta, shock_idx=0, shock_size=1.0, H=40)
        >>> irf_shock1 = lss.compute_irf(theta, shock_idx=1, shock_size=1.0, H=40)
        >>> irfs = {"Shock 0": irf_shock0, "Shock 1": irf_shock1}
        >>> fig = plot_irf_panel(irfs, title="LSS IRFs")
        >>> fig.savefig("lss_irfs.png")
    """
    setup_plot_style()

    if obs_names is None:
        obs_names = CANONICAL_OBS

    # Determine dimensionality and validate inputs
    first_label = list(irfs.keys())[0]
    first_irf = irfs[first_label]

    if first_irf.ndim == 2:
        # 2D: (H+1, n_obs) - single IRF per label
        mode = "2d"
        n_obs = first_irf.shape[1]
        H_full = first_irf.shape[0] - 1
        n_labels = len(irfs)
        n_shocks = 1
    elif first_irf.ndim == 3:
        # 3D: (n_shocks, H+1, n_obs) - multiple IRFs per label
        mode = "3d"
        n_shocks, H_full_plus1, n_obs = first_irf.shape
        H_full = H_full_plus1 - 1
        n_labels = len(irfs)
    else:
        raise ValueError(f"IRF arrays must be 2D or 3D, got {first_irf.ndim}D")

    # Validate all IRFs have consistent shape
    for label, irf in irfs.items():
        if irf.shape != first_irf.shape:
            raise ValueError(
                f"All IRFs must have the same shape. Got {irf.shape} for '{label}', "
                f"expected {first_irf.shape}"
            )

    if horizon is None:
        horizon = H_full

    # Create subplot grid
    if mode == "2d" or layout == "obs_rows":
        n_rows = n_obs
        n_cols = n_labels if mode == "2d" else n_shocks
        figsize = (4 * n_cols, 3 * n_rows)
    else:  # layout == "shock_rows" and mode == "3d"
        n_rows = n_shocks
        n_cols = n_obs
        figsize = (4 * n_cols, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    h_grid = np.arange(horizon + 1)

    if mode == "2d":
        # Plot each label (simulator) in a separate column
        for col_idx, (label, irf) in enumerate(irfs.items()):
            color = COLOR_PALETTE.get(label.lower(), f"C{col_idx}")

            for row_idx in range(n_obs):
                ax = axes[row_idx, col_idx]
                ax.plot(h_grid, irf[:horizon + 1, row_idx], color=color, label=label)
                ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

                # Labels
                if row_idx == 0:
                    ax.set_title(label, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(obs_names[row_idx])
                if row_idx == n_obs - 1:
                    ax.set_xlabel("Horizon")

                ax.grid(True, alpha=0.3)

    elif layout == "obs_rows":
        # 3D IRFs: rows = observables, columns = shocks
        if shock_labels is None:
            shock_labels = [f"Shock {i}" for i in range(n_shocks)]

        for row_idx in range(n_obs):
            for col_idx in range(n_shocks):
                ax = axes[row_idx, col_idx]

                # Plot all labels on the same axis
                for label_idx, (label, irf) in enumerate(irfs.items()):
                    color = COLOR_PALETTE.get(label.lower(), f"C{label_idx}")
                    ax.plot(
                        h_grid,
                        irf[col_idx, :horizon + 1, row_idx],
                        color=color,
                        label=label,
                    )

                ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

                # Labels
                if row_idx == 0:
                    ax.set_title(shock_labels[col_idx], fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(obs_names[row_idx])
                if row_idx == n_obs - 1:
                    ax.set_xlabel("Horizon")

                # Legend only on first subplot
                if row_idx == 0 and col_idx == 0:
                    ax.legend(loc="best")

                ax.grid(True, alpha=0.3)

    else:  # layout == "shock_rows"
        # 3D IRFs: rows = shocks, columns = observables
        if shock_labels is None:
            shock_labels = [f"Shock {i}" for i in range(n_shocks)]

        for row_idx in range(n_shocks):
            for col_idx in range(n_obs):
                ax = axes[row_idx, col_idx]

                # Plot all labels on the same axis
                for label_idx, (label, irf) in enumerate(irfs.items()):
                    color = COLOR_PALETTE.get(label.lower(), f"C{label_idx}")
                    ax.plot(
                        h_grid,
                        irf[row_idx, :horizon + 1, col_idx],
                        color=color,
                        label=label,
                    )

                ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

                # Labels
                if row_idx == 0:
                    ax.set_title(obs_names[col_idx], fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(shock_labels[row_idx])
                if row_idx == n_shocks - 1:
                    ax.set_xlabel("Horizon")

                # Legend only on first subplot
                if row_idx == 0 and col_idx == 0:
                    ax.legend(loc="best")

                ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_irf_comparison(
    irf_true: npt.NDArray[np.float64],
    irf_pred: npt.NDArray[np.float64],
    obs_names: Sequence[str] | None = None,
    title: str = "IRF Comparison",
) -> plt.Figure:
    """Plot ground truth vs predicted IRFs side by side.

    Args:
        irf_true: Ground truth IRF, shape (H+1, n_obs)
        irf_pred: Predicted IRF, shape (H+1, n_obs)
        obs_names: Observable names (default: canonical)
        title: Figure title

    Returns:
        Matplotlib figure object
    """
    setup_plot_style()

    if obs_names is None:
        obs_names = CANONICAL_OBS

    H = irf_true.shape[0] - 1
    n_obs = irf_true.shape[1]

    fig, axes = plt.subplots(1, n_obs, figsize=(12, 4))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    if n_obs == 1:
        axes = [axes]

    h_grid = np.arange(H + 1)

    for idx, ax in enumerate(axes):
        ax.plot(h_grid, irf_true[:, idx], color=COLOR_PALETTE["ground_truth"],
                linewidth=2, label="Ground Truth", marker="o", markersize=3)
        ax.plot(h_grid, irf_pred[:, idx], color=COLOR_PALETTE["prediction"],
                linewidth=2, label="Prediction", linestyle="--", marker="x", markersize=3)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_xlabel("Horizon")
        ax.set_ylabel(obs_names[idx])
        ax.set_title(obs_names[idx], fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def save_figure(
    fig: plt.Figure,
    output_path: str | Path,
    formats: list[str] | None = None,
    dpi: int = 300,
    close: bool = True,
) -> None:
    """Save figure to file with consistent settings.

    Args:
        fig: Matplotlib figure to save
        output_path: Output file path (extension determines format if formats=None)
        formats: List of formats to save (e.g., ["png", "pdf"]).
                 If None, inferred from output_path extension.
        dpi: Resolution for raster formats (default: 300)
        close: Whether to close the figure after saving (default: True)

    Example:
        >>> fig = plot_irf_panel(irfs)
        >>> save_figure(fig, "output.png")
        >>> # Or save in multiple formats:
        >>> save_figure(fig, "output", formats=["png", "pdf"])
    """
    output_path = Path(output_path)

    if formats is None:
        # Infer format from extension
        formats = [output_path.suffix.lstrip(".")]
        if not formats[0]:
            formats = ["png"]  # Default to PNG if no extension

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        if len(formats) == 1 and output_path.suffix:
            # Use provided path as-is
            save_path = output_path
        else:
            # Append format extension
            save_path = output_path.with_suffix(f".{fmt}")

        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    if close:
        plt.close(fig)


def main() -> None:
    """Command-line interface for generating IRF plots."""
    parser = argparse.ArgumentParser(
        description="Generate IRF panel plots from simulators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot IRFs from LSS, VAR, NK simulators
  python -m emulator.eval.plots --simulators lss,var,nk --output sprint1_irf_panel.png

  # Plot with custom horizon and shock size
  python -m emulator.eval.plots --simulators lss --horizon 80 --shock-size 2.0 --output lss_large_shock.png

  # Generate multiple shocks per simulator
  python -m emulator.eval.plots --simulators lss,var --all-shocks --output multi_shock_panel.png
        """,
    )

    parser.add_argument(
        "--simulators",
        type=str,
        required=True,
        help="Comma-separated list of simulators (e.g., 'lss,var,nk')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="irf_panel.png",
        help="Output file path (default: irf_panel.png)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=40,
        help="IRF horizon length (default: 40)",
    )
    parser.add_argument(
        "--shock-idx",
        type=int,
        default=0,
        help="Shock index to plot (default: 0, ignored if --all-shocks)",
    )
    parser.add_argument(
        "--shock-size",
        type=float,
        default=1.0,
        help="Shock size in standard deviations (default: 1.0)",
    )
    parser.add_argument(
        "--all-shocks",
        action="store_true",
        help="Plot all shocks for each simulator",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for parameter sampling (default: 42)",
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=["obs_rows", "shock_rows"],
        default="obs_rows",
        help="Layout mode (default: obs_rows)",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png",
        help="Comma-separated output formats (e.g., 'png,pdf')",
    )

    args = parser.parse_args()

    # Import simulators
    from simulators import LSSSimulator, NKSimulator, VARSimulator

    simulator_map = {
        "lss": LSSSimulator,
        "var": VARSimulator,
        "nk": NKSimulator,
    }

    # Parse simulator names
    simulator_names = [s.strip().lower() for s in args.simulators.split(",")]

    # Validate simulator names
    for name in simulator_names:
        if name not in simulator_map:
            available = ", ".join(simulator_map.keys())
            raise ValueError(
                f"Unknown simulator '{name}'. Available: {available}"
            )

    # Generate IRFs
    rng = np.random.default_rng(args.seed)
    irfs = {}

    print(f"Generating IRFs with seed={args.seed}, H={args.horizon}...")

    for name in simulator_names:
        simulator = simulator_map[name]()
        theta = simulator.sample_parameters(rng)

        print(f"  {name.upper()}: sampled parameters, computing IRFs...")

        if args.all_shocks:
            # Compute IRFs for all shocks
            n_shocks = simulator.shock_manifest.n_shocks
            irf_array = np.zeros((n_shocks, args.horizon + 1, 3))

            for shock_idx in range(n_shocks):
                irf_array[shock_idx] = simulator.compute_irf(
                    theta=theta,
                    shock_idx=shock_idx,
                    shock_size=args.shock_size,
                    H=args.horizon,
                )

            irfs[name.upper()] = irf_array
        else:
            # Single shock
            irf = simulator.compute_irf(
                theta=theta,
                shock_idx=args.shock_idx,
                shock_size=args.shock_size,
                H=args.horizon,
            )
            irfs[name.upper()] = irf

    # Create plot
    title = f"Impulse Response Functions (shock size = {args.shock_size}σ)"
    fig = plot_irf_panel(
        irfs=irfs,
        horizon=args.horizon,
        title=title,
        layout=args.layout,
    )

    # Save figure
    formats = [f.strip() for f in args.formats.split(",")]
    save_figure(fig, args.output, formats=formats)

    print(f"\nPlot generation complete!")


if __name__ == "__main__":
    main()
