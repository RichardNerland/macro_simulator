"""Evaluation harness and visualization for the Universal Macro Emulator."""

from emulator.eval.plots import (
    generate_sanity_plots,
    load_dataset_for_plots,
    plot_irf_comparison,
    plot_irf_panel,
    plot_irf_statistics,
    plot_parameter_histograms,
    plot_sample_irfs,
    save_figure,
    setup_plot_style,
)
from emulator.eval.metrics import (
    compute_sigma_from_data,
    exponential_weights,
    gap_metric,
    get_weight_scheme,
    hf_ratio,
    iae,
    impact_weighted,
    nrmse,
    overshoot_ratio,
    sign_at_impact,
    sign_flip_count,
    uniform_weights,
)
from emulator.eval.leaderboard import (
    LeaderboardGenerator,
    compare_models,
    compute_success_criteria,
)

__all__ = [
    # Plotting
    "plot_irf_panel",
    "plot_irf_comparison",
    "plot_parameter_histograms",
    "plot_sample_irfs",
    "plot_irf_statistics",
    "load_dataset_for_plots",
    "generate_sanity_plots",
    "save_figure",
    "setup_plot_style",
    # Metrics
    "nrmse",
    "iae",
    "sign_at_impact",
    "gap_metric",
    "hf_ratio",
    "overshoot_ratio",
    "sign_flip_count",
    "uniform_weights",
    "exponential_weights",
    "impact_weighted",
    "get_weight_scheme",
    "compute_sigma_from_data",
    # Leaderboard
    "LeaderboardGenerator",
    "compare_models",
    "compute_success_criteria",
]
