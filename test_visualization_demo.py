#!/usr/bin/env python
"""
Demonstration script for the IRF visualization module.

This script generates sample IRF plots using the implemented simulators
to verify that S1.28 and S1.29 are working correctly.
"""

import numpy as np
from simulators import LSSSimulator, VARSimulator, NKSimulator
from emulator.eval.plots import plot_irf_panel, save_figure


def main():
    """Generate demonstration IRF plots."""
    print("="*70)
    print("IRF Visualization Demonstration")
    print("="*70)
    print()

    # Configuration
    seed = 42
    H = 40
    shock_size = 1.0

    print(f"Configuration:")
    print(f"  Random seed: {seed}")
    print(f"  Horizon: {H}")
    print(f"  Shock size: {shock_size} std dev")
    print()

    # Initialize RNG
    rng = np.random.default_rng(seed)

    # Initialize simulators
    print("Initializing simulators...")
    lss = LSSSimulator()
    var = VARSimulator()
    nk = NKSimulator()

    simulators = {
        "LSS": lss,
        "VAR": var,
        "NK": nk,
    }

    # Sample parameters and compute IRFs for shock 0
    print("\nGenerating IRFs for shock 0...")
    irfs_shock0 = {}

    for name, sim in simulators.items():
        print(f"  {name}: sampling parameters...")
        theta = sim.sample_parameters(rng)

        print(f"  {name}: computing IRF...")
        irf = sim.compute_irf(
            theta=theta,
            shock_idx=0,
            shock_size=shock_size,
            H=H,
        )
        irfs_shock0[name] = irf
        print(f"  {name}: IRF shape = {irf.shape}")

    # Create and save plot for shock 0
    print("\nCreating IRF panel plot (shock 0)...")
    fig1 = plot_irf_panel(
        irfs=irfs_shock0,
        title=f"Impulse Response Functions - Shock 0 (size = {shock_size}σ)",
    )

    output_path1 = "sprint1_irf_panel_shock0.png"
    save_figure(fig1, output_path1)
    print(f"Saved: {output_path1}")

    # Generate IRFs for all shocks from LSS
    print("\nGenerating all shocks for LSS simulator...")
    theta_lss = lss.sample_parameters(rng)
    n_shocks = lss.shock_manifest.n_shocks

    irf_lss_all = np.zeros((n_shocks, H + 1, 3))
    for shock_idx in range(n_shocks):
        irf_lss_all[shock_idx] = lss.compute_irf(
            theta=theta_lss,
            shock_idx=shock_idx,
            shock_size=shock_size,
            H=H,
        )
        print(f"  Shock {shock_idx}: shape = {irf_lss_all[shock_idx].shape}")

    # Create multi-shock panel
    print("\nCreating multi-shock panel plot...")
    fig2 = plot_irf_panel(
        irfs={"LSS": irf_lss_all},
        title=f"LSS Model - All Shocks (size = {shock_size}σ)",
        shock_labels=[f"Shock {i}" for i in range(n_shocks)],
        layout="obs_rows",
    )

    output_path2 = "sprint1_irf_panel_lss_all_shocks.png"
    save_figure(fig2, output_path2)
    print(f"Saved: {output_path2}")

    # Compare multiple simulators with all shocks
    print("\nGenerating comparison across all simulators...")
    irfs_all = {}

    for name, sim in simulators.items():
        theta = sim.sample_parameters(rng)
        n_shocks = sim.shock_manifest.n_shocks
        irf_all = np.zeros((n_shocks, H + 1, 3))

        for shock_idx in range(n_shocks):
            irf_all[shock_idx] = sim.compute_irf(
                theta=theta,
                shock_idx=shock_idx,
                shock_size=shock_size,
                H=H,
            )

        irfs_all[name] = irf_all
        print(f"  {name}: generated {n_shocks} IRFs")

    print("\nCreating comprehensive comparison plot...")
    fig3 = plot_irf_panel(
        irfs=irfs_all,
        title=f"Cross-Simulator IRF Comparison (size = {shock_size}σ)",
        layout="obs_rows",
    )

    output_path3 = "sprint1_irf_panel_comprehensive.png"
    save_figure(fig3, output_path3)
    print(f"Saved: {output_path3}")

    print()
    print("="*70)
    print("Demonstration complete!")
    print("="*70)
    print()
    print("Generated files:")
    print(f"  1. {output_path1} - Single shock comparison")
    print(f"  2. {output_path2} - LSS all shocks")
    print(f"  3. {output_path3} - Cross-simulator comprehensive")
    print()


if __name__ == "__main__":
    main()
