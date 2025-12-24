"""Example script demonstrating the evaluation pipeline.

This script shows how to:
1. Evaluate a trained UniversalEmulator on dataset splits
2. Generate comparison figures
3. Create a leaderboard comparing multiple models

Usage:
    # Evaluate single model
    python examples/evaluate_model.py --checkpoint runs/smoke_test_regime_A/best.pt --dataset datasets/v1.0-dev/

    # Generate figures
    python examples/evaluate_model.py --checkpoint runs/smoke_test_regime_A/best.pt --dataset datasets/v1.0-dev/ --figures

    # Compare multiple models
    python examples/evaluate_model.py --compare results/
"""

import argparse
import json
from pathlib import Path

from emulator.eval.evaluate import evaluate_all_worlds, load_universal_model
from emulator.eval.figures import (
    plot_irf_panel,
    plot_nrmse_bar_chart,
    plot_shape_comparison,
)
from emulator.eval.leaderboard import LeaderboardGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate UniversalEmulator and generate figures"
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--worlds",
        type=str,
        default="lss,var,nk",
        help="Comma-separated list of worlds to evaluate (default: lss,var,nk)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="test_interpolation",
        help="Comma-separated list of splits (default: test_interpolation)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for results and figures (default: results/)",
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Generate visualization figures",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="Directory with multiple result JSONs to compare",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        # Comparison mode: load and compare multiple results
        print(f"Comparing models from: {args.compare}")

        leaderboard = LeaderboardGenerator()

        for json_file in sorted(args.compare.glob("*.json")):
            model_name = json_file.stem
            is_oracle = "oracle" in model_name.lower()
            try:
                leaderboard.add_model(model_name, json_file, is_oracle=is_oracle)
                print(f"  Added: {model_name}")
            except Exception as e:
                print(f"  Skipped {model_name}: {e}")

        # Print summary
        leaderboard.print_summary()

        # Save outputs
        leaderboard.save_csv(args.output_dir / "leaderboard.csv")
        leaderboard.save_markdown(args.output_dir / "leaderboard.md")

        print(f"\nLeaderboard saved to: {args.output_dir}/")
        return

    # Single model evaluation
    world_ids = [w.strip() for w in args.worlds.split(",")]
    split_names = [s.strip() for s in args.splits.split(",")]

    # Load model
    model = None
    regime = "A"
    if args.checkpoint:
        print(f"Loading model from: {args.checkpoint}")
        try:
            model = load_universal_model(args.checkpoint)
            print(f"Loaded UniversalEmulator with {model.get_num_parameters():,} parameters")

            # Extract regime
            import torch
            checkpoint = torch.load(args.checkpoint, map_location="cpu")
            if "config" in checkpoint:
                regime = checkpoint["config"].get("regime", "A")
                if hasattr(regime, "value"):
                    regime = regime.value
                print(f"Detected regime: {regime}")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("Using oracle mode (ground truth)")
            model = None
    else:
        print("No checkpoint provided, using oracle mode")

    # Evaluate
    print(f"\nEvaluating on worlds: {world_ids}")
    print(f"Splits: {split_names}")

    results = evaluate_all_worlds(
        model=model,
        dataset_path=args.dataset,
        world_ids=world_ids,
        split_names=split_names,
        shock_idx=0,
        weight_scheme="uniform",
        regime=regime,
    )

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    agg = results.get("aggregate", {})
    print(f"NRMSE:           {agg.get('nrmse', 0.0):.4f}")
    print(f"IAE:             {agg.get('iae', 0.0):.4f}")
    print(f"Sign@Impact:     {agg.get('sign_at_impact', 0.0):.4f}")
    print(f"HF-ratio:        {agg.get('hf_ratio', 0.0):.4f}")
    print(f"Overshoot:       {agg.get('overshoot_ratio', 0.0):.4f}")
    print("="*80)

    # Save results
    results_path = args.output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Generate figures if requested
    if args.figures:
        print("\nGenerating figures...")

        # 1. NRMSE bar chart
        print("  - NRMSE bar chart")
        fig = plot_nrmse_bar_chart(
            results,
            args.output_dir / "nrmse_bar_chart.png",
        )

        # 2. Shape comparison
        print("  - Shape metrics")
        fig = plot_shape_comparison(
            results,
            args.output_dir / "shape_comparison.png",
        )

        # 3. Sample IRF panel (if we have predictions)
        if model is not None:
            print("  - Sample IRF panel")
            # Load one sample for visualization
            import numpy as np
            import zarr

            world_id = world_ids[0]
            world_dir = args.dataset / world_id

            irfs_store = zarr.open(str(world_dir / "irfs.zarr"), mode="r")
            theta_store = zarr.open(str(world_dir / "theta.zarr"), mode="r")

            # Get first 5 samples from test split
            manifest_path = args.dataset / "manifest.json"
            with open(manifest_path) as f:
                manifest = json.load(f)

            split_indices = manifest["splits"][world_id][split_names[0]][:5]
            y_true = np.array(irfs_store[split_indices])[:, 0, :, :]  # (5, H+1, 3)
            theta = np.array(theta_store[split_indices])

            # Compute predictions
            from emulator.eval.evaluate import compute_predictions
            y_pred = compute_predictions(
                model=model,
                theta=theta,
                world_id=world_id,
                shock_idx=0,
                H=y_true.shape[1] - 1,
                regime=regime,
            )

            # Plot
            fig = plot_irf_panel(
                y_pred=y_pred,
                y_true=y_true,
                world_id=world_id,
                output_path=args.output_dir / "irf_panel.png",
            )

        print(f"\nFigures saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
