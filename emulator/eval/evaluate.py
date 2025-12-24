"""
Evaluation harness for the Universal Macro Emulator.

This module provides a CLI for evaluating trained models on dataset splits:
- Load model checkpoint and dataset
- Compute all metrics per world, per regime, per split
- Output structured JSON results
- Optionally generate diagnostic plots

Usage:
    python -m emulator.eval.evaluate --checkpoint path/to/checkpoint.pt --dataset path/to/dataset/

Output structure:
    {
        "aggregate": {"nrmse": 0.15, "iae": 2.3, ...},
        "per_world": {"lss": {...}, "var": {...}, ...},
        "per_split": {"interpolation": {...}, "extrapolation_slice": {...}, "lowo": {...}}
    }
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import zarr

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

from emulator.eval.metrics import (
    compute_sigma_from_data,
    gap_metric,
    hf_ratio,
    iae,
    nrmse,
    overshoot_ratio,
    sign_at_impact,
    sign_flip_count,
    uniform_weights,
    exponential_weights,
)


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (.pt)

    Returns:
        Dictionary with model state_dict and metadata
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for loading checkpoints. Install with: pip install torch")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")  # type: ignore
    return checkpoint


def load_dataset_split(
    dataset_path: Path,
    world_id: str,
    split_name: str,
) -> dict[str, npt.NDArray]:
    """Load a specific split from the dataset.

    Args:
        dataset_path: Root dataset directory
        world_id: Simulator identifier (e.g., "lss", "var", "nk")
        split_name: Split to load (e.g., "test_interpolation", "test_extrapolation_slice")

    Returns:
        Dictionary with:
            - irfs: IRF arrays, shape (n_samples, n_shocks, H+1, 3)
            - theta: Parameters, shape (n_samples, n_params)
            - shocks: Shock sequences, shape (n_samples, T, n_shocks)
    """
    world_dir = dataset_path / world_id

    # Load manifest to get split indices
    manifest_path = dataset_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Get split indices for this world
    if world_id not in manifest.get("splits", {}):
        raise ValueError(f"World {world_id} not found in manifest splits")

    world_splits = manifest["splits"][world_id]
    if split_name not in world_splits:
        raise ValueError(f"Split {split_name} not found for world {world_id}")

    split_indices = np.array(world_splits[split_name])

    # Load data
    irfs_store = zarr.open(world_dir / "irfs.zarr", mode="r")
    theta_store = zarr.open(world_dir / "theta.zarr", mode="r")
    shocks_store = zarr.open(world_dir / "shocks.zarr", mode="r")

    # Index into the split
    irfs = np.array(irfs_store[split_indices])
    theta = np.array(theta_store[split_indices])
    shocks = np.array(shocks_store[split_indices])

    return {
        "irfs": irfs,
        "theta": theta,
        "shocks": shocks,
        "n_samples": len(split_indices),
    }


def compute_predictions(
    model: Any,  # torch.nn.Module when available
    theta: npt.NDArray,
    world_id: str,
    shock_idx: int = 0,
    H: int = 40,
    batch_size: int = 32,
) -> npt.NDArray:
    """Compute model predictions for a batch of parameters.

    Args:
        model: Trained emulator model
        theta: Parameter array, shape (n_samples, n_params)
        world_id: Simulator identifier
        shock_idx: Which shock IRF to compute
        H: Horizon length
        batch_size: Batch size for inference

    Returns:
        Predictions, shape (n_samples, H+1, 3)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for model inference. Install with: pip install torch")

    model.eval()
    n_samples = theta.shape[0]
    predictions = []

    with torch.no_grad():  # type: ignore
        for i in range(0, n_samples, batch_size):
            batch_theta = torch.from_numpy(theta[i:i+batch_size]).float()  # type: ignore

            # Assume model has a forward signature: (theta, world_id, shock_idx, H)
            # This may need adjustment based on actual model interface
            batch_preds = model(
                theta=batch_theta,
                world_id=world_id,
                shock_idx=shock_idx,
                H=H,
            )

            predictions.append(batch_preds.cpu().numpy())

    return np.concatenate(predictions, axis=0)


def compute_all_metrics(
    y_pred: npt.NDArray,
    y_true: npt.NDArray,
    weight_scheme: str = "uniform",
) -> dict[str, float]:
    """Compute all metrics for a set of predictions.

    Args:
        y_pred: Predictions, shape (n_samples, H+1, 3) or (n_samples, n_shocks, H+1, 3)
        y_true: Ground truth, same shape as y_pred
        weight_scheme: Horizon weighting scheme ("uniform" or "exponential")

    Returns:
        Dictionary with all metric values
    """
    # Handle multi-shock case: average over shocks first
    if y_pred.ndim == 4:
        # (n_samples, n_shocks, H+1, 3) -> average over shocks
        y_pred_avg = np.mean(y_pred, axis=1)  # (n_samples, H+1, 3)
        y_true_avg = np.mean(y_true, axis=1)
    else:
        y_pred_avg = y_pred
        y_true_avg = y_true

    H = y_pred_avg.shape[1] - 1

    # Get weights
    if weight_scheme == "uniform":
        weights = uniform_weights(H)
    elif weight_scheme == "exponential":
        weights = exponential_weights(H, tau=20.0)
    else:
        raise ValueError(f"Unknown weight scheme: {weight_scheme}")

    # Compute sigma from data
    sigma = compute_sigma_from_data(y_true_avg)

    # Accuracy metrics
    nrmse_val = nrmse(y_pred_avg, y_true_avg, sigma=sigma, weights=weights)
    iae_val = iae(y_pred_avg, y_true_avg)
    sign_acc = sign_at_impact(y_pred_avg, y_true_avg, horizon_window=3)

    # Shape metrics
    hf_ratio_val = hf_ratio(y_pred_avg, cutoff_period=6.0)
    overshoot_val = overshoot_ratio(y_pred_avg)
    sign_flips = sign_flip_count(y_pred_avg)

    return {
        "nrmse": float(nrmse_val),
        "nrmse_weighted": float(nrmse_val),  # Same as nrmse with specified weights
        "iae": float(iae_val),
        "sign_at_impact": float(sign_acc),
        "hf_ratio": float(hf_ratio_val),
        "overshoot_ratio": float(overshoot_val),
        "sign_flip_count": float(sign_flips),
    }


def evaluate_on_split(
    model: Any | None,  # torch.nn.Module when available, or None for oracle
    dataset_path: Path,
    world_id: str,
    split_name: str,
    shock_idx: int = 0,
    weight_scheme: str = "uniform",
) -> dict[str, Any]:
    """Evaluate model on a specific split.

    Args:
        model: Trained emulator model
        dataset_path: Root dataset directory
        world_id: Simulator identifier
        split_name: Split to evaluate on
        shock_idx: Which shock IRF to compute
        weight_scheme: Horizon weighting scheme

    Returns:
        Dictionary with metrics and metadata
    """
    # Load data
    data = load_dataset_split(dataset_path, world_id, split_name)

    # Ground truth: extract single shock IRF
    y_true = data["irfs"][:, shock_idx, :, :]  # (n_samples, H+1, 3)
    theta = data["theta"]

    H = y_true.shape[1] - 1

    # Compute predictions
    # For now, we'll use ground truth as a placeholder
    # In actual use, replace with: y_pred = compute_predictions(model, theta, world_id, shock_idx, H)
    y_pred = y_true  # PLACEHOLDER: Replace with actual model predictions

    # Compute metrics
    metrics = compute_all_metrics(y_pred, y_true, weight_scheme=weight_scheme)

    # Add metadata
    metrics["n_samples"] = data["n_samples"]
    metrics["world_id"] = world_id
    metrics["split_name"] = split_name
    metrics["shock_idx"] = shock_idx

    return metrics


def evaluate_all_worlds(
    model: Any | None,  # torch.nn.Module when available, or None for oracle
    dataset_path: Path,
    world_ids: list[str],
    split_names: list[str],
    shock_idx: int = 0,
    weight_scheme: str = "uniform",
) -> dict[str, Any]:
    """Evaluate model on all worlds and splits.

    Args:
        model: Trained emulator model
        dataset_path: Root dataset directory
        world_ids: List of simulator identifiers to evaluate
        split_names: List of splits to evaluate
        shock_idx: Which shock IRF to compute
        weight_scheme: Horizon weighting scheme

    Returns:
        Nested dictionary with results:
        {
            "aggregate": {...},
            "per_world": {world_id: {...}},
            "per_split": {split_name: {...}},
            "per_world_per_split": {world_id: {split_name: {...}}}
        }
    """
    results = {
        "per_world": {},
        "per_split": {},
        "per_world_per_split": {},
    }

    # Collect all metrics
    all_metrics = []

    for world_id in world_ids:
        results["per_world_per_split"][world_id] = {}

        for split_name in split_names:
            try:
                metrics = evaluate_on_split(
                    model, dataset_path, world_id, split_name, shock_idx, weight_scheme
                )

                results["per_world_per_split"][world_id][split_name] = metrics
                all_metrics.append(metrics)

            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: Could not evaluate {world_id}/{split_name}: {e}")
                continue

    # Aggregate per world
    for world_id in world_ids:
        world_metrics_list = [
            m for m in all_metrics if m["world_id"] == world_id
        ]
        if world_metrics_list:
            results["per_world"][world_id] = aggregate_metrics(world_metrics_list)

    # Aggregate per split
    for split_name in split_names:
        split_metrics_list = [
            m for m in all_metrics if m["split_name"] == split_name
        ]
        if split_metrics_list:
            results["per_split"][split_name] = aggregate_metrics(split_metrics_list)

    # Global aggregate
    if all_metrics:
        results["aggregate"] = aggregate_metrics(all_metrics)
    else:
        results["aggregate"] = {}

    return results


def aggregate_metrics(metrics_list: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate metrics across multiple evaluations.

    Takes mean of each metric, weighted by n_samples.

    Args:
        metrics_list: List of metric dictionaries

    Returns:
        Aggregated metrics
    """
    if not metrics_list:
        return {}

    # Extract metric names (exclude metadata)
    metric_names = [
        "nrmse", "nrmse_weighted", "iae", "sign_at_impact",
        "hf_ratio", "overshoot_ratio", "sign_flip_count"
    ]

    # Compute weighted average
    total_samples = sum(m.get("n_samples", 1) for m in metrics_list)

    aggregated = {}
    for name in metric_names:
        values = [m.get(name, 0.0) for m in metrics_list]
        weights = [m.get("n_samples", 1) for m in metrics_list]

        if total_samples > 0:
            aggregated[name] = sum(v * w for v, w in zip(values, weights)) / total_samples
        else:
            aggregated[name] = 0.0

    aggregated["n_samples"] = total_samples

    return aggregated


def save_results(results: dict[str, Any], output_path: Path) -> None:
    """Save evaluation results to JSON file.

    Args:
        results: Results dictionary
        output_path: Path to output JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")


def print_summary(results: dict[str, Any]) -> None:
    """Print summary table of results.

    Args:
        results: Results dictionary
    """
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    # Aggregate results
    if "aggregate" in results and results["aggregate"]:
        print("\nAggregate Metrics:")
        agg = results["aggregate"]
        print(f"  NRMSE:           {agg.get('nrmse', 0.0):.4f}")
        print(f"  IAE:             {agg.get('iae', 0.0):.4f}")
        print(f"  Sign@Impact:     {agg.get('sign_at_impact', 0.0):.4f}")
        print(f"  HF-ratio:        {agg.get('hf_ratio', 0.0):.4f}")
        print(f"  Overshoot:       {agg.get('overshoot_ratio', 0.0):.4f}")
        print(f"  Sign flips:      {agg.get('sign_flip_count', 0.0):.2f}")
        print(f"  Samples:         {agg.get('n_samples', 0)}")

    # Per-world breakdown
    if "per_world" in results and results["per_world"]:
        print("\nPer-World Metrics:")
        print(f"{'World':<12} {'NRMSE':>8} {'IAE':>8} {'Sign@0':>8} {'HF-ratio':>8} {'Samples':>8}")
        print("-" * 80)

        for world_id, metrics in sorted(results["per_world"].items()):
            print(
                f"{world_id:<12} "
                f"{metrics.get('nrmse', 0.0):>8.4f} "
                f"{metrics.get('iae', 0.0):>8.4f} "
                f"{metrics.get('sign_at_impact', 0.0):>8.4f} "
                f"{metrics.get('hf_ratio', 0.0):>8.4f} "
                f"{metrics.get('n_samples', 0):>8}"
            )

    # Per-split breakdown
    if "per_split" in results and results["per_split"]:
        print("\nPer-Split Metrics:")
        print(f"{'Split':<25} {'NRMSE':>8} {'IAE':>8} {'Sign@0':>8} {'Samples':>8}")
        print("-" * 80)

        for split_name, metrics in sorted(results["per_split"].items()):
            print(
                f"{split_name:<25} "
                f"{metrics.get('nrmse', 0.0):>8.4f} "
                f"{metrics.get('iae', 0.0):>8.4f} "
                f"{metrics.get('sign_at_impact', 0.0):>8.4f} "
                f"{metrics.get('n_samples', 0):>8}"
            )

    print("="*80 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate Universal Macro Emulator on dataset splits"
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to model checkpoint (.pt file). If not provided, uses ground truth (oracle).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset directory (contains manifest.json and world subdirs)",
    )
    parser.add_argument(
        "--worlds",
        type=str,
        default="lss,var,nk,rbc,switching,zlb",
        help="Comma-separated list of world_ids to evaluate (default: all 6)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="test_interpolation",
        help="Comma-separated list of splits to evaluate (default: test_interpolation)",
    )
    parser.add_argument(
        "--shock-idx",
        type=int,
        default=0,
        help="Which shock IRF to evaluate (default: 0)",
    )
    parser.add_argument(
        "--weight-scheme",
        type=str,
        default="uniform",
        choices=["uniform", "exponential"],
        help="Horizon weighting scheme for NRMSE (default: uniform)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_results.json"),
        help="Path to output JSON file (default: eval_results.json)",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate diagnostic plots (requires --checkpoint)",
    )

    args = parser.parse_args()

    # Parse world IDs and splits
    world_ids = [w.strip() for w in args.worlds.split(",")]
    split_names = [s.strip() for s in args.splits.split(",")]

    print(f"Evaluating on {len(world_ids)} worlds, {len(split_names)} splits")
    print(f"World IDs: {world_ids}")
    print(f"Splits: {split_names}")

    # Load model
    model = None
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        checkpoint = load_checkpoint(args.checkpoint)

        # TODO: Instantiate actual model from checkpoint
        # For now, we use None and fall back to ground truth
        print("Warning: Model loading not implemented, using ground truth (oracle)")
    else:
        print("\nNo checkpoint provided, using ground truth (oracle mode)")

    # Evaluate
    print("\nRunning evaluation...")
    results = evaluate_all_worlds(
        model=model,
        dataset_path=args.dataset,
        world_ids=world_ids,
        split_names=split_names,
        shock_idx=args.shock_idx,
        weight_scheme=args.weight_scheme,
    )

    # Print summary
    print_summary(results)

    # Save results
    save_results(results, args.output)

    # Generate plots if requested
    if args.plots:
        if args.checkpoint is None:
            print("Warning: --plots requires --checkpoint, skipping plot generation")
        else:
            print("\nGenerating diagnostic plots...")
            # TODO: Implement plot generation
            print("Warning: Plot generation not yet implemented")


if __name__ == "__main__":
    main()
