#!/usr/bin/env python
"""Dataset generation script for Universal Macro Emulator.

This script generates datasets for all or specific simulators, with support for:
- Deterministic seeding (per spec §4.2)
- Parallel generation via multiprocessing
- Split construction (train/val/test/extrapolation)
- Manifest generation with version tracking

Usage:
    python -m data.scripts.generate_dataset --world all --n_samples 10000 --seed 42 --output datasets/v1.0/
    python -m data.scripts.generate_dataset --world nk --n_samples 100000 --seed 42 --output datasets/v1.0/
    python -m data.scripts.generate_dataset --world lss,var,nk --n_samples 50000 --workers 4
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np

# Import simulators
try:
    from simulators import (
        LSSSimulator,
        NKSimulator,
        RBCSimulator,
        SwitchingSimulator,
        VARSimulator,
        ZLBSimulator,
    )
    from simulators.base import SimulatorAdapter
except ImportError as e:
    print(f"Error importing simulators: {e}")
    print("Make sure simulators are implemented.")
    exit(1)

from data.manifest import ManifestGenerator
from data.splits import construct_all_splits
from data.writer import DatasetWriter

# Registry of available simulators
SIMULATOR_REGISTRY: dict[str, type[SimulatorAdapter]] = {
    "lss": LSSSimulator,
    "var": VARSimulator,
    "nk": NKSimulator,
    "rbc": RBCSimulator,
    "switching": SwitchingSimulator,
    "zlb": ZLBSimulator,
}


def generate_sample(
    simulator: SimulatorAdapter,
    sample_idx: int,
    global_seed: int,
    T: int,
    H: int,
) -> dict[str, Any]:
    """Generate a single sample (theta, trajectories, IRFs, shocks).

    This function is deterministic given sample_idx and global_seed.

    Args:
        simulator: Simulator instance
        sample_idx: Index of this sample
        global_seed: Global seed for reproducibility
        T: Trajectory length
        H: IRF horizon

    Returns:
        Dictionary with keys: theta, trajectories, irfs, shocks, metadata
    """
    # Derive per-sample seed deterministically
    sample_seed = global_seed + sample_idx
    rng = np.random.default_rng(sample_seed)

    # 1. Sample parameters
    theta = simulator.sample_parameters(rng)

    # 2. Generate shock sequence (in std dev units)
    n_shocks = simulator.shock_manifest.n_shocks
    eps = rng.standard_normal((T, n_shocks))

    # 3. Simulate trajectory
    output = simulator.simulate(theta=theta, eps=eps, T=T, x0=None)
    trajectories = output.y_canonical  # Shape: (T, 3)

    # 4. Compute IRFs for all shocks
    irfs = np.zeros((n_shocks, H + 1, 3))
    for shock_idx in range(n_shocks):
        irf = simulator.compute_irf(
            theta=theta,
            shock_idx=shock_idx,
            shock_size=1.0,  # 1 std dev shock
            H=H,
            x0=None,  # Steady-state IRF
        )
        irfs[shock_idx, :, :] = irf

    # 5. Collect metadata (if applicable)
    metadata = {}
    if output.regime_seq is not None:
        metadata["regime_seq"] = output.regime_seq

    return {
        "theta": theta,
        "trajectories": trajectories,
        "irfs": irfs,
        "shocks": eps,
        "metadata": metadata if metadata else None,
    }


def generate_samples_worker(args: tuple) -> list[dict[str, Any]]:
    """Worker function for parallel sample generation.

    Args:
        args: Tuple of (simulator_class, world_id, start_idx, end_idx, global_seed, T, H)

    Returns:
        List of sample dictionaries
    """
    simulator_class, world_id, start_idx, end_idx, global_seed, T, H = args

    # Create simulator instance
    simulator = simulator_class()

    # Generate samples
    samples = []
    for i in range(start_idx, end_idx):
        sample = generate_sample(simulator, i, global_seed, T, H)
        samples.append(sample)

        # Progress logging (every 100 samples)
        if (i - start_idx + 1) % 100 == 0:
            print(f"  [{world_id}] Worker: {i - start_idx + 1}/{end_idx - start_idx} samples")

    return samples


def generate_dataset_for_world(
    world_id: str,
    n_samples: int,
    global_seed: int,
    output_dir: Path,
    T: int = 200,
    H: int = 40,
    workers: int = 1,
    version: str = "1.0.0",
) -> dict[str, Any]:
    """Generate dataset for a single simulator world.

    Args:
        world_id: Simulator identifier (e.g., "nk", "var")
        n_samples: Number of samples to generate
        global_seed: Global random seed
        output_dir: Output directory for dataset
        T: Trajectory length
        H: IRF horizon
        workers: Number of parallel workers
        version: Dataset version string

    Returns:
        Summary statistics dictionary
    """
    print(f"\n{'='*80}")
    print(f"Generating dataset for {world_id.upper()} simulator")
    print(f"{'='*80}")
    print(f"  Samples: {n_samples}")
    print(f"  T: {T}, H: {H}")
    print(f"  Seed: {global_seed}")
    print(f"  Workers: {workers}")

    # Get simulator class
    if world_id not in SIMULATOR_REGISTRY:
        print(f"  ERROR: Simulator '{world_id}' not found in registry")
        return {"error": f"Unknown simulator: {world_id}"}

    simulator_class = SIMULATOR_REGISTRY[world_id]

    # Create simulator instance to get manifests
    simulator = simulator_class()

    # Initialize DatasetWriter
    n_params = len(simulator.param_manifest.names)
    n_shocks = simulator.shock_manifest.n_shocks

    writer = DatasetWriter(
        output_dir=output_dir,
        world_id=world_id,
        n_samples=n_samples,
        T=T,
        H=H,
        n_params=n_params,
        n_shocks=n_shocks,
        n_obs=3,  # Canonical observables
        dtype="float32",
    )

    print(f"  Initialized Zarr arrays")

    # Generate samples (parallel or sequential)
    start_time = time.time()

    if workers > 1:
        # Parallel generation
        print(f"  Generating samples in parallel ({workers} workers)...")

        # Split work across workers
        chunk_size = n_samples // workers
        worker_args = []

        for w in range(workers):
            start_idx = w * chunk_size
            end_idx = (w + 1) * chunk_size if w < workers - 1 else n_samples
            worker_args.append((simulator_class, world_id, start_idx, end_idx, global_seed, T, H))

        # Run workers
        with mp.Pool(processes=workers) as pool:
            results = pool.map(generate_samples_worker, worker_args)

        # Flatten results and write to Zarr
        print(f"  Writing samples to Zarr...")
        sample_idx = 0
        for worker_samples in results:
            for sample in worker_samples:
                writer.write_sample(
                    sample_idx=sample_idx,
                    theta=sample["theta"],
                    trajectories=sample["trajectories"],
                    irfs=sample["irfs"],
                    shocks=sample["shocks"],
                    metadata=sample["metadata"],
                )
                sample_idx += 1

    else:
        # Sequential generation
        print(f"  Generating samples sequentially...")
        for i in range(n_samples):
            sample = generate_sample(simulator, i, global_seed, T, H)

            writer.write_sample(
                sample_idx=i,
                theta=sample["theta"],
                trajectories=sample["trajectories"],
                irfs=sample["irfs"],
                shocks=sample["shocks"],
                metadata=sample["metadata"],
            )

            # Progress logging
            if (i + 1) % 100 == 0 or (i + 1) == n_samples:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"  Progress: {i+1}/{n_samples} ({100*(i+1)/n_samples:.1f}%) - {rate:.1f} samples/sec")

    elapsed = time.time() - start_time
    print(f"  Generation complete: {elapsed:.2f}s ({n_samples/elapsed:.1f} samples/sec)")

    # Finalize writer
    summary = writer.finalize()

    # Collect theta for split construction
    print(f"  Constructing train/val/test splits...")
    theta_all = writer.theta[:]  # Load all theta

    # Construct splits
    splits = construct_all_splits(
        theta=theta_all,
        param_manifest=simulator.param_manifest,
        world_id=world_id,
        seed=global_seed,
    )

    # Save splits to disk
    splits_file = output_dir / world_id / "splits.json"
    splits_serializable = {k: v.tolist() for k, v in splits.items()}
    with open(splits_file, "w") as f:
        json.dump(splits_serializable, f, indent=2)

    print(f"  Split sizes:")
    for split_name, split_indices in splits.items():
        frac = len(split_indices) / n_samples
        print(f"    {split_name}: {len(split_indices)} ({frac*100:.1f}%)")

    # Add summary info
    summary["splits"] = {k: len(v) for k, v in splits.items()}
    summary["generation_time_sec"] = elapsed
    summary["samples_per_sec"] = n_samples / elapsed

    print(f"  Dataset for {world_id} complete!")

    return summary


def generate_manifest(
    output_dir: Path,
    world_summaries: dict[str, dict],
    global_seed: int,
    version: str,
    T: int,
    H: int,
) -> Path:
    """Generate manifest.json for the dataset.

    Args:
        output_dir: Output directory for dataset
        world_summaries: Dictionary of world_id -> summary stats
        global_seed: Global random seed
        version: Dataset version
        T: Trajectory length
        H: IRF horizon

    Returns:
        Path to saved manifest
    """
    print(f"\n{'='*80}")
    print(f"Generating manifest.json")
    print(f"{'='*80}")

    gen = ManifestGenerator(
        version=version,
        output_dir=output_dir,
        global_seed=global_seed,
        T=T,
        H=H,
    )

    # Add simulators
    for world_id, summary in world_summaries.items():
        if "error" in summary:
            continue

        # Recreate simulator to get manifests
        simulator_class = SIMULATOR_REGISTRY[world_id]
        simulator = simulator_class()

        gen.add_simulator(
            world_id=world_id,
            n_samples=summary["n_samples"],
            param_manifest=simulator.param_manifest,
            shock_manifest=simulator.shock_manifest,
            obs_manifest=simulator.obs_manifest,
            config={
                "T": T,
                "H": H,
                "dtype": summary["dtype"],
            },
        )

    # Add standard splits
    gen.add_standard_splits(seed=global_seed)

    # Add notes
    gen.add_notes(
        f"Generated with generate_dataset.py. "
        f"Total samples: {sum(s.get('n_samples', 0) for s in world_summaries.values())}. "
        f"Worlds: {', '.join(world_summaries.keys())}."
    )

    # Save manifest
    manifest_path = gen.save(validate=True)
    print(f"  Manifest saved: {manifest_path}")

    return manifest_path


def main():
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate datasets for Universal Macro Emulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dataset for all simulators
  python -m data.scripts.generate_dataset --world all --n_samples 10000 --seed 42 --output datasets/v1.0-dev/

  # Generate dataset for specific simulator
  python -m data.scripts.generate_dataset --world nk --n_samples 100000 --seed 42 --output datasets/v1.0/

  # Generate with parallel workers
  python -m data.scripts.generate_dataset --world lss,var,nk --n_samples 50000 --workers 4 --output datasets/v1.0/
        """,
    )

    parser.add_argument(
        "--world",
        type=str,
        required=True,
        help="Simulator(s) to generate data for. Use 'all' for all, or comma-separated list (e.g., 'lss,var,nk')",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        required=True,
        help="Number of samples to generate per simulator",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for dataset (e.g., datasets/v1.0/)",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=200,
        help="Trajectory length (default: 200)",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=40,
        help="IRF horizon (default: 40)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for generation (default: 1 = sequential)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="Dataset version string (default: 1.0.0)",
    )

    args = parser.parse_args()

    # Parse world list
    if args.world == "all":
        world_ids = list(SIMULATOR_REGISTRY.keys())
    else:
        world_ids = [w.strip() for w in args.world.split(",")]

    # Validate world IDs
    for world_id in world_ids:
        if world_id not in SIMULATOR_REGISTRY:
            print(f"ERROR: Unknown simulator '{world_id}'")
            print(f"Available simulators: {', '.join(SIMULATOR_REGISTRY.keys())}")
            return 1

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*80}")
    print(f"Dataset Generation Pipeline")
    print(f"{'='*80}")
    print(f"  Output directory: {output_dir}")
    print(f"  Worlds: {', '.join(world_ids)}")
    print(f"  Samples per world: {args.n_samples}")
    print(f"  T={args.T}, H={args.H}")
    print(f"  Global seed: {args.seed}")
    print(f"  Workers: {args.workers}")
    print(f"  Version: {args.version}")

    # Generate datasets for each world
    world_summaries = {}
    total_start = time.time()

    for world_id in world_ids:
        try:
            summary = generate_dataset_for_world(
                world_id=world_id,
                n_samples=args.n_samples,
                global_seed=args.seed,
                output_dir=output_dir,
                T=args.T,
                H=args.H,
                workers=args.workers,
                version=args.version,
            )
            world_summaries[world_id] = summary

        except Exception as e:
            print(f"  ERROR generating dataset for {world_id}: {e}")
            import traceback
            traceback.print_exc()
            world_summaries[world_id] = {"error": str(e)}

    total_elapsed = time.time() - total_start

    # Generate manifest
    try:
        manifest_path = generate_manifest(
            output_dir=output_dir,
            world_summaries=world_summaries,
            global_seed=args.seed,
            version=args.version,
            T=args.T,
            H=args.H,
        )
    except Exception as e:
        print(f"ERROR generating manifest: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print summary
    print(f"\n{'='*80}")
    print(f"Dataset Generation Complete")
    print(f"{'='*80}")
    print(f"  Total time: {total_elapsed:.2f}s")
    print(f"  Worlds processed: {len([s for s in world_summaries.values() if 'error' not in s])}/{len(world_ids)}")
    print(f"  Total samples: {sum(s.get('n_samples', 0) for s in world_summaries.values() if 'error' not in s)}")
    print(f"  Dataset location: {output_dir}")
    print(f"  Manifest: {manifest_path}")

    # Print per-world summaries
    print(f"\nPer-World Summary:")
    for world_id, summary in world_summaries.items():
        if "error" in summary:
            print(f"  {world_id}: ERROR - {summary['error']}")
        else:
            print(f"  {world_id}:")
            print(f"    Samples: {summary['n_samples']}")
            print(f"    Time: {summary['generation_time_sec']:.2f}s ({summary['samples_per_sec']:.1f} samples/sec)")
            print(f"    Splits: train={summary['splits']['train']}, val={summary['splits']['val']}, "
                  f"test_interp={summary['splits']['test_interpolation']}")

    return 0


if __name__ == "__main__":
    exit(main())
