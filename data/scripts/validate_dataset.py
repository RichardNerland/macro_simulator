#!/usr/bin/env python
"""Dataset validation script for Universal Macro Emulator.

This script validates datasets to ensure:
1. Zarr array integrity (shapes, dtypes, no NaNs)
2. Manifest consistency (metadata matches actual data)
3. Split consistency (disjoint, correct fractions)
4. Statistical sanity checks (value ranges, distributions)

Usage:
    python -m data.scripts.validate_dataset --path datasets/v1.0/
    python -m data.scripts.validate_dataset --path datasets/v1.0/ --world nk
    python -m data.scripts.validate_dataset --path datasets/v1.0/ --verbose
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from data.manifest import ManifestGenerator
from data.splits import validate_splits


class DatasetValidator:
    """Validator for Universal Macro Emulator datasets.

    This class performs comprehensive validation of generated datasets.
    """

    def __init__(self, dataset_path: Path, verbose: bool = False):
        """Initialize validator.

        Args:
            dataset_path: Path to dataset root directory
            verbose: Whether to print verbose output
        """
        self.dataset_path = Path(dataset_path)
        self.verbose = verbose
        self.errors = []
        self.warnings = []

    def log(self, message: str) -> None:
        """Log message if verbose."""
        if self.verbose:
            print(f"  {message}")

    def error(self, message: str) -> None:
        """Record an error."""
        self.errors.append(message)
        print(f"  ERROR: {message}")

    def warning(self, message: str) -> None:
        """Record a warning."""
        self.warnings.append(message)
        print(f"  WARNING: {message}")

    def validate_manifest_exists(self) -> bool:
        """Validate that manifest.json exists and is valid JSON."""
        manifest_path = self.dataset_path / "manifest.json"

        if not manifest_path.exists():
            self.error(f"Manifest not found: {manifest_path}")
            return False

        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            self.log(f"Manifest loaded: {manifest_path}")
            return True
        except json.JSONDecodeError as e:
            self.error(f"Manifest is not valid JSON: {e}")
            return False

    def validate_manifest_schema(self) -> bool:
        """Validate manifest against schema."""
        manifest_path = self.dataset_path / "manifest.json"

        try:
            ManifestGenerator.validate_file(manifest_path)
            self.log("Manifest schema validation passed")
            return True
        except ValueError as e:
            self.error(f"Manifest schema validation failed: {e}")
            return False

    def validate_zarr_arrays(self, world_id: str, manifest: dict) -> bool:
        """Validate Zarr arrays for a specific world.

        Args:
            world_id: Simulator identifier
            manifest: Loaded manifest dictionary

        Returns:
            True if validation passed, False otherwise
        """
        world_dir = self.dataset_path / world_id

        if not world_dir.exists():
            self.error(f"World directory not found: {world_dir}")
            return False

        # Get expected dimensions from manifest
        world_config = manifest["simulators"][world_id]
        n_samples = world_config["n_samples"]
        n_params = len(world_config["param_manifest"]["names"])
        n_shocks = world_config["shock_manifest"]["n_shocks"]
        T = manifest.get("T", 200)
        H = manifest.get("H", 40)

        passed = True

        # Validate trajectories.zarr
        try:
            traj = zarr.open(str(world_dir / "trajectories.zarr"), mode="r")
            expected_shape = (n_samples, T, 3)

            if traj.shape != expected_shape:
                self.error(f"{world_id}/trajectories.zarr: shape {traj.shape} != expected {expected_shape}")
                passed = False
            else:
                self.log(f"{world_id}/trajectories.zarr: shape OK {traj.shape}")

            # Check for NaNs
            if np.any(np.isnan(traj[:])):
                self.error(f"{world_id}/trajectories.zarr contains NaN values")
                passed = False

            # Check for infs
            if np.any(np.isinf(traj[:])):
                self.error(f"{world_id}/trajectories.zarr contains Inf values")
                passed = False

        except Exception as e:
            self.error(f"Failed to load {world_id}/trajectories.zarr: {e}")
            passed = False

        # Validate irfs.zarr
        try:
            irfs = zarr.open(str(world_dir / "irfs.zarr"), mode="r")
            expected_shape = (n_samples, n_shocks, H + 1, 3)

            if irfs.shape != expected_shape:
                self.error(f"{world_id}/irfs.zarr: shape {irfs.shape} != expected {expected_shape}")
                passed = False
            else:
                self.log(f"{world_id}/irfs.zarr: shape OK {irfs.shape}")

            # Check for NaNs
            if np.any(np.isnan(irfs[:])):
                self.error(f"{world_id}/irfs.zarr contains NaN values")
                passed = False

            # Check for infs
            if np.any(np.isinf(irfs[:])):
                self.error(f"{world_id}/irfs.zarr contains Inf values")
                passed = False

        except Exception as e:
            self.error(f"Failed to load {world_id}/irfs.zarr: {e}")
            passed = False

        # Validate shocks.zarr
        try:
            shocks = zarr.open(str(world_dir / "shocks.zarr"), mode="r")
            expected_shape = (n_samples, T, n_shocks)

            if shocks.shape != expected_shape:
                self.error(f"{world_id}/shocks.zarr: shape {shocks.shape} != expected {expected_shape}")
                passed = False
            else:
                self.log(f"{world_id}/shocks.zarr: shape OK {shocks.shape}")

            # Check for NaNs
            if np.any(np.isnan(shocks[:])):
                self.error(f"{world_id}/shocks.zarr contains NaN values")
                passed = False

        except Exception as e:
            self.error(f"Failed to load {world_id}/shocks.zarr: {e}")
            passed = False

        # Validate theta.zarr
        try:
            theta = zarr.open(str(world_dir / "theta.zarr"), mode="r")
            expected_shape = (n_samples, n_params)

            if theta.shape != expected_shape:
                self.error(f"{world_id}/theta.zarr: shape {theta.shape} != expected {expected_shape}")
                passed = False
            else:
                self.log(f"{world_id}/theta.zarr: shape OK {theta.shape}")

            # Check for NaNs
            if np.any(np.isnan(theta[:])):
                self.error(f"{world_id}/theta.zarr contains NaN values")
                passed = False

            # Check bounds
            bounds = np.array(world_config["param_manifest"]["bounds"])
            theta_data = theta[:]

            if np.any(theta_data < bounds[:, 0]) or np.any(theta_data > bounds[:, 1]):
                self.error(f"{world_id}/theta.zarr: some parameters out of bounds")
                passed = False

        except Exception as e:
            self.error(f"Failed to load {world_id}/theta.zarr: {e}")
            passed = False

        return passed

    def validate_splits(self, world_id: str, manifest: dict) -> bool:
        """Validate split files for a specific world.

        Args:
            world_id: Simulator identifier
            manifest: Loaded manifest dictionary

        Returns:
            True if validation passed, False otherwise
        """
        splits_file = self.dataset_path / world_id / "splits.json"

        if not splits_file.exists():
            self.warning(f"Splits file not found: {splits_file}")
            return False

        try:
            with open(splits_file, "r") as f:
                splits_dict = json.load(f)

            # Convert to numpy arrays
            splits = {k: np.array(v) for k, v in splits_dict.items()}

            # Get n_samples
            n_samples = manifest["simulators"][world_id]["n_samples"]

            # Validate splits
            validate_splits(splits, n_samples)
            self.log(f"{world_id}/splits.json: validation passed")

            # Check fractions
            for split_name, split_indices in splits.items():
                frac = len(split_indices) / n_samples
                self.log(f"  {split_name}: {len(split_indices)} ({frac*100:.1f}%)")

            return True

        except Exception as e:
            self.error(f"Split validation failed for {world_id}: {e}")
            return False

    def validate_statistical_sanity(self, world_id: str, manifest: dict) -> bool:
        """Perform basic statistical sanity checks.

        Args:
            world_id: Simulator identifier
            manifest: Loaded manifest dictionary

        Returns:
            True if checks passed, False otherwise
        """
        world_dir = self.dataset_path / world_id

        try:
            # Load data
            traj = zarr.open(str(world_dir / "trajectories.zarr"), mode="r")
            irfs = zarr.open(str(world_dir / "irfs.zarr"), mode="r")

            # Check trajectory statistics
            traj_data = traj[:]
            traj_mean = np.mean(traj_data, axis=(0, 1))
            traj_std = np.std(traj_data, axis=(0, 1))

            self.log(f"{world_id} trajectory stats:")
            self.log(f"  Mean: {traj_mean}")
            self.log(f"  Std:  {traj_std}")

            # Check for zero variance (bad)
            if np.any(traj_std < 1e-6):
                self.warning(f"{world_id}: trajectories have near-zero variance on some observables")

            # Check for extremely large values
            traj_max = np.max(np.abs(traj_data))
            if traj_max > 1000:
                self.warning(f"{world_id}: trajectories contain very large values (max abs = {traj_max})")

            # Check IRF statistics
            irfs_data = irfs[:]
            irfs_max = np.max(np.abs(irfs_data))

            self.log(f"{world_id} IRF max abs value: {irfs_max}")

            if irfs_max > 100:
                self.warning(f"{world_id}: IRFs contain very large values (max abs = {irfs_max})")

            # Check that IRFs decay (most should go to zero eventually)
            # Sample a few IRFs and check if they decay
            sample_irfs = irfs_data[0, 0, :, :]  # First sample, first shock
            initial_abs = np.mean(np.abs(sample_irfs[0, :]))
            final_abs = np.mean(np.abs(sample_irfs[-1, :]))

            if final_abs > initial_abs:
                self.warning(f"{world_id}: Sample IRF does not decay (initial={initial_abs}, final={final_abs})")

            return True

        except Exception as e:
            self.error(f"Statistical sanity check failed for {world_id}: {e}")
            return False

    def validate_world(self, world_id: str, manifest: dict) -> bool:
        """Validate all aspects of a single world.

        Args:
            world_id: Simulator identifier
            manifest: Loaded manifest dictionary

        Returns:
            True if all validations passed, False otherwise
        """
        print(f"\n{'='*80}")
        print(f"Validating {world_id.upper()}")
        print(f"{'='*80}")

        passed = True

        # Validate Zarr arrays
        if not self.validate_zarr_arrays(world_id, manifest):
            passed = False

        # Validate splits
        if not self.validate_splits(world_id, manifest):
            passed = False

        # Statistical sanity checks
        if not self.validate_statistical_sanity(world_id, manifest):
            passed = False

        return passed

    def validate_all(self, world_id: str | None = None) -> bool:
        """Validate entire dataset or specific world.

        Args:
            world_id: Optional specific world to validate (None = all)

        Returns:
            True if all validations passed, False otherwise
        """
        print(f"{'='*80}")
        print(f"Dataset Validation")
        print(f"{'='*80}")
        print(f"  Dataset path: {self.dataset_path}")

        # 1. Validate manifest exists
        if not self.validate_manifest_exists():
            return False

        # 2. Validate manifest schema
        if not self.validate_manifest_schema():
            return False

        # 3. Load manifest
        manifest_path = self.dataset_path / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # 4. Determine which worlds to validate
        if world_id is not None:
            if world_id not in manifest["simulators"]:
                self.error(f"World '{world_id}' not found in manifest")
                return False
            worlds_to_validate = [world_id]
        else:
            worlds_to_validate = list(manifest["simulators"].keys())

        # 5. Validate each world
        all_passed = True
        for wid in worlds_to_validate:
            if not self.validate_world(wid, manifest):
                all_passed = False

        # 6. Print summary
        print(f"\n{'='*80}")
        print(f"Validation Summary")
        print(f"{'='*80}")
        print(f"  Worlds validated: {len(worlds_to_validate)}")
        print(f"  Errors: {len(self.errors)}")
        print(f"  Warnings: {len(self.warnings)}")

        if self.errors:
            print(f"\n  Errors found:")
            for err in self.errors:
                print(f"    - {err}")

        if self.warnings:
            print(f"\n  Warnings:")
            for warn in self.warnings:
                print(f"    - {warn}")

        if all_passed and len(self.errors) == 0:
            print(f"\n  VALIDATION PASSED")
            return True
        else:
            print(f"\n  VALIDATION FAILED")
            return False


def main():
    """Main entry point for validation script."""
    parser = argparse.ArgumentParser(
        description="Validate datasets for Universal Macro Emulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to dataset root directory (e.g., datasets/v1.0/)",
    )
    parser.add_argument(
        "--world",
        type=str,
        default=None,
        help="Specific world to validate (default: all worlds)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )

    args = parser.parse_args()

    # Create validator
    validator = DatasetValidator(
        dataset_path=Path(args.path),
        verbose=args.verbose,
    )

    # Run validation
    passed = validator.validate_all(world_id=args.world)

    return 0 if passed else 1


if __name__ == "__main__":
    exit(main())
