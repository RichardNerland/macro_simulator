"""Manifest generation for Universal Macro Emulator datasets.

This module creates and validates manifest.json files that serve as the
authoritative metadata source for datasets.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Stub imports for simulators.base (being built by another agent)
try:
    from simulators.base import (
        ObservableManifest,
        ParameterManifest,
        ShockManifest,
    )
except ImportError:
    # Stubs for development
    from dataclasses import dataclass, field

    @dataclass
    class ParameterManifest:
        """Stub for ParameterManifest during development."""

        names: list[str]
        units: list[str]
        bounds: np.ndarray
        defaults: np.ndarray
        priors: list[dict] = field(default_factory=list)

    @dataclass
    class ShockManifest:
        """Stub for ShockManifest during development."""

        names: list[str]
        n_shocks: int
        sigma: np.ndarray
        default_size: float = 1.0

    @dataclass
    class ObservableManifest:
        """Stub for ObservableManifest during development."""

        canonical_names: list[str]
        extra_names: list[str]
        n_canonical: int = 3
        n_extra: int = 0


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class ManifestGenerator:
    """Generate valid manifest.json for datasets.

    The manifest is the authoritative metadata source per spec §4.2.
    All dataset queries should reference the manifest first.

    Attributes:
        version: Dataset version in semver format
        output_dir: Root directory for dataset
        git_hash: Git commit hash or "unknown"
        simulators: Dictionary of simulator configurations
        splits: Dictionary of split configurations
        rng_seeds: RNG seed configuration
    """

    def __init__(
        self,
        version: str,
        output_dir: str | Path,
        global_seed: int = 42,
        T: int | None = None,
        H: int = 40,
    ):
        """Initialize manifest generator.

        Args:
            version: Dataset version in semver format (e.g., "1.0.0")
            output_dir: Root directory for dataset (e.g., datasets/v1.0/)
            global_seed: Global RNG seed for reproducibility
            T: Trajectory length (optional, can vary by simulator)
            H: IRF horizon (default 40)
        """
        self.version = version
        self.output_dir = Path(output_dir)
        self.global_seed = global_seed
        self.T = T
        self.H = H

        # Initialize manifest structure
        self.manifest: dict[str, Any] = {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_hash": self._get_git_hash(),
            "simulators": {},
            "splits": {},
            "rng_seeds": {
                "global": global_seed,
                "per_sample": True,
                "derivation_algorithm": "global_seed + sample_index",
            },
        }

        # Add optional fields
        if T is not None:
            self.manifest["T"] = T
        if H != 40:  # Only include if non-default
            self.manifest["H"] = H

    def add_simulator(
        self,
        world_id: str,
        n_samples: int,
        param_manifest: ParameterManifest,
        shock_manifest: ShockManifest,
        obs_manifest: ObservableManifest,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Add a simulator configuration to the manifest.

        Args:
            world_id: Unique simulator identifier (e.g., "nk", "var")
            n_samples: Number of samples generated
            param_manifest: Parameter metadata
            shock_manifest: Shock metadata
            obs_manifest: Observable metadata
            config: Optional simulator-specific configuration
        """
        # Compute config hash for reproducibility
        config_hash = self._compute_config_hash(config) if config else "default"

        self.manifest["simulators"][world_id] = {
            "n_samples": n_samples,
            "param_manifest": self._serialize_param_manifest(param_manifest),
            "shock_manifest": self._serialize_shock_manifest(shock_manifest),
            "obs_manifest": self._serialize_obs_manifest(obs_manifest),
            "config_hash": config_hash,
        }

        # Store full config if provided (optional, for reference)
        if config is not None:
            self.manifest["simulators"][world_id]["config"] = config

    def add_split(
        self,
        split_name: str,
        fraction: float,
        seed: int | None = None,
        algorithm: str | None = None,
        **kwargs,
    ) -> None:
        """Add a train/val/test split configuration.

        Args:
            split_name: Name of split (e.g., "train", "val", "test_interpolation")
            fraction: Fraction of data in this split
            seed: Random seed for split (defaults to global_seed)
            algorithm: Optional split algorithm identifier
            **kwargs: Additional split-specific parameters
        """
        if seed is None:
            seed = self.global_seed

        split_config: dict[str, Any] = {
            "fraction": fraction,
            "seed": seed,
        }

        if algorithm is not None:
            split_config["algorithm"] = algorithm

        # Add any additional parameters
        split_config.update(kwargs)

        self.manifest["splits"][split_name] = split_config

    def add_standard_splits(self, seed: int | None = None) -> None:
        """Add standard train/val/test splits per spec §4.5.

        Args:
            seed: Random seed (defaults to global_seed)
        """
        if seed is None:
            seed = self.global_seed

        self.add_split("train", fraction=0.8, seed=seed, algorithm="random_interpolation")
        self.add_split("val", fraction=0.1, seed=seed, algorithm="random_interpolation")
        self.add_split(
            "test_interpolation", fraction=0.05, seed=seed, algorithm="random_interpolation"
        )
        self.add_split(
            "test_extrapolation_slice",
            fraction=0.025,
            seed=seed,
            algorithm="slice_predicate",
        )
        self.add_split(
            "test_extrapolation_corner",
            fraction=0.025,
            seed=seed,
            algorithm="corner_extremes",
        )

    def add_notes(self, notes: str) -> None:
        """Add optional notes to the manifest.

        Args:
            notes: Free-form notes about the dataset
        """
        self.manifest["notes"] = notes

    def save(self, validate: bool = True) -> Path:
        """Save manifest to JSON file.

        Args:
            validate: Whether to validate against schema before saving

        Returns:
            Path to saved manifest file

        Raises:
            ValueError: If manifest is invalid and validate=True
        """
        manifest_path = self.output_dir / "manifest.json"

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate if requested
        if validate:
            self.validate()

        # Write to file with pretty formatting
        with open(manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2, cls=NumpyEncoder)

        return manifest_path

    def validate(self) -> None:
        """Validate manifest against schema.

        Raises:
            ValueError: If manifest is invalid
        """
        # Check required fields
        required_fields = [
            "version",
            "created_at",
            "git_hash",
            "simulators",
            "splits",
            "rng_seeds",
        ]
        for field in required_fields:
            if field not in self.manifest:
                raise ValueError(f"Missing required field: {field}")

        # Check version format (semver)
        version_parts = self.manifest["version"].split(".")
        if len(version_parts) != 3:
            raise ValueError(f"Invalid version format: {self.manifest['version']}")

        # Check simulators
        if not self.manifest["simulators"]:
            raise ValueError("No simulators added to manifest")

        for world_id, sim_config in self.manifest["simulators"].items():
            self._validate_simulator_config(world_id, sim_config)

        # Check splits
        required_splits = ["train", "val", "test_interpolation"]
        for split_name in required_splits:
            if split_name not in self.manifest["splits"]:
                raise ValueError(f"Missing required split: {split_name}")

        # Check RNG seeds
        if "global" not in self.manifest["rng_seeds"]:
            raise ValueError("Missing global seed in rng_seeds")
        if "per_sample" not in self.manifest["rng_seeds"]:
            raise ValueError("Missing per_sample flag in rng_seeds")

    def _validate_simulator_config(self, world_id: str, config: dict) -> None:
        """Validate a single simulator configuration."""
        required_fields = [
            "n_samples",
            "param_manifest",
            "shock_manifest",
            "obs_manifest",
            "config_hash",
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Simulator {world_id} missing field: {field}")

        # Validate param_manifest
        param_manifest = config["param_manifest"]
        if len(param_manifest["names"]) != len(param_manifest["defaults"]):
            raise ValueError(
                f"Simulator {world_id}: param names/defaults length mismatch"
            )

        # Validate shock_manifest
        shock_manifest = config["shock_manifest"]
        if shock_manifest["n_shocks"] != len(shock_manifest["names"]):
            raise ValueError(
                f"Simulator {world_id}: n_shocks != len(shock names)"
            )

        # Validate obs_manifest
        obs_manifest = config["obs_manifest"]
        if obs_manifest["n_canonical"] != 3:
            raise ValueError(
                f"Simulator {world_id}: n_canonical must be 3, got {obs_manifest['n_canonical']}"
            )
        if len(obs_manifest["canonical_names"]) != 3:
            raise ValueError(
                f"Simulator {world_id}: canonical_names must have 3 elements"
            )

    @staticmethod
    def _serialize_param_manifest(manifest: ParameterManifest) -> dict[str, Any]:
        """Serialize ParameterManifest to JSON-compatible dict."""
        return {
            "names": manifest.names,
            "units": manifest.units,
            "bounds": manifest.bounds.tolist(),
            "defaults": manifest.defaults.tolist(),
            "priors": manifest.priors if manifest.priors else [],
        }

    @staticmethod
    def _serialize_shock_manifest(manifest: ShockManifest) -> dict[str, Any]:
        """Serialize ShockManifest to JSON-compatible dict."""
        return {
            "names": manifest.names,
            "n_shocks": manifest.n_shocks,
            "sigma": manifest.sigma.tolist(),
            "default_size": manifest.default_size,
        }

    @staticmethod
    def _serialize_obs_manifest(manifest: ObservableManifest) -> dict[str, Any]:
        """Serialize ObservableManifest to JSON-compatible dict."""
        return {
            "canonical_names": manifest.canonical_names,
            "extra_names": manifest.extra_names,
            "n_canonical": manifest.n_canonical,
            "n_extra": manifest.n_extra,
        }

    @staticmethod
    def _get_git_hash() -> str:
        """Get current git commit hash, or "unknown" if not in git repo."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            return result.stdout.strip()[:40]  # Full hash
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return "unknown"

    @staticmethod
    def _compute_config_hash(config: dict[str, Any]) -> str:
        """Compute hash of configuration for reproducibility.

        Args:
            config: Configuration dictionary

        Returns:
            SHA256 hash (first 16 characters)
        """
        # Sort keys for deterministic hashing
        config_str = json.dumps(config, sort_keys=True, cls=NumpyEncoder)
        hash_obj = hashlib.sha256(config_str.encode())
        return hash_obj.hexdigest()[:16]

    @staticmethod
    def load(manifest_path: str | Path) -> dict[str, Any]:
        """Load manifest from JSON file.

        Args:
            manifest_path: Path to manifest.json

        Returns:
            Manifest dictionary

        Raises:
            FileNotFoundError: If manifest file doesn't exist
            json.JSONDecodeError: If manifest is not valid JSON
        """
        manifest_path = Path(manifest_path)
        with open(manifest_path, "r") as f:
            return json.load(f)

    @staticmethod
    def validate_file(manifest_path: str | Path) -> None:
        """Validate a manifest file against schema.

        Args:
            manifest_path: Path to manifest.json

        Raises:
            ValueError: If manifest is invalid
        """
        manifest = ManifestGenerator.load(manifest_path)

        # Create temporary ManifestGenerator to use validation logic
        temp_gen = ManifestGenerator(
            version=manifest.get("version", "1.0.0"),
            output_dir=Path(manifest_path).parent,
        )
        temp_gen.manifest = manifest
        temp_gen.validate()
