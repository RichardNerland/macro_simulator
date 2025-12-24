"""Zarr dataset writer for Universal Macro Emulator.

This module provides functionality to write simulator outputs to Zarr format
with proper chunking, compression, and metadata handling.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

# Stub imports for simulators.base (being built by another agent)
# These will be replaced with actual imports when simulators.base is ready
try:
    from simulators.base import SimulatorOutput
except ImportError:
    # Stub for development
    from dataclasses import dataclass

    @dataclass
    class SimulatorOutput:
        """Stub for SimulatorOutput during development."""

        y_canonical: np.ndarray
        y_extra: np.ndarray | None = None
        x_state: np.ndarray | None = None
        regime_seq: np.ndarray | None = None


class DatasetWriter:
    """Write simulator outputs to Zarr arrays with efficient chunking.

    This class handles writing trajectories, IRFs, shocks, and parameters
    to Zarr format following the schema defined in spec §4.1.

    The storage schema is:
        {world_id}/
            trajectories.zarr  # (n_samples, T, n_obs)
            irfs.zarr          # (n_samples, n_shocks, H+1, 3)
            shocks.zarr        # (n_samples, T, n_shocks)
            theta.zarr         # (n_samples, n_params)
            metadata.zarr      # Optional: regime sequences, flags, etc.
    """

    def __init__(
        self,
        output_dir: str | Path,
        world_id: str,
        n_samples: int,
        T: int,
        H: int,
        n_params: int,
        n_shocks: int,
        n_obs: int = 3,
        dtype: str = "float32",
        chunk_size: int | None = None,
    ):
        """Initialize dataset writer.

        Args:
            output_dir: Root directory for dataset (e.g., datasets/v1.0/)
            world_id: Simulator identifier (e.g., "nk", "var", "lss")
            n_samples: Total number of samples to be written
            T: Trajectory length
            H: IRF horizon (output will be H+1 points)
            n_params: Number of parameters
            n_shocks: Number of shock types
            n_obs: Number of observables (default 3 for canonical)
            dtype: Data type for storage (default "float32" per spec §4.4)
            chunk_size: Chunk size along sample dimension (default: min(1000, n_samples))
        """
        self.output_dir = Path(output_dir)
        self.world_id = world_id
        self.n_samples = n_samples
        self.T = T
        self.H = H
        self.n_params = n_params
        self.n_shocks = n_shocks
        self.n_obs = n_obs
        self.dtype = dtype

        # Default chunk size: 1000 samples or total samples, whichever is smaller
        self.chunk_size = chunk_size if chunk_size is not None else min(1000, n_samples)

        # Create world directory
        self.world_dir = self.output_dir / world_id
        self.world_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Zarr arrays
        self._init_zarr_arrays()

        # Track number of samples written
        self._samples_written = 0

    def _init_zarr_arrays(self) -> None:
        """Initialize Zarr arrays with proper shapes and chunking."""
        # Trajectories: (n_samples, T, n_obs)
        # Chunk along sample dimension for efficient per-sample writes
        self.trajectories = zarr.open(
            str(self.world_dir / "trajectories.zarr"),
            mode="w",
            shape=(self.n_samples, self.T, self.n_obs),
            chunks=(self.chunk_size, self.T, self.n_obs),
            dtype=self.dtype,
        )
        self.trajectories.attrs["description"] = "Trajectories: (n_samples, T, n_obs)"
        self.trajectories.attrs["dimensions"] = ["sample", "time", "observable"]
        self.trajectories.attrs["observable_names"] = [
            "output",
            "inflation",
            "rate",
        ]

        # IRFs: (n_samples, n_shocks, H+1, 3)
        # Store all canonical observables, chunk along sample dimension
        self.irfs = zarr.open(
            str(self.world_dir / "irfs.zarr"),
            mode="w",
            shape=(self.n_samples, self.n_shocks, self.H + 1, 3),
            chunks=(self.chunk_size, self.n_shocks, self.H + 1, 3),
            dtype=self.dtype,
        )
        self.irfs.attrs["description"] = "IRFs: (n_samples, n_shocks, H+1, 3)"
        self.irfs.attrs["dimensions"] = ["sample", "shock", "horizon", "observable"]
        self.irfs.attrs["horizon_range"] = f"0..{self.H}"
        self.irfs.attrs["convention"] = "IRF[h] = y_shocked[h] - y_baseline[h]"

        # Shocks: (n_samples, T, n_shocks)
        # Shock sequences in standard deviation units
        self.shocks = zarr.open(
            str(self.world_dir / "shocks.zarr"),
            mode="w",
            shape=(self.n_samples, self.T, self.n_shocks),
            chunks=(self.chunk_size, self.T, self.n_shocks),
            dtype=self.dtype,
        )
        self.shocks.attrs["description"] = "Shock sequences: (n_samples, T, n_shocks)"
        self.shocks.attrs["dimensions"] = ["sample", "time", "shock"]
        self.shocks.attrs["units"] = "standard deviations"

        # Theta: (n_samples, n_params)
        # Parameters in natural units (ground truth)
        self.theta = zarr.open(
            str(self.world_dir / "theta.zarr"),
            mode="w",
            shape=(self.n_samples, self.n_params),
            chunks=(self.chunk_size, self.n_params),
            dtype=self.dtype,
        )
        self.theta.attrs["description"] = "Parameters: (n_samples, n_params)"
        self.theta.attrs["dimensions"] = ["sample", "parameter"]
        self.theta.attrs["units"] = "natural units (ground truth)"

        # Metadata array (optional): for regime sequences, binding flags, etc.
        # We'll create this on-demand if needed
        self.metadata_group = None

    def write_sample(
        self,
        sample_idx: int,
        theta: np.ndarray,
        trajectories: np.ndarray,
        irfs: np.ndarray,
        shocks: np.ndarray,
        metadata: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Write a single sample to the dataset.

        Args:
            sample_idx: Index of this sample (0-indexed)
            theta: Parameters, shape (n_params,)
            trajectories: Trajectory, shape (T, n_obs)
            irfs: IRFs for all shocks, shape (n_shocks, H+1, 3)
            shocks: Shock sequence, shape (T, n_shocks)
            metadata: Optional metadata dict (e.g., regime_seq, binding_flags)

        Raises:
            ValueError: If shapes don't match expected dimensions
            IndexError: If sample_idx is out of bounds
        """
        if sample_idx < 0 or sample_idx >= self.n_samples:
            raise IndexError(
                f"sample_idx {sample_idx} out of bounds [0, {self.n_samples})"
            )

        # Validate shapes
        self._validate_shapes(theta, trajectories, irfs, shocks)

        # Convert to target dtype
        theta = theta.astype(self.dtype, copy=False)
        trajectories = trajectories.astype(self.dtype, copy=False)
        irfs = irfs.astype(self.dtype, copy=False)
        shocks = shocks.astype(self.dtype, copy=False)

        # Write to Zarr arrays
        self.theta[sample_idx, :] = theta
        self.trajectories[sample_idx, :, :] = trajectories
        self.irfs[sample_idx, :, :, :] = irfs
        self.shocks[sample_idx, :, :] = shocks

        # Handle metadata if provided
        if metadata is not None:
            self._write_metadata(sample_idx, metadata)

        self._samples_written += 1

    def write_batch(
        self,
        start_idx: int,
        theta_batch: np.ndarray,
        trajectories_batch: np.ndarray,
        irfs_batch: np.ndarray,
        shocks_batch: np.ndarray,
        metadata_batch: list[dict[str, np.ndarray]] | None = None,
    ) -> None:
        """Write a batch of samples to the dataset.

        This is more efficient than repeated write_sample calls.

        Args:
            start_idx: Starting index for this batch
            theta_batch: Parameters, shape (batch_size, n_params)
            trajectories_batch: Trajectories, shape (batch_size, T, n_obs)
            irfs_batch: IRFs, shape (batch_size, n_shocks, H+1, 3)
            shocks_batch: Shocks, shape (batch_size, T, n_shocks)
            metadata_batch: Optional list of metadata dicts, one per sample
        """
        batch_size = theta_batch.shape[0]
        end_idx = start_idx + batch_size

        if end_idx > self.n_samples:
            raise IndexError(
                f"Batch [{start_idx}, {end_idx}) exceeds n_samples {self.n_samples}"
            )

        # Convert to target dtype
        theta_batch = theta_batch.astype(self.dtype, copy=False)
        trajectories_batch = trajectories_batch.astype(self.dtype, copy=False)
        irfs_batch = irfs_batch.astype(self.dtype, copy=False)
        shocks_batch = shocks_batch.astype(self.dtype, copy=False)

        # Write batch to Zarr arrays
        self.theta[start_idx:end_idx, :] = theta_batch
        self.trajectories[start_idx:end_idx, :, :] = trajectories_batch
        self.irfs[start_idx:end_idx, :, :, :] = irfs_batch
        self.shocks[start_idx:end_idx, :, :] = shocks_batch

        # Handle metadata batch if provided
        if metadata_batch is not None:
            for i, metadata in enumerate(metadata_batch):
                self._write_metadata(start_idx + i, metadata)

        self._samples_written += batch_size

    def _validate_shapes(
        self,
        theta: np.ndarray,
        trajectories: np.ndarray,
        irfs: np.ndarray,
        shocks: np.ndarray,
    ) -> None:
        """Validate that input arrays have correct shapes."""
        if theta.shape != (self.n_params,):
            raise ValueError(
                f"theta shape {theta.shape} != expected ({self.n_params},)"
            )

        if trajectories.shape != (self.T, self.n_obs):
            raise ValueError(
                f"trajectories shape {trajectories.shape} != expected ({self.T}, {self.n_obs})"
            )

        if irfs.shape != (self.n_shocks, self.H + 1, 3):
            raise ValueError(
                f"irfs shape {irfs.shape} != expected ({self.n_shocks}, {self.H + 1}, 3)"
            )

        if shocks.shape != (self.T, self.n_shocks):
            raise ValueError(
                f"shocks shape {shocks.shape} != expected ({self.T}, {self.n_shocks})"
            )

    def _write_metadata(self, sample_idx: int, metadata: dict[str, np.ndarray]) -> None:
        """Write optional metadata for a sample.

        Args:
            sample_idx: Index of the sample
            metadata: Dictionary of metadata arrays (e.g., regime_seq, binding_flags)
        """
        if self.metadata_group is None:
            # Initialize metadata group on first use
            self.metadata_group = zarr.open_group(
                str(self.world_dir / "metadata.zarr"), mode="a"
            )

        for key, value in metadata.items():
            if key not in self.metadata_group:
                # Create array for this metadata field
                # Assume 1D arrays for now, shape (n_samples, T) or (n_samples,)
                value_shape = value.shape
                full_shape = (self.n_samples,) + value_shape
                chunks = (self.chunk_size,) + value_shape

                self.metadata_group.create_dataset(
                    key,
                    shape=full_shape,
                    chunks=chunks,
                    dtype=value.dtype,
                )

            # Write metadata value
            self.metadata_group[key][sample_idx] = value

    def finalize(self) -> dict[str, Any]:
        """Finalize the dataset and return summary statistics.

        Returns:
            Dictionary with summary information about the written dataset
        """
        # Get stored bytes (method in zarr v3, property in v2)
        def get_nbytes_stored(arr):
            """Get nbytes_stored, handling both zarr v2 (property) and v3 (method)."""
            attr = arr.nbytes_stored
            return attr() if callable(attr) else attr

        traj_stored = get_nbytes_stored(self.trajectories)
        irfs_stored = get_nbytes_stored(self.irfs)
        shocks_stored = get_nbytes_stored(self.shocks)
        theta_stored = get_nbytes_stored(self.theta)

        summary = {
            "world_id": self.world_id,
            "n_samples": self.n_samples,
            "samples_written": self._samples_written,
            "T": self.T,
            "H": self.H,
            "n_params": self.n_params,
            "n_shocks": self.n_shocks,
            "n_obs": self.n_obs,
            "dtype": self.dtype,
            "chunk_size": self.chunk_size,
            "arrays": {
                "trajectories": {
                    "shape": self.trajectories.shape,
                    "nbytes": self.trajectories.nbytes,
                    "nbytes_stored": traj_stored,
                    "compression_ratio": (
                        self.trajectories.nbytes / traj_stored
                        if traj_stored > 0
                        else 0
                    ),
                },
                "irfs": {
                    "shape": self.irfs.shape,
                    "nbytes": self.irfs.nbytes,
                    "nbytes_stored": irfs_stored,
                    "compression_ratio": (
                        self.irfs.nbytes / irfs_stored
                        if irfs_stored > 0
                        else 0
                    ),
                },
                "shocks": {
                    "shape": self.shocks.shape,
                    "nbytes": self.shocks.nbytes,
                    "nbytes_stored": shocks_stored,
                    "compression_ratio": (
                        self.shocks.nbytes / shocks_stored
                        if shocks_stored > 0
                        else 0
                    ),
                },
                "theta": {
                    "shape": self.theta.shape,
                    "nbytes": self.theta.nbytes,
                    "nbytes_stored": theta_stored,
                    "compression_ratio": (
                        self.theta.nbytes / theta_stored
                        if theta_stored > 0
                        else 0
                    ),
                },
            },
        }

        # Check if all samples were written
        if self._samples_written != self.n_samples:
            summary["warning"] = (
                f"Only {self._samples_written}/{self.n_samples} samples written"
            )

        return summary

    @staticmethod
    def load_dataset(
        dataset_dir: str | Path, world_id: str
    ) -> tuple[zarr.Array, zarr.Array, zarr.Array, zarr.Array]:
        """Load a dataset from disk.

        Args:
            dataset_dir: Root directory for dataset
            world_id: Simulator identifier

        Returns:
            Tuple of (trajectories, irfs, shocks, theta) Zarr arrays
        """
        dataset_dir = Path(dataset_dir)
        world_dir = dataset_dir / world_id

        trajectories = zarr.open(str(world_dir / "trajectories.zarr"), mode="r")
        irfs = zarr.open(str(world_dir / "irfs.zarr"), mode="r")
        shocks = zarr.open(str(world_dir / "shocks.zarr"), mode="r")
        theta = zarr.open(str(world_dir / "theta.zarr"), mode="r")

        return trajectories, irfs, shocks, theta
