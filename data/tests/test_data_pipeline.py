"""Tests for data pipeline: Zarr writer and manifest generator.

These tests verify that:
1. Zarr arrays are written with correct shapes
2. Manifest is valid JSON and matches schema
3. Round-trip: write → read produces same data
4. Deterministic seeding works correctly
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from data.manifest import ManifestGenerator, ParameterManifest, ShockManifest, ObservableManifest
from data.writer import DatasetWriter


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_param_manifest():
    """Create sample parameter manifest."""
    return ParameterManifest(
        names=["beta", "sigma", "phi_pi"],
        units=["-", "-", "-"],
        bounds=np.array([[0.985, 0.995], [0.5, 2.5], [1.05, 3.5]]),
        defaults=np.array([0.99, 1.0, 1.5]),
        priors=[],
    )


@pytest.fixture
def sample_shock_manifest():
    """Create sample shock manifest."""
    return ShockManifest(
        names=["monetary", "technology", "cost_push"],
        n_shocks=3,
        sigma=np.array([0.001, 0.005, 0.001]),
        default_size=1.0,
    )


@pytest.fixture
def sample_obs_manifest():
    """Create sample observable manifest."""
    return ObservableManifest(
        canonical_names=["output", "inflation", "rate"],
        extra_names=[],
        n_canonical=3,
        n_extra=0,
    )


@pytest.mark.fast
class TestDatasetWriter:
    """Test suite for DatasetWriter."""

    def test_init_creates_directories(self, temp_dir):
        """Test that initialization creates necessary directories."""
        writer = DatasetWriter(
            output_dir=temp_dir,
            world_id="test_world",
            n_samples=10,
            T=100,
            H=40,
            n_params=5,
            n_shocks=3,
        )

        assert (temp_dir / "test_world").exists()
        assert (temp_dir / "test_world" / "trajectories.zarr").exists()
        assert (temp_dir / "test_world" / "irfs.zarr").exists()
        assert (temp_dir / "test_world" / "shocks.zarr").exists()
        assert (temp_dir / "test_world" / "theta.zarr").exists()

    def test_zarr_arrays_have_correct_shapes(self, temp_dir):
        """Test that Zarr arrays are created with correct shapes."""
        n_samples, T, H, n_params, n_shocks = 100, 200, 40, 8, 3

        writer = DatasetWriter(
            output_dir=temp_dir,
            world_id="test_world",
            n_samples=n_samples,
            T=T,
            H=H,
            n_params=n_params,
            n_shocks=n_shocks,
        )

        assert writer.trajectories.shape == (n_samples, T, 3)
        assert writer.irfs.shape == (n_samples, n_shocks, H + 1, 3)
        assert writer.shocks.shape == (n_samples, T, n_shocks)
        assert writer.theta.shape == (n_samples, n_params)

    def test_write_single_sample(self, temp_dir):
        """Test writing a single sample."""
        n_samples, T, H, n_params, n_shocks = 10, 100, 40, 5, 3

        writer = DatasetWriter(
            output_dir=temp_dir,
            world_id="test_world",
            n_samples=n_samples,
            T=T,
            H=H,
            n_params=n_params,
            n_shocks=n_shocks,
        )

        # Create sample data
        theta = np.random.randn(n_params)
        trajectories = np.random.randn(T, 3)
        irfs = np.random.randn(n_shocks, H + 1, 3)
        shocks = np.random.randn(T, n_shocks)

        # Write sample
        writer.write_sample(
            sample_idx=0,
            theta=theta,
            trajectories=trajectories,
            irfs=irfs,
            shocks=shocks,
        )

        # Verify data was written
        np.testing.assert_allclose(writer.theta[0], theta, rtol=1e-5)
        np.testing.assert_allclose(writer.trajectories[0], trajectories, rtol=1e-5)
        np.testing.assert_allclose(writer.irfs[0], irfs, rtol=1e-5)
        np.testing.assert_allclose(writer.shocks[0], shocks, rtol=1e-5)

    def test_write_batch(self, temp_dir):
        """Test writing a batch of samples."""
        n_samples, T, H, n_params, n_shocks = 100, 100, 40, 5, 3
        batch_size = 10

        writer = DatasetWriter(
            output_dir=temp_dir,
            world_id="test_world",
            n_samples=n_samples,
            T=T,
            H=H,
            n_params=n_params,
            n_shocks=n_shocks,
        )

        # Create batch data
        theta_batch = np.random.randn(batch_size, n_params)
        trajectories_batch = np.random.randn(batch_size, T, 3)
        irfs_batch = np.random.randn(batch_size, n_shocks, H + 1, 3)
        shocks_batch = np.random.randn(batch_size, T, n_shocks)

        # Write batch
        writer.write_batch(
            start_idx=0,
            theta_batch=theta_batch,
            trajectories_batch=trajectories_batch,
            irfs_batch=irfs_batch,
            shocks_batch=shocks_batch,
        )

        # Verify data was written
        np.testing.assert_allclose(writer.theta[:batch_size], theta_batch, rtol=1e-5)
        np.testing.assert_allclose(
            writer.trajectories[:batch_size], trajectories_batch, rtol=1e-5
        )
        np.testing.assert_allclose(writer.irfs[:batch_size], irfs_batch, rtol=1e-5)
        np.testing.assert_allclose(writer.shocks[:batch_size], shocks_batch, rtol=1e-5)

    def test_dtype_conversion(self, temp_dir):
        """Test that data is correctly converted to target dtype."""
        writer = DatasetWriter(
            output_dir=temp_dir,
            world_id="test_world",
            n_samples=10,
            T=100,
            H=40,
            n_params=5,
            n_shocks=3,
            dtype="float32",
        )

        # Create float64 data
        theta = np.random.randn(5).astype(np.float64)
        trajectories = np.random.randn(100, 3).astype(np.float64)
        irfs = np.random.randn(3, 41, 3).astype(np.float64)
        shocks = np.random.randn(100, 3).astype(np.float64)

        writer.write_sample(
            sample_idx=0,
            theta=theta,
            trajectories=trajectories,
            irfs=irfs,
            shocks=shocks,
        )

        # Verify dtype is float32
        assert writer.theta.dtype == np.float32
        assert writer.trajectories.dtype == np.float32
        assert writer.irfs.dtype == np.float32
        assert writer.shocks.dtype == np.float32

    def test_shape_validation(self, temp_dir):
        """Test that shape validation catches mismatched shapes."""
        writer = DatasetWriter(
            output_dir=temp_dir,
            world_id="test_world",
            n_samples=10,
            T=100,
            H=40,
            n_params=5,
            n_shocks=3,
        )

        # Wrong theta shape
        with pytest.raises(ValueError, match="theta shape"):
            writer.write_sample(
                sample_idx=0,
                theta=np.random.randn(10),  # Wrong size
                trajectories=np.random.randn(100, 3),
                irfs=np.random.randn(3, 41, 3),
                shocks=np.random.randn(100, 3),
            )

        # Wrong trajectories shape
        with pytest.raises(ValueError, match="trajectories shape"):
            writer.write_sample(
                sample_idx=0,
                theta=np.random.randn(5),
                trajectories=np.random.randn(50, 3),  # Wrong T
                irfs=np.random.randn(3, 41, 3),
                shocks=np.random.randn(100, 3),
            )

    def test_round_trip(self, temp_dir):
        """Test write → read produces same data."""
        n_samples, T, H, n_params, n_shocks = 10, 100, 40, 5, 3

        # Write data
        writer = DatasetWriter(
            output_dir=temp_dir,
            world_id="test_world",
            n_samples=n_samples,
            T=T,
            H=H,
            n_params=n_params,
            n_shocks=n_shocks,
        )

        original_data = []
        for i in range(n_samples):
            theta = np.random.randn(n_params)
            trajectories = np.random.randn(T, 3)
            irfs = np.random.randn(n_shocks, H + 1, 3)
            shocks = np.random.randn(T, n_shocks)

            writer.write_sample(i, theta, trajectories, irfs, shocks)
            original_data.append((theta, trajectories, irfs, shocks))

        summary = writer.finalize()
        assert summary["samples_written"] == n_samples

        # Read data back
        traj_read, irfs_read, shocks_read, theta_read = DatasetWriter.load_dataset(
            temp_dir, "test_world"
        )

        # Verify all samples match
        for i in range(n_samples):
            theta, trajectories, irfs, shocks = original_data[i]
            np.testing.assert_allclose(theta_read[i], theta, rtol=1e-5)
            np.testing.assert_allclose(traj_read[i], trajectories, rtol=1e-5)
            np.testing.assert_allclose(irfs_read[i], irfs, rtol=1e-5)
            np.testing.assert_allclose(shocks_read[i], shocks, rtol=1e-5)

    def test_compression_ratio(self, temp_dir):
        """Test that compression is working."""
        n_samples, T, H, n_params, n_shocks = 100, 200, 40, 8, 3

        writer = DatasetWriter(
            output_dir=temp_dir,
            world_id="test_world",
            n_samples=n_samples,
            T=T,
            H=H,
            n_params=n_params,
            n_shocks=n_shocks,
        )

        # Write some compressible data (zeros)
        for i in range(n_samples):
            writer.write_sample(
                i,
                theta=np.zeros(n_params),
                trajectories=np.zeros((T, 3)),
                irfs=np.zeros((n_shocks, H + 1, 3)),
                shocks=np.zeros((T, n_shocks)),
            )

        summary = writer.finalize()

        # Check that compression ratio is > 1 (data is compressed)
        assert summary["arrays"]["trajectories"]["compression_ratio"] > 1.0
        assert summary["arrays"]["irfs"]["compression_ratio"] > 1.0


@pytest.mark.fast
class TestManifestGenerator:
    """Test suite for ManifestGenerator."""

    def test_init_creates_valid_manifest(self, temp_dir):
        """Test that initialization creates a valid manifest structure."""
        gen = ManifestGenerator(
            version="1.0.0",
            output_dir=temp_dir,
            global_seed=42,
        )

        assert gen.manifest["version"] == "1.0.0"
        assert "created_at" in gen.manifest
        assert "git_hash" in gen.manifest
        assert gen.manifest["rng_seeds"]["global"] == 42
        assert gen.manifest["rng_seeds"]["per_sample"] is True

    def test_add_simulator(
        self, temp_dir, sample_param_manifest, sample_shock_manifest, sample_obs_manifest
    ):
        """Test adding a simulator to the manifest."""
        gen = ManifestGenerator(
            version="1.0.0",
            output_dir=temp_dir,
        )

        gen.add_simulator(
            world_id="nk",
            n_samples=100000,
            param_manifest=sample_param_manifest,
            shock_manifest=sample_shock_manifest,
            obs_manifest=sample_obs_manifest,
        )

        assert "nk" in gen.manifest["simulators"]
        assert gen.manifest["simulators"]["nk"]["n_samples"] == 100000
        assert len(gen.manifest["simulators"]["nk"]["param_manifest"]["names"]) == 3

    def test_add_standard_splits(self, temp_dir):
        """Test adding standard train/val/test splits."""
        gen = ManifestGenerator(
            version="1.0.0",
            output_dir=temp_dir,
        )

        gen.add_standard_splits(seed=42)

        assert "train" in gen.manifest["splits"]
        assert "val" in gen.manifest["splits"]
        assert "test_interpolation" in gen.manifest["splits"]
        assert "test_extrapolation_slice" in gen.manifest["splits"]
        assert "test_extrapolation_corner" in gen.manifest["splits"]

        # Check fractions sum to 1.0
        total_fraction = sum(
            split["fraction"] for split in gen.manifest["splits"].values()
        )
        assert abs(total_fraction - 1.0) < 1e-6

    def test_manifest_validation_success(
        self, temp_dir, sample_param_manifest, sample_shock_manifest, sample_obs_manifest
    ):
        """Test that valid manifests pass validation."""
        gen = ManifestGenerator(
            version="1.0.0",
            output_dir=temp_dir,
        )

        gen.add_simulator(
            world_id="nk",
            n_samples=100000,
            param_manifest=sample_param_manifest,
            shock_manifest=sample_shock_manifest,
            obs_manifest=sample_obs_manifest,
        )
        gen.add_standard_splits()

        # Should not raise
        gen.validate()

    def test_manifest_validation_missing_simulator(self, temp_dir):
        """Test that manifest without simulators fails validation."""
        gen = ManifestGenerator(
            version="1.0.0",
            output_dir=temp_dir,
        )
        gen.add_standard_splits()

        with pytest.raises(ValueError, match="No simulators"):
            gen.validate()

    def test_manifest_validation_missing_splits(
        self, temp_dir, sample_param_manifest, sample_shock_manifest, sample_obs_manifest
    ):
        """Test that manifest without required splits fails validation."""
        gen = ManifestGenerator(
            version="1.0.0",
            output_dir=temp_dir,
        )

        gen.add_simulator(
            world_id="nk",
            n_samples=100000,
            param_manifest=sample_param_manifest,
            shock_manifest=sample_shock_manifest,
            obs_manifest=sample_obs_manifest,
        )

        with pytest.raises(ValueError, match="Missing required split"):
            gen.validate()

    def test_save_and_load(
        self, temp_dir, sample_param_manifest, sample_shock_manifest, sample_obs_manifest
    ):
        """Test saving and loading manifest."""
        gen = ManifestGenerator(
            version="1.0.0",
            output_dir=temp_dir,
        )

        gen.add_simulator(
            world_id="nk",
            n_samples=100000,
            param_manifest=sample_param_manifest,
            shock_manifest=sample_shock_manifest,
            obs_manifest=sample_obs_manifest,
        )
        gen.add_standard_splits()

        # Save
        manifest_path = gen.save()
        assert manifest_path.exists()

        # Load
        loaded = ManifestGenerator.load(manifest_path)
        assert loaded["version"] == "1.0.0"
        assert "nk" in loaded["simulators"]
        assert loaded["simulators"]["nk"]["n_samples"] == 100000

    def test_manifest_is_valid_json(
        self, temp_dir, sample_param_manifest, sample_shock_manifest, sample_obs_manifest
    ):
        """Test that saved manifest is valid JSON."""
        gen = ManifestGenerator(
            version="1.0.0",
            output_dir=temp_dir,
        )

        gen.add_simulator(
            world_id="nk",
            n_samples=100000,
            param_manifest=sample_param_manifest,
            shock_manifest=sample_shock_manifest,
            obs_manifest=sample_obs_manifest,
        )
        gen.add_standard_splits()

        manifest_path = gen.save()

        # Should be able to parse as JSON
        with open(manifest_path, "r") as f:
            data = json.load(f)

        assert isinstance(data, dict)
        assert data["version"] == "1.0.0"

    def test_config_hash_deterministic(self, temp_dir):
        """Test that config hash is deterministic."""
        config = {"param1": 1.0, "param2": "value", "param3": [1, 2, 3]}

        hash1 = ManifestGenerator._compute_config_hash(config)
        hash2 = ManifestGenerator._compute_config_hash(config)

        assert hash1 == hash2

    def test_config_hash_different_for_different_configs(self, temp_dir):
        """Test that different configs produce different hashes."""
        config1 = {"param1": 1.0, "param2": "value"}
        config2 = {"param1": 1.0, "param2": "different"}

        hash1 = ManifestGenerator._compute_config_hash(config1)
        hash2 = ManifestGenerator._compute_config_hash(config2)

        assert hash1 != hash2


@pytest.mark.fast
class TestDeterministicSeeding:
    """Test deterministic seeding and reproducibility."""

    def test_same_seed_produces_same_data(self, temp_dir):
        """Test that same global seed produces identical data."""
        seed = 42
        n_samples = 10

        # Generate data with seed
        rng1 = np.random.default_rng(seed)
        data1 = [rng1.standard_normal(5) for _ in range(n_samples)]

        # Generate again with same seed
        rng2 = np.random.default_rng(seed)
        data2 = [rng2.standard_normal(5) for _ in range(n_samples)]

        # Should be identical
        for d1, d2 in zip(data1, data2):
            np.testing.assert_array_equal(d1, d2)

    def test_per_sample_seed_derivation(self, temp_dir):
        """Test per-sample seed derivation from global seed."""
        global_seed = 42
        n_samples = 10

        # Derive per-sample seeds
        sample_seeds = [global_seed + i for i in range(n_samples)]

        # Generate data for each sample
        data1 = []
        for seed in sample_seeds:
            rng = np.random.default_rng(seed)
            data1.append(rng.standard_normal(5))

        # Regenerate with same logic
        data2 = []
        for i in range(n_samples):
            rng = np.random.default_rng(global_seed + i)
            data2.append(rng.standard_normal(5))

        # Should be identical
        for d1, d2 in zip(data1, data2):
            np.testing.assert_array_equal(d1, d2)


@pytest.mark.fast
class TestIntegration:
    """Integration tests for full pipeline."""

    def test_write_and_manifest_consistency(
        self, temp_dir, sample_param_manifest, sample_shock_manifest, sample_obs_manifest
    ):
        """Test that writer and manifest are consistent."""
        n_samples, T, H, n_params, n_shocks = 100, 200, 40, 3, 3

        # Create writer
        writer = DatasetWriter(
            output_dir=temp_dir,
            world_id="nk",
            n_samples=n_samples,
            T=T,
            H=H,
            n_params=n_params,
            n_shocks=n_shocks,
        )

        # Write some data
        for i in range(n_samples):
            writer.write_sample(
                i,
                theta=np.random.randn(n_params),
                trajectories=np.random.randn(T, 3),
                irfs=np.random.randn(n_shocks, H + 1, 3),
                shocks=np.random.randn(T, n_shocks),
            )

        summary = writer.finalize()

        # Create manifest
        gen = ManifestGenerator(
            version="1.0.0",
            output_dir=temp_dir,
            T=T,
            H=H,
        )

        gen.add_simulator(
            world_id="nk",
            n_samples=n_samples,
            param_manifest=sample_param_manifest,
            shock_manifest=sample_shock_manifest,
            obs_manifest=sample_obs_manifest,
        )
        gen.add_standard_splits()

        manifest_path = gen.save()

        # Verify consistency
        manifest = ManifestGenerator.load(manifest_path)
        assert manifest["simulators"]["nk"]["n_samples"] == summary["n_samples"]
        assert manifest["T"] == T
        # H is only included in manifest if non-default (40)
        assert manifest.get("H", 40) == H


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
