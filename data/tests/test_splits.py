"""Tests for split construction algorithms.

These tests verify that:
1. Splits are disjoint (no sample in multiple splits)
2. Splits have correct fractions
3. Split construction is deterministic (same seed → same splits)
4. Predicates work correctly
"""

import numpy as np
import pytest

from data.splits import (
    SLICE_PREDICATES,
    compute_split_algorithm_hash,
    compute_summary_stats,
    construct_all_splits,
    get_lowo_world_lists,
    split_extrapolation_corner,
    split_extrapolation_slice,
    split_interpolation,
    split_lowo,
    validate_splits,
)

try:
    from simulators.base import ParameterManifest
except ImportError:
    from dataclasses import dataclass

    @dataclass
    class ParameterManifest:
        """Stub for ParameterManifest during development."""
        names: list[str]
        units: list[str]
        bounds: np.ndarray
        defaults: np.ndarray
        priors: list[dict] | None = None


@pytest.fixture
def sample_param_manifest():
    """Create sample parameter manifest for NK simulator."""
    return ParameterManifest(
        names=["beta", "sigma", "phi_pi", "phi_y", "rho_i", "rho_a", "sigma_a", "sigma_m"],
        units=["-", "-", "-", "-", "-", "-", "-", "-"],
        bounds=np.array([
            [0.985, 0.995],  # beta
            [0.5, 2.5],      # sigma
            [1.05, 3.5],     # phi_pi
            [0.0, 1.0],      # phi_y
            [0.0, 0.9],      # rho_i
            [0.0, 0.95],     # rho_a
            [0.005, 0.02],   # sigma_a
            [0.001, 0.01],   # sigma_m
        ]),
        defaults=np.array([0.99, 1.0, 1.5, 0.125, 0.8, 0.9, 0.01, 0.005]),
    )


@pytest.mark.fast
class TestSplitInterpolation:
    """Test suite for random interpolation split."""

    def test_split_fractions(self):
        """Test that split fractions are correct."""
        n_samples = 10000
        seed = 42

        splits = split_interpolation(n_samples, seed)

        assert "train" in splits
        assert "val" in splits
        assert "test_interpolation" in splits

        # Check fractions (allow 1% tolerance)
        train_frac = len(splits["train"]) / n_samples
        val_frac = len(splits["val"]) / n_samples
        test_frac = len(splits["test_interpolation"]) / n_samples

        assert abs(train_frac - 0.80) < 0.01
        assert abs(val_frac - 0.10) < 0.01
        assert abs(test_frac - 0.05) < 0.01

    def test_split_disjoint(self):
        """Test that splits are disjoint (no overlap)."""
        n_samples = 1000
        seed = 42

        splits = split_interpolation(n_samples, seed)

        # Check pairwise disjointness
        train_val_overlap = np.intersect1d(splits["train"], splits["val"])
        train_test_overlap = np.intersect1d(splits["train"], splits["test_interpolation"])
        val_test_overlap = np.intersect1d(splits["val"], splits["test_interpolation"])

        assert len(train_val_overlap) == 0
        assert len(train_test_overlap) == 0
        assert len(val_test_overlap) == 0

    def test_split_coverage(self):
        """Test that splits cover 95% of samples (5% reserved for extrapolation)."""
        n_samples = 1000
        seed = 42

        splits = split_interpolation(n_samples, seed)

        all_indices = np.concatenate([splits["train"], splits["val"], splits["test_interpolation"]])
        unique_indices = np.unique(all_indices)

        # Should cover 95% of samples (within rounding)
        coverage = len(unique_indices) / n_samples
        assert abs(coverage - 0.95) < 0.01

    def test_determinism(self):
        """Test that same seed produces same splits."""
        n_samples = 1000
        seed = 42

        splits1 = split_interpolation(n_samples, seed)
        splits2 = split_interpolation(n_samples, seed)

        # Should be identical
        np.testing.assert_array_equal(splits1["train"], splits2["train"])
        np.testing.assert_array_equal(splits1["val"], splits2["val"])
        np.testing.assert_array_equal(splits1["test_interpolation"], splits2["test_interpolation"])

    def test_different_seeds_different_splits(self):
        """Test that different seeds produce different splits."""
        n_samples = 1000

        splits1 = split_interpolation(n_samples, seed=42)
        splits2 = split_interpolation(n_samples, seed=43)

        # Should be different
        assert not np.array_equal(splits1["train"], splits2["train"])


@pytest.mark.fast
class TestSlicePredicates:
    """Test suite for slice predicates."""

    def test_nk_slice_predicate(self, sample_param_manifest):
        """Test NK slice predicate: φ_π > 2.0."""
        # Create sample with φ_π = 2.5 (should be in slice)
        theta_high = np.array([0.99, 1.0, 2.5, 0.125, 0.8, 0.9, 0.01, 0.005])
        assert SLICE_PREDICATES["nk"](theta_high, sample_param_manifest)

        # Create sample with φ_π = 1.5 (should not be in slice)
        theta_low = np.array([0.99, 1.0, 1.5, 0.125, 0.8, 0.9, 0.01, 0.005])
        assert not SLICE_PREDICATES["nk"](theta_low, sample_param_manifest)

    def test_rbc_slice_predicate(self):
        """Test RBC slice predicate: ρ_a > 0.95."""
        manifest = ParameterManifest(
            names=["beta", "alpha", "delta", "rho_a", "sigma_a"],
            units=["-", "-", "-", "-", "-"],
            bounds=np.array([[0.985, 0.995], [0.25, 0.40], [0.02, 0.03], [0.0, 0.99], [0.005, 0.02]]),
            defaults=np.array([0.99, 0.33, 0.025, 0.9, 0.01]),
        )

        # High persistence
        theta_high = np.array([0.99, 0.33, 0.025, 0.97, 0.01])
        assert SLICE_PREDICATES["rbc"](theta_high, manifest)

        # Low persistence
        theta_low = np.array([0.99, 0.33, 0.025, 0.85, 0.01])
        assert not SLICE_PREDICATES["rbc"](theta_low, manifest)

    def test_zlb_slice_predicate(self):
        """Test ZLB slice predicate: r_ss < 1.0."""
        manifest = ParameterManifest(
            names=["beta", "sigma", "phi_pi", "r_ss"],
            units=["-", "-", "-", "%"],
            bounds=np.array([[0.985, 0.995], [0.5, 2.5], [1.05, 3.5], [0.5, 4.0]]),
            defaults=np.array([0.99, 1.0, 1.5, 2.0]),
        )

        # Low steady-state rate
        theta_low = np.array([0.99, 1.0, 1.5, 0.8])
        assert SLICE_PREDICATES["zlb"](theta_low, manifest)

        # High steady-state rate
        theta_high = np.array([0.99, 1.0, 1.5, 2.5])
        assert not SLICE_PREDICATES["zlb"](theta_high, manifest)


@pytest.mark.fast
class TestExtrapolationSlice:
    """Test suite for extrapolation-slice split."""

    def test_slice_identifies_correct_samples(self, sample_param_manifest):
        """Test that slice correctly identifies samples matching predicate."""
        n_samples = 100
        seed = 42

        # Create theta with some high φ_π values
        rng = np.random.default_rng(seed)
        theta = rng.uniform(
            sample_param_manifest.bounds[:, 0],
            sample_param_manifest.bounds[:, 1],
            size=(n_samples, len(sample_param_manifest.names))
        )

        # Manually set MANY samples to have φ_π > 2.0 to ensure they're in slice
        phi_pi_idx = sample_param_manifest.names.index("phi_pi")
        for i in range(10):  # Set first 10 samples
            theta[i, phi_pi_idx] = 2.5 + i * 0.1
        theta[50, phi_pi_idx] = 1.5  # This one below threshold

        slice_idx = split_extrapolation_slice(theta, sample_param_manifest, "nk", seed)

        # Should include some of the first 10 samples (high phi_pi)
        assert len(slice_idx) > 0
        # At least some of the slice should be from high phi_pi samples
        high_phi_pi_in_slice = sum(1 for idx in slice_idx if idx < 10)
        assert high_phi_pi_in_slice > 0
        # Sample 50 should not be in slice (low phi_pi)
        assert 50 not in slice_idx

    def test_slice_respects_target_fraction(self, sample_param_manifest):
        """Test that slice returns at most 2.5% of samples."""
        n_samples = 1000
        seed = 42

        # Create theta with many samples matching predicate
        rng = np.random.default_rng(seed)
        theta = rng.uniform(
            sample_param_manifest.bounds[:, 0],
            sample_param_manifest.bounds[:, 1],
            size=(n_samples, len(sample_param_manifest.names))
        )

        # Set many samples to have φ_π > 2.0
        phi_pi_idx = sample_param_manifest.names.index("phi_pi")
        theta[:200, phi_pi_idx] = 2.5  # 20% of samples match predicate

        slice_idx = split_extrapolation_slice(theta, sample_param_manifest, "nk", seed)

        # Should return at most 2.5% = 25 samples
        assert len(slice_idx) <= int(0.025 * n_samples) + 1  # +1 for rounding

    def test_slice_determinism(self, sample_param_manifest):
        """Test that same seed produces same slice."""
        n_samples = 100
        seed = 42

        rng = np.random.default_rng(seed)
        theta = rng.uniform(
            sample_param_manifest.bounds[:, 0],
            sample_param_manifest.bounds[:, 1],
            size=(n_samples, len(sample_param_manifest.names))
        )

        slice1 = split_extrapolation_slice(theta, sample_param_manifest, "nk", seed)
        slice2 = split_extrapolation_slice(theta, sample_param_manifest, "nk", seed)

        np.testing.assert_array_equal(slice1, slice2)

    def test_slice_empty_for_unknown_world(self, sample_param_manifest):
        """Test that slice returns empty array for unknown world_id."""
        n_samples = 100
        rng = np.random.default_rng(42)
        theta = rng.uniform(
            sample_param_manifest.bounds[:, 0],
            sample_param_manifest.bounds[:, 1],
            size=(n_samples, len(sample_param_manifest.names))
        )

        slice_idx = split_extrapolation_slice(theta, sample_param_manifest, "unknown_world", 42)
        assert len(slice_idx) == 0


@pytest.mark.fast
class TestSummaryStats:
    """Test suite for summary statistics computation."""

    def test_nk_summary_stats(self, sample_param_manifest):
        """Test NK summary statistics computation."""
        n_samples = 10
        rng = np.random.default_rng(42)
        theta = rng.uniform(
            sample_param_manifest.bounds[:, 0],
            sample_param_manifest.bounds[:, 1],
            size=(n_samples, len(sample_param_manifest.names))
        )

        stats = compute_summary_stats(theta, sample_param_manifest, "nk")

        assert stats is not None
        assert "persistence" in stats
        assert "volatility" in stats
        assert "policy_strength" in stats

        assert stats["persistence"].shape == (n_samples,)
        assert stats["volatility"].shape == (n_samples,)
        assert stats["policy_strength"].shape == (n_samples,)

    def test_summary_stats_none_for_unknown_world(self, sample_param_manifest):
        """Test that summary stats returns None for unknown world."""
        n_samples = 10
        rng = np.random.default_rng(42)
        theta = rng.uniform(
            sample_param_manifest.bounds[:, 0],
            sample_param_manifest.bounds[:, 1],
            size=(n_samples, len(sample_param_manifest.names))
        )

        stats = compute_summary_stats(theta, sample_param_manifest, "unknown_world")
        assert stats is None


@pytest.mark.fast
class TestExtrapolationCorner:
    """Test suite for extrapolation-corner split."""

    def test_corner_identifies_joint_extremes(self, sample_param_manifest):
        """Test that corner correctly identifies joint extremes."""
        n_samples = 100
        seed = 42

        rng = np.random.default_rng(seed)
        theta = rng.uniform(
            sample_param_manifest.bounds[:, 0],
            sample_param_manifest.bounds[:, 1],
            size=(n_samples, len(sample_param_manifest.names))
        )

        # Manually create a corner sample: high persistence AND high volatility
        rho_a_idx = sample_param_manifest.names.index("rho_a")
        sigma_a_idx = sample_param_manifest.names.index("sigma_a")

        theta[0, rho_a_idx] = 0.94  # High persistence
        theta[0, sigma_a_idx] = 0.019  # High volatility

        corner_idx = split_extrapolation_corner(theta, sample_param_manifest, "nk", seed, quantile=0.9)

        # Sample 0 should likely be in corner (if thresholds align)
        # At minimum, should return some corner samples
        assert len(corner_idx) > 0

    def test_corner_respects_target_fraction(self, sample_param_manifest):
        """Test that corner returns at most 2.5% of samples."""
        n_samples = 1000
        seed = 42

        rng = np.random.default_rng(seed)
        theta = rng.uniform(
            sample_param_manifest.bounds[:, 0],
            sample_param_manifest.bounds[:, 1],
            size=(n_samples, len(sample_param_manifest.names))
        )

        corner_idx = split_extrapolation_corner(theta, sample_param_manifest, "nk", seed, quantile=0.9)

        # Should return at most 2.5% = 25 samples
        assert len(corner_idx) <= int(0.025 * n_samples) + 1

    def test_corner_determinism(self, sample_param_manifest):
        """Test that same seed produces same corner."""
        n_samples = 100
        seed = 42

        rng = np.random.default_rng(seed)
        theta = rng.uniform(
            sample_param_manifest.bounds[:, 0],
            sample_param_manifest.bounds[:, 1],
            size=(n_samples, len(sample_param_manifest.names))
        )

        corner1 = split_extrapolation_corner(theta, sample_param_manifest, "nk", seed)
        corner2 = split_extrapolation_corner(theta, sample_param_manifest, "nk", seed)

        np.testing.assert_array_equal(corner1, corner2)


@pytest.mark.fast
class TestConstructAllSplits:
    """Test suite for construct_all_splits."""

    def test_all_splits_disjoint(self, sample_param_manifest):
        """Test that all splits are disjoint."""
        n_samples = 1000
        seed = 42

        rng = np.random.default_rng(seed)
        theta = rng.uniform(
            sample_param_manifest.bounds[:, 0],
            sample_param_manifest.bounds[:, 1],
            size=(n_samples, len(sample_param_manifest.names))
        )

        splits = construct_all_splits(theta, sample_param_manifest, "nk", seed)

        # Use validate_splits to check disjointness
        # Should not raise
        validate_splits(splits, n_samples)

    def test_all_splits_coverage(self, sample_param_manifest):
        """Test that all splits cover all samples."""
        n_samples = 1000
        seed = 42

        rng = np.random.default_rng(seed)
        theta = rng.uniform(
            sample_param_manifest.bounds[:, 0],
            sample_param_manifest.bounds[:, 1],
            size=(n_samples, len(sample_param_manifest.names))
        )

        splits = construct_all_splits(theta, sample_param_manifest, "nk", seed)

        # Concatenate all splits
        all_indices = np.concatenate([
            splits["train"],
            splits["val"],
            splits["test_interpolation"],
            splits["test_extrapolation_slice"],
            splits["test_extrapolation_corner"],
        ])

        unique_indices = np.unique(all_indices)

        # Should cover all samples
        assert len(unique_indices) == n_samples
        assert np.array_equal(unique_indices, np.arange(n_samples))

    def test_all_splits_determinism(self, sample_param_manifest):
        """Test that construct_all_splits is deterministic."""
        n_samples = 1000
        seed = 42

        rng = np.random.default_rng(seed)
        theta = rng.uniform(
            sample_param_manifest.bounds[:, 0],
            sample_param_manifest.bounds[:, 1],
            size=(n_samples, len(sample_param_manifest.names))
        )

        splits1 = construct_all_splits(theta, sample_param_manifest, "nk", seed)
        splits2 = construct_all_splits(theta, sample_param_manifest, "nk", seed)

        # All splits should be identical
        for split_name in splits1.keys():
            np.testing.assert_array_equal(splits1[split_name], splits2[split_name])

    def test_all_splits_correct_keys(self, sample_param_manifest):
        """Test that construct_all_splits returns correct split names."""
        n_samples = 1000
        seed = 42

        rng = np.random.default_rng(seed)
        theta = rng.uniform(
            sample_param_manifest.bounds[:, 0],
            sample_param_manifest.bounds[:, 1],
            size=(n_samples, len(sample_param_manifest.names))
        )

        splits = construct_all_splits(theta, sample_param_manifest, "nk", seed)

        expected_keys = {
            "train",
            "val",
            "test_interpolation",
            "test_extrapolation_slice",
            "test_extrapolation_corner",
        }

        assert set(splits.keys()) == expected_keys


@pytest.mark.fast
class TestValidateSplits:
    """Test suite for validate_splits."""

    def test_validate_success(self, sample_param_manifest):
        """Test that valid splits pass validation."""
        n_samples = 1000
        seed = 42

        rng = np.random.default_rng(seed)
        theta = rng.uniform(
            sample_param_manifest.bounds[:, 0],
            sample_param_manifest.bounds[:, 1],
            size=(n_samples, len(sample_param_manifest.names))
        )

        splits = construct_all_splits(theta, sample_param_manifest, "nk", seed)

        # Should not raise
        validate_splits(splits, n_samples)

    def test_validate_detects_overlap(self):
        """Test that validation detects overlapping splits."""
        n_samples = 100

        # Create overlapping splits
        splits = {
            "train": np.arange(80),
            "val": np.arange(70, 90),  # Overlaps with train
            "test_interpolation": np.arange(90, 95),
            "test_extrapolation_slice": np.array([]),
            "test_extrapolation_corner": np.arange(95, 100),
        }

        with pytest.raises(ValueError, match="overlap"):
            validate_splits(splits, n_samples)

    def test_validate_detects_incomplete_coverage(self):
        """Test that validation detects incomplete coverage."""
        n_samples = 100

        # Create splits that don't cover all samples
        splits = {
            "train": np.arange(70),
            "val": np.arange(70, 80),
            "test_interpolation": np.arange(80, 85),
            "test_extrapolation_slice": np.array([]),
            "test_extrapolation_corner": np.array([]),
            # Missing samples 85-99
        }

        with pytest.raises(ValueError, match="cover"):
            validate_splits(splits, n_samples)


@pytest.mark.fast
class TestSplitAlgorithmHash:
    """Test suite for split algorithm hash computation."""

    def test_hash_deterministic(self):
        """Test that hash is deterministic."""
        config = {"algorithm": "random", "seed": 42, "fractions": [0.8, 0.1, 0.05, 0.025, 0.025]}

        hash1 = compute_split_algorithm_hash(config)
        hash2 = compute_split_algorithm_hash(config)

        assert hash1 == hash2

    def test_hash_different_for_different_configs(self):
        """Test that different configs produce different hashes."""
        config1 = {"algorithm": "random", "seed": 42}
        config2 = {"algorithm": "random", "seed": 43}

        hash1 = compute_split_algorithm_hash(config1)
        hash2 = compute_split_algorithm_hash(config2)

        assert hash1 != hash2


@pytest.mark.fast
class TestLOWO:
    """Test suite for Leave-One-World-Out (LOWO) split."""

    def test_lowo_basic_functionality(self):
        """Test basic LOWO split functionality."""
        world_ids = ["lss", "var", "nk", "rbc", "switching", "zlb"]
        held_out_world = "nk"

        split = split_lowo(world_ids, held_out_world)

        assert "train_worlds" in split
        assert "test_worlds" in split
        assert held_out_world not in split["train_worlds"]
        assert held_out_world in split["test_worlds"]
        assert len(split["test_worlds"]) == 1
        assert len(split["train_worlds"]) == 5

    def test_lowo_train_worlds_correct(self):
        """Test that train_worlds contains all worlds except held_out_world."""
        world_ids = ["lss", "var", "nk", "rbc", "switching", "zlb"]
        held_out_world = "rbc"

        split = split_lowo(world_ids, held_out_world)

        expected_train = ["lss", "var", "nk", "switching", "zlb"]
        assert set(split["train_worlds"]) == set(expected_train)
        assert split["test_worlds"] == ["rbc"]

    def test_lowo_disjoint_worlds(self):
        """Test that train and test worlds are disjoint."""
        world_ids = ["lss", "var", "nk", "rbc", "switching", "zlb"]

        for held_out_world in world_ids:
            split = split_lowo(world_ids, held_out_world)

            train_set = set(split["train_worlds"])
            test_set = set(split["test_worlds"])

            # Check disjointness
            overlap = train_set & test_set
            assert len(overlap) == 0, f"Overlap found: {overlap}"

            # Check coverage
            total = train_set | test_set
            assert total == set(world_ids)

    def test_lowo_all_worlds_as_holdout(self):
        """Test LOWO for each world as holdout."""
        world_ids = ["lss", "var", "nk", "rbc", "switching", "zlb"]

        for held_out_world in world_ids:
            split = split_lowo(world_ids, held_out_world)

            # Verify held-out world is in test
            assert held_out_world in split["test_worlds"]
            assert held_out_world not in split["train_worlds"]

            # Verify counts
            assert len(split["train_worlds"]) == len(world_ids) - 1
            assert len(split["test_worlds"]) == 1

    def test_lowo_invalid_world(self):
        """Test that LOWO raises ValueError for invalid held_out_world."""
        world_ids = ["lss", "var", "nk", "rbc", "switching", "zlb"]
        invalid_world = "invalid_simulator"

        with pytest.raises(ValueError, match="not in world_ids"):
            split_lowo(world_ids, invalid_world)

    def test_lowo_preserves_world_order(self):
        """Test that LOWO preserves world order (excluding held-out world)."""
        world_ids = ["lss", "var", "nk", "rbc", "switching", "zlb"]
        held_out_world = "nk"

        split = split_lowo(world_ids, held_out_world)

        # Train worlds should preserve order
        expected_train = ["lss", "var", "rbc", "switching", "zlb"]
        assert split["train_worlds"] == expected_train

    def test_lowo_single_world_holdout_size(self):
        """Test that test_worlds always contains exactly one world."""
        world_ids = ["lss", "var", "nk", "rbc", "switching", "zlb"]

        for held_out_world in world_ids:
            split = split_lowo(world_ids, held_out_world)
            assert len(split["test_worlds"]) == 1
            assert split["test_worlds"][0] == held_out_world

    def test_lowo_two_worlds(self):
        """Test LOWO with minimal case (2 worlds)."""
        world_ids = ["lss", "var"]
        held_out_world = "lss"

        split = split_lowo(world_ids, held_out_world)

        assert split["train_worlds"] == ["var"]
        assert split["test_worlds"] == ["lss"]

    def test_lowo_deterministic(self):
        """Test that LOWO split is deterministic (no randomness)."""
        world_ids = ["lss", "var", "nk", "rbc", "switching", "zlb"]
        held_out_world = "switching"

        split1 = split_lowo(world_ids, held_out_world)
        split2 = split_lowo(world_ids, held_out_world)

        assert split1["train_worlds"] == split2["train_worlds"]
        assert split1["test_worlds"] == split2["test_worlds"]

    def test_get_lowo_world_lists_convenience(self):
        """Test the convenience function get_lowo_world_lists."""
        world_ids = ["lss", "var", "nk", "rbc", "switching", "zlb"]
        held_out_world = "nk"

        train_worlds, test_worlds = get_lowo_world_lists(world_ids, held_out_world)

        assert train_worlds == ["lss", "var", "rbc", "switching", "zlb"]
        assert test_worlds == ["nk"]
        assert len(train_worlds) == 5
        assert len(test_worlds) == 1

    def test_get_lowo_world_lists_matches_split_lowo(self):
        """Test that convenience function matches split_lowo output."""
        world_ids = ["lss", "var", "nk", "rbc", "switching", "zlb"]

        for held_out_world in world_ids:
            split = split_lowo(world_ids, held_out_world)
            train_worlds, test_worlds = get_lowo_world_lists(world_ids, held_out_world)

            assert train_worlds == split["train_worlds"]
            assert test_worlds == split["test_worlds"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
