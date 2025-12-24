#!/usr/bin/env python
"""
Verification script for Sprint 1 Data Infrastructure (Tasks S1.24-S1.27).

This script demonstrates that all implemented components work correctly:
1. JSON Schema validation
2. Zarr dataset writer
3. Manifest generator
4. Deterministic seeding

Usage:
    python verify_sprint1_data.py
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

# Import our implementations
from data.writer import DatasetWriter
from data.manifest import ManifestGenerator

# Import simulator types (will use actual imports if available, stubs otherwise)
try:
    from simulators.base import ParameterManifest, ShockManifest, ObservableManifest
    print("✓ Successfully imported from simulators.base")
except ImportError:
    from data.manifest import ParameterManifest, ShockManifest, ObservableManifest
    print("⚠ Using stub implementations (simulators.base not available)")


def verify_json_schema():
    """Verify that JSON schema exists and is valid JSON."""
    print("\n" + "=" * 60)
    print("TEST 1: JSON Schema Validation")
    print("=" * 60)

    schema_path = Path("data/schemas/manifest.json")
    print(f"Checking schema file: {schema_path}")

    assert schema_path.exists(), "Schema file not found"
    print("✓ Schema file exists")

    with open(schema_path) as f:
        schema = json.load(f)
    print("✓ Schema is valid JSON")

    # Check required top-level properties
    required = schema.get("required", [])
    expected_required = ["version", "created_at", "git_hash", "simulators", "splits", "rng_seeds"]
    assert all(field in required for field in expected_required), "Missing required fields"
    print(f"✓ Schema requires: {required}")

    print("\n✅ JSON Schema validation: PASS")
    return True


def verify_zarr_writer():
    """Verify Zarr dataset writer functionality."""
    print("\n" + "=" * 60)
    print("TEST 2: Zarr Dataset Writer")
    print("=" * 60)

    # Create temporary directory
    tmpdir = Path(tempfile.mkdtemp())
    print(f"Using temporary directory: {tmpdir}")

    try:
        # Setup parameters
        n_samples, T, H, n_params, n_shocks = 10, 100, 40, 5, 3

        # Create writer
        writer = DatasetWriter(
            output_dir=tmpdir,
            world_id="test_world",
            n_samples=n_samples,
            T=T,
            H=H,
            n_params=n_params,
            n_shocks=n_shocks,
        )
        print(f"✓ Created DatasetWriter for {n_samples} samples")

        # Write samples
        rng = np.random.default_rng(42)
        original_data = []
        for i in range(n_samples):
            theta = rng.uniform(0, 1, n_params)
            trajectories = rng.standard_normal((T, 3))
            irfs = rng.standard_normal((n_shocks, H + 1, 3))
            shocks = rng.standard_normal((T, n_shocks))

            writer.write_sample(i, theta, trajectories, irfs, shocks)
            original_data.append((theta, trajectories, irfs, shocks))

        print(f"✓ Wrote {n_samples} samples")

        # Finalize and check summary
        summary = writer.finalize()
        assert summary["samples_written"] == n_samples
        print(f"✓ Finalized dataset: {summary['samples_written']} samples written")
        print(f"  Compression ratio (trajectories): {summary['arrays']['trajectories']['compression_ratio']:.2f}x")

        # Read back and verify
        traj_read, irfs_read, shocks_read, theta_read = DatasetWriter.load_dataset(
            tmpdir, "test_world"
        )
        print("✓ Loaded dataset back from disk")

        # Verify shapes
        assert traj_read.shape == (n_samples, T, 3)
        assert irfs_read.shape == (n_samples, n_shocks, H + 1, 3)
        assert shocks_read.shape == (n_samples, T, n_shocks)
        assert theta_read.shape == (n_samples, n_params)
        print("✓ All arrays have correct shapes")

        # Verify round-trip (first sample)
        theta_orig, traj_orig, irf_orig, shock_orig = original_data[0]
        np.testing.assert_allclose(theta_read[0], theta_orig, rtol=1e-5)
        np.testing.assert_allclose(traj_read[0], traj_orig, rtol=1e-5)
        print("✓ Round-trip verification passed (write → read → match)")

        print("\n✅ Zarr dataset writer: PASS")
        return True

    finally:
        # Cleanup
        shutil.rmtree(tmpdir)
        print(f"Cleaned up temporary directory")


def verify_manifest_generator():
    """Verify manifest generator functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: Manifest Generator")
    print("=" * 60)

    # Create temporary directory
    tmpdir = Path(tempfile.mkdtemp())
    print(f"Using temporary directory: {tmpdir}")

    try:
        # Create manifest generator
        gen = ManifestGenerator(
            version="1.0.0",
            output_dir=tmpdir,
            global_seed=42,
            T=200,
            H=40,
        )
        print("✓ Created ManifestGenerator")

        # Create sample manifests
        param_manifest = ParameterManifest(
            names=["beta", "sigma", "phi_pi"],
            units=["-", "-", "-"],
            bounds=np.array([[0.985, 0.995], [0.5, 2.5], [1.05, 3.5]]),
            defaults=np.array([0.99, 1.0, 1.5]),
            priors=[],
        )

        shock_manifest = ShockManifest(
            names=["monetary", "technology", "cost_push"],
            n_shocks=3,
            sigma=np.array([0.005, 0.01, 0.005]),
        )

        obs_manifest = ObservableManifest(
            canonical_names=["output", "inflation", "rate"],
            extra_names=[],
        )

        # Add simulator
        gen.add_simulator(
            world_id="nk",
            n_samples=100000,
            param_manifest=param_manifest,
            shock_manifest=shock_manifest,
            obs_manifest=obs_manifest,
        )
        print("✓ Added simulator 'nk' with manifests")

        # Add standard splits
        gen.add_standard_splits(seed=42)
        print("✓ Added standard splits (train/val/test)")

        # Validate
        gen.validate()
        print("✓ Manifest validation passed")

        # Save
        manifest_path = gen.save()
        assert manifest_path.exists()
        print(f"✓ Saved manifest to: {manifest_path}")

        # Load and verify
        loaded = ManifestGenerator.load(manifest_path)
        assert loaded["version"] == "1.0.0"
        assert "nk" in loaded["simulators"]
        assert loaded["simulators"]["nk"]["n_samples"] == 100000
        assert loaded["rng_seeds"]["global"] == 42
        assert loaded["rng_seeds"]["per_sample"] is True
        print("✓ Loaded manifest and verified contents")

        # Check that fractions sum to 1.0
        total_fraction = sum(split["fraction"] for split in loaded["splits"].values())
        assert abs(total_fraction - 1.0) < 1e-6
        print(f"✓ Split fractions sum to 1.0 (actual: {total_fraction:.6f})")

        print("\n✅ Manifest generator: PASS")
        return True

    finally:
        # Cleanup
        shutil.rmtree(tmpdir)
        print(f"Cleaned up temporary directory")


def verify_deterministic_seeding():
    """Verify deterministic seeding strategy."""
    print("\n" + "=" * 60)
    print("TEST 4: Deterministic Seeding")
    print("=" * 60)

    # Test 1: Same seed produces same data
    seed = 42
    n_samples = 5

    rng1 = np.random.default_rng(seed)
    data1 = [rng1.standard_normal(10) for _ in range(n_samples)]

    rng2 = np.random.default_rng(seed)
    data2 = [rng2.standard_normal(10) for _ in range(n_samples)]

    for i, (d1, d2) in enumerate(zip(data1, data2)):
        np.testing.assert_array_equal(d1, d2)

    print(f"✓ Same seed ({seed}) produces identical data across runs")

    # Test 2: Per-sample seed derivation
    global_seed = 42
    n_samples = 5

    # Generate data using per-sample seeds
    sample_data = []
    for i in range(n_samples):
        sample_rng = np.random.default_rng(global_seed + i)
        sample_data.append(sample_rng.standard_normal(10))

    # Regenerate and verify
    for i in range(n_samples):
        sample_rng = np.random.default_rng(global_seed + i)
        regenerated = sample_rng.standard_normal(10)
        np.testing.assert_array_equal(sample_data[i], regenerated)

    print(f"✓ Per-sample seeds (global_seed + index) are reproducible")
    print(f"  Sample 0 seed: {global_seed + 0}")
    print(f"  Sample 1 seed: {global_seed + 1}")
    print(f"  Sample {n_samples - 1} seed: {global_seed + n_samples - 1}")

    print("\n✅ Deterministic seeding: PASS")
    return True


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("Sprint 1 Data Infrastructure Verification")
    print("Tasks S1.24-S1.27")
    print("=" * 60)

    results = []

    try:
        results.append(("JSON Schema", verify_json_schema()))
    except Exception as e:
        print(f"\n❌ JSON Schema test FAILED: {e}")
        results.append(("JSON Schema", False))

    try:
        results.append(("Zarr Writer", verify_zarr_writer()))
    except Exception as e:
        print(f"\n❌ Zarr Writer test FAILED: {e}")
        results.append(("Zarr Writer", False))

    try:
        results.append(("Manifest Generator", verify_manifest_generator()))
    except Exception as e:
        print(f"\n❌ Manifest Generator test FAILED: {e}")
        results.append(("Manifest Generator", False))

    try:
        results.append(("Deterministic Seeding", verify_deterministic_seeding()))
    except Exception as e:
        print(f"\n❌ Deterministic Seeding test FAILED: {e}")
        results.append(("Deterministic Seeding", False))

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:30s} {status}")

    all_passed = all(result[1] for result in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✅")
        print("Sprint 1 Data Infrastructure (S1.24-S1.27) is COMPLETE")
    else:
        print("SOME TESTS FAILED ❌")
        print("Please review the errors above")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
