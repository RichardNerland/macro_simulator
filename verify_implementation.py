#!/usr/bin/env python3
"""
Verification script for Sprint 1 implementation.

This script tests all the simulators to ensure they work correctly.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, '/Users/nerland/macro_simulator')

def main():
    print("=" * 80)
    print("Sprint 1: Simulator Infrastructure Verification")
    print("=" * 80)

    # Test 1: Import base classes
    print("\n[1/10] Testing base module imports...")
    try:
        from simulators.base import (
            SimulatorAdapter,
            ParameterManifest,
            ShockManifest,
            ObservableManifest,
            SimulatorOutput,
            normalize_param,
            normalize_bounded,
            check_bounds,
            check_spectral_radius,
        )
        print("✓ Base module imports successful")
    except Exception as e:
        print(f"✗ Base module import failed: {e}")
        return False

    # Test 2: Import simulators
    print("\n[2/10] Testing simulator imports...")
    try:
        from simulators.lss import LSSSimulator
        from simulators.var import VARSimulator
        from simulators.nk import NKSimulator
        print("✓ All simulator imports successful")
    except Exception as e:
        print(f"✗ Simulator import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: LSS instantiation and basic operations
    print("\n[3/10] Testing LSS simulator...")
    try:
        import numpy as np
        lss = LSSSimulator(n_state=4, n_shocks=2)
        assert lss.world_id == "lss"
        rng = np.random.default_rng(42)
        theta = lss.sample_parameters(rng)
        assert theta.shape == (lss.n_params,)

        # Test simulation
        eps = rng.standard_normal((50, 2)) * 0.01
        output = lss.simulate(theta, eps, 50)
        assert output.y_canonical.shape == (50, 3)

        # Test IRF
        irf = lss.compute_irf(theta, 0, 1.0, 40)
        assert irf.shape == (41, 3)

        # Test analytic IRF
        irf_analytic = lss.get_analytic_irf(theta, 0, 1.0, 40)
        assert irf_analytic.shape == (41, 3)
        assert np.allclose(irf, irf_analytic, rtol=1e-10)

        print("✓ LSS simulator works correctly")
    except Exception as e:
        print(f"✗ LSS simulator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: VAR instantiation and basic operations
    print("\n[4/10] Testing VAR simulator...")
    try:
        var = VARSimulator(n_vars=3, lag_order=2)
        assert var.world_id == "var"
        rng = np.random.default_rng(42)
        theta = var.sample_parameters(rng)
        assert theta.shape == (var.n_params,)

        # Test simulation
        eps = rng.standard_normal((50, 3)) * 0.01
        output = var.simulate(theta, eps, 50)
        assert output.y_canonical.shape == (50, 3)

        # Test IRF
        irf = var.compute_irf(theta, 0, 1.0, 40)
        assert irf.shape == (41, 3)

        # Test analytic IRF
        irf_analytic = var.get_analytic_irf(theta, 0, 1.0, 40)
        assert irf_analytic.shape == (41, 3)
        assert np.allclose(irf, irf_analytic, rtol=1e-9)

        print("✓ VAR simulator works correctly")
    except Exception as e:
        print(f"✗ VAR simulator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: NK instantiation and basic operations
    print("\n[5/10] Testing NK simulator...")
    try:
        nk = NKSimulator()
        assert nk.world_id == "nk"
        rng = np.random.default_rng(42)
        theta = nk.sample_parameters(rng)
        assert theta.shape == (nk.n_params,)

        # Verify determinacy
        assert nk.validate_parameters(theta)

        # Test simulation
        eps = rng.standard_normal((50, 3)) * 0.01
        output = nk.simulate(theta, eps, 50)
        assert output.y_canonical.shape == (50, 3)

        # Test IRF
        irf = nk.compute_irf(theta, 0, 1.0, 40)
        assert irf.shape == (41, 3)

        print("✓ NK simulator works correctly")
    except Exception as e:
        print(f"✗ NK simulator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 6: Determinism
    print("\n[6/10] Testing determinism...")
    try:
        lss = LSSSimulator(n_state=4, n_shocks=2)
        rng1 = np.random.default_rng(123)
        theta1 = lss.sample_parameters(rng1)
        rng2 = np.random.default_rng(123)
        theta2 = lss.sample_parameters(rng2)
        assert np.array_equal(theta1, theta2)
        print("✓ Determinism verified")
    except Exception as e:
        print(f"✗ Determinism test failed: {e}")
        return False

    # Test 7: IRF baseline subtraction
    print("\n[7/10] Testing IRF baseline subtraction...")
    try:
        lss = LSSSimulator(n_state=4, n_shocks=2)
        rng = np.random.default_rng(42)
        theta = lss.sample_parameters(rng)

        # Zero shock should give zero IRF
        irf_zero = lss.compute_irf(theta, 0, 0.0, 40)
        assert np.allclose(irf_zero, 0, atol=1e-12)

        # Initial state should cancel out
        x0_zero = np.zeros(4)
        x0_random = rng.standard_normal(4) * 0.1
        irf1 = lss.compute_irf(theta, 0, 1.0, 40, x0=x0_zero)
        irf2 = lss.compute_irf(theta, 0, 1.0, 40, x0=x0_random)
        assert np.allclose(irf1, irf2, rtol=1e-10)

        print("✓ IRF baseline subtraction correct")
    except Exception as e:
        print(f"✗ IRF baseline test failed: {e}")
        return False

    # Test 8: Parameter validation
    print("\n[8/10] Testing parameter validation...")
    try:
        lss = LSSSimulator()
        rng = np.random.default_rng(42)
        theta = lss.sample_parameters(rng)
        assert lss.validate_parameters(theta)

        # Test out-of-bounds
        theta_bad = theta.copy()
        theta_bad[0] = 1e10  # Way out of bounds
        assert not lss.validate_parameters(theta_bad)

        print("✓ Parameter validation works")
    except Exception as e:
        print(f"✗ Parameter validation test failed: {e}")
        return False

    # Test 9: Stability checks
    print("\n[9/10] Testing stability checks...")
    try:
        from simulators.base import check_spectral_radius

        # Stable matrix
        A_stable = np.array([[0.5, 0.1], [0.1, 0.5]])
        assert check_spectral_radius(A_stable, 1.0)

        # Unstable matrix
        A_unstable = np.array([[1.5, 0.1], [0.1, 1.5]])
        assert not check_spectral_radius(A_unstable, 1.0)

        print("✓ Stability checks work")
    except Exception as e:
        print(f"✗ Stability check test failed: {e}")
        return False

    # Test 10: Manifests
    print("\n[10/10] Testing manifests...")
    try:
        lss = LSSSimulator(n_state=4, n_shocks=2)

        # Parameter manifest
        pm = lss.param_manifest
        assert len(pm.names) == lss.n_params
        assert pm.bounds.shape == (lss.n_params, 2)

        # Shock manifest
        sm = lss.shock_manifest
        assert sm.n_shocks == 2
        assert len(sm.names) == 2

        # Observable manifest
        om = lss.obs_manifest
        assert om.n_canonical == 3
        assert om.canonical_names == ["output", "inflation", "rate"]

        print("✓ Manifests correct")
    except Exception as e:
        print(f"✗ Manifest test failed: {e}")
        return False

    print("\n" + "=" * 80)
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
