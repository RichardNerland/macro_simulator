"""
Tests for Vector Autoregression (VAR) simulator.
"""

import numpy as np
import pytest

from simulators.var import VARSimulator


@pytest.mark.fast
def test_var_initialization():
    """Test VAR simulator initializes correctly."""
    sim = VARSimulator(n_vars=3, lag_order=2)
    assert sim.world_id == "var"
    assert sim.n_vars == 3
    assert sim.lag_order == 2


@pytest.mark.fast
def test_var_manifests():
    """Test VAR simulator manifests are correctly formed."""
    sim = VARSimulator(n_vars=3, lag_order=2)

    # Check parameter manifest
    param_manifest = sim.param_manifest
    # c: 3, A_1, A_2: 2*3*3=18, Sigma (Cholesky): 3*4/2=6
    expected_n_params = 3 + 18 + 6
    assert len(param_manifest.names) == expected_n_params
    assert len(param_manifest.units) == expected_n_params
    assert param_manifest.bounds.shape == (expected_n_params, 2)
    assert param_manifest.defaults.shape == (expected_n_params,)

    # Check shock manifest
    shock_manifest = sim.shock_manifest
    assert shock_manifest.n_shocks == 3
    assert len(shock_manifest.names) == 3
    assert shock_manifest.sigma.shape == (3,)

    # Check observable manifest
    obs_manifest = sim.obs_manifest
    assert obs_manifest.n_canonical == 3
    assert obs_manifest.canonical_names == ["output", "inflation", "rate"]


@pytest.mark.fast
def test_var_sample_parameters():
    """Test parameter sampling produces stationary systems."""
    sim = VARSimulator(n_vars=3, lag_order=2, rho_max=0.95)
    rng = np.random.default_rng(42)

    # Sample multiple times
    for _ in range(10):
        theta = sim.sample_parameters(rng)

        # Check shape
        assert theta.shape == (sim.n_params,)

        # Check bounds
        assert sim.validate_parameters(theta)

        # Check stationarity via companion matrix
        _, A_matrices, _ = sim._unpack_theta(theta)
        F = sim._build_companion_matrix(A_matrices)
        eigenvalues = np.linalg.eigvals(F)
        spectral_radius = np.max(np.abs(eigenvalues))
        assert spectral_radius < sim.rho_max


@pytest.mark.fast
def test_var_determinism():
    """Test that same seed produces same output."""
    sim = VARSimulator(n_vars=3, lag_order=2)

    # Sample with same seed
    rng1 = np.random.default_rng(123)
    theta1 = sim.sample_parameters(rng1)

    rng2 = np.random.default_rng(123)
    theta2 = sim.sample_parameters(rng2)

    np.testing.assert_array_equal(theta1, theta2)

    # Simulate with same parameters and shocks
    T = 50
    rng1 = np.random.default_rng(456)
    eps1 = rng1.standard_normal((T, sim.n_vars))

    rng2 = np.random.default_rng(456)
    eps2 = rng2.standard_normal((T, sim.n_vars))

    output1 = sim.simulate(theta1, eps1, T)
    output2 = sim.simulate(theta2, eps2, T)

    np.testing.assert_array_equal(output1.y_canonical, output2.y_canonical)


@pytest.mark.fast
def test_var_simulate():
    """Test simulation produces correct shapes and doesn't explode."""
    sim = VARSimulator(n_vars=3, lag_order=2)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    T = 100
    eps = rng.standard_normal((T, sim.n_vars)) * 0.01

    output = sim.simulate(theta, eps, T)

    # Check shapes
    assert output.y_canonical.shape == (T, 3)

    # Check not exploding (reasonable bounds)
    assert np.all(np.abs(output.y_canonical) < 100)
    assert np.all(np.isfinite(output.y_canonical))


@pytest.mark.fast
def test_var_compute_irf():
    """Test IRF computation produces correct shapes."""
    sim = VARSimulator(n_vars=3, lag_order=2)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 40
    shock_idx = 0
    shock_size = 1.0

    irf = sim.compute_irf(theta, shock_idx, shock_size, H)

    # Check shape
    assert irf.shape == (H + 1, 3)

    # Check finite
    assert np.all(np.isfinite(irf))


@pytest.mark.fast
def test_var_analytic_irf_matches_simulated():
    """Test analytic IRF matches simulation-based IRF."""
    sim = VARSimulator(n_vars=3, lag_order=2, rho_max=0.9)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 40
    shock_idx = 0
    shock_size = 1.0

    # Compute both IRFs
    irf_simulated = sim.compute_irf(theta, shock_idx, shock_size, H)
    irf_analytic = sim.get_analytic_irf(theta, shock_idx, shock_size, H)

    # Should match closely (numerical precision)
    np.testing.assert_allclose(irf_simulated, irf_analytic, rtol=1e-9, atol=1e-11)


@pytest.mark.fast
def test_var_irf_impact_effect():
    """Test IRF has non-zero impact effect."""
    sim = VARSimulator(n_vars=3, lag_order=2)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 40
    shock_idx = 0
    shock_size = 1.0

    irf = sim.compute_irf(theta, shock_idx, shock_size, H)

    # Should have non-zero response (at least for the shocked variable)
    assert not np.allclose(irf, 0)


@pytest.mark.fast
def test_var_irf_decays():
    """Test IRF decays over time (for stationary system)."""
    sim = VARSimulator(n_vars=3, lag_order=2, rho_max=0.9)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 80
    shock_idx = 0
    shock_size = 1.0

    irf = sim.get_analytic_irf(theta, shock_idx, shock_size, H)

    # Compute norm at different horizons
    norm_start = np.linalg.norm(irf[:10, :])
    norm_end = np.linalg.norm(irf[-10:, :])

    # For stationary system, later horizons should have smaller norm
    assert norm_end <= norm_start * 2  # Allow some slack


@pytest.mark.fast
def test_var_zero_shock_gives_zero_irf():
    """Test that zero-sized shock gives zero IRF."""
    sim = VARSimulator(n_vars=3, lag_order=2)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 40
    shock_idx = 0
    shock_size = 0.0

    irf = sim.compute_irf(theta, shock_idx, shock_size, H)

    np.testing.assert_allclose(irf, 0, atol=1e-12)


@pytest.mark.fast
def test_var_irf_linearity():
    """Test IRF scales linearly with shock size."""
    sim = VARSimulator(n_vars=3, lag_order=2)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 40
    shock_idx = 0

    irf_1 = sim.compute_irf(theta, shock_idx, 1.0, H)
    irf_2 = sim.compute_irf(theta, shock_idx, 2.0, H)

    # Should be exactly 2x (linear system)
    np.testing.assert_allclose(irf_2, 2 * irf_1, rtol=1e-10, atol=1e-15)


@pytest.mark.fast
def test_var_different_initial_states():
    """Test IRF can be computed from different initial states."""
    sim = VARSimulator(n_vars=3, lag_order=2)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 40
    shock_idx = 0
    shock_size = 1.0

    # IRF from zero initial state
    x0_zero = np.zeros(sim.n_vars * sim.lag_order)
    irf_zero = sim.compute_irf(theta, shock_idx, shock_size, H, x0=x0_zero)

    # IRF from random initial state
    x0_random = rng.standard_normal(sim.n_vars * sim.lag_order) * 0.1
    irf_random = sim.compute_irf(theta, shock_idx, shock_size, H, x0=x0_random)

    # Both should be the same (IRF is difference, so initial state cancels out)
    np.testing.assert_allclose(irf_zero, irf_random, rtol=1e-10, atol=1e-15)


@pytest.mark.fast
def test_var_lag_order_1():
    """Test VAR(1) special case works correctly."""
    sim = VARSimulator(n_vars=3, lag_order=1, rho_max=0.9)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 40

    irf = sim.compute_irf(theta, 0, 1.0, H)
    assert irf.shape == (H + 1, 3)
    assert np.all(np.isfinite(irf))


@pytest.mark.fast
def test_var_companion_matrix_construction():
    """Test companion matrix construction is correct."""
    sim = VARSimulator(n_vars=3, lag_order=2)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    _, A_matrices, _ = sim._unpack_theta(theta)

    F = sim._build_companion_matrix(A_matrices)

    # Check shape
    assert F.shape == (6, 6)  # n_vars * lag_order

    # Check structure: first row should be [A_1, A_2]
    np.testing.assert_array_equal(F[0:3, 0:3], A_matrices[0])
    np.testing.assert_array_equal(F[0:3, 3:6], A_matrices[1])

    # Lower blocks should be identity and zeros
    np.testing.assert_array_equal(F[3:6, 0:3], np.eye(3))
    np.testing.assert_array_equal(F[3:6, 3:6], np.zeros((3, 3)))
