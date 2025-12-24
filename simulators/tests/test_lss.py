"""
Tests for Linear State-Space (LSS) simulator.
"""

import numpy as np
import pytest

from simulators.lss import LSSSimulator


@pytest.mark.fast
def test_lss_initialization():
    """Test LSS simulator initializes correctly."""
    sim = LSSSimulator(n_state=4, n_shocks=2)
    assert sim.world_id == "lss"
    assert sim.n_state == 4
    assert sim.n_shocks == 2


@pytest.mark.fast
def test_lss_manifests():
    """Test LSS simulator manifests are correctly formed."""
    sim = LSSSimulator(n_state=3, n_shocks=2)

    # Check parameter manifest
    param_manifest = sim.param_manifest
    expected_n_params = 3 * 3 + 3 * 2 + 3 * 3  # A + B + C
    assert len(param_manifest.names) == expected_n_params
    assert len(param_manifest.units) == expected_n_params
    assert param_manifest.bounds.shape == (expected_n_params, 2)
    assert param_manifest.defaults.shape == (expected_n_params,)

    # Check shock manifest
    shock_manifest = sim.shock_manifest
    assert shock_manifest.n_shocks == 2
    assert len(shock_manifest.names) == 2
    assert shock_manifest.sigma.shape == (2,)

    # Check observable manifest
    obs_manifest = sim.obs_manifest
    assert obs_manifest.n_canonical == 3
    assert obs_manifest.canonical_names == ["output", "inflation", "rate"]
    assert obs_manifest.n_extra == 0


@pytest.mark.fast
def test_lss_sample_parameters():
    """Test parameter sampling produces stable systems."""
    sim = LSSSimulator(n_state=4, n_shocks=2, rho_max=0.95)
    rng = np.random.default_rng(42)

    # Sample multiple times
    for _ in range(10):
        theta = sim.sample_parameters(rng)

        # Check shape
        assert theta.shape == (sim.n_params,)

        # Check bounds
        assert sim.validate_parameters(theta)

        # Check stability
        A, _, _ = sim._unpack_theta(theta)
        eigenvalues = np.linalg.eigvals(A)
        spectral_radius = np.max(np.abs(eigenvalues))
        assert spectral_radius < sim.rho_max


@pytest.mark.fast
def test_lss_determinism():
    """Test that same seed produces same output."""
    sim = LSSSimulator(n_state=4, n_shocks=2)

    # Sample with same seed
    rng1 = np.random.default_rng(123)
    theta1 = sim.sample_parameters(rng1)

    rng2 = np.random.default_rng(123)
    theta2 = sim.sample_parameters(rng2)

    np.testing.assert_array_equal(theta1, theta2)

    # Simulate with same parameters and shocks
    T = 50
    rng1 = np.random.default_rng(456)
    eps1 = rng1.standard_normal((T, sim.n_shocks))

    rng2 = np.random.default_rng(456)
    eps2 = rng2.standard_normal((T, sim.n_shocks))

    output1 = sim.simulate(theta1, eps1, T)
    output2 = sim.simulate(theta2, eps2, T)

    np.testing.assert_array_equal(output1.y_canonical, output2.y_canonical)


@pytest.mark.fast
def test_lss_simulate():
    """Test simulation produces correct shapes and doesn't explode."""
    sim = LSSSimulator(n_state=4, n_shocks=2)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    T = 100
    eps = rng.standard_normal((T, sim.n_shocks)) * 0.01

    output = sim.simulate(theta, eps, T)

    # Check shapes
    assert output.y_canonical.shape == (T, 3)
    assert output.x_state is not None
    assert output.x_state.shape == (T, sim.n_state)

    # Check not exploding (reasonable bounds)
    assert np.all(np.abs(output.y_canonical) < 100)
    assert np.all(np.isfinite(output.y_canonical))


@pytest.mark.fast
def test_lss_compute_irf():
    """Test IRF computation produces correct shapes."""
    sim = LSSSimulator(n_state=4, n_shocks=2)
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
def test_lss_analytic_irf_matches_simulated():
    """Test analytic IRF matches simulation-based IRF."""
    sim = LSSSimulator(n_state=4, n_shocks=2, rho_max=0.9)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 40
    shock_idx = 0
    shock_size = 1.0

    # Compute both IRFs
    irf_simulated = sim.compute_irf(theta, shock_idx, shock_size, H)
    irf_analytic = sim.get_analytic_irf(theta, shock_idx, shock_size, H)

    # Should match very closely (numerical precision)
    np.testing.assert_allclose(irf_simulated, irf_analytic, rtol=1e-10, atol=1e-12)


@pytest.mark.fast
def test_lss_irf_impact_effect():
    """Test IRF has non-zero impact effect for responsive observables."""
    sim = LSSSimulator(n_state=4, n_shocks=2)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 40
    shock_idx = 0
    shock_size = 1.0

    irf = sim.compute_irf(theta, shock_idx, shock_size, H)

    # At least one observable should respond at impact (unless C @ B is zero, which is unlikely)
    # We just check that the IRF is not all zeros
    assert not np.allclose(irf, 0)


@pytest.mark.fast
def test_lss_irf_decays():
    """Test IRF decays over time (for stable system)."""
    sim = LSSSimulator(n_state=4, n_shocks=2, rho_max=0.9)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 80
    shock_idx = 0
    shock_size = 1.0

    irf = sim.get_analytic_irf(theta, shock_idx, shock_size, H)

    # Compute norm at different horizons
    norm_start = np.linalg.norm(irf[:10, :])
    norm_end = np.linalg.norm(irf[-10:, :])

    # For stable system, later horizons should have smaller norm
    # (or at least not be significantly larger)
    assert norm_end <= norm_start * 2  # Allow some slack


@pytest.mark.fast
def test_lss_zero_shock_gives_zero_irf():
    """Test that zero-sized shock gives zero IRF."""
    sim = LSSSimulator(n_state=4, n_shocks=2)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 40
    shock_idx = 0
    shock_size = 0.0

    irf = sim.compute_irf(theta, shock_idx, shock_size, H)

    np.testing.assert_allclose(irf, 0, atol=1e-12)


@pytest.mark.fast
def test_lss_irf_linearity():
    """Test IRF scales linearly with shock size."""
    sim = LSSSimulator(n_state=4, n_shocks=2)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 40
    shock_idx = 0

    irf_1 = sim.compute_irf(theta, shock_idx, 1.0, H)
    irf_2 = sim.compute_irf(theta, shock_idx, 2.0, H)

    # Should be exactly 2x (linear system)
    np.testing.assert_allclose(irf_2, 2 * irf_1, rtol=1e-10)


@pytest.mark.fast
def test_lss_different_initial_states():
    """Test IRF can be computed from different initial states."""
    sim = LSSSimulator(n_state=4, n_shocks=2)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 40
    shock_idx = 0
    shock_size = 1.0

    # IRF from zero initial state
    x0_zero = np.zeros(sim.n_state)
    irf_zero = sim.compute_irf(theta, shock_idx, shock_size, H, x0=x0_zero)

    # IRF from random initial state
    x0_random = rng.standard_normal(sim.n_state) * 0.1
    irf_random = sim.compute_irf(theta, shock_idx, shock_size, H, x0=x0_random)

    # Both should be the same (IRF is difference, so initial state cancels out)
    np.testing.assert_allclose(irf_zero, irf_random, rtol=1e-10)
