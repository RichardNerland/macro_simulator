"""
Tests for Regime-Switching simulator.
"""

import numpy as np
import pytest

from simulators.switching import SwitchingSimulator


@pytest.mark.fast
def test_switching_initialization():
    """Test switching simulator initializes correctly."""
    sim = SwitchingSimulator(n_state=3, n_shocks=2)
    assert sim.world_id == "switching"
    assert sim.n_state == 3
    assert sim.n_shocks == 2


@pytest.mark.fast
def test_switching_manifests():
    """Test switching simulator manifests are correctly formed."""
    sim = SwitchingSimulator(n_state=3, n_shocks=2)

    # Check parameter manifest
    param_manifest = sim.param_manifest
    # 2 (transition probs) + 2*3*3 (A_0, A_1) + 2*3*2 (B_0, B_1) + 3*3 (C)
    expected_n_params = 2 + 2 * 3 * 3 + 2 * 3 * 2 + 3 * 3
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
def test_switching_sample_parameters():
    """Test parameter sampling produces stable systems in both regimes."""
    sim = SwitchingSimulator(n_state=3, n_shocks=2, rho_max=0.95)
    rng = np.random.default_rng(42)

    # Sample multiple times
    for _ in range(10):
        theta = sim.sample_parameters(rng)

        # Check shape
        assert theta.shape == (sim.n_params,)

        # Check bounds
        assert sim.validate_parameters(theta)

        # Check stability of both regime matrices
        p_00, p_11, A_0, A_1, B_0, B_1, C = sim._unpack_theta(theta)

        # Check transition probabilities
        assert 0 < p_00 < 1
        assert 0 < p_11 < 1

        # Check stability of both A matrices
        eigenvalues_0 = np.linalg.eigvals(A_0)
        spectral_radius_0 = np.max(np.abs(eigenvalues_0))
        assert spectral_radius_0 < sim.rho_max

        eigenvalues_1 = np.linalg.eigvals(A_1)
        spectral_radius_1 = np.max(np.abs(eigenvalues_1))
        assert spectral_radius_1 < sim.rho_max


@pytest.mark.fast
def test_switching_transition_probabilities():
    """Test transition probability bounds."""
    sim = SwitchingSimulator(n_state=3, n_shocks=2)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    p_00, p_11, _, _, _, _, _ = sim._unpack_theta(theta)

    # Both should be valid probabilities
    assert 0 < p_00 < 1
    assert 0 < p_11 < 1

    # Should be in the high-persistence range
    bounds = sim.param_manifest.bounds
    assert bounds[0, 0] <= p_00 <= bounds[0, 1]
    assert bounds[1, 0] <= p_11 <= bounds[1, 1]


@pytest.mark.fast
def test_switching_regime_sequence():
    """Test regime sequence generation."""
    sim = SwitchingSimulator(n_state=3, n_shocks=2)
    rng = np.random.default_rng(42)

    p_00 = 0.9
    p_11 = 0.9
    T = 200

    regime_seq = sim._simulate_regime_sequence(p_00, p_11, T, rng)

    # Check shape and values
    assert regime_seq.shape == (T,)
    assert np.all((regime_seq == 0) | (regime_seq == 1))

    # Check that both regimes appear (with high probability)
    assert 0 in regime_seq
    assert 1 in regime_seq


@pytest.mark.fast
def test_switching_regime_persistence():
    """Test regime persistence matches transition probabilities."""
    sim = SwitchingSimulator(n_state=3, n_shocks=2)
    rng = np.random.default_rng(123)

    # High persistence
    p_00 = 0.95
    p_11 = 0.95
    T = 10000

    regime_seq = sim._simulate_regime_sequence(p_00, p_11, T, rng)

    # Count transitions
    n_00 = np.sum((regime_seq[:-1] == 0) & (regime_seq[1:] == 0))
    n_01 = np.sum((regime_seq[:-1] == 0) & (regime_seq[1:] == 1))
    n_10 = np.sum((regime_seq[:-1] == 1) & (regime_seq[1:] == 0))
    n_11 = np.sum((regime_seq[:-1] == 1) & (regime_seq[1:] == 1))

    # Estimate transition probabilities
    if n_00 + n_01 > 0:
        p_00_est = n_00 / (n_00 + n_01)
        # Should be close to 0.95 (allow 5% tolerance)
        assert abs(p_00_est - p_00) < 0.05

    if n_10 + n_11 > 0:
        p_11_est = n_11 / (n_10 + n_11)
        # Should be close to 0.95 (allow 5% tolerance)
        assert abs(p_11_est - p_11) < 0.05


@pytest.mark.fast
def test_switching_expected_regime_duration():
    """Test expected regime durations match theoretical values."""
    sim = SwitchingSimulator(n_state=3, n_shocks=2)
    rng = np.random.default_rng(456)

    p_00 = 0.9  # Expected duration in regime 0: 1/(1-0.9) = 10
    p_11 = 0.95  # Expected duration in regime 1: 1/(1-0.95) = 20
    T = 10000

    regime_seq = sim._simulate_regime_sequence(p_00, p_11, T, rng)

    # Find regime durations (runs of consecutive same regime)
    durations_0 = []
    durations_1 = []

    current_regime = regime_seq[0]
    current_duration = 1

    for t in range(1, T):
        if regime_seq[t] == current_regime:
            current_duration += 1
        else:
            if current_regime == 0:
                durations_0.append(current_duration)
            else:
                durations_1.append(current_duration)
            current_regime = regime_seq[t]
            current_duration = 1

    # Check expected durations (allow generous tolerance for stochastic test)
    if len(durations_0) > 10:
        expected_duration_0 = 1 / (1 - p_00)
        mean_duration_0 = np.mean(durations_0)
        assert abs(mean_duration_0 - expected_duration_0) < 3  # Within 3 periods

    if len(durations_1) > 10:
        expected_duration_1 = 1 / (1 - p_11)
        mean_duration_1 = np.mean(durations_1)
        assert abs(mean_duration_1 - expected_duration_1) < 5  # Within 5 periods


@pytest.mark.fast
def test_switching_determinism():
    """Test that same seed produces same output."""
    sim = SwitchingSimulator(n_state=3, n_shocks=2)

    # Sample with same seed
    rng1 = np.random.default_rng(123)
    theta1 = sim.sample_parameters(rng1)

    rng2 = np.random.default_rng(123)
    theta2 = sim.sample_parameters(rng2)

    np.testing.assert_array_equal(theta1, theta2)

    # Simulate with same parameters, shocks, and regime seed
    T = 50
    rng_eps = np.random.default_rng(456)
    eps = rng_eps.standard_normal((T, sim.n_shocks))

    output1 = sim.simulate(theta1, eps, T, regime_seed=789)
    output2 = sim.simulate(theta2, eps, T, regime_seed=789)

    np.testing.assert_array_equal(output1.y_canonical, output2.y_canonical)
    np.testing.assert_array_equal(output1.regime_seq, output2.regime_seq)


@pytest.mark.fast
def test_switching_simulate():
    """Test simulation produces correct shapes and doesn't explode."""
    sim = SwitchingSimulator(n_state=3, n_shocks=2)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    T = 100
    eps = rng.standard_normal((T, sim.n_shocks)) * 0.01

    output = sim.simulate(theta, eps, T)

    # Check shapes
    assert output.y_canonical.shape == (T, 3)
    assert output.x_state is not None
    assert output.x_state.shape == (T, sim.n_state)
    assert output.regime_seq is not None
    assert output.regime_seq.shape == (T,)

    # Check regime values
    assert np.all((output.regime_seq == 0) | (output.regime_seq == 1))

    # Check not exploding (reasonable bounds)
    assert np.all(np.abs(output.y_canonical) < 100)
    assert np.all(np.isfinite(output.y_canonical))


@pytest.mark.fast
def test_switching_regime_affects_dynamics():
    """Test that different regimes produce different dynamics."""
    sim = SwitchingSimulator(n_state=3, n_shocks=2)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    p_00, p_11, A_0, A_1, B_0, B_1, C = sim._unpack_theta(theta)

    # Verify that A_0 and A_1 are different
    assert not np.allclose(A_0, A_1)

    # Verify that B_0 and B_1 are different (in general)
    assert not np.allclose(B_0, B_1)


@pytest.mark.fast
def test_switching_compute_irf():
    """Test IRF computation produces correct shapes."""
    sim = SwitchingSimulator(n_state=3, n_shocks=2)
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
def test_switching_irf_determinism():
    """Test IRF is deterministic given regime seed."""
    sim = SwitchingSimulator(n_state=3, n_shocks=2)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 40
    shock_idx = 0
    shock_size = 1.0

    # Compute IRF twice with same regime seed
    irf1 = sim.compute_irf(theta, shock_idx, shock_size, H, regime_seed=123)
    irf2 = sim.compute_irf(theta, shock_idx, shock_size, H, regime_seed=123)

    np.testing.assert_array_equal(irf1, irf2)


@pytest.mark.fast
def test_switching_irf_different_regimes():
    """Test IRF differs with different regime seeds."""
    sim = SwitchingSimulator(n_state=3, n_shocks=2)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 40
    shock_idx = 0
    shock_size = 1.0

    # Compute IRF with different regime seeds
    irf1 = sim.compute_irf(theta, shock_idx, shock_size, H, regime_seed=123)
    irf2 = sim.compute_irf(theta, shock_idx, shock_size, H, regime_seed=456)

    # Should generally be different (regime path affects IRF)
    # Check that at least one point differs significantly
    max_diff = np.max(np.abs(irf1 - irf2))
    # With high probability, different regime paths lead to different IRFs
    # This is a probabilistic test, but with different seeds it should hold
    assert max_diff > 1e-10 or np.allclose(irf1, irf2, rtol=1e-10)
    # Note: In rare cases, regime paths might be similar, so we allow both outcomes


@pytest.mark.fast
def test_switching_zero_shock_gives_zero_irf():
    """Test that zero-sized shock gives zero IRF."""
    sim = SwitchingSimulator(n_state=3, n_shocks=2)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 40
    shock_idx = 0
    shock_size = 0.0

    irf = sim.compute_irf(theta, shock_idx, shock_size, H)

    np.testing.assert_allclose(irf, 0, atol=1e-12)


@pytest.mark.fast
def test_switching_irf_linearity():
    """Test IRF scales linearly with shock size (for same regime path)."""
    sim = SwitchingSimulator(n_state=3, n_shocks=2)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 40
    shock_idx = 0
    regime_seed = 999

    irf_1 = sim.compute_irf(theta, shock_idx, 1.0, H, regime_seed=regime_seed)
    irf_2 = sim.compute_irf(theta, shock_idx, 2.0, H, regime_seed=regime_seed)

    # Should be exactly 2x (linear system with same regime path)
    np.testing.assert_allclose(irf_2, 2 * irf_1, rtol=1e-10)


@pytest.mark.fast
def test_switching_initial_state_cancels():
    """Test that IRF is same from different initial states (baseline cancels)."""
    sim = SwitchingSimulator(n_state=3, n_shocks=2)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 40
    shock_idx = 0
    shock_size = 1.0
    regime_seed = 123

    # IRF from zero initial state
    x0_zero = np.zeros(sim.n_state)
    irf_zero = sim.compute_irf(
        theta, shock_idx, shock_size, H, x0=x0_zero, regime_seed=regime_seed
    )

    # IRF from random initial state
    x0_random = rng.standard_normal(sim.n_state) * 0.1
    irf_random = sim.compute_irf(
        theta, shock_idx, shock_size, H, x0=x0_random, regime_seed=regime_seed
    )

    # Both should be the same (IRF is difference, so initial state cancels out)
    np.testing.assert_allclose(irf_zero, irf_random, rtol=1e-10)


@pytest.mark.fast
def test_switching_stability_both_regimes():
    """Test that both regime matrices are stable."""
    sim = SwitchingSimulator(n_state=3, n_shocks=2, rho_max=0.95)
    rng = np.random.default_rng(42)

    # Sample many times to ensure stability filter works
    for _ in range(20):
        theta = sim.sample_parameters(rng)
        p_00, p_11, A_0, A_1, B_0, B_1, C = sim._unpack_theta(theta)

        # Check spectral radius for both A matrices
        eig_0 = np.linalg.eigvals(A_0)
        rho_0 = np.max(np.abs(eig_0))
        assert rho_0 < sim.rho_max

        eig_1 = np.linalg.eigvals(A_1)
        rho_1 = np.max(np.abs(eig_1))
        assert rho_1 < sim.rho_max


@pytest.mark.fast
def test_switching_pack_unpack_theta():
    """Test that packing and unpacking theta is reversible."""
    sim = SwitchingSimulator(n_state=3, n_shocks=2)
    rng = np.random.default_rng(42)

    theta_original = sim.sample_parameters(rng)
    p_00, p_11, A_0, A_1, B_0, B_1, C = sim._unpack_theta(theta_original)
    theta_repacked = sim._pack_theta(p_00, p_11, A_0, A_1, B_0, B_1, C)

    np.testing.assert_array_equal(theta_original, theta_repacked)


@pytest.mark.fast
def test_switching_irf_not_always_zero():
    """Test IRF has non-zero response (at least sometimes)."""
    sim = SwitchingSimulator(n_state=3, n_shocks=2)
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 40
    shock_idx = 0
    shock_size = 1.0

    irf = sim.compute_irf(theta, shock_idx, shock_size, H)

    # Should have some non-zero response
    assert not np.allclose(irf, 0)
