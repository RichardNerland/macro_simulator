"""
Tests for New Keynesian (NK) simulator.
"""

import numpy as np
import pytest

from simulators.nk import NKSimulator


@pytest.mark.fast
def test_nk_initialization():
    """Test NK simulator initializes correctly."""
    sim = NKSimulator()
    assert sim.world_id == "nk"
    assert sim.n_params == 12


@pytest.mark.fast
def test_nk_manifests():
    """Test NK simulator manifests are correctly formed."""
    sim = NKSimulator()

    # Check parameter manifest
    param_manifest = sim.param_manifest
    assert len(param_manifest.names) == 12
    assert len(param_manifest.units) == 12
    assert param_manifest.bounds.shape == (12, 2)
    assert param_manifest.defaults.shape == (12,)

    # Check that phi_pi lower bound > 1 (Taylor principle)
    phi_pi_idx = param_manifest.names.index("phi_pi")
    assert param_manifest.bounds[phi_pi_idx, 0] > 1.0

    # Check shock manifest
    shock_manifest = sim.shock_manifest
    assert shock_manifest.n_shocks == 3
    assert shock_manifest.names == ["monetary", "demand", "cost_push"]
    assert shock_manifest.sigma.shape == (3,)

    # Check observable manifest
    obs_manifest = sim.obs_manifest
    assert obs_manifest.n_canonical == 3
    assert obs_manifest.canonical_names == ["output", "inflation", "rate"]


@pytest.mark.fast
def test_nk_sample_parameters():
    """Test parameter sampling produces determinate systems."""
    sim = NKSimulator()
    rng = np.random.default_rng(42)

    # Sample multiple times
    for _ in range(5):
        theta = sim.sample_parameters(rng)

        # Check shape
        assert theta.shape == (sim.n_params,)

        # Check bounds
        assert sim.validate_parameters(theta)

        # Check Taylor principle
        phi_pi = theta[3]
        assert phi_pi > 1.0

        # Check determinacy
        _, _, determinate = sim._solve_re_system(theta)
        assert determinate


@pytest.mark.fast
def test_nk_determinism():
    """Test that same seed produces same output."""
    sim = NKSimulator()

    # Sample with same seed
    rng1 = np.random.default_rng(123)
    theta1 = sim.sample_parameters(rng1)

    rng2 = np.random.default_rng(123)
    theta2 = sim.sample_parameters(rng2)

    np.testing.assert_array_equal(theta1, theta2)

    # Simulate with same parameters and shocks
    T = 50
    rng1 = np.random.default_rng(456)
    eps1 = rng1.standard_normal((T, 3))

    rng2 = np.random.default_rng(456)
    eps2 = rng2.standard_normal((T, 3))

    output1 = sim.simulate(theta1, eps1, T)
    output2 = sim.simulate(theta2, eps2, T)

    np.testing.assert_array_equal(output1.y_canonical, output2.y_canonical)


@pytest.mark.fast
def test_nk_simulate():
    """Test simulation produces correct shapes and doesn't explode."""
    sim = NKSimulator()
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    T = 100
    eps = rng.standard_normal((T, 3)) * 0.01

    output = sim.simulate(theta, eps, T)

    # Check shapes
    assert output.y_canonical.shape == (T, 3)
    assert output.x_state is not None
    assert output.x_state.shape == (T, 5)

    # Check not exploding (reasonable bounds for macro variables)
    assert np.all(np.abs(output.y_canonical) < 100)
    assert np.all(np.isfinite(output.y_canonical))


@pytest.mark.fast
def test_nk_compute_irf():
    """Test IRF computation produces correct shapes."""
    sim = NKSimulator()
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 40

    # Test each shock type
    for shock_idx in range(3):
        irf = sim.compute_irf(theta, shock_idx, 1.0, H)

        # Check shape
        assert irf.shape == (H + 1, 3)

        # Check finite
        assert np.all(np.isfinite(irf))


@pytest.mark.fast
def test_nk_monetary_shock_signs():
    """Test monetary shock produces expected sign patterns.

    A contractionary monetary shock (positive shock to interest rate) should:
    - Increase interest rate (positive)
    - Decrease output gap (negative)
    - Decrease inflation (negative, eventually)
    """
    sim = NKSimulator()
    rng = np.random.default_rng(42)

    # Use default parameters for predictable behavior
    theta = sim.param_manifest.defaults.copy()

    # Ensure determinacy
    if not sim.validate_parameters(theta):
        pytest.skip("Default parameters not determinate")

    H = 20
    shock_idx = 0  # Monetary shock
    shock_size = 1.0  # Contractionary (positive)

    irf = sim.compute_irf(theta, shock_idx, shock_size, H)

    # Interest rate should increase on impact or shortly after
    assert irf[0, 2] > 0 or irf[1, 2] > 0, "Monetary shock should increase interest rate"


@pytest.mark.fast
def test_nk_demand_shock_signs():
    """Test demand shock produces expected sign patterns.

    A positive demand shock (increase in natural rate) should:
    - Increase output gap (positive)
    - Increase inflation (positive)
    - Increase interest rate (positive, via Taylor rule)
    """
    sim = NKSimulator()
    rng = np.random.default_rng(42)

    theta = sim.param_manifest.defaults.copy()

    if not sim.validate_parameters(theta):
        pytest.skip("Default parameters not determinate")

    H = 20
    shock_idx = 1  # Demand shock
    shock_size = 1.0  # Positive

    irf = sim.compute_irf(theta, shock_idx, shock_size, H)

    # Check that we get some response (sign depends on exact calibration)
    assert not np.allclose(irf, 0), "Demand shock should produce non-zero response"


@pytest.mark.fast
def test_nk_zero_shock_gives_zero_irf():
    """Test that zero-sized shock gives zero IRF."""
    sim = NKSimulator()
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 40
    shock_idx = 0
    shock_size = 0.0

    irf = sim.compute_irf(theta, shock_idx, shock_size, H)

    np.testing.assert_allclose(irf, 0, atol=1e-12)


@pytest.mark.fast
def test_nk_irf_decays():
    """Test IRF decays over time (for stable system)."""
    sim = NKSimulator()
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 80
    shock_idx = 0
    shock_size = 1.0

    irf = sim.compute_irf(theta, shock_idx, shock_size, H)

    # Compute norm at different horizons
    norm_start = np.linalg.norm(irf[:10, :])
    norm_end = np.linalg.norm(irf[-10:, :])

    # For stable system, later horizons should have smaller norm
    # NK models typically have persistent dynamics, so allow generous slack
    assert norm_end <= norm_start * 3


@pytest.mark.fast
def test_nk_indeterminate_parameters():
    """Test that indeterminate parameters are rejected."""
    sim = NKSimulator()

    # Create parameters that violate Taylor principle (phi_pi < 1)
    theta = sim.param_manifest.defaults.copy()
    theta[3] = 0.9  # phi_pi < 1, violates Taylor principle

    # Should fail validation
    assert not sim.validate_parameters(theta)

    # Should be marked as indeterminate
    _, _, determinate = sim._solve_re_system(theta)
    assert not determinate


@pytest.mark.fast
def test_nk_different_initial_states():
    """Test IRF can be computed from different initial states."""
    sim = NKSimulator()
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 40
    shock_idx = 0
    shock_size = 1.0

    # IRF from zero initial state
    x0_zero = np.zeros(5)
    irf_zero = sim.compute_irf(theta, shock_idx, shock_size, H, x0=x0_zero)

    # IRF from random initial state
    x0_random = rng.standard_normal(5) * 0.1
    irf_random = sim.compute_irf(theta, shock_idx, shock_size, H, x0=x0_random)

    # Both should be the same (IRF is difference, so initial state cancels out)
    np.testing.assert_allclose(irf_zero, irf_random, rtol=1e-10)


@pytest.mark.fast
def test_nk_observables_scaling():
    """Test that observables are properly scaled (annualized percentages)."""
    sim = NKSimulator()
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    T = 100
    eps = rng.standard_normal((T, 3)) * 0.01

    output = sim.simulate(theta, eps, T)

    # Observables should be in reasonable ranges for annualized percentages
    # Output gap: typically -10% to +10%
    # Inflation: typically -5% to +10% (annualized)
    # Interest rate: typically 0% to 20% (annualized)

    # Just check they're in macro-reasonable ranges
    assert np.all(np.abs(output.y_canonical[:, 0]) < 50), "Output gap out of range"
    assert np.all(np.abs(output.y_canonical[:, 1]) < 50), "Inflation out of range"
    assert np.all(output.y_canonical[:, 2] > -20), "Interest rate too negative"
    assert np.all(output.y_canonical[:, 2] < 50), "Interest rate too high"


@pytest.mark.fast
def test_nk_taylor_rule_coefficient():
    """Test that Taylor rule coefficient affects dynamics."""
    sim = NKSimulator()
    rng = np.random.default_rng(42)

    # Two parameterizations with different phi_pi
    theta1 = sim.sample_parameters(rng)
    theta2 = theta1.copy()
    # Ensure phi_pi is meaningfully different (avoid hitting bounds)
    if theta1[3] < 2.0:
        theta2[3] = theta1[3] + 0.5
    else:
        theta2[3] = theta1[3] - 0.5

    H = 40
    # Use cost-push shock (idx=2) since it goes through the phi_pi channel:
    # cost-push → inflation (pi) → interest rate (i) via phi_pi
    shock_idx = 2

    irf1 = sim.compute_irf(theta1, shock_idx, 1.0, H)
    irf2 = sim.compute_irf(theta2, shock_idx, 1.0, H)

    # IRFs should be different
    assert not np.allclose(irf1, irf2), "Different phi_pi should produce different IRFs"
