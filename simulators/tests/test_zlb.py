"""
Tests for Zero Lower Bound (ZLB) Toy simulator.
"""

import numpy as np
import pytest

from simulators.zlb import ZLBSimulator


@pytest.mark.fast
def test_zlb_initialization():
    """Test ZLB simulator initializes correctly."""
    sim = ZLBSimulator()
    assert sim.world_id == "zlb"
    assert sim.n_params == 12  # Same as NK


@pytest.mark.fast
def test_zlb_manifests():
    """Test ZLB simulator manifests are correctly formed."""
    sim = ZLBSimulator()

    # Check parameter manifest (same as NK)
    param_manifest = sim.param_manifest
    assert len(param_manifest.names) == 12
    assert len(param_manifest.units) == 12
    assert param_manifest.bounds.shape == (12, 2)
    assert param_manifest.defaults.shape == (12,)

    # Check shock manifest (same as NK)
    shock_manifest = sim.shock_manifest
    assert shock_manifest.n_shocks == 3
    assert shock_manifest.names == ["monetary", "demand", "cost_push"]
    assert shock_manifest.sigma.shape == (3,)

    # Check observable manifest (same as NK)
    obs_manifest = sim.obs_manifest
    assert obs_manifest.n_canonical == 3
    assert obs_manifest.canonical_names == ["output", "inflation", "rate"]


@pytest.mark.fast
def test_zlb_sample_parameters():
    """Test parameter sampling produces determinate systems."""
    sim = ZLBSimulator()
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
def test_zlb_determinism():
    """Test that same seed produces same output."""
    sim = ZLBSimulator()

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

    # Outputs should be identical
    np.testing.assert_array_equal(output1.y_canonical, output2.y_canonical)

    # Binding fractions should be identical
    assert output1.metadata is not None
    assert output2.metadata is not None
    assert output1.metadata["binding_fraction"] == output2.metadata["binding_fraction"]


@pytest.mark.fast
def test_zlb_simulate():
    """Test simulation produces correct shapes and doesn't explode."""
    sim = ZLBSimulator()
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    T = 100
    eps = rng.standard_normal((T, 3)) * 0.01

    output = sim.simulate(theta, eps, T)

    # Check shapes
    assert output.y_canonical.shape == (T, 3)
    assert output.x_state is not None
    assert output.x_state.shape == (T, 5)

    # Check metadata
    assert output.metadata is not None
    assert "binding_fraction" in output.metadata
    assert 0.0 <= output.metadata["binding_fraction"] <= 1.0

    # Check not exploding
    assert np.all(np.abs(output.y_canonical) < 100)
    assert np.all(np.isfinite(output.y_canonical))


@pytest.mark.fast
def test_zlb_constraint_binds():
    """Test that ZLB constraint binds when rates go negative."""
    sim = ZLBSimulator()
    rng = np.random.default_rng(42)

    # Use parameters with low steady-state rate to increase binding probability
    theta = sim.param_manifest.defaults.copy()
    theta[11] = 0.5  # r_ss = 0.5% annualized (very low)

    # Ensure still determinate
    if not sim.validate_parameters(theta):
        pytest.skip("Low r_ss parameters not determinate")

    T = 200

    # Generate large negative monetary shocks to push rates down
    # Note: shocks are scaled by sigma_m (default 0.0025) inside simulate()
    # To get -1% quarterly shock, need eps = -400 (since -400 * 0.0025 = -1.0 quarterly = -4% annualized)
    eps = np.zeros((T, 3))
    eps[:, 0] = -500.0  # Very large negative monetary shock (500 std devs)

    output = sim.simulate(theta, eps, T)

    # Check that ZLB binds at some point
    assert output.metadata is not None
    binding_fraction = output.metadata["binding_fraction"]

    # With very large negative shocks and low r_ss, should bind most of the time
    assert binding_fraction > 0.5, "ZLB should bind frequently with large negative shocks"

    # Interest rate should never be negative
    assert np.all(output.y_canonical[:, 2] >= 0.0), "Interest rate should never be negative"


@pytest.mark.fast
def test_zlb_binding_fraction_computed():
    """Test that binding_fraction is computed correctly."""
    sim = ZLBSimulator()
    rng = np.random.default_rng(42)

    theta = sim.param_manifest.defaults.copy()
    theta[11] = 0.5  # Low r_ss

    if not sim.validate_parameters(theta):
        pytest.skip("Parameters not determinate")

    T = 100

    # All large negative monetary shocks (scaled by sigma)
    eps = np.zeros((T, 3))
    eps[:, 0] = -500.0  # Very large negative (500 std devs)

    output = sim.simulate(theta, eps, T)

    assert output.metadata is not None
    binding_fraction = output.metadata["binding_fraction"]

    # Should bind in most/all periods with such large negative shocks
    assert binding_fraction > 0.8, "Should bind frequently with very large negative shocks"

    # Verify: count how many periods have rate at exactly 0
    rate_at_zero = np.sum(output.y_canonical[:, 2] < 1e-10)  # Near zero (accounting for float precision)
    expected_fraction = rate_at_zero / T

    # Binding fraction should approximately match periods at zero
    # (May not be exact due to dynamics, but should be close)
    assert abs(binding_fraction - expected_fraction) < 0.2, \
        "binding_fraction should match fraction of periods at ZLB"


@pytest.mark.fast
def test_zlb_non_binding_matches_nk():
    """Test that when ZLB doesn't bind, dynamics match NK simulator.

    Note: ZLB outputs absolute interest rates, NK outputs deviations from steady state.
    So we compare (ZLB rate) vs (NK deviation + r_ss).
    """
    from simulators.nk import NKSimulator

    sim_zlb = ZLBSimulator()
    sim_nk = NKSimulator()
    rng = np.random.default_rng(42)

    # Use high steady-state rate to avoid binding
    theta = sim_zlb.param_manifest.defaults.copy()
    theta[11] = 3.5  # r_ss = 3.5% annualized (high)

    if not sim_zlb.validate_parameters(theta):
        pytest.skip("Parameters not determinate")

    T = 100

    # Small positive shocks (won't push rates negative)
    eps = rng.standard_normal((T, 3)) * 0.001  # Very small shocks

    output_zlb = sim_zlb.simulate(theta, eps, T)
    output_nk = sim_nk.simulate(theta, eps, T)

    # Should not bind
    assert output_zlb.metadata is not None
    assert output_zlb.metadata["binding_fraction"] == 0.0, \
        "ZLB should not bind with high r_ss and small shocks"

    # When not binding, dynamics should match
    # ZLB outputs absolute rates, NK outputs deviations
    # So ZLB rate = NK deviation + r_ss
    r_ss = theta[11]

    # Output gap and inflation should match exactly
    np.testing.assert_allclose(
        output_zlb.y_canonical[:, :2],
        output_nk.y_canonical[:, :2],
        rtol=1e-10,
        err_msg="Output gap and inflation should match between ZLB and NK"
    )

    # Interest rate: ZLB absolute = NK deviation + r_ss
    nk_absolute_rate = output_nk.y_canonical[:, 2] + r_ss
    np.testing.assert_allclose(
        output_zlb.y_canonical[:, 2],
        nk_absolute_rate,
        rtol=1e-10,
        err_msg="ZLB absolute rate should match NK deviation + r_ss when not binding"
    )


@pytest.mark.fast
def test_zlb_compute_irf():
    """Test IRF computation produces correct shapes."""
    sim = ZLBSimulator()
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
def test_zlb_zero_shock_gives_zero_irf():
    """Test that zero-sized shock gives zero IRF."""
    sim = ZLBSimulator()
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 40
    shock_idx = 0
    shock_size = 0.0

    irf = sim.compute_irf(theta, shock_idx, shock_size, H)

    np.testing.assert_allclose(irf, 0, atol=1e-12)


@pytest.mark.fast
def test_zlb_irf_differs_from_nk_when_binding():
    """Test that IRF differs from NK when ZLB binds."""
    from simulators.nk import NKSimulator

    sim_zlb = ZLBSimulator()
    sim_nk = NKSimulator()
    rng = np.random.default_rng(42)

    # Use low steady-state rate
    theta = sim_zlb.param_manifest.defaults.copy()
    theta[11] = 0.5  # r_ss = 0.5% (low)

    if not sim_zlb.validate_parameters(theta):
        pytest.skip("Parameters not determinate")

    H = 40
    shock_idx = 0  # Monetary shock
    shock_size = -500.0  # Very large negative shock (pushes rates to ZLB)

    irf_zlb = sim_zlb.compute_irf(theta, shock_idx, shock_size, H)
    irf_nk = sim_nk.compute_irf(theta, shock_idx, shock_size, H)

    # IRFs should differ when ZLB binds
    # (At least for some horizons, the dynamics will be different)
    max_diff = np.max(np.abs(irf_zlb - irf_nk))
    assert max_diff > 0.1, \
        "ZLB IRF should differ from NK IRF when constraint binds"


@pytest.mark.fast
def test_zlb_interest_rate_never_negative():
    """Test that interest rate never goes negative in any scenario."""
    sim = ZLBSimulator()
    rng = np.random.default_rng(42)

    # Test multiple scenarios
    for scenario in range(5):
        theta = sim.sample_parameters(rng)

        # Try to make binding more likely
        theta[11] = rng.uniform(0.5, 2.0)  # Lower r_ss values

        T = 100
        # Include some very large negative shocks to stress-test the constraint
        eps = rng.standard_normal((T, 3)) * 10.0  # Larger shocks
        eps[:20, 0] = -200.0  # Very large negative monetary shocks early on

        output = sim.simulate(theta, eps, T)

        # Interest rate must always be >= 0
        assert np.all(output.y_canonical[:, 2] >= -1e-10), \
            f"Interest rate went negative in scenario {scenario}"


@pytest.mark.fast
def test_zlb_irf_baseline_difference():
    """Test that IRF is computed as baseline-difference."""
    sim = ZLBSimulator()
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 40
    shock_idx = 1  # Demand shock
    shock_size = 1.0

    # Compute IRF
    irf = sim.compute_irf(theta, shock_idx, shock_size, H)

    # Manually compute baseline and shocked paths
    x0 = np.zeros(5)
    eps_baseline = np.zeros((H + 1, 3))
    eps_shocked = np.zeros((H + 1, 3))
    eps_shocked[0, shock_idx] = shock_size

    output_baseline = sim.simulate(theta, eps_baseline, H + 1, x0)
    output_shocked = sim.simulate(theta, eps_shocked, H + 1, x0)

    irf_manual = output_shocked.y_canonical - output_baseline.y_canonical

    # IRF should match manual computation
    np.testing.assert_allclose(
        irf,
        irf_manual,
        rtol=1e-12,
        err_msg="IRF should be computed as baseline-difference"
    )


@pytest.mark.fast
def test_zlb_different_initial_states():
    """Test IRF computation from different initial states.

    Note: Due to ZLB nonlinearity, IRFs may differ slightly from different
    initial states (unlike linear NK model where they exactly match).
    """
    sim = ZLBSimulator()
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    H = 40
    shock_idx = 0
    shock_size = 1.0

    # IRF from zero initial state
    x0_zero = np.zeros(5)
    irf_zero = sim.compute_irf(theta, shock_idx, shock_size, H, x0=x0_zero)

    # IRF from random initial state
    x0_random = rng.standard_normal(5) * 0.05
    irf_random = sim.compute_irf(theta, shock_idx, shock_size, H, x0=x0_random)

    # Both should compute successfully
    assert irf_zero.shape == (H + 1, 3)
    assert irf_random.shape == (H + 1, 3)

    # For small shocks and initial states, should be close (but may not be exact due to nonlinearity)
    # We just verify they're both valid, not necessarily identical


@pytest.mark.fast
def test_zlb_observables_scaling():
    """Test that observables are properly scaled (annualized percentages)."""
    sim = ZLBSimulator()
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    T = 100
    eps = rng.standard_normal((T, 3)) * 0.01

    output = sim.simulate(theta, eps, T)

    # Observables should be in reasonable ranges for annualized percentages
    assert np.all(np.abs(output.y_canonical[:, 0]) < 50), "Output gap out of range"
    assert np.all(np.abs(output.y_canonical[:, 1]) < 50), "Inflation out of range"

    # Interest rate: must be >= 0 (ZLB) and < 50% (reasonable upper bound)
    assert np.all(output.y_canonical[:, 2] >= 0), "Interest rate negative (violates ZLB)"
    assert np.all(output.y_canonical[:, 2] < 50), "Interest rate too high"


@pytest.mark.fast
def test_zlb_metadata_structure():
    """Test that metadata is correctly structured."""
    sim = ZLBSimulator()
    rng = np.random.default_rng(42)

    theta = sim.sample_parameters(rng)
    T = 50
    eps = rng.standard_normal((T, 3)) * 0.01

    output = sim.simulate(theta, eps, T)

    # Check metadata exists and has correct structure
    assert output.metadata is not None
    assert isinstance(output.metadata, dict)
    assert "binding_fraction" in output.metadata

    # Check binding_fraction is valid
    binding_fraction = output.metadata["binding_fraction"]
    assert isinstance(binding_fraction, float)
    assert 0.0 <= binding_fraction <= 1.0


@pytest.mark.fast
def test_zlb_validate_parameters():
    """Test parameter validation works correctly."""
    sim = ZLBSimulator()

    # Valid parameters (default)
    theta_valid = sim.param_manifest.defaults.copy()
    assert sim.validate_parameters(theta_valid)

    # Invalid: phi_pi < 1 (violates Taylor principle)
    theta_invalid = theta_valid.copy()
    theta_invalid[3] = 0.9
    assert not sim.validate_parameters(theta_invalid)

    # Invalid: out of bounds
    theta_oob = theta_valid.copy()
    theta_oob[0] = 10.0  # sigma way out of bounds
    assert not sim.validate_parameters(theta_oob)
