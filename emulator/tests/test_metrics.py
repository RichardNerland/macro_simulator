"""
Unit tests for evaluation metrics.

Tests cover weighting schemes, accuracy metrics, shape metrics,
and edge cases with known inputs.
"""

import numpy as np
import pytest

from emulator.eval.metrics import (
    exponential_weights,
    gap_metric,
    get_weight_scheme,
    hf_ratio,
    iae,
    impact_weighted,
    nrmse,
    overshoot_ratio,
    sign_at_impact,
    sign_flip_count,
    uniform_weights,
)

# ============================================================================
# Weighting Schemes
# ============================================================================


@pytest.mark.fast
def test_uniform_weights_shape():
    """Test uniform weights return correct shape."""
    H = 40
    weights = uniform_weights(H)

    assert weights.shape == (H + 1,)
    assert weights.dtype == np.float64


@pytest.mark.fast
def test_uniform_weights_sum():
    """Test uniform weights sum to 1.0."""
    H = 40
    weights = uniform_weights(H)

    assert np.allclose(weights.sum(), 1.0)


@pytest.mark.fast
def test_uniform_weights_values():
    """Test uniform weights have equal values."""
    H = 40
    weights = uniform_weights(H)

    expected = 1.0 / (H + 1)
    assert np.allclose(weights, expected)


@pytest.mark.fast
def test_exponential_weights_shape():
    """Test exponential weights return correct shape."""
    H = 40
    weights = exponential_weights(H)

    assert weights.shape == (H + 1,)
    assert weights.dtype == np.float64


@pytest.mark.fast
def test_exponential_weights_sum():
    """Test exponential weights sum to 1.0."""
    H = 40
    weights = exponential_weights(H, tau=20.0)

    assert np.allclose(weights.sum(), 1.0)


@pytest.mark.fast
def test_exponential_weights_decay():
    """Test exponential weights decay monotonically."""
    H = 40
    weights = exponential_weights(H, tau=20.0)

    # Weights should be monotonically decreasing
    assert np.all(weights[:-1] >= weights[1:])

    # First weight should be largest
    assert weights[0] == weights.max()


@pytest.mark.fast
def test_exponential_weights_tau_parameter():
    """Test tau parameter affects decay rate."""
    H = 40
    weights_fast = exponential_weights(H, tau=5.0)
    weights_slow = exponential_weights(H, tau=50.0)

    # Faster decay (smaller tau) should have higher initial weight
    assert weights_fast[0] > weights_slow[0]

    # Slower decay should have higher weight at end
    assert weights_slow[-1] > weights_fast[-1]


@pytest.mark.fast
def test_impact_weighted_shape():
    """Test impact-weighted scheme returns correct shape."""
    H = 40
    weights = impact_weighted(H, impact_length=5)

    assert weights.shape == (H + 1,)
    assert weights.dtype == np.float64


@pytest.mark.fast
def test_impact_weighted_sum():
    """Test impact-weighted scheme sums to 1.0."""
    H = 40
    weights = impact_weighted(H, impact_length=5)

    assert np.allclose(weights.sum(), 1.0)


@pytest.mark.fast
def test_impact_weighted_upweights_early():
    """Test impact-weighted scheme upweights early horizons."""
    H = 40
    impact_length = 5
    weights = impact_weighted(H, impact_length=impact_length)

    # First impact_length weights should be larger than later ones
    # Compare mean of early vs late weights
    mean_early = np.mean(weights[:impact_length])
    mean_late = np.mean(weights[impact_length:])
    assert mean_early > mean_late


@pytest.mark.fast
def test_get_weight_scheme_uniform():
    """Test factory function returns uniform weights."""
    H = 40
    weights = get_weight_scheme("uniform", H)
    expected = uniform_weights(H)

    assert np.allclose(weights, expected)


@pytest.mark.fast
def test_get_weight_scheme_exponential():
    """Test factory function returns exponential weights."""
    H = 40
    tau = 15.0
    weights = get_weight_scheme("exponential", H, tau=tau)
    expected = exponential_weights(H, tau=tau)

    assert np.allclose(weights, expected)


@pytest.mark.fast
def test_get_weight_scheme_impact():
    """Test factory function returns impact weights."""
    H = 40
    impact_length = 8
    weights = get_weight_scheme("impact", H, impact_length=impact_length)
    expected = impact_weighted(H, impact_length=impact_length)

    assert np.allclose(weights, expected)


@pytest.mark.fast
def test_get_weight_scheme_invalid():
    """Test factory function raises on invalid scheme."""
    with pytest.raises(ValueError, match="Unknown weight scheme"):
        get_weight_scheme("invalid_scheme", 40)


# ============================================================================
# NRMSE Metric
# ============================================================================


@pytest.mark.fast
def test_nrmse_perfect_prediction():
    """Test NRMSE = 0 when prediction equals truth."""
    np.random.seed(42)
    H = 40
    n_obs = 3
    n_samples = 10

    y_true = np.random.randn(n_samples, H + 1, n_obs)
    y_pred = y_true.copy()

    result = nrmse(y_pred, y_true)

    assert np.isclose(result, 0.0, atol=1e-10)


@pytest.mark.fast
def test_nrmse_single_irf():
    """Test NRMSE handles single IRF (2D input)."""
    np.random.seed(42)
    H = 40
    n_obs = 3

    y_true = np.random.randn(H + 1, n_obs)
    y_pred = y_true + 0.1 * np.random.randn(H + 1, n_obs)

    result = nrmse(y_pred, y_true)

    assert isinstance(result, float)
    assert result > 0.0


@pytest.mark.fast
def test_nrmse_batch():
    """Test NRMSE handles batch of IRFs (3D input)."""
    np.random.seed(42)
    H = 40
    n_obs = 3
    n_samples = 10

    y_true = np.random.randn(n_samples, H + 1, n_obs)
    y_pred = y_true + 0.1 * np.random.randn(n_samples, H + 1, n_obs)

    result = nrmse(y_pred, y_true)

    assert isinstance(result, float)
    assert result > 0.0


@pytest.mark.fast
def test_nrmse_per_variable():
    """Test NRMSE per_variable flag returns array."""
    np.random.seed(42)
    H = 40
    n_obs = 3
    n_samples = 10

    y_true = np.random.randn(n_samples, H + 1, n_obs)
    y_pred = y_true + 0.1 * np.random.randn(n_samples, H + 1, n_obs)

    result = nrmse(y_pred, y_true, per_variable=True)

    assert isinstance(result, np.ndarray)
    assert result.shape == (n_obs,)
    assert np.all(result > 0.0)


@pytest.mark.fast
def test_nrmse_with_weights():
    """Test NRMSE respects horizon weights."""
    np.random.seed(42)
    H = 40
    n_obs = 3

    y_true = np.zeros((H + 1, n_obs))
    y_pred = np.zeros((H + 1, n_obs))

    # Add error only at first horizon
    y_pred[0, :] = 1.0

    # With uniform weights, all horizons matter equally
    weights_uniform = uniform_weights(H)
    nrmse_uniform = nrmse(y_pred, y_true, weights=weights_uniform)

    # With impact weights, first horizon has more weight
    weights_impact = impact_weighted(H, impact_length=5)
    nrmse_impact = nrmse(y_pred, y_true, weights=weights_impact)

    # Impact weighting should give higher NRMSE for error at h=0
    assert nrmse_impact > nrmse_uniform


@pytest.mark.fast
def test_nrmse_with_sigma():
    """Test NRMSE uses provided sigma for normalization."""
    np.random.seed(42)
    H = 40
    n_obs = 3
    n_samples = 10

    y_true = np.random.randn(n_samples, H + 1, n_obs)
    y_pred = y_true + 0.1 * np.random.randn(n_samples, H + 1, n_obs)

    # Provide custom sigma
    sigma = np.array([1.0, 2.0, 0.5])
    result = nrmse(y_pred, y_true, sigma=sigma)

    assert isinstance(result, float)
    assert result > 0.0


@pytest.mark.fast
def test_nrmse_zero_sigma_handling():
    """Test NRMSE handles zero sigma gracefully."""
    np.random.seed(42)
    H = 40
    n_obs = 3

    # Constant true values (zero variance)
    y_true = np.ones((10, H + 1, n_obs))
    y_pred = y_true + 0.1 * np.random.randn(10, H + 1, n_obs)

    # Should not raise division by zero
    result = nrmse(y_pred, y_true)
    assert np.isfinite(result)


# ============================================================================
# IAE Metric
# ============================================================================


@pytest.mark.fast
def test_iae_perfect_prediction():
    """Test IAE = 0 when prediction equals truth."""
    np.random.seed(42)
    H = 40
    n_obs = 3
    n_samples = 10

    y_true = np.random.randn(n_samples, H + 1, n_obs)
    y_pred = y_true.copy()

    result = iae(y_pred, y_true)

    assert np.isclose(result, 0.0, atol=1e-10)


@pytest.mark.fast
def test_iae_known_value():
    """Test IAE with known constant error."""
    H = 10
    n_obs = 2

    y_true = np.zeros((H + 1, n_obs))
    y_pred = np.ones((H + 1, n_obs))  # Constant error of 1.0

    result = iae(y_pred, y_true)

    # IAE = Σ_h |error| = (H+1) * 1.0 = 11
    expected = float(H + 1)
    assert np.isclose(result, expected)


@pytest.mark.fast
def test_iae_single_irf():
    """Test IAE handles single IRF (2D input)."""
    np.random.seed(42)
    H = 40
    n_obs = 3

    y_true = np.random.randn(H + 1, n_obs)
    y_pred = y_true + 0.1 * np.random.randn(H + 1, n_obs)

    result = iae(y_pred, y_true)

    assert isinstance(result, float)
    assert result > 0.0


@pytest.mark.fast
def test_iae_batch():
    """Test IAE handles batch of IRFs (3D input)."""
    np.random.seed(42)
    H = 40
    n_obs = 3
    n_samples = 10

    y_true = np.random.randn(n_samples, H + 1, n_obs)
    y_pred = y_true + 0.1 * np.random.randn(n_samples, H + 1, n_obs)

    result = iae(y_pred, y_true)

    assert isinstance(result, float)
    assert result > 0.0


@pytest.mark.fast
def test_iae_per_variable():
    """Test IAE per_variable flag returns array."""
    np.random.seed(42)
    H = 40
    n_obs = 3
    n_samples = 10

    y_true = np.random.randn(n_samples, H + 1, n_obs)
    y_pred = y_true + 0.1 * np.random.randn(n_samples, H + 1, n_obs)

    result = iae(y_pred, y_true, per_variable=True)

    assert isinstance(result, np.ndarray)
    assert result.shape == (n_obs,)
    assert np.all(result > 0.0)


# ============================================================================
# Sign-at-Impact Metric
# ============================================================================


@pytest.mark.fast
def test_sign_at_impact_perfect():
    """Test sign accuracy = 1.0 when signs match perfectly."""
    np.random.seed(42)
    H = 40
    n_obs = 3
    n_samples = 10

    y_true = np.random.randn(n_samples, H + 1, n_obs)
    y_pred = y_true.copy()

    result = sign_at_impact(y_pred, y_true)

    assert np.isclose(result, 1.0)


@pytest.mark.fast
def test_sign_at_impact_opposite():
    """Test sign accuracy = 0.0 when signs are opposite."""
    H = 40
    n_obs = 3
    n_samples = 10

    y_true = np.ones((n_samples, H + 1, n_obs))
    y_pred = -np.ones((n_samples, H + 1, n_obs))

    result = sign_at_impact(y_pred, y_true)

    assert np.isclose(result, 0.0)


@pytest.mark.fast
def test_sign_at_impact_single_irf():
    """Test sign accuracy handles single IRF (2D input)."""
    H = 40
    n_obs = 3

    y_true = np.ones((H + 1, n_obs))
    y_pred = np.ones((H + 1, n_obs))

    result = sign_at_impact(y_pred, y_true)

    assert isinstance(result, float)
    assert np.isclose(result, 1.0)


@pytest.mark.fast
def test_sign_at_impact_batch():
    """Test sign accuracy handles batch of IRFs (3D input)."""
    np.random.seed(42)
    H = 40
    n_obs = 3
    n_samples = 10

    y_true = np.random.randn(n_samples, H + 1, n_obs)
    y_pred = y_true * 1.2  # Scale but preserve sign

    result = sign_at_impact(y_pred, y_true)

    assert isinstance(result, float)
    assert np.isclose(result, 1.0)


@pytest.mark.fast
def test_sign_at_impact_per_variable():
    """Test sign accuracy per_variable flag returns array."""
    np.random.seed(42)
    H = 40
    n_obs = 3
    n_samples = 10

    y_true = np.random.randn(n_samples, H + 1, n_obs)
    y_pred = y_true.copy()

    result = sign_at_impact(y_pred, y_true, per_variable=True)

    assert isinstance(result, np.ndarray)
    assert result.shape == (n_obs,)
    assert np.allclose(result, 1.0)


@pytest.mark.fast
def test_sign_at_impact_horizon_window():
    """Test sign accuracy respects horizon_window parameter."""
    H = 40
    n_obs = 3

    # Create IRF where sign flips after h=2
    y_true = np.ones((H + 1, n_obs))
    y_pred = np.ones((H + 1, n_obs))
    y_pred[3:, :] = -1.0  # Flip sign after h=2

    # With window=3 (h=0,1,2), all signs match
    result_window3 = sign_at_impact(y_pred, y_true, horizon_window=3)
    assert np.isclose(result_window3, 1.0)

    # With window=5 (h=0,1,2,3,4), some signs don't match
    result_window5 = sign_at_impact(y_pred, y_true, horizon_window=5)
    assert result_window5 < 1.0


@pytest.mark.fast
def test_sign_at_impact_zero_handling():
    """Test sign accuracy handles zeros correctly."""
    H = 40
    n_obs = 3

    y_true = np.zeros((H + 1, n_obs))
    y_pred = np.zeros((H + 1, n_obs))

    # sign(0) == 0, so 0 == 0 should be True
    result = sign_at_impact(y_pred, y_true)
    assert np.isclose(result, 1.0)


# ============================================================================
# Gap Metric
# ============================================================================


@pytest.mark.fast
def test_gap_metric_equal_performance():
    """Test gap = 0% when models perform equally."""
    nrmse_universal = 0.5
    nrmse_specialist = 0.5

    gap = gap_metric(nrmse_universal, nrmse_specialist)

    assert np.isclose(gap, 0.0)


@pytest.mark.fast
def test_gap_metric_universal_better():
    """Test gap < 0% when universal beats specialist."""
    nrmse_universal = 0.4
    nrmse_specialist = 0.5

    gap = gap_metric(nrmse_universal, nrmse_specialist)

    expected = (0.4 - 0.5) / 0.5 * 100.0  # -20%
    assert np.isclose(gap, expected)
    assert gap < 0.0


@pytest.mark.fast
def test_gap_metric_universal_worse():
    """Test gap > 0% when universal is worse than specialist."""
    nrmse_universal = 0.6
    nrmse_specialist = 0.5

    gap = gap_metric(nrmse_universal, nrmse_specialist)

    expected = (0.6 - 0.5) / 0.5 * 100.0  # 20%
    assert np.isclose(gap, expected)
    assert gap > 0.0


@pytest.mark.fast
def test_gap_metric_known_value():
    """Test gap metric with known value."""
    nrmse_universal = 0.3
    nrmse_specialist = 0.2

    gap = gap_metric(nrmse_universal, nrmse_specialist)

    expected = 50.0  # 50% worse
    assert np.isclose(gap, expected)


# ============================================================================
# HF-Ratio Metric
# ============================================================================


@pytest.mark.fast
def test_hf_ratio_constant_irf():
    """Test HF-ratio ≈ 0 for constant IRF (no oscillation)."""
    H = 40
    n_obs = 3

    # Constant IRF (DC component only)
    irf = np.ones((H + 1, n_obs))

    result = hf_ratio(irf, cutoff_period=6.0)

    # All energy in DC (freq=0), no high-frequency content
    assert result < 0.01  # Essentially zero


@pytest.mark.fast
def test_hf_ratio_smooth_decay():
    """Test HF-ratio low for smooth exponential decay."""
    H = 40

    # Smooth exponential decay
    h = np.arange(H + 1)
    irf = np.column_stack([
        np.exp(-0.1 * h),
        np.exp(-0.15 * h),
        np.exp(-0.2 * h),
    ])

    result = hf_ratio(irf, cutoff_period=6.0)

    # Smooth decay should have low HF content
    assert result < 0.3


@pytest.mark.fast
def test_hf_ratio_oscillating_irf():
    """Test HF-ratio high for oscillating IRF."""
    H = 40

    # High-frequency oscillation
    h = np.arange(H + 1)
    irf = np.column_stack([
        np.sin(2 * np.pi * h / 4),  # Period = 4 quarters
        np.cos(2 * np.pi * h / 5),  # Period = 5 quarters
        np.sin(2 * np.pi * h / 3),  # Period = 3 quarters
    ])

    result = hf_ratio(irf, cutoff_period=6.0)

    # Oscillations with periods < 6 should have high HF content
    assert result > 0.5


@pytest.mark.fast
def test_hf_ratio_single_irf():
    """Test HF-ratio handles single IRF (2D input)."""
    H = 40

    h = np.arange(H + 1)
    irf = np.column_stack([
        np.exp(-0.1 * h),
        np.exp(-0.15 * h),
        np.exp(-0.2 * h),
    ])

    result = hf_ratio(irf)

    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


@pytest.mark.fast
def test_hf_ratio_batch():
    """Test HF-ratio handles batch of IRFs (3D input)."""
    np.random.seed(42)
    H = 40
    n_obs = 3
    n_samples = 10

    h = np.arange(H + 1)
    irf = np.zeros((n_samples, H + 1, n_obs))
    for i in range(n_samples):
        irf[i] = np.column_stack([
            np.exp(-0.1 * h),
            np.exp(-0.15 * h),
            np.exp(-0.2 * h),
        ])

    result = hf_ratio(irf)

    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


@pytest.mark.fast
def test_hf_ratio_per_variable():
    """Test HF-ratio per_variable flag returns array."""
    np.random.seed(42)
    H = 40
    n_obs = 3
    n_samples = 10

    h = np.arange(H + 1)
    irf = np.zeros((n_samples, H + 1, n_obs))
    for i in range(n_samples):
        irf[i] = np.column_stack([
            np.exp(-0.1 * h),
            np.exp(-0.15 * h),
            np.exp(-0.2 * h),
        ])

    result = hf_ratio(irf, per_variable=True)

    assert isinstance(result, np.ndarray)
    assert result.shape == (n_obs,)
    assert np.all((result >= 0.0) & (result <= 1.0))


@pytest.mark.fast
def test_hf_ratio_cutoff_sensitivity():
    """Test HF-ratio changes with cutoff_period."""
    H = 40

    # IRF with period = 8 quarters
    h = np.arange(H + 1)
    irf = np.sin(2 * np.pi * h / 8).reshape(-1, 1)

    # With cutoff=10, period=8 is high-frequency
    hf_ratio_10 = hf_ratio(irf, cutoff_period=10.0)

    # With cutoff=6, period=8 is low-frequency
    hf_ratio_6 = hf_ratio(irf, cutoff_period=6.0)

    # More HF content with higher cutoff
    assert hf_ratio_10 > hf_ratio_6


# ============================================================================
# Overshoot Ratio Metric
# ============================================================================


@pytest.mark.fast
def test_overshoot_ratio_no_overshoot():
    """Test overshoot = 1.0 when peak equals impact."""
    H = 40

    # Monotonic decay (peak at impact)
    h = np.arange(H + 1)
    irf = np.column_stack([
        np.exp(-0.1 * h),
        np.exp(-0.15 * h),
        np.exp(-0.2 * h),
    ])

    result = overshoot_ratio(irf)

    # Peak is at h=0, so overshoot = |irf[0]| / |irf[0]| ≈ 1.0
    assert np.isclose(result, 1.0, rtol=1e-3)


@pytest.mark.fast
def test_overshoot_ratio_with_overshoot():
    """Test overshoot > 1.0 when peak exceeds impact."""
    H = 40

    # IRF with hump shape (peak after impact)
    h = np.arange(H + 1)
    irf = np.column_stack([
        h * np.exp(-0.1 * h),  # Hump-shaped
        h * np.exp(-0.15 * h),
        h * np.exp(-0.2 * h),
    ])

    result = overshoot_ratio(irf)

    # Peak is greater than impact, so overshoot > 1.0
    assert result > 1.0


@pytest.mark.fast
def test_overshoot_ratio_known_value():
    """Test overshoot with known values."""
    H = 10
    n_obs = 1

    # Simple case: impact=1.0, peak=2.0
    irf = np.zeros((H + 1, n_obs))
    irf[0, 0] = 1.0  # Impact
    irf[5, 0] = 2.0  # Peak

    result = overshoot_ratio(irf)

    expected = 2.0  # Peak / impact
    assert np.isclose(result, expected)


@pytest.mark.fast
def test_overshoot_ratio_single_irf():
    """Test overshoot handles single IRF (2D input)."""
    H = 40

    h = np.arange(H + 1)
    irf = np.column_stack([
        np.exp(-0.1 * h),
        np.exp(-0.15 * h),
        np.exp(-0.2 * h),
    ])

    result = overshoot_ratio(irf)

    assert isinstance(result, float)
    # Allow small numerical error (overshoot should be ~1.0 for monotonic decay)
    assert result >= 0.99


@pytest.mark.fast
def test_overshoot_ratio_batch():
    """Test overshoot handles batch of IRFs (3D input)."""
    np.random.seed(42)
    H = 40
    n_obs = 3
    n_samples = 10

    h = np.arange(H + 1)
    irf = np.zeros((n_samples, H + 1, n_obs))
    for i in range(n_samples):
        irf[i] = np.column_stack([
            np.exp(-0.1 * h),
            np.exp(-0.15 * h),
            np.exp(-0.2 * h),
        ])

    result = overshoot_ratio(irf)

    assert isinstance(result, float)
    # Allow small numerical error (overshoot should be ~1.0 for monotonic decay)
    assert result >= 0.99


@pytest.mark.fast
def test_overshoot_ratio_per_variable():
    """Test overshoot per_variable flag returns array."""
    np.random.seed(42)
    H = 40
    n_obs = 3
    n_samples = 10

    h = np.arange(H + 1)
    irf = np.zeros((n_samples, H + 1, n_obs))
    for i in range(n_samples):
        irf[i] = np.column_stack([
            np.exp(-0.1 * h),
            np.exp(-0.15 * h),
            np.exp(-0.2 * h),
        ])

    result = overshoot_ratio(irf, per_variable=True)

    assert isinstance(result, np.ndarray)
    assert result.shape == (n_obs,)
    # Allow small numerical error (overshoot should be ~1.0 for monotonic decay)
    assert np.all(result >= 0.99)


@pytest.mark.fast
def test_overshoot_ratio_zero_impact():
    """Test overshoot handles zero impact with epsilon."""
    H = 40
    n_obs = 3

    # IRF starting at zero
    irf = np.zeros((H + 1, n_obs))
    irf[10:20, :] = 1.0  # Non-zero values later

    # Should not raise division by zero
    result = overshoot_ratio(irf, epsilon=1e-8)
    assert np.isfinite(result)


# ============================================================================
# Sign-Flip Count Metric
# ============================================================================


@pytest.mark.fast
def test_sign_flip_count_monotonic():
    """Test sign-flip count = 0 for monotonic IRF."""
    H = 40

    # Monotonic decay (no sign flips in first diff)
    h = np.arange(H + 1)
    irf = np.column_stack([
        np.exp(-0.1 * h),
        np.exp(-0.15 * h),
        np.exp(-0.2 * h),
    ])

    result = sign_flip_count(irf)

    # Monotonic decay has no sign flips in differences
    assert np.isclose(result, 0.0)


@pytest.mark.fast
def test_sign_flip_count_oscillating():
    """Test sign-flip count > 0 for oscillating IRF."""
    H = 40

    # Oscillating IRF
    h = np.arange(H + 1)
    irf = np.column_stack([
        np.sin(2 * np.pi * h / 8),
        np.cos(2 * np.pi * h / 8),
        np.sin(2 * np.pi * h / 10),
    ])

    result = sign_flip_count(irf)

    # Oscillations should produce multiple sign flips
    assert result > 1.0


@pytest.mark.fast
def test_sign_flip_count_single_hump():
    """Test sign-flip count for hump-shaped IRF."""
    H = 20

    # Hump: increases then decreases (one sign flip in diff)
    h = np.arange(H + 1)
    irf = (h * np.exp(-0.2 * h)).reshape(-1, 1)

    result = sign_flip_count(irf)

    # One sign flip when going from increase to decrease
    assert np.isclose(result, 1.0, atol=0.5)


@pytest.mark.fast
def test_sign_flip_count_single_irf():
    """Test sign-flip count handles single IRF (2D input)."""
    H = 40

    h = np.arange(H + 1)
    irf = np.column_stack([
        np.exp(-0.1 * h),
        np.exp(-0.15 * h),
        np.exp(-0.2 * h),
    ])

    result = sign_flip_count(irf)

    assert isinstance(result, float)
    assert result >= 0.0


@pytest.mark.fast
def test_sign_flip_count_batch():
    """Test sign-flip count handles batch of IRFs (3D input)."""
    np.random.seed(42)
    H = 40
    n_obs = 3
    n_samples = 10

    h = np.arange(H + 1)
    irf = np.zeros((n_samples, H + 1, n_obs))
    for i in range(n_samples):
        irf[i] = np.column_stack([
            np.exp(-0.1 * h),
            np.exp(-0.15 * h),
            np.exp(-0.2 * h),
        ])

    result = sign_flip_count(irf)

    assert isinstance(result, float)
    assert result >= 0.0


@pytest.mark.fast
def test_sign_flip_count_per_variable():
    """Test sign-flip count per_variable flag returns array."""
    np.random.seed(42)
    H = 40
    n_obs = 3
    n_samples = 10

    h = np.arange(H + 1)
    irf = np.zeros((n_samples, H + 1, n_obs))
    for i in range(n_samples):
        irf[i] = np.column_stack([
            np.exp(-0.1 * h),
            np.exp(-0.15 * h),
            np.exp(-0.2 * h),
        ])

    result = sign_flip_count(irf, per_variable=True)

    assert isinstance(result, np.ndarray)
    assert result.shape == (n_obs,)
    assert np.all(result >= 0.0)


@pytest.mark.fast
def test_sign_flip_count_known_value():
    """Test sign-flip count with known pattern."""

    # Create IRF with exactly 2 sign flips in differences
    # Pattern: [0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2]
    irf = np.array([0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2]).reshape(-1, 1).astype(float)

    result = sign_flip_count(irf)

    # Differences: [1, 1, -1, -1, 1, 1, -1, -1, 1, 1]
    # Sign changes: 4 flips
    assert np.isclose(result, 4.0)


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


@pytest.mark.fast
def test_metrics_consistency_across_dimensions():
    """Test metrics give same results for single vs batch with one sample."""
    np.random.seed(42)
    H = 40
    n_obs = 3

    # Single IRF (2D)
    y_true_single = np.random.randn(H + 1, n_obs)
    y_pred_single = y_true_single + 0.1 * np.random.randn(H + 1, n_obs)

    # Batch with one sample (3D)
    y_true_batch = y_true_single[np.newaxis, ...]
    y_pred_batch = y_pred_single[np.newaxis, ...]

    # Test NRMSE
    nrmse_single = nrmse(y_pred_single, y_true_single)
    nrmse_batch = nrmse(y_pred_batch, y_true_batch)
    assert np.isclose(nrmse_single, nrmse_batch)

    # Test IAE
    iae_single = iae(y_pred_single, y_true_single)
    iae_batch = iae(y_pred_batch, y_true_batch)
    assert np.isclose(iae_single, iae_batch)

    # Test sign accuracy
    sign_single = sign_at_impact(y_pred_single, y_true_single)
    sign_batch = sign_at_impact(y_pred_batch, y_true_batch)
    assert np.isclose(sign_single, sign_batch)


@pytest.mark.fast
def test_shape_metrics_consistency_across_dimensions():
    """Test shape metrics give same results for single vs batch."""
    np.random.seed(42)
    H = 40

    # Single IRF
    h = np.arange(H + 1)
    irf_single = np.column_stack([
        np.exp(-0.1 * h),
        np.exp(-0.15 * h),
        np.exp(-0.2 * h),
    ])

    # Batch with one sample
    irf_batch = irf_single[np.newaxis, ...]

    # Test HF-ratio
    hf_single = hf_ratio(irf_single)
    hf_batch = hf_ratio(irf_batch)
    assert np.isclose(hf_single, hf_batch, rtol=1e-5)

    # Test overshoot
    overshoot_single = overshoot_ratio(irf_single)
    overshoot_batch = overshoot_ratio(irf_batch)
    assert np.isclose(overshoot_single, overshoot_batch)

    # Test sign-flip count
    flip_single = sign_flip_count(irf_single)
    flip_batch = sign_flip_count(irf_batch)
    assert np.isclose(flip_single, flip_batch)


@pytest.mark.fast
def test_all_metrics_return_valid_ranges():
    """Test all metrics return values in expected ranges."""
    np.random.seed(42)
    H = 40
    n_obs = 3
    n_samples = 10

    y_true = np.random.randn(n_samples, H + 1, n_obs)
    y_pred = y_true + 0.1 * np.random.randn(n_samples, H + 1, n_obs)

    # NRMSE should be non-negative
    nrmse_val = nrmse(y_pred, y_true)
    assert nrmse_val >= 0.0

    # IAE should be non-negative
    iae_val = iae(y_pred, y_true)
    assert iae_val >= 0.0

    # Sign accuracy should be in [0, 1]
    sign_val = sign_at_impact(y_pred, y_true)
    assert 0.0 <= sign_val <= 1.0

    # HF-ratio should be in [0, 1]
    hf_val = hf_ratio(y_true)
    assert 0.0 <= hf_val <= 1.0

    # Overshoot should be >= 1.0
    overshoot_val = overshoot_ratio(y_true)
    assert overshoot_val >= 1.0

    # Sign-flip count should be non-negative
    flip_val = sign_flip_count(y_true)
    assert flip_val >= 0.0
