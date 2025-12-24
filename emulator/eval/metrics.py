"""
Evaluation metrics for the Universal Macro Emulator.

This module implements accuracy and shape metrics for assessing IRF prediction quality.
All metrics follow spec §7 conventions.

Metric Categories:
1. Accuracy Metrics: NRMSE, IAE, Sign-at-Impact
2. Shape Metrics: HF-ratio, Overshoot, Sign-flip count
3. Gap Metrics: Universal vs specialist comparison

All metrics operate on IRF arrays with shape:
- Single IRF: (H+1, n_obs)
- Batch of IRFs: (n_samples, H+1, n_obs)
- Multi-shock batch: (n_samples, n_shocks, H+1, n_obs)
"""

from typing import Literal

import numpy as np
import numpy.typing as npt


# ============================================================================
# Horizon Weighting Schemes
# ============================================================================


def uniform_weights(H: int) -> npt.NDArray[np.float64]:
    """Uniform horizon weights.

    Args:
        H: Horizon length (number of periods after impact)

    Returns:
        Weights of shape (H+1,) with w[h] = 1/(H+1)
    """
    return np.ones(H + 1, dtype=np.float64) / (H + 1)


def exponential_weights(H: int, tau: float = 20.0) -> npt.NDArray[np.float64]:
    """Exponential horizon weights w[h] ∝ exp(-h/tau).

    Args:
        H: Horizon length
        tau: Decay parameter (default: 20.0 quarters)

    Returns:
        Normalized weights of shape (H+1,)
    """
    h = np.arange(H + 1, dtype=np.float64)
    w = np.exp(-h / tau)
    return w / w.sum()


def impact_weighted(H: int, impact_length: int = 5) -> npt.NDArray[np.float64]:
    """Impact-weighted horizon weights.

    Higher weight on first `impact_length` horizons.

    Args:
        H: Horizon length
        impact_length: Number of horizons to upweight (default: 5)

    Returns:
        Normalized weights of shape (H+1,)
    """
    w = np.ones(H + 1, dtype=np.float64)
    w[:impact_length] = 2.0  # Double weight on impact period
    return w / w.sum()


def get_weight_scheme(
    scheme: Literal["uniform", "exponential", "impact"],
    H: int,
    **kwargs,
) -> npt.NDArray[np.float64]:
    """Get horizon weighting scheme by name.

    Args:
        scheme: One of "uniform", "exponential", "impact"
        H: Horizon length
        **kwargs: Additional arguments for specific schemes

    Returns:
        Weights of shape (H+1,)
    """
    if scheme == "uniform":
        return uniform_weights(H)
    elif scheme == "exponential":
        tau = kwargs.get("tau", 20.0)
        return exponential_weights(H, tau=tau)
    elif scheme == "impact":
        impact_length = kwargs.get("impact_length", 5)
        return impact_weighted(H, impact_length=impact_length)
    else:
        raise ValueError(f"Unknown weight scheme: {scheme}")


# ============================================================================
# Accuracy Metrics
# ============================================================================


def nrmse(
    y_pred: npt.NDArray[np.float64],
    y_true: npt.NDArray[np.float64],
    sigma: npt.NDArray[np.float64] | None = None,
    weights: npt.NDArray[np.float64] | None = None,
    per_variable: bool = False,
) -> float | npt.NDArray[np.float64]:
    """Normalized Root Mean Square Error (spec §7.2.1).

    Formula:
        NRMSE = sqrt(Σ_h w[h] * (ŷ[h,v] - y[h,v])²) / σ[v]

    Aggregated over samples and variables (unless per_variable=True).

    Args:
        y_pred: Predictions, shape (n_samples, H+1, n_obs) or (H+1, n_obs)
        y_true: Ground truth, same shape as y_pred
        sigma: Per-variable standard deviation for normalization, shape (n_obs,).
               If None, computed from y_true.
        weights: Horizon weights, shape (H+1,). If None, uses uniform.
        per_variable: If True, return per-variable NRMSE, shape (n_obs,)

    Returns:
        Scalar NRMSE (averaged over variables) or array of shape (n_obs,)
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    # Handle single IRF (H+1, n_obs) by adding batch dimension
    if y_pred.ndim == 2:
        y_pred = y_pred[np.newaxis, ...]
        y_true = y_true[np.newaxis, ...]

    n_samples, H_plus_1, n_obs = y_pred.shape
    H = H_plus_1 - 1

    # Default weights: uniform
    if weights is None:
        weights = uniform_weights(H)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    # Compute sigma if not provided
    if sigma is None:
        # Compute per-variable std across samples and horizons
        sigma = np.std(y_true, axis=(0, 1))  # shape: (n_obs,)
        # Avoid division by zero
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
    else:
        sigma = np.asarray(sigma, dtype=np.float64)

    # Squared errors: (n_samples, H+1, n_obs)
    sq_errors = (y_pred - y_true) ** 2

    # Weighted sum over horizons: (n_samples, n_obs)
    weighted_sq_errors = np.einsum("h,shv->sv", weights, sq_errors)

    # RMSE per sample per variable
    rmse_per_sample_var = np.sqrt(weighted_sq_errors)  # (n_samples, n_obs)

    # Normalize by sigma: (n_samples, n_obs)
    nrmse_per_sample_var = rmse_per_sample_var / sigma[np.newaxis, :]

    # Average over samples: (n_obs,)
    nrmse_per_var = np.mean(nrmse_per_sample_var, axis=0)

    if per_variable:
        return nrmse_per_var
    else:
        # Average over variables
        return float(np.mean(nrmse_per_var))


def iae(
    y_pred: npt.NDArray[np.float64],
    y_true: npt.NDArray[np.float64],
    per_variable: bool = False,
) -> float | npt.NDArray[np.float64]:
    """Integrated Absolute Error (spec §7.1.1).

    Formula:
        IAE = Σ_h |ŷ[h,v] - y[h,v]|

    Args:
        y_pred: Predictions, shape (n_samples, H+1, n_obs) or (H+1, n_obs)
        y_true: Ground truth, same shape as y_pred
        per_variable: If True, return per-variable IAE, shape (n_obs,)

    Returns:
        Scalar IAE (averaged over samples and variables) or array of shape (n_obs,)
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    # Handle single IRF
    if y_pred.ndim == 2:
        y_pred = y_pred[np.newaxis, ...]
        y_true = y_true[np.newaxis, ...]

    # Sum absolute errors over horizons: (n_samples, n_obs)
    abs_errors = np.abs(y_pred - y_true)
    iae_per_sample_var = np.sum(abs_errors, axis=1)

    # Average over samples: (n_obs,)
    iae_per_var = np.mean(iae_per_sample_var, axis=0)

    if per_variable:
        return iae_per_var
    else:
        return float(np.mean(iae_per_var))


def sign_at_impact(
    y_pred: npt.NDArray[np.float64],
    y_true: npt.NDArray[np.float64],
    horizon_window: int = 3,
    per_variable: bool = False,
) -> float | npt.NDArray[np.float64]:
    """Sign-at-impact accuracy (spec §7.1.1).

    Measures whether prediction gets the direction correct for first few horizons.

    Formula:
        accuracy(sign(ŷ[0:horizon_window]) == sign(y[0:horizon_window]))

    Args:
        y_pred: Predictions, shape (n_samples, H+1, n_obs) or (H+1, n_obs)
        y_true: Ground truth, same shape as y_pred
        horizon_window: Number of horizons to check (default: 3, i.e., h=0,1,2)
        per_variable: If True, return per-variable accuracy, shape (n_obs,)

    Returns:
        Scalar accuracy in [0, 1] or array of shape (n_obs,)
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    # Handle single IRF
    if y_pred.ndim == 2:
        y_pred = y_pred[np.newaxis, ...]
        y_true = y_true[np.newaxis, ...]

    # Check sign agreement for first `horizon_window` horizons
    signs_match = np.sign(y_pred[:, :horizon_window, :]) == np.sign(
        y_true[:, :horizon_window, :]
    )

    # Average over horizons and samples: (n_obs,)
    accuracy_per_var = np.mean(signs_match, axis=(0, 1))

    if per_variable:
        return accuracy_per_var
    else:
        return float(np.mean(accuracy_per_var))


def gap_metric(
    nrmse_universal: float,
    nrmse_specialist: float,
) -> float:
    """Gap between universal and specialist model (spec §7.2.1).

    Formula:
        Gap = (NRMSE_universal - NRMSE_specialist) / NRMSE_specialist * 100%

    Args:
        nrmse_universal: NRMSE of universal model
        nrmse_specialist: NRMSE of specialist model

    Returns:
        Gap percentage (can be negative if universal beats specialist)
    """
    return (nrmse_universal - nrmse_specialist) / nrmse_specialist * 100.0


# ============================================================================
# Shape Metrics
# ============================================================================


def hf_ratio(
    irf: npt.NDArray[np.float64],
    cutoff_period: float = 6.0,
    per_variable: bool = False,
) -> float | npt.NDArray[np.float64]:
    """High-frequency energy ratio via DFT (spec §7.1.2).

    Measures oscillation/wiggliness by computing fraction of spectral energy
    in high frequencies (periods < cutoff_period).

    Formula:
        HF-ratio = Σ_{f≥f_c} |Y(f)|² / Σ_f |Y(f)|²

    Where f_c corresponds to periods < cutoff_period.

    Args:
        irf: IRF array, shape (n_samples, H+1, n_obs) or (H+1, n_obs)
        cutoff_period: Period threshold in quarters (default: 6.0)
        per_variable: If True, return per-variable ratio, shape (n_obs,)

    Returns:
        HF-ratio in [0, 1], averaged over samples and variables
    """
    irf = np.asarray(irf, dtype=np.float64)

    # Handle single IRF
    if irf.ndim == 2:
        irf = irf[np.newaxis, ...]

    n_samples, H_plus_1, n_obs = irf.shape

    # Compute FFT for each sample and variable
    fft_coeffs = np.fft.rfft(irf, axis=1)  # (n_samples, freqs, n_obs)
    power_spectrum = np.abs(fft_coeffs) ** 2

    # Frequency grid (cycles per H+1 samples)
    freqs = np.fft.rfftfreq(H_plus_1)

    # Cutoff frequency (cycles per sample)
    cutoff_freq = 1.0 / cutoff_period

    # Identify high-frequency bins
    is_high_freq = freqs >= cutoff_freq

    # Total and high-frequency power
    total_power = np.sum(power_spectrum, axis=1)  # (n_samples, n_obs)
    hf_power = np.sum(power_spectrum[:, is_high_freq, :], axis=1)  # (n_samples, n_obs)

    # Avoid division by zero
    total_power = np.where(total_power < 1e-12, 1.0, total_power)

    # HF ratio per sample and variable
    hf_ratio_per_sample_var = hf_power / total_power

    # Average over samples
    hf_ratio_per_var = np.mean(hf_ratio_per_sample_var, axis=0)  # (n_obs,)

    if per_variable:
        return hf_ratio_per_var
    else:
        return float(np.mean(hf_ratio_per_var))


def overshoot_ratio(
    irf: npt.NDArray[np.float64],
    epsilon: float = 1e-8,
    per_variable: bool = False,
) -> float | npt.NDArray[np.float64]:
    """Overshoot ratio (spec §7.1.2).

    Measures peak-to-impact ratio.

    Formula:
        overshoot = max_h |y[h]| / (|y[0]| + ε)

    Args:
        irf: IRF array, shape (n_samples, H+1, n_obs) or (H+1, n_obs)
        epsilon: Small constant to avoid division by zero (default: 1e-8)
        per_variable: If True, return per-variable ratio, shape (n_obs,)

    Returns:
        Overshoot ratio, averaged over samples and variables
    """
    irf = np.asarray(irf, dtype=np.float64)

    # Handle single IRF
    if irf.ndim == 2:
        irf = irf[np.newaxis, ...]

    n_samples, H_plus_1, n_obs = irf.shape

    # Peak absolute value over all horizons: (n_samples, n_obs)
    peak_abs = np.max(np.abs(irf), axis=1)

    # Impact absolute value: (n_samples, n_obs)
    impact_abs = np.abs(irf[:, 0, :]) + epsilon

    # Overshoot ratio per sample and variable
    overshoot_per_sample_var = peak_abs / impact_abs

    # Average over samples
    overshoot_per_var = np.mean(overshoot_per_sample_var, axis=0)  # (n_obs,)

    if per_variable:
        return overshoot_per_var
    else:
        return float(np.mean(overshoot_per_var))


def sign_flip_count(
    irf: npt.NDArray[np.float64],
    per_variable: bool = False,
) -> float | npt.NDArray[np.float64]:
    """Count sign flips in IRF (spec §7.1.2).

    Counts oscillations by detecting sign changes in the first difference.

    Formula:
        sign_flips = Σ_{h>1} 1[sign(Δy[h]) ≠ sign(Δy[h-1])]

    Args:
        irf: IRF array, shape (n_samples, H+1, n_obs) or (H+1, n_obs)
        per_variable: If True, return per-variable count, shape (n_obs,)

    Returns:
        Average sign flip count, over samples and variables
    """
    irf = np.asarray(irf, dtype=np.float64)

    # Handle single IRF
    if irf.ndim == 2:
        irf = irf[np.newaxis, ...]

    # First differences: Δy[h] = y[h] - y[h-1]
    diff = np.diff(irf, axis=1)  # (n_samples, H, n_obs)

    # Sign of differences
    sign_diff = np.sign(diff)

    # Sign changes: sign(Δy[h]) ≠ sign(Δy[h-1])
    sign_changes = sign_diff[:, 1:, :] != sign_diff[:, :-1, :]

    # Count flips per sample and variable: (n_samples, n_obs)
    flip_count_per_sample_var = np.sum(sign_changes, axis=1)

    # Average over samples: (n_obs,)
    flip_count_per_var = np.mean(flip_count_per_sample_var, axis=0)

    if per_variable:
        return flip_count_per_var
    else:
        return float(np.mean(flip_count_per_var))


# ============================================================================
# Utility Functions
# ============================================================================


def compute_sigma_from_data(
    y: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute per-variable standard deviation for NRMSE normalization.

    Args:
        y: IRF array, shape (n_samples, H+1, n_obs) or (H+1, n_obs)

    Returns:
        Standard deviation per variable, shape (n_obs,)
    """
    y = np.asarray(y, dtype=np.float64)

    if y.ndim == 2:
        # Single IRF: std over horizons
        return np.std(y, axis=0)
    else:
        # Batch: std over samples and horizons
        return np.std(y, axis=(0, 1))
