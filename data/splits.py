"""Split construction algorithms for dataset generation.

This module implements train/val/test split construction per spec §4.6.
All split algorithms are deterministic given the seed.

Split types:
1. Interpolation (random): Standard random train/val/test
2. Extrapolation-slice: Hold out samples based on parameter predicates
3. Extrapolation-corner: Hold out samples at joint extremes (high persistence + high volatility)
"""

import hashlib
import json
from collections.abc import Callable

import numpy as np

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


# Type alias for predicate functions
PredicateFn = Callable[[np.ndarray, ParameterManifest], bool]


def split_interpolation(n_samples: int, seed: int) -> dict[str, np.ndarray]:
    """Standard random split for interpolation evaluation.

    Per spec §4.6.1: 80% train, 10% val, 5% test_interpolation.
    Remaining 5% is reserved for extrapolation splits.

    Args:
        n_samples: Total number of samples
        seed: Random seed for reproducibility

    Returns:
        Dictionary with keys "train", "val", "test_interpolation" mapping to index arrays
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    # Split fractions: 80% train, 10% val, 5% test, 5% reserved
    train_end = int(0.80 * n_samples)
    val_end = int(0.90 * n_samples)
    test_end = int(0.95 * n_samples)

    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test_interpolation": indices[val_end:test_end],
    }


def split_extrapolation_slice(
    theta: np.ndarray,
    param_manifest: ParameterManifest,
    world_id: str,
    seed: int,
) -> np.ndarray:
    """Hold out samples where specific parameters exceed thresholds.

    Per spec §4.6.2: Use per-simulator predicates to identify economically
    meaningful parameter regions (e.g., NK: φ_π > 2.0, high policy response).

    Args:
        theta: Parameter array, shape (n_samples, n_params)
        param_manifest: Parameter metadata with names and bounds
        world_id: Simulator identifier (e.g., "nk", "var", "lss")
        seed: Random seed for subsampling if more slice samples than target

    Returns:
        Indices for slice holdout (2.5% of total, or all slice samples if fewer)
    """
    # Get predicate for this world
    if world_id not in SLICE_PREDICATES:
        # No slice predicate defined for this world, return empty array
        return np.array([], dtype=np.int64)

    predicate = SLICE_PREDICATES[world_id]

    # Apply predicate to each sample
    slice_mask = np.array([predicate(t, param_manifest) for t in theta])
    slice_indices = np.where(slice_mask)[0]

    # Take 2.5% of total, or all slice samples if fewer
    rng = np.random.default_rng(seed)
    target_n = int(0.025 * len(theta))

    if len(slice_indices) > target_n:
        # Subsample randomly from slice
        slice_indices = rng.choice(slice_indices, target_n, replace=False)

    return slice_indices


def split_extrapolation_corner(
    theta: np.ndarray,
    param_manifest: ParameterManifest,
    world_id: str,
    seed: int,
    quantile: float = 0.9,
) -> np.ndarray:
    """Hold out samples at joint extremes of multiple summary statistics.

    Per spec §4.6.3: Identify samples that are high on multiple axes
    (e.g., high persistence AND high volatility).

    Summary statistics:
    1. persistence: Maximum AR coefficient or eigenvalue magnitude
    2. volatility: Shock standard deviation (or max across shocks)

    Args:
        theta: Parameter array, shape (n_samples, n_params)
        param_manifest: Parameter metadata
        world_id: Simulator identifier
        seed: Random seed for subsampling
        quantile: Threshold for "high" (default 0.9 = top 10%)

    Returns:
        Indices for corner holdout (2.5% of total, or all corner samples if fewer)
    """
    # Compute summary statistics
    stats = compute_summary_stats(theta, param_manifest, world_id)

    if stats is None or len(stats) == 0:
        # No summary statistics defined for this world
        return np.array([], dtype=np.int64)

    # Identify corner samples: high on multiple axes
    persistence = stats.get("persistence")
    volatility = stats.get("volatility")

    if persistence is None or volatility is None:
        # Need both persistence and volatility for corner
        return np.array([], dtype=np.int64)

    # Compute thresholds
    persistence_threshold = np.quantile(persistence, quantile)
    volatility_threshold = np.quantile(volatility, quantile)

    # Corner = high on both axes
    corner_mask = (persistence > persistence_threshold) & (volatility > volatility_threshold)
    corner_indices = np.where(corner_mask)[0]

    # Take 2.5% of total, or all corner samples if fewer
    rng = np.random.default_rng(seed)
    target_n = int(0.025 * len(theta))

    if len(corner_indices) > target_n:
        corner_indices = rng.choice(corner_indices, target_n, replace=False)

    return corner_indices


def split_lowo(
    world_ids: list[str],
    held_out_world: str,
) -> dict[str, list[str]]:
    """Leave-One-World-Out (LOWO) split for cross-world generalization.

    Per spec §7.3: LOWO evaluation tests generalization by training on all
    worlds except one, and testing on the held-out world.

    Args:
        world_ids: List of all available world identifiers
        held_out_world: World to hold out for testing

    Returns:
        Dictionary with keys:
        - "train_worlds": List of world_ids for training (all except held_out_world)
        - "test_worlds": List containing only the held_out_world

    Raises:
        ValueError: If held_out_world not in world_ids

    Example:
        >>> split_lowo(["lss", "var", "nk", "rbc", "switching", "zlb"], "nk")
        {
            "train_worlds": ["lss", "var", "rbc", "switching", "zlb"],
            "test_worlds": ["nk"]
        }
    """
    if held_out_world not in world_ids:
        raise ValueError(
            f"held_out_world '{held_out_world}' not in world_ids {world_ids}"
        )

    train_worlds = [w for w in world_ids if w != held_out_world]
    test_worlds = [held_out_world]

    return {
        "train_worlds": train_worlds,
        "test_worlds": test_worlds,
    }


def get_lowo_world_lists(all_worlds: list[str], held_out_world: str) -> tuple[list[str], list[str]]:
    """Convenience function to get train and test world lists for LOWO.

    This is a simplified interface for the common use case of getting
    just the train and test world lists.

    Args:
        all_worlds: List of all available world identifiers
        held_out_world: World to hold out for testing

    Returns:
        Tuple of (train_worlds, test_worlds)

    Example:
        >>> get_lowo_world_lists(["lss", "var", "nk", "rbc"], "nk")
        (["lss", "var", "rbc"], ["nk"])
    """
    split = split_lowo(all_worlds, held_out_world)
    return split["train_worlds"], split["test_worlds"]


def construct_all_splits(
    theta: np.ndarray,
    param_manifest: ParameterManifest,
    world_id: str,
    seed: int,
) -> dict[str, np.ndarray]:
    """Construct all train/val/test splits ensuring disjointness.

    Per spec §4.6.4: Extrapolation splits are constructed first, then removed
    from the pool before random splitting.

    Args:
        theta: Parameter array, shape (n_samples, n_params)
        param_manifest: Parameter metadata
        world_id: Simulator identifier
        seed: Global seed for reproducibility

    Returns:
        Dictionary mapping split names to index arrays:
        - "train": Training set (most of data)
        - "val": Validation set
        - "test_interpolation": Random holdout
        - "test_extrapolation_slice": Parameter slice holdout
        - "test_extrapolation_corner": Joint extremes holdout
    """
    n_samples = len(theta)

    # 1. Identify extrapolation samples first
    slice_idx = split_extrapolation_slice(theta, param_manifest, world_id, seed)
    corner_idx = split_extrapolation_corner(theta, param_manifest, world_id, seed)

    # 1b. Ensure disjoint: remove overlap from corner (keep slice as-is)
    # This handles cases where predicates and summary stats use the same params
    # (e.g., RBC: rho_a > 0.95 is both slice predicate AND persistence stat)
    overlap = np.intersect1d(slice_idx, corner_idx)
    corner_idx = np.setdiff1d(corner_idx, overlap)

    # 2. Ensure disjoint extrapolation sets
    extrap_idx = np.union1d(slice_idx, corner_idx)

    # 3. Remove extrapolation samples from pool
    pool_idx = np.setdiff1d(np.arange(n_samples), extrap_idx)

    # 4. Random split on remaining pool
    # We need to adjust fractions since we removed some samples
    # Original: 80% train, 10% val, 5% test from full dataset
    # After removing 5% (extrap), we split remaining 95% as: 80/95, 10/95, 5/95
    n_pool = len(pool_idx)
    rng = np.random.default_rng(seed)
    pool_permuted = rng.permutation(pool_idx)

    # Calculate split points on the pool
    # We want final fractions relative to total: 80%, 10%, 5%
    train_end = int(0.80 * n_samples)
    val_end = int(0.90 * n_samples)

    # Adjust for actual pool size
    train_end = min(train_end, n_pool)
    val_end = min(val_end, train_end + (n_pool - train_end) // 2)

    return {
        "train": pool_permuted[:train_end],
        "val": pool_permuted[train_end:val_end],
        "test_interpolation": pool_permuted[val_end:],
        "test_extrapolation_slice": slice_idx,
        "test_extrapolation_corner": corner_idx,
    }


def validate_splits(splits: dict[str, np.ndarray], n_samples: int) -> None:
    """Validate that splits are disjoint and cover all samples.

    Args:
        splits: Dictionary of split name to index arrays
        n_samples: Total number of samples

    Raises:
        ValueError: If splits overlap or don't cover all samples
    """
    # Check disjointness
    all_splits = ["train", "val", "test_interpolation", "test_extrapolation_slice", "test_extrapolation_corner"]

    for i, split_i in enumerate(all_splits):
        if split_i not in splits:
            continue

        for split_j in all_splits[i+1:]:
            if split_j not in splits:
                continue

            overlap = np.intersect1d(splits[split_i], splits[split_j])
            if len(overlap) > 0:
                raise ValueError(f"Splits '{split_i}' and '{split_j}' overlap with {len(overlap)} samples")

    # Check coverage
    all_indices = np.concatenate([splits[s] for s in all_splits if s in splits])
    unique_indices = np.unique(all_indices)

    if len(unique_indices) != n_samples:
        raise ValueError(f"Splits cover {len(unique_indices)} samples, expected {n_samples}")

    if not np.array_equal(unique_indices, np.arange(n_samples)):
        missing = np.setdiff1d(np.arange(n_samples), unique_indices)
        raise ValueError(f"Splits missing {len(missing)} samples: {missing[:10]}...")


def compute_split_algorithm_hash(split_config: dict) -> str:
    """Compute hash of split algorithm for manifest.

    This allows detecting if split logic has changed between dataset versions.

    Args:
        split_config: Dictionary with split algorithm details

    Returns:
        SHA256 hash (first 16 characters)
    """
    config_str = json.dumps(split_config, sort_keys=True)
    hash_obj = hashlib.sha256(config_str.encode())
    return hash_obj.hexdigest()[:16]


# ============================================================================
# Per-Simulator Slice Predicates (spec §4.6.2)
# ============================================================================

def nk_slice_predicate(theta: np.ndarray, manifest: ParameterManifest) -> bool:
    """NK slice: φ_π > 2.0 (aggressive policy response)."""
    try:
        phi_pi_idx = manifest.names.index("phi_pi")
        return theta[phi_pi_idx] > 2.0
    except (ValueError, IndexError):
        return False


def rbc_slice_predicate(theta: np.ndarray, manifest: ParameterManifest) -> bool:
    """RBC slice: ρ_a > 0.95 (high persistence)."""
    try:
        rho_a_idx = manifest.names.index("rho_a")
        return theta[rho_a_idx] > 0.95
    except (ValueError, IndexError):
        return False


def var_slice_predicate(theta: np.ndarray, manifest: ParameterManifest) -> bool:
    """VAR slice: max_eigenvalue(A) > 0.95 (near unit root).

    Note: This requires computing eigenvalues from the VAR coefficients.
    We approximate by checking if any AR coefficient is > 0.95.
    """
    try:
        # For VAR, we look for high persistence in any coefficient
        # This is a simplification; full implementation would reconstruct companion matrix
        # For now, check if any parameter labeled with "rho" or "persistence" is > 0.95
        for i, name in enumerate(manifest.names):
            if "rho" in name.lower() or "persistence" in name.lower():
                if theta[i] > 0.95:
                    return True

        # Alternative: check if any coefficient is > 0.95
        return np.any(theta > 0.95)
    except (ValueError, IndexError):
        return False


def lss_slice_predicate(theta: np.ndarray, manifest: ParameterManifest) -> bool:
    """LSS slice: max_eigenvalue(A) > 0.95 (near unit root).

    Similar to VAR, we use a heuristic based on parameter values.
    """
    try:
        # Check for high persistence parameters
        for i, name in enumerate(manifest.names):
            if "rho" in name.lower() or "persistence" in name.lower() or "eigenvalue" in name.lower():
                if theta[i] > 0.95:
                    return True

        # Alternative heuristic
        return np.any(theta > 0.95)
    except (ValueError, IndexError):
        return False


def switching_slice_predicate(theta: np.ndarray, manifest: ParameterManifest) -> bool:
    """Switching slice: p_stay > 0.95 (sticky regimes, either regime)."""
    try:
        # Look for transition probabilities
        for i, name in enumerate(manifest.names):
            if "p_stay" in name.lower() or "p_00" in name or "p_11" in name:
                if theta[i] > 0.95:
                    return True
        return False
    except (ValueError, IndexError):
        return False


def zlb_slice_predicate(theta: np.ndarray, manifest: ParameterManifest) -> bool:
    """ZLB slice: r_ss < 1.0 (annualized, low steady-state rate)."""
    try:
        r_ss_idx = manifest.names.index("r_ss")
        return theta[r_ss_idx] < 1.0
    except (ValueError, IndexError):
        return False


# Registry of slice predicates per world_id
SLICE_PREDICATES: dict[str, PredicateFn] = {
    "nk": nk_slice_predicate,
    "rbc": rbc_slice_predicate,
    "var": var_slice_predicate,
    "lss": lss_slice_predicate,
    "switching": switching_slice_predicate,
    "zlb": zlb_slice_predicate,
}


# ============================================================================
# Summary Statistics for Corner Split (spec §4.6.3)
# ============================================================================

def compute_summary_stats(
    theta: np.ndarray,
    manifest: ParameterManifest,
    world_id: str,
) -> dict[str, np.ndarray] | None:
    """Compute summary statistics for corner split.

    Computes:
    1. persistence: Maximum AR coefficient or eigenvalue magnitude
    2. volatility: Shock standard deviation (or max across shocks)
    3. policy_strength: Policy response coefficient (where applicable)

    Args:
        theta: Parameter array, shape (n_samples, n_params)
        manifest: Parameter metadata
        world_id: Simulator identifier

    Returns:
        Dictionary with statistics arrays, or None if not applicable
    """
    if world_id not in SUMMARY_STAT_FUNCTIONS:
        return None

    return SUMMARY_STAT_FUNCTIONS[world_id](theta, manifest)


def nk_summary_stats(theta: np.ndarray, manifest: ParameterManifest) -> dict[str, np.ndarray]:
    """NK summary statistics: persistence (ρ_a, ρ_i), volatility (σ), policy (φ_π)."""
    n_samples = theta.shape[0]

    # Persistence: max of shock persistence parameters
    persistence = np.zeros(n_samples)
    for name in ["rho_a", "rho_m", "rho_u", "rho_i"]:
        if name in manifest.names:
            idx = manifest.names.index(name)
            persistence = np.maximum(persistence, theta[:, idx])

    # Volatility: max of shock std devs
    volatility = np.zeros(n_samples)
    for name in ["sigma_a", "sigma_m", "sigma_u"]:
        if name in manifest.names:
            idx = manifest.names.index(name)
            volatility = np.maximum(volatility, theta[:, idx])

    # Policy strength
    policy_strength = np.zeros(n_samples)
    if "phi_pi" in manifest.names:
        idx = manifest.names.index("phi_pi")
        policy_strength = theta[:, idx]

    return {
        "persistence": persistence,
        "volatility": volatility,
        "policy_strength": policy_strength,
    }


def rbc_summary_stats(theta: np.ndarray, manifest: ParameterManifest) -> dict[str, np.ndarray]:
    """RBC summary statistics: persistence (ρ_a), volatility (σ_a)."""
    n_samples = theta.shape[0]

    # Persistence
    persistence = np.zeros(n_samples)
    if "rho_a" in manifest.names:
        idx = manifest.names.index("rho_a")
        persistence = theta[:, idx]

    # Volatility
    volatility = np.zeros(n_samples)
    if "sigma_a" in manifest.names:
        idx = manifest.names.index("sigma_a")
        volatility = theta[:, idx]

    return {
        "persistence": persistence,
        "volatility": volatility,
    }


def var_summary_stats(theta: np.ndarray, manifest: ParameterManifest) -> dict[str, np.ndarray]:
    """VAR summary statistics: max coefficient (proxy for persistence), shock std dev."""
    # Persistence: max AR coefficient (simplified)
    persistence = np.max(np.abs(theta), axis=1)

    # Volatility: heuristic based on parameter scale
    volatility = np.std(theta, axis=1)

    return {
        "persistence": persistence,
        "volatility": volatility,
    }


def lss_summary_stats(theta: np.ndarray, manifest: ParameterManifest) -> dict[str, np.ndarray]:
    """LSS summary statistics: max eigenvalue (persistence), process noise (volatility)."""
    # Similar heuristics to VAR
    persistence = np.max(np.abs(theta), axis=1)
    volatility = np.std(theta, axis=1)

    return {
        "persistence": persistence,
        "volatility": volatility,
    }


def switching_summary_stats(theta: np.ndarray, manifest: ParameterManifest) -> dict[str, np.ndarray]:
    """Switching summary statistics: regime persistence, regime volatility."""
    n_samples = theta.shape[0]

    # Persistence: transition probabilities
    persistence = np.zeros(n_samples)
    for name in ["p_00", "p_11", "p_stay"]:
        if name in manifest.names:
            idx = manifest.names.index(name)
            persistence = np.maximum(persistence, theta[:, idx])

    # Volatility: max regime volatility
    volatility = np.zeros(n_samples)
    for name in ["sigma_0", "sigma_1", "sigma"]:
        if name in manifest.names:
            idx = manifest.names.index(name)
            volatility = np.maximum(volatility, theta[:, idx])

    return {
        "persistence": persistence,
        "volatility": volatility,
    }


def zlb_summary_stats(theta: np.ndarray, manifest: ParameterManifest) -> dict[str, np.ndarray]:
    """ZLB summary statistics: same as NK plus steady-state rate."""
    # Reuse NK stats
    stats = nk_summary_stats(theta, manifest)

    # Add steady-state rate info
    if "r_ss" in manifest.names:
        idx = manifest.names.index("r_ss")
        stats["r_ss"] = theta[:, idx]

    return stats


# Registry of summary stat functions per world_id
SUMMARY_STAT_FUNCTIONS = {
    "nk": nk_summary_stats,
    "rbc": rbc_summary_stats,
    "var": var_summary_stats,
    "lss": lss_summary_stats,
    "switching": switching_summary_stats,
    "zlb": zlb_summary_stats,
}
