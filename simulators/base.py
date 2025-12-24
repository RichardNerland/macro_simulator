"""
Base classes and protocols for simulator adapters.

This module defines the core interface that all macroeconomic simulators must implement,
along with supporting dataclasses and utility functions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class ParameterManifest:
    """Metadata for simulator parameters.

    Attributes:
        names: Parameter names (e.g., ["beta", "sigma", "phi_pi"])
        units: Physical units (e.g., ["-", "-", "-"] for dimensionless)
        bounds: Parameter bounds, shape (n_params, 2) with [lower, upper] for each
        defaults: Default/calibration values, shape (n_params,)
        priors: Prior specifications (optional), list of dicts with distribution info
    """
    names: list[str]
    units: list[str]
    bounds: npt.NDArray[np.float64]  # Shape: (n_params, 2)
    defaults: npt.NDArray[np.float64]  # Shape: (n_params,)
    priors: list[dict[str, Any]] | None = None


@dataclass
class ShockManifest:
    """Metadata for structural shocks.

    Attributes:
        names: Shock names (e.g., ["monetary", "technology", "cost_push"])
        n_shocks: Number of shock types
        sigma: Standard deviation of each shock, shape (n_shocks,)
        default_size: Default shock size in standard deviation units (typically 1.0)
    """
    names: list[str]
    n_shocks: int
    sigma: npt.NDArray[np.float64]  # Shape: (n_shocks,)
    default_size: float = 1.0


@dataclass
class ObservableManifest:
    """Metadata for observables.

    Attributes:
        canonical_names: Always ["output", "inflation", "rate"]
        extra_names: Simulator-specific observables (up to 10)
        n_canonical: Number of canonical observables (always 3)
        n_extra: Number of extra observables
    """
    canonical_names: list[str]
    extra_names: list[str]
    n_canonical: int = 3
    n_extra: int = 0


@dataclass
class SimulatorOutput:
    """Output from a single simulation run.

    Attributes:
        y_canonical: Canonical observables, shape (T, 3) - output, inflation, rate
        y_extra: Simulator-specific observables, shape (T, n_extra) or None
        x_state: Internal state trajectory, shape (T, n_state) (optional)
        regime_seq: Regime indicators for regime-switching models, shape (T,) (optional)
        metadata: Additional simulator-specific metadata (e.g., binding_fraction for ZLB)
    """
    y_canonical: npt.NDArray[np.float64]  # Shape: (T, 3)
    y_extra: npt.NDArray[np.float64] | None = None  # Shape: (T, n_extra)
    x_state: npt.NDArray[np.float64] | None = None  # Shape: (T, n_state)
    regime_seq: npt.NDArray[np.int32] | None = None  # Shape: (T,)
    metadata: dict[str, Any] | None = None  # Simulator-specific metadata


class SimulatorAdapter(ABC):
    """Base class for all simulator adapters.

    All simulators must implement this interface to ensure compatibility with
    the dataset generation and emulator training pipeline.
    """

    @property
    @abstractmethod
    def world_id(self) -> str:
        """Unique identifier for this simulator (e.g., 'lss', 'var', 'nk')."""
        pass

    @property
    @abstractmethod
    def param_manifest(self) -> ParameterManifest:
        """Parameter metadata."""
        pass

    @property
    @abstractmethod
    def shock_manifest(self) -> ShockManifest:
        """Shock metadata."""
        pass

    @property
    @abstractmethod
    def obs_manifest(self) -> ObservableManifest:
        """Observable metadata."""
        pass

    @abstractmethod
    def sample_parameters(self, rng: np.random.Generator) -> npt.NDArray[np.float64]:
        """Sample valid parameters from prior.

        Parameters must satisfy bounds and stability/determinacy constraints.

        Args:
            rng: Numpy random number generator for reproducibility

        Returns:
            Parameter vector, shape (n_params,)
        """
        pass

    @abstractmethod
    def simulate(
        self,
        theta: npt.NDArray[np.float64],
        eps: npt.NDArray[np.float64],
        T: int,
        x0: npt.NDArray[np.float64] | None = None,
    ) -> SimulatorOutput:
        """Run simulation for T periods.

        Args:
            theta: Parameters, shape (n_params,)
            eps: Shock sequence, shape (T, n_shocks)
            T: Number of time steps
            x0: Initial state (optional), shape (n_state,)

        Returns:
            SimulatorOutput with trajectories
        """
        pass

    @abstractmethod
    def compute_irf(
        self,
        theta: npt.NDArray[np.float64],
        shock_idx: int,
        shock_size: float,
        H: int,
        x0: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute impulse response function via baseline subtraction.

        IRF[h] = y_shocked[h] - y_baseline[h]

        where:
        - y_baseline: simulation with eps=0 for all t >= 0
        - y_shocked: simulation with shock at t=0, eps=0 for t > 0

        Args:
            theta: Parameters, shape (n_params,)
            shock_idx: Index of shock to perturb (0 to n_shocks-1)
            shock_size: Size of shock in standard deviation units
            H: Horizon length (IRF computed for h=0..H, so H+1 points)
            x0: Initial state (optional), shape (n_state,). If None, use steady state

        Returns:
            IRF array, shape (H+1, 3) for canonical observables
        """
        pass

    def validate_parameters(self, theta: npt.NDArray[np.float64]) -> bool:
        """Check parameter validity (bounds check).

        Args:
            theta: Parameters to validate

        Returns:
            True if parameters are within bounds, False otherwise
        """
        bounds = self.param_manifest.bounds
        return bool(np.all(theta >= bounds[:, 0]) and np.all(theta <= bounds[:, 1]))

    def get_analytic_irf(
        self,
        theta: npt.NDArray[np.float64],
        shock_idx: int,
        shock_size: float,
        H: int,
    ) -> npt.NDArray[np.float64] | None:
        """Return closed-form IRF if available (for oracle baseline).

        Args:
            theta: Parameters
            shock_idx: Index of shock to perturb
            shock_size: Size of shock in std devs
            H: Horizon length

        Returns:
            IRF array shape (H+1, 3), or None if no closed form available
        """
        return None


# Parameter normalization utilities

def normalize_param(
    x: float,
    default: float,
    bounds: tuple[float, float],
) -> float:
    """Normalize parameter to approximately [-3, 3] range using z-score.

    Uses bounds-derived scale: scale = (upper - lower) / 6, where 6 sigma
    covers most of the [lower, upper] range.

    Args:
        x: Parameter value to normalize
        default: Default/center value for normalization
        bounds: (lower, upper) bounds for the parameter

    Returns:
        Normalized value, clipped to [-5, 5]
    """
    scale = (bounds[1] - bounds[0]) / 6.0  # ~6 sigma covers most of range
    z = (x - default) / scale
    return float(np.clip(z, -5.0, 5.0))


def normalize_bounded(x: float, lower: float, upper: float) -> float:
    """Normalize strictly bounded parameter via logit transform.

    For parameters like probabilities that must stay in (lower, upper),
    this applies a logit transform after mapping to (0, 1).

    Args:
        x: Parameter value to normalize
        lower: Lower bound
        upper: Upper bound

    Returns:
        Logit-transformed value
    """
    # Map to (0, 1)
    p = (x - lower) / (upper - lower)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    # Logit transform
    return float(np.log(p / (1 - p)))


def check_bounds(theta: npt.NDArray[np.float64], bounds: npt.NDArray[np.float64]) -> bool:
    """Check if parameters are within bounds.

    Args:
        theta: Parameters to check, shape (n_params,)
        bounds: Bounds array, shape (n_params, 2)

    Returns:
        True if all parameters within bounds
    """
    return bool(np.all(theta >= bounds[:, 0]) and np.all(theta <= bounds[:, 1]))


def check_spectral_radius(A: npt.NDArray[np.float64], max_radius: float = 1.0) -> bool:
    """Check if matrix has spectral radius below threshold (stability check).

    Args:
        A: Square matrix to check
        max_radius: Maximum allowed spectral radius (default 1.0 for stability)

    Returns:
        True if spectral radius < max_radius
    """
    eigenvalues = np.linalg.eigvals(A)
    spectral_radius = np.max(np.abs(eigenvalues))
    return bool(spectral_radius < max_radius)
