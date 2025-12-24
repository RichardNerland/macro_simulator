"""
Vector Autoregression (VAR) simulator.

Implements a VAR(p) model:
    y[t] = c + A_1 @ y[t-1] + A_2 @ y[t-2] + ... + A_p @ y[t-p] + eps[t]

where the companion matrix eigenvalues satisfy the stationarity constraint.
"""

import numpy as np
import numpy.typing as npt
from scipy.linalg import companion

from .base import (
    ObservableManifest,
    ParameterManifest,
    ShockManifest,
    SimulatorAdapter,
    SimulatorOutput,
    check_spectral_radius,
)


class VARSimulator(SimulatorAdapter):
    """Vector Autoregression simulator.

    The model has:
    - n_vars endogenous variables (we use 3 for canonical observables)
    - lag order p
    - n_vars exogenous shocks

    Parameters (stored in theta):
    - c: constant term (n_vars,)
    - A_1, ..., A_p: coefficient matrices (n_vars x n_vars each)
    - Sigma: innovation covariance (stored as Cholesky factor for positive definiteness)
    """

    def __init__(
        self,
        n_vars: int = 3,
        lag_order: int = 2,
        rho_max: float = 0.95,
        seed: int | None = None,
    ):
        """Initialize VAR simulator.

        Args:
            n_vars: Number of endogenous variables (default 3 for canonical observables)
            lag_order: VAR lag order p (default 2)
            rho_max: Maximum eigenvalue magnitude for stationarity (default 0.95)
            seed: Random seed for reproducibility (optional)
        """
        self.n_vars = n_vars
        self.lag_order = lag_order
        self.rho_max = rho_max
        self._seed = seed

        # Calculate number of parameters
        # c: n_vars, A_1...A_p: p * n_vars^2, Sigma (Cholesky): n_vars*(n_vars+1)/2
        self.n_params = (
            n_vars
            + lag_order * n_vars * n_vars
            + n_vars * (n_vars + 1) // 2  # Lower triangular Cholesky
        )

    @property
    def world_id(self) -> str:
        return "var"

    @property
    def param_manifest(self) -> ParameterManifest:
        """Parameter manifest for VAR simulator."""
        names = []
        units = []
        bounds_list = []
        defaults_list = []

        # Constant term
        for i in range(self.n_vars):
            names.append(f"c_{i}")
            units.append("-")
            bounds_list.append([-1.0, 1.0])
            defaults_list.append(0.0)

        # Coefficient matrices A_1, ..., A_p
        for lag in range(1, self.lag_order + 1):
            for i in range(self.n_vars):
                for j in range(self.n_vars):
                    names.append(f"A{lag}_{i}{j}")
                    units.append("-")
                    bounds_list.append([-2.0, 2.0])
                    # Default: A_1 is diagonal with 0.5, others are zero
                    if lag == 1 and i == j:
                        defaults_list.append(0.5)
                    else:
                        defaults_list.append(0.0)

        # Cholesky factor of covariance (lower triangular)
        for i in range(self.n_vars):
            for j in range(i + 1):  # Lower triangular
                names.append(f"L_{i}{j}")
                units.append("-")
                if i == j:
                    bounds_list.append([0.001, 0.5])  # Diagonal elements positive
                    defaults_list.append(0.01)
                else:
                    bounds_list.append([-0.5, 0.5])
                    defaults_list.append(0.0)

        bounds = np.array(bounds_list, dtype=np.float64)
        defaults = np.array(defaults_list, dtype=np.float64)

        return ParameterManifest(
            names=names,
            units=units,
            bounds=bounds,
            defaults=defaults,
        )

    @property
    def shock_manifest(self) -> ShockManifest:
        """Shock manifest for VAR simulator."""
        names = [f"shock_{i}" for i in range(self.n_vars)]
        sigma = np.ones(self.n_vars, dtype=np.float64) * 0.01  # Small shocks

        return ShockManifest(
            names=names,
            n_shocks=self.n_vars,
            sigma=sigma,
            default_size=1.0,
        )

    @property
    def obs_manifest(self) -> ObservableManifest:
        """Observable manifest for VAR simulator."""
        return ObservableManifest(
            canonical_names=["output", "inflation", "rate"],
            extra_names=[],
            n_canonical=3,
            n_extra=0,
        )

    def _unpack_theta(
        self, theta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Unpack parameter vector into c, A_matrices, Sigma.

        Args:
            theta: Parameter vector

        Returns:
            Tuple of (c, [A_1, ..., A_p], Sigma)
        """
        idx = 0

        # Extract constant
        c = theta[idx : idx + self.n_vars]
        idx += self.n_vars

        # Extract coefficient matrices
        A_matrices = []
        for _ in range(self.lag_order):
            A_size = self.n_vars * self.n_vars
            A = theta[idx : idx + A_size].reshape(self.n_vars, self.n_vars)
            A_matrices.append(A)
            idx += A_size

        # Extract Cholesky factor and reconstruct Sigma
        L_size = self.n_vars * (self.n_vars + 1) // 2
        L_vec = theta[idx : idx + L_size]

        # Reconstruct lower triangular matrix
        L = np.zeros((self.n_vars, self.n_vars), dtype=np.float64)
        idx_L = 0
        for i in range(self.n_vars):
            for j in range(i + 1):
                L[i, j] = L_vec[idx_L]
                idx_L += 1

        # Sigma = L @ L.T
        Sigma = L @ L.T

        return c, A_matrices, Sigma

    def _pack_theta(
        self,
        c: npt.NDArray[np.float64],
        A_matrices: list[npt.NDArray[np.float64]],
        Sigma: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Pack c, A_matrices, Sigma into parameter vector.

        Args:
            c: Constant term (n_vars,)
            A_matrices: List of coefficient matrices
            Sigma: Innovation covariance

        Returns:
            Parameter vector
        """
        # Pack c and A matrices
        parts = [c]
        for A in A_matrices:
            parts.append(A.ravel())

        # Compute Cholesky factor of Sigma
        L = np.linalg.cholesky(Sigma)

        # Pack lower triangular elements
        L_vec = []
        for i in range(self.n_vars):
            for j in range(i + 1):
                L_vec.append(L[i, j])

        parts.append(np.array(L_vec))

        return np.concatenate(parts)

    def _build_companion_matrix(
        self, A_matrices: list[npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]:
        """Build companion matrix for VAR(p).

        Args:
            A_matrices: List of coefficient matrices [A_1, ..., A_p]

        Returns:
            Companion matrix of size (n_vars*p, n_vars*p)
        """
        n = self.n_vars
        p = self.lag_order

        # Companion form:
        # [A_1  A_2  ...  A_p]
        # [I    0    ...   0 ]
        # [0    I    ...   0 ]
        # ...
        # [0    0    ...   I  0]

        F = np.zeros((n * p, n * p), dtype=np.float64)

        # First row: [A_1, A_2, ..., A_p]
        for i, A in enumerate(A_matrices):
            F[0:n, i * n : (i + 1) * n] = A

        # Lower blocks: identity matrices
        for i in range(1, p):
            F[i * n : (i + 1) * n, (i - 1) * n : i * n] = np.eye(n)

        return F

    def sample_parameters(self, rng: np.random.Generator) -> npt.NDArray[np.float64]:
        """Sample stationary VAR parameters.

        Uses eigenvalue placement to ensure stationarity.

        Args:
            rng: Random number generator

        Returns:
            Parameter vector with stationary VAR
        """
        max_attempts = 100

        for attempt in range(max_attempts):
            # Sample companion matrix eigenvalues
            n_state = self.n_vars * self.lag_order
            n_complex_pairs = n_state // 2
            n_real = n_state % 2

            eigenvalues = []

            # Complex conjugate pairs
            for _ in range(n_complex_pairs):
                radius = rng.uniform(0.3, self.rho_max)
                angle = rng.uniform(0, 2 * np.pi)
                eigenvalues.append(radius * np.exp(1j * angle))
                eigenvalues.append(radius * np.exp(-1j * angle))

            # Real eigenvalues
            for _ in range(n_real):
                eigenvalues.append(rng.uniform(-self.rho_max, self.rho_max))

            eigenvalues = np.array(eigenvalues[:n_state])

            # Construct companion matrix with desired eigenvalues
            V = rng.standard_normal((n_state, n_state))
            V, _ = np.linalg.qr(V)
            F_target = (V @ np.diag(eigenvalues) @ V.T).real.astype(np.float64)

            # Extract A_1, ..., A_p from first block row of companion matrix
            A_matrices = []
            for i in range(self.lag_order):
                A = F_target[0 : self.n_vars, i * self.n_vars : (i + 1) * self.n_vars]
                A_matrices.append(A)

            # Sample constant (small)
            c = rng.uniform(-0.1, 0.1, size=self.n_vars)

            # Sample innovation covariance
            # Use diagonal for simplicity, with small variances
            diag_elements = rng.uniform(0.001, 0.01, size=self.n_vars)
            Sigma = np.diag(diag_elements**2)

            # Pack parameters
            theta = self._pack_theta(c, A_matrices, Sigma)

            # Verify stationarity
            F_check = self._build_companion_matrix(A_matrices)
            if check_spectral_radius(F_check, self.rho_max) and self.validate_parameters(theta):
                return theta

        raise ValueError(
            f"Failed to generate stationary VAR parameters after {max_attempts} attempts"
        )

    def simulate(
        self,
        theta: npt.NDArray[np.float64],
        eps: npt.NDArray[np.float64],
        T: int,
        x0: npt.NDArray[np.float64] | None = None,
    ) -> SimulatorOutput:
        """Simulate VAR model.

        Args:
            theta: Parameters
            eps: Shock sequence, shape (T, n_vars)
            T: Number of time steps
            x0: Initial state (optional), shape (n_vars * lag_order,). If None, use zeros

        Returns:
            SimulatorOutput with trajectories
        """
        c, A_matrices, Sigma = self._unpack_theta(theta)

        # Cholesky factor for shock scaling
        L = np.linalg.cholesky(Sigma)

        # Initialize history
        if x0 is None:
            # Start from zero
            y_history = np.zeros((self.lag_order, self.n_vars), dtype=np.float64)
        else:
            # Reshape x0 into (lag_order, n_vars)
            y_history = x0.reshape(self.lag_order, self.n_vars).copy()

        # Pre-allocate output
        y_canonical = np.zeros((T, self.n_vars), dtype=np.float64)

        # Simulate
        for t in range(T):
            # Compute y[t] = c + A_1 @ y[t-1] + ... + A_p @ y[t-p] + eps[t]
            y_t = c.copy()

            for lag in range(self.lag_order):
                # y[t-lag-1] is y_history[-(lag+1)]
                y_t += A_matrices[lag] @ y_history[-(lag + 1), :]

            # Add shock (scaled by Cholesky factor)
            y_t += L @ eps[t, :]

            # Store
            y_canonical[t, :] = y_t

            # Update history (shift and append)
            y_history = np.vstack([y_history[1:, :], y_t])

        # For VAR, the canonical observables are just the first 3 variables
        # (or all variables if n_vars == 3)
        return SimulatorOutput(y_canonical=y_canonical[:, :3])

    def compute_irf(
        self,
        theta: npt.NDArray[np.float64],
        shock_idx: int,
        shock_size: float,
        H: int,
        x0: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute IRF via baseline subtraction.

        Args:
            theta: Parameters
            shock_idx: Index of shock to perturb
            shock_size: Size of shock in std devs
            H: Horizon length
            x0: Initial state (optional)

        Returns:
            IRF array, shape (H+1, 3)
        """
        if x0 is None:
            x0 = np.zeros(self.n_vars * self.lag_order, dtype=np.float64)

        # Baseline simulation (no shocks)
        eps_baseline = np.zeros((H + 1, self.n_vars), dtype=np.float64)
        output_baseline = self.simulate(theta, eps_baseline, H + 1, x0)

        # Shocked simulation (shock at t=0 only)
        eps_shocked = np.zeros((H + 1, self.n_vars), dtype=np.float64)
        sigma = self.shock_manifest.sigma
        eps_shocked[0, shock_idx] = shock_size * sigma[shock_idx]
        output_shocked = self.simulate(theta, eps_shocked, H + 1, x0)

        # IRF = difference
        irf = output_shocked.y_canonical - output_baseline.y_canonical

        return irf

    def get_analytic_irf(
        self,
        theta: npt.NDArray[np.float64],
        shock_idx: int,
        shock_size: float,
        H: int,
    ) -> npt.NDArray[np.float64]:
        """Compute analytic IRF using companion matrix.

        The IRF can be computed as powers of the companion matrix.

        Args:
            theta: Parameters
            shock_idx: Index of shock to perturb
            shock_size: Size of shock in std devs
            H: Horizon length

        Returns:
            IRF array, shape (H+1, 3)
        """
        c, A_matrices, Sigma = self._unpack_theta(theta)
        sigma = self.shock_manifest.sigma

        # Build companion matrix
        F = self._build_companion_matrix(A_matrices)

        # Cholesky factor for shock scaling
        L = np.linalg.cholesky(Sigma)

        # Initial shock vector (expanded to companion form)
        e_shock = np.zeros(self.n_vars, dtype=np.float64)
        e_shock[shock_idx] = shock_size * sigma[shock_idx]
        shock_vec = L @ e_shock

        # Expanded shock vector (only first n_vars elements are non-zero)
        shock_vec_expanded = np.zeros(self.n_vars * self.lag_order, dtype=np.float64)
        shock_vec_expanded[: self.n_vars] = shock_vec

        # Compute IRF
        irf = np.zeros((H + 1, 3), dtype=np.float64)
        F_power = np.eye(self.n_vars * self.lag_order, dtype=np.float64)

        for h in range(H + 1):
            # Response at horizon h
            response = F_power @ shock_vec_expanded
            irf[h, :] = response[: min(3, self.n_vars)]  # First 3 variables

            if h < H:
                F_power = F_power @ F

        return irf
