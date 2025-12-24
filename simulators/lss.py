"""
Linear State-Space (LSS) simulator.

Implements a random stable linear state-space model:
    x[t+1] = A @ x[t] + B @ eps[t]
    y[t] = C @ x[t]

where A is stable (spectral radius < rho_max).
"""

import numpy as np
import numpy.typing as npt

from .base import (
    ObservableManifest,
    ParameterManifest,
    ShockManifest,
    SimulatorAdapter,
    SimulatorOutput,
    check_spectral_radius,
)


class LSSSimulator(SimulatorAdapter):
    """Linear State-Space simulator with random stable dynamics.

    The model has:
    - n_state dimensional state vector
    - n_shocks exogenous shocks
    - 3 canonical observables (output, inflation, rate)

    Parameters (stored in theta):
    - A matrix: n_state x n_state (stored row-major as vector)
    - B matrix: n_state x n_shocks (stored row-major as vector)
    - C matrix: 3 x n_state (stored row-major as vector)
    """

    def __init__(
        self,
        n_state: int = 4,
        n_shocks: int = 2,
        rho_max: float = 0.95,
        seed: int | None = None,
    ):
        """Initialize LSS simulator.

        Args:
            n_state: State dimension (default 4)
            n_shocks: Number of shocks (default 2)
            rho_max: Maximum spectral radius for stability (default 0.95)
            seed: Random seed for reproducibility (optional)
        """
        self.n_state = n_state
        self.n_shocks = n_shocks
        self.rho_max = rho_max
        self._seed = seed

        # Calculate number of parameters
        self.n_params = n_state * n_state + n_state * n_shocks + 3 * n_state

    @property
    def world_id(self) -> str:
        return "lss"

    @property
    def param_manifest(self) -> ParameterManifest:
        """Parameter manifest for LSS simulator."""
        names = []
        units = []
        bounds_list = []
        defaults_list = []

        # A matrix parameters (n_state x n_state)
        for i in range(self.n_state):
            for j in range(self.n_state):
                names.append(f"A_{i}{j}")
                units.append("-")
                bounds_list.append([-2.0, 2.0])  # Wide bounds, stability enforced separately
                defaults_list.append(0.5 if i == j else 0.0)  # Default to diagonal

        # B matrix parameters (n_state x n_shocks)
        for i in range(self.n_state):
            for j in range(self.n_shocks):
                names.append(f"B_{i}{j}")
                units.append("-")
                bounds_list.append([-1.0, 1.0])
                defaults_list.append(0.1)

        # C matrix parameters (3 x n_state)
        for i in range(3):
            for j in range(self.n_state):
                names.append(f"C_{i}{j}")
                units.append("-")
                bounds_list.append([-2.0, 2.0])
                defaults_list.append(1.0 if j == i else 0.0)  # Default to identity-like

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
        """Shock manifest for LSS simulator."""
        names = [f"shock_{i}" for i in range(self.n_shocks)]
        sigma = np.ones(self.n_shocks, dtype=np.float64) * 0.01  # Small shocks

        return ShockManifest(
            names=names,
            n_shocks=self.n_shocks,
            sigma=sigma,
            default_size=1.0,
        )

    @property
    def obs_manifest(self) -> ObservableManifest:
        """Observable manifest for LSS simulator."""
        return ObservableManifest(
            canonical_names=["output", "inflation", "rate"],
            extra_names=[],
            n_canonical=3,
            n_extra=0,
        )

    def _unpack_theta(
        self, theta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Unpack parameter vector into A, B, C matrices.

        Args:
            theta: Parameter vector

        Returns:
            Tuple of (A, B, C) matrices
        """
        idx = 0

        # Extract A matrix
        A_size = self.n_state * self.n_state
        A = theta[idx : idx + A_size].reshape(self.n_state, self.n_state)
        idx += A_size

        # Extract B matrix
        B_size = self.n_state * self.n_shocks
        B = theta[idx : idx + B_size].reshape(self.n_state, self.n_shocks)
        idx += B_size

        # Extract C matrix
        C_size = 3 * self.n_state
        C = theta[idx : idx + C_size].reshape(3, self.n_state)

        return A, B, C

    def _pack_theta(
        self,
        A: npt.NDArray[np.float64],
        B: npt.NDArray[np.float64],
        C: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Pack A, B, C matrices into parameter vector.

        Args:
            A: State transition matrix (n_state, n_state)
            B: Shock loading matrix (n_state, n_shocks)
            C: Observation matrix (3, n_state)

        Returns:
            Parameter vector
        """
        return np.concatenate([A.ravel(), B.ravel(), C.ravel()])

    def sample_parameters(self, rng: np.random.Generator) -> npt.NDArray[np.float64]:
        """Sample stable LSS parameters.

        Uses eigenvalue placement to ensure stability.

        Args:
            rng: Random number generator

        Returns:
            Parameter vector with stable A matrix
        """
        # Sample stable A matrix via eigenvalue placement
        # Generate eigenvalues uniformly in disk of radius rho_max
        n_complex_pairs = self.n_state // 2
        n_real = self.n_state % 2

        eigenvalues = []

        # Complex conjugate pairs
        for _ in range(n_complex_pairs):
            radius = rng.uniform(0.3, self.rho_max)  # Avoid very small eigenvalues
            angle = rng.uniform(0, 2 * np.pi)
            eigenvalues.append(radius * np.exp(1j * angle))
            eigenvalues.append(radius * np.exp(-1j * angle))

        # Real eigenvalues
        for _ in range(n_real):
            eigenvalues.append(rng.uniform(-self.rho_max, self.rho_max))

        eigenvalues = np.array(eigenvalues[:self.n_state])

        # Construct A with desired eigenvalues using random orthogonal basis
        V = rng.standard_normal((self.n_state, self.n_state))
        V, _ = np.linalg.qr(V)  # Orthonormalize
        A = (V @ np.diag(eigenvalues) @ V.T).real.astype(np.float64)

        # Sample B matrix
        B = rng.standard_normal((self.n_state, self.n_shocks)) * 0.1

        # Sample C matrix - make it well-scaled for canonical observables
        C = rng.standard_normal((3, self.n_state)) * 0.5

        # Ensure C has reasonable row norms (for output scaling)
        for i in range(3):
            row_norm = np.linalg.norm(C[i, :])
            if row_norm > 0:
                C[i, :] = C[i, :] / row_norm * 0.5

        theta = self._pack_theta(A, B, C)

        # Verify stability
        assert check_spectral_radius(A, self.rho_max), "Failed to generate stable A"
        assert self.validate_parameters(theta), "Generated parameters out of bounds"

        return theta

    def simulate(
        self,
        theta: npt.NDArray[np.float64],
        eps: npt.NDArray[np.float64],
        T: int,
        x0: npt.NDArray[np.float64] | None = None,
    ) -> SimulatorOutput:
        """Simulate LSS model.

        Dynamics: x[t+1] = A @ x[t] + B @ eps[t]
                  y[t] = C @ x[t]

        Args:
            theta: Parameters
            eps: Shock sequence, shape (T, n_shocks)
            T: Number of time steps
            x0: Initial state (optional), defaults to zero

        Returns:
            SimulatorOutput with trajectories
        """
        A, B, C = self._unpack_theta(theta)

        # Initialize state
        if x0 is None:
            x0 = np.zeros(self.n_state, dtype=np.float64)

        # Pre-allocate arrays
        x_state = np.zeros((T, self.n_state), dtype=np.float64)
        y_canonical = np.zeros((T, 3), dtype=np.float64)

        # Simulate
        x_t = x0.copy()
        for t in range(T):
            # Observation
            y_canonical[t, :] = C @ x_t

            # Store state
            x_state[t, :] = x_t

            # State transition
            if t < T - 1:  # No need to update on last iteration
                x_t = A @ x_t + B @ eps[t, :]

        return SimulatorOutput(
            y_canonical=y_canonical,
            x_state=x_state,
        )

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
            x0 = np.zeros(self.n_state, dtype=np.float64)

        # Baseline simulation (no shocks)
        eps_baseline = np.zeros((H + 1, self.n_shocks), dtype=np.float64)
        output_baseline = self.simulate(theta, eps_baseline, H + 1, x0)

        # Shocked simulation (shock at t=0 only)
        eps_shocked = np.zeros((H + 1, self.n_shocks), dtype=np.float64)
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
        """Compute analytic IRF using closed-form solution.

        For LSS with dynamics x[t+1] = A @ x[t] + B @ eps[t], y[t] = C @ x[t]:
        - Shock at t=0 affects x[1], so y[0] = 0 (no effect yet)
        - IRF[h] = C @ A^(h-1) @ B @ e for h >= 1
        - This matches the simulation where y[0] is observed before shock applies

        Args:
            theta: Parameters
            shock_idx: Index of shock to perturb
            shock_size: Size of shock in std devs
            H: Horizon length

        Returns:
            IRF array, shape (H+1, 3)
        """
        A, B, C = self._unpack_theta(theta)
        sigma = self.shock_manifest.sigma

        # Shock vector (one-hot)
        e_shock = np.zeros(self.n_shocks, dtype=np.float64)
        e_shock[shock_idx] = shock_size * sigma[shock_idx]

        # Compute IRF for each horizon
        # Note: IRF[0] = 0 because shock at t=0 affects y at t=1
        irf = np.zeros((H + 1, 3), dtype=np.float64)
        A_power = np.eye(self.n_state, dtype=np.float64)  # A^0 = I

        for h in range(1, H + 1):
            # irf[h] = C @ A^(h-1) @ B @ e
            irf[h, :] = C @ A_power @ B @ e_shock
            A_power = A_power @ A  # A^h = A^(h-1) @ A

        return irf
