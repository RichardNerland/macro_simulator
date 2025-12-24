"""
Regime-Switching simulator.

Implements a Markov-switching linear state-space model with 2 regimes:
    s[t] ~ Markov(P)  (regime indicator, 0 or 1)
    x[t+1] = A_{s[t]} @ x[t] + B_{s[t]} @ eps[t]
    y[t] = C @ x[t]

where P is a 2x2 transition matrix and A_0, A_1 are regime-specific state matrices.
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


class SwitchingSimulator(SimulatorAdapter):
    """Regime-switching linear state-space simulator.

    The model has:
    - 2 discrete regimes (low/high volatility or persistence)
    - n_state dimensional state vector
    - n_shocks exogenous shocks
    - 3 canonical observables (output, inflation, rate)
    - Markov transition matrix P with persistence probabilities p_00, p_11

    Parameters (stored in theta):
    - p_00: Probability of staying in regime 0
    - p_11: Probability of staying in regime 1
    - A_0 matrix: n_state x n_state for regime 0 (stored row-major)
    - A_1 matrix: n_state x n_state for regime 1 (stored row-major)
    - B_0 matrix: n_state x n_shocks for regime 0 (stored row-major)
    - B_1 matrix: n_state x n_shocks for regime 1 (stored row-major)
    - C matrix: 3 x n_state (observation, shared across regimes)
    """

    def __init__(
        self,
        n_state: int = 3,
        n_shocks: int = 2,
        rho_max: float = 0.95,
        seed: int | None = None,
    ):
        """Initialize regime-switching simulator.

        Args:
            n_state: State dimension (default 3, kept small for tractability)
            n_shocks: Number of shocks (default 2)
            rho_max: Maximum spectral radius for stability (default 0.95)
            seed: Random seed for reproducibility (optional)
        """
        self.n_state = n_state
        self.n_shocks = n_shocks
        self.rho_max = rho_max
        self._seed = seed

        # Calculate number of parameters:
        # - 2 transition probabilities (p_00, p_11)
        # - 2 A matrices: 2 * n_state * n_state
        # - 2 B matrices: 2 * n_state * n_shocks
        # - 1 C matrix: 3 * n_state
        self.n_params = (
            2
            + 2 * n_state * n_state
            + 2 * n_state * n_shocks
            + 3 * n_state
        )

    @property
    def world_id(self) -> str:
        return "switching"

    @property
    def param_manifest(self) -> ParameterManifest:
        """Parameter manifest for regime-switching simulator."""
        names = []
        units = []
        bounds_list = []
        defaults_list = []

        # Transition probabilities
        names.extend(["p_00", "p_11"])
        units.extend(["-", "-"])
        bounds_list.extend([[0.7, 0.99], [0.7, 0.99]])  # High persistence regimes
        defaults_list.extend([0.9, 0.9])

        # A_0 matrix parameters (regime 0: lower persistence)
        for i in range(self.n_state):
            for j in range(self.n_state):
                names.append(f"A0_{i}{j}")
                units.append("-")
                bounds_list.append([-2.0, 2.0])
                defaults_list.append(0.4 if i == j else 0.0)

        # A_1 matrix parameters (regime 1: higher persistence)
        for i in range(self.n_state):
            for j in range(self.n_state):
                names.append(f"A1_{i}{j}")
                units.append("-")
                bounds_list.append([-2.0, 2.0])
                defaults_list.append(0.7 if i == j else 0.0)

        # B_0 matrix parameters (regime 0: lower volatility)
        for i in range(self.n_state):
            for j in range(self.n_shocks):
                names.append(f"B0_{i}{j}")
                units.append("-")
                bounds_list.append([-1.0, 1.0])
                defaults_list.append(0.05)

        # B_1 matrix parameters (regime 1: higher volatility)
        for i in range(self.n_state):
            for j in range(self.n_shocks):
                names.append(f"B1_{i}{j}")
                units.append("-")
                bounds_list.append([-1.0, 1.0])
                defaults_list.append(0.15)

        # C matrix parameters (shared observation matrix)
        for i in range(3):
            for j in range(self.n_state):
                names.append(f"C_{i}{j}")
                units.append("-")
                bounds_list.append([-2.0, 2.0])
                defaults_list.append(1.0 if j == i else 0.0)

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
        """Shock manifest for regime-switching simulator."""
        names = [f"shock_{i}" for i in range(self.n_shocks)]
        sigma = np.ones(self.n_shocks, dtype=np.float64) * 0.01

        return ShockManifest(
            names=names,
            n_shocks=self.n_shocks,
            sigma=sigma,
            default_size=1.0,
        )

    @property
    def obs_manifest(self) -> ObservableManifest:
        """Observable manifest for regime-switching simulator."""
        return ObservableManifest(
            canonical_names=["output", "inflation", "rate"],
            extra_names=[],
            n_canonical=3,
            n_extra=0,
        )

    def _unpack_theta(
        self, theta: npt.NDArray[np.float64]
    ) -> tuple[
        float,
        float,
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Unpack parameter vector into transition probs and matrices.

        Args:
            theta: Parameter vector

        Returns:
            Tuple of (p_00, p_11, A_0, A_1, B_0, B_1, C)
        """
        idx = 0

        # Extract transition probabilities
        p_00 = theta[idx]
        p_11 = theta[idx + 1]
        idx += 2

        # Extract A_0 matrix
        A0_size = self.n_state * self.n_state
        A_0 = theta[idx : idx + A0_size].reshape(self.n_state, self.n_state)
        idx += A0_size

        # Extract A_1 matrix
        A_1 = theta[idx : idx + A0_size].reshape(self.n_state, self.n_state)
        idx += A0_size

        # Extract B_0 matrix
        B_size = self.n_state * self.n_shocks
        B_0 = theta[idx : idx + B_size].reshape(self.n_state, self.n_shocks)
        idx += B_size

        # Extract B_1 matrix
        B_1 = theta[idx : idx + B_size].reshape(self.n_state, self.n_shocks)
        idx += B_size

        # Extract C matrix
        C_size = 3 * self.n_state
        C = theta[idx : idx + C_size].reshape(3, self.n_state)

        return p_00, p_11, A_0, A_1, B_0, B_1, C

    def _pack_theta(
        self,
        p_00: float,
        p_11: float,
        A_0: npt.NDArray[np.float64],
        A_1: npt.NDArray[np.float64],
        B_0: npt.NDArray[np.float64],
        B_1: npt.NDArray[np.float64],
        C: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Pack transition probs and matrices into parameter vector.

        Args:
            p_00: Probability of staying in regime 0
            p_11: Probability of staying in regime 1
            A_0: State transition matrix for regime 0
            A_1: State transition matrix for regime 1
            B_0: Shock loading matrix for regime 0
            B_1: Shock loading matrix for regime 1
            C: Observation matrix (shared)

        Returns:
            Parameter vector
        """
        return np.concatenate([
            np.array([p_00, p_11], dtype=np.float64),
            A_0.ravel(),
            A_1.ravel(),
            B_0.ravel(),
            B_1.ravel(),
            C.ravel(),
        ])

    def _sample_transition_matrix(
        self, rng: np.random.Generator
    ) -> tuple[float, float]:
        """Sample transition matrix probabilities.

        Args:
            rng: Random number generator

        Returns:
            Tuple of (p_00, p_11) - probabilities of staying in each regime
        """
        bounds = self.param_manifest.bounds
        p_00_bounds = bounds[0]
        p_11_bounds = bounds[1]

        p_00 = rng.uniform(p_00_bounds[0], p_00_bounds[1])
        p_11 = rng.uniform(p_11_bounds[0], p_11_bounds[1])

        return p_00, p_11

    def _sample_stable_matrix(
        self, rng: np.random.Generator, target_rho: float
    ) -> npt.NDArray[np.float64]:
        """Sample a stable A matrix via eigenvalue placement.

        Args:
            rng: Random number generator
            target_rho: Target maximum spectral radius

        Returns:
            Stable A matrix
        """
        # Generate eigenvalues uniformly in disk of radius target_rho
        n_complex_pairs = self.n_state // 2
        n_real = self.n_state % 2

        eigenvalues = []

        # Complex conjugate pairs
        for _ in range(n_complex_pairs):
            radius = rng.uniform(0.2, target_rho)  # Avoid very small eigenvalues
            angle = rng.uniform(0, 2 * np.pi)
            eigenvalues.append(radius * np.exp(1j * angle))
            eigenvalues.append(radius * np.exp(-1j * angle))

        # Real eigenvalues
        for _ in range(n_real):
            eigenvalues.append(rng.uniform(-target_rho, target_rho))

        eigenvalues = np.array(eigenvalues[: self.n_state])

        # Construct A with desired eigenvalues using random orthogonal basis
        V = rng.standard_normal((self.n_state, self.n_state))
        V, _ = np.linalg.qr(V)  # Orthonormalize
        A = (V @ np.diag(eigenvalues) @ V.T).real.astype(np.float64)

        return A

    def sample_parameters(self, rng: np.random.Generator) -> npt.NDArray[np.float64]:
        """Sample valid regime-switching parameters.

        Ensures:
        - Valid transition probabilities in (0, 1)
        - Both regime matrices A_0 and A_1 are stable
        - Regime 0 has lower persistence/volatility
        - Regime 1 has higher persistence/volatility

        Args:
            rng: Random number generator

        Returns:
            Parameter vector with stable regime dynamics
        """
        # Sample transition probabilities
        p_00, p_11 = self._sample_transition_matrix(rng)

        # Sample regime-specific dynamics
        # Regime 0: lower persistence (rho ~ 0.3-0.7)
        # Regime 1: higher persistence (rho ~ 0.7-0.95)
        rho_0 = rng.uniform(0.3, 0.7)
        rho_1 = rng.uniform(0.7, self.rho_max)

        A_0 = self._sample_stable_matrix(rng, rho_0)
        A_1 = self._sample_stable_matrix(rng, rho_1)

        # Sample B matrices
        # Regime 0: lower volatility
        B_0 = rng.standard_normal((self.n_state, self.n_shocks)) * 0.05

        # Regime 1: higher volatility
        B_1 = rng.standard_normal((self.n_state, self.n_shocks)) * 0.15

        # Sample C matrix - shared across regimes, well-scaled
        C = rng.standard_normal((3, self.n_state)) * 0.5

        # Ensure C has reasonable row norms
        for i in range(3):
            row_norm = np.linalg.norm(C[i, :])
            if row_norm > 0:
                C[i, :] = C[i, :] / row_norm * 0.5

        theta = self._pack_theta(p_00, p_11, A_0, A_1, B_0, B_1, C)

        # Verify stability and bounds
        assert check_spectral_radius(A_0, self.rho_max), "A_0 not stable"
        assert check_spectral_radius(A_1, self.rho_max), "A_1 not stable"
        assert self.validate_parameters(theta), "Generated parameters out of bounds"

        return theta

    def _simulate_regime_sequence(
        self,
        p_00: float,
        p_11: float,
        T: int,
        rng: np.random.Generator,
        s0: int = 0,
    ) -> npt.NDArray[np.int32]:
        """Simulate Markov regime sequence.

        Args:
            p_00: Probability of staying in regime 0
            p_11: Probability of staying in regime 1
            T: Number of time steps
            rng: Random number generator
            s0: Initial regime (default 0)

        Returns:
            Regime sequence, shape (T,)
        """
        regime_seq = np.zeros(T, dtype=np.int32)
        regime_seq[0] = s0

        # Transition probabilities
        P = np.array([[p_00, 1 - p_00], [1 - p_11, p_11]], dtype=np.float64)

        for t in range(1, T):
            # Sample next regime based on current regime
            current_regime = regime_seq[t - 1]
            prob_stay = P[current_regime, current_regime]
            regime_seq[t] = current_regime if rng.random() < prob_stay else 1 - current_regime

        return regime_seq

    def simulate(
        self,
        theta: npt.NDArray[np.float64],
        eps: npt.NDArray[np.float64],
        T: int,
        x0: npt.NDArray[np.float64] | None = None,
        regime_seed: int | None = None,
    ) -> SimulatorOutput:
        """Simulate regime-switching model.

        Dynamics:
            s[t] ~ Markov(P)
            x[t+1] = A_{s[t]} @ x[t] + B_{s[t]} @ eps[t]
            y[t] = C @ x[t]

        Args:
            theta: Parameters
            eps: Shock sequence, shape (T, n_shocks)
            T: Number of time steps
            x0: Initial state (optional), defaults to zero
            regime_seed: Seed for regime sequence generation (optional)

        Returns:
            SimulatorOutput with trajectories and regime_seq
        """
        p_00, p_11, A_0, A_1, B_0, B_1, C = self._unpack_theta(theta)

        # Initialize state
        if x0 is None:
            x0 = np.zeros(self.n_state, dtype=np.float64)

        # Generate regime sequence
        if regime_seed is not None:
            regime_rng = np.random.default_rng(regime_seed)
        else:
            # Use a derived seed from the shock sequence for determinism
            regime_rng = np.random.default_rng(
                hash(eps.tobytes()) % (2**32)
            )

        regime_seq = self._simulate_regime_sequence(p_00, p_11, T, regime_rng)

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

            # State transition with regime-dependent matrices
            if t < T - 1:  # No need to update on last iteration
                regime = regime_seq[t]
                if regime == 0:
                    x_t = A_0 @ x_t + B_0 @ eps[t, :]
                else:
                    x_t = A_1 @ x_t + B_1 @ eps[t, :]

        return SimulatorOutput(
            y_canonical=y_canonical,
            x_state=x_state,
            regime_seq=regime_seq,
        )

    def compute_irf(
        self,
        theta: npt.NDArray[np.float64],
        shock_idx: int,
        shock_size: float,
        H: int,
        x0: npt.NDArray[np.float64] | None = None,
        regime_seed: int = 42,
    ) -> npt.NDArray[np.float64]:
        """Compute IRF via baseline subtraction.

        For regime-switching models, IRF depends on the regime path.
        We use a fixed regime sequence for both baseline and shocked simulations
        to isolate the shock effect.

        Args:
            theta: Parameters
            shock_idx: Index of shock to perturb
            shock_size: Size of shock in std devs
            H: Horizon length
            x0: Initial state (optional)
            regime_seed: Seed for regime sequence (for determinism)

        Returns:
            IRF array, shape (H+1, 3)
        """
        if x0 is None:
            x0 = np.zeros(self.n_state, dtype=np.float64)

        # Baseline simulation (no shocks)
        eps_baseline = np.zeros((H + 1, self.n_shocks), dtype=np.float64)
        output_baseline = self.simulate(
            theta, eps_baseline, H + 1, x0, regime_seed=regime_seed
        )

        # Shocked simulation (shock at t=0 only)
        eps_shocked = np.zeros((H + 1, self.n_shocks), dtype=np.float64)
        sigma = self.shock_manifest.sigma
        eps_shocked[0, shock_idx] = shock_size * sigma[shock_idx]
        output_shocked = self.simulate(
            theta, eps_shocked, H + 1, x0, regime_seed=regime_seed
        )

        # IRF = difference
        irf = output_shocked.y_canonical - output_baseline.y_canonical

        return irf
