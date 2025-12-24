"""
Real Business Cycle (RBC) simulator.

Implements a standard RBC model with:
- Capital accumulation
- Technology shock (productivity)
- Household optimization (consumption-leisure choice)

Model equations (steady state):
    Y = A * K^alpha * L^(1-alpha)
    K' = (1-delta) * K + I
    Y = C + I

Linearized dynamics around steady state:
    State: [k_t, a_t] (capital deviation, productivity deviation)
    Observables: output, consumption growth (as inflation proxy), interest rate
"""

import numpy as np
import numpy.typing as npt
from scipy.linalg import eig

from .base import (
    ObservableManifest,
    ParameterManifest,
    ShockManifest,
    SimulatorAdapter,
    SimulatorOutput,
)


class RBCSimulator(SimulatorAdapter):
    """Real Business Cycle simulator.

    Standard RBC model with capital and technology shocks.

    Parameters:
    - beta: Discount factor (quarterly)
    - alpha: Capital share (Cobb-Douglas)
    - delta: Depreciation rate (quarterly)
    - gamma: Risk aversion (CRRA)
    - chi: Labor disutility weight
    - rho_a: Technology shock persistence
    - sigma_a: Technology shock std dev

    Shocks:
    - Technology (productivity): eps_a[t]
    """

    def __init__(self, seed: int | None = None):
        """Initialize RBC simulator.

        Args:
            seed: Random seed for reproducibility (optional)
        """
        self._seed = seed
        self.n_params = 7  # beta, alpha, delta, gamma, chi, rho_a, sigma_a

    @property
    def world_id(self) -> str:
        return "rbc"

    @property
    def param_manifest(self) -> ParameterManifest:
        """Parameter manifest for RBC simulator."""
        names = [
            "beta",    # 0: Discount factor
            "alpha",   # 1: Capital share
            "delta",   # 2: Depreciation rate
            "gamma",   # 3: Risk aversion (CRRA)
            "chi",     # 4: Labor disutility weight
            "rho_a",   # 5: Technology persistence
            "sigma_a", # 6: Technology shock std dev
        ]

        units = ["-", "-", "-", "-", "-", "-", "%"]

        bounds = np.array([
            [0.985, 0.995],  # beta (quarterly)
            [0.25, 0.40],    # alpha
            [0.02, 0.03],    # delta (quarterly)
            [1.0, 3.0],      # gamma
            [0.5, 5.0],      # chi
            [0.0, 0.99],     # rho_a
            [0.005, 0.02],   # sigma_a
        ], dtype=np.float64)

        defaults = np.array([
            0.99,   # beta
            0.33,   # alpha
            0.025,  # delta
            1.0,    # gamma (log utility)
            1.0,    # chi
            0.95,   # rho_a
            0.01,   # sigma_a
        ], dtype=np.float64)

        return ParameterManifest(
            names=names,
            units=units,
            bounds=bounds,
            defaults=defaults,
        )

    @property
    def shock_manifest(self) -> ShockManifest:
        """Shock manifest for RBC simulator."""
        names = ["technology"]
        sigma = np.array([0.01], dtype=np.float64)  # Default 1% productivity shock

        return ShockManifest(
            names=names,
            n_shocks=1,
            sigma=sigma,
            default_size=1.0,
        )

    @property
    def obs_manifest(self) -> ObservableManifest:
        """Observable manifest for RBC simulator."""
        return ObservableManifest(
            canonical_names=["output", "inflation", "rate"],
            extra_names=["capital", "consumption", "labor", "investment"],
            n_canonical=3,
            n_extra=4,
        )

    def _compute_steady_state(
        self, theta: npt.NDArray[np.float64]
    ) -> dict[str, float]:
        """Compute deterministic steady state.

        Args:
            theta: Parameter vector

        Returns:
            Dictionary with steady-state values
        """
        # Unpack parameters
        beta = theta[0]
        alpha = theta[1]
        delta = theta[2]
        gamma = theta[3]
        chi = theta[4]

        # Steady-state calculations
        # From Euler equation: beta * (1 - delta + alpha * Y/K) = 1
        # => Y/K = (1/beta - 1 + delta) / alpha
        y_k_ratio = (1.0 / beta - 1.0 + delta) / alpha

        # From production function with A=1, Y = K^alpha * L^(1-alpha)
        # => Y/K = (L)^(1-alpha) * K^(alpha-1) = (L/K)^(1-alpha)
        # => L/K = (Y/K)^(1/(1-alpha))
        l_k_ratio = y_k_ratio ** (1.0 / (1.0 - alpha))

        # From resource constraint: Y = C + I, I = delta * K
        # => C/K = Y/K - delta
        c_k_ratio = y_k_ratio - delta

        # From labor FOC: chi * L^(1/gamma) = (1-alpha) * Y / L
        # Normalize by setting K=1 (or solve for absolute values)
        # We'll use normalized values (all relative to K=1)

        # Solve for L using labor FOC and production function
        # chi * L^(1/gamma) = (1-alpha) * Y / L
        # Y = K^alpha * L^(1-alpha) = L^(1-alpha) with K=1
        # => chi * L^(1/gamma) = (1-alpha) * L^(1-alpha) / L = (1-alpha) * L^(-alpha)
        # => chi * L^(1/gamma + alpha) = (1-alpha)
        # => L = [(1-alpha) / chi]^(gamma / (1 + gamma*alpha))

        L_ss = ((1.0 - alpha) / chi) ** (gamma / (1.0 + gamma * alpha))

        # With K=1 normalization:
        K_ss = 1.0
        Y_ss = K_ss ** alpha * L_ss ** (1.0 - alpha)
        I_ss = delta * K_ss
        C_ss = Y_ss - I_ss  # Use resource constraint directly

        # Interest rate: R = 1 + MPK - delta = 1 + alpha * Y/K - delta
        # Gross return on capital
        R_ss = 1.0 + alpha * y_k_ratio - delta

        return {
            "K": K_ss,
            "Y": Y_ss,
            "C": C_ss,
            "L": L_ss,
            "I": I_ss,
            "R": R_ss,
            "A": 1.0,  # Normalized
        }

    def _linearize(
        self, theta: npt.NDArray[np.float64], ss: dict[str, float]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Linearize RBC model around steady state.

        State vector: s[t] = [k_t, a_t]  (log deviations from steady state)

        Dynamics (simplified RBC):
        k[t+1] = persistence * k[t] + k_loading * a[t]
        a[t+1] = rho_a * a[t] + eps_a[t]

        Observation: y[t] = C @ s[t]

        Args:
            theta: Parameter vector
            ss: Steady-state values

        Returns:
            Tuple of (A, B, C) matrices where:
            - A: State transition (2, 2)
            - B: Shock loading (2, 1)
            - C_obs: Observation matrix (7, 2) for [Y, C, L, I, R, C_growth, K]
        """
        # Unpack parameters
        beta = theta[0]
        alpha = theta[1]
        delta = theta[2]
        gamma = theta[3]
        rho_a = theta[5]

        # State transition matrix
        # Simplified dynamics: k[t+1] depends on k[t] and a[t]
        # Using approximate linearization coefficients

        # Capital accumulation coefficient (persistence)
        k_persist = (1.0 - delta) + alpha * delta

        # Technology impact on capital
        k_tech_load = (1.0 - alpha) * 0.5  # Simplified loading

        # State matrix: s[t] = [k[t], a[t]]
        A = np.array([
            [k_persist, k_tech_load],  # k[t+1] equation
            [0.0, rho_a],              # a[t+1] = rho_a * a[t] + shock
        ], dtype=np.float64)

        # Shock loading matrix
        B = np.array([
            [0.0],  # k[t+1] not directly affected by shock (only through a)
            [1.0],  # a[t+1] affected by technology shock
        ], dtype=np.float64)

        # Observation matrix (7 variables: Y, C, L, I, R, C_growth, K)
        # y[t] = alpha * k[t] + (1-alpha) * l[t] + a[t]  (production function)
        # l[t] ≈ -alpha/(1-alpha) * k[t] + 1/(1-alpha) * a[t]  (labor FOC)
        # c[t] ≈ k[t] effects  (consumption)

        # Simplified observation coefficients
        y_k_coef = alpha  # Output response to capital
        y_a_coef = 1.0    # Output response to technology

        c_k_coef = 0.5    # Consumption response to capital
        c_a_coef = 0.7    # Consumption response to technology

        l_k_coef = -0.3   # Labor response to capital
        l_a_coef = 0.8    # Labor response to technology

        i_k_coef = 1.5    # Investment response to capital (more volatile)
        i_a_coef = 1.2    # Investment response to technology

        # Interest rate: R[t] = alpha * (y[t] - k[t])
        r_k_coef = -(1.0 - alpha)  # Rate falls when capital rises
        r_a_coef = 1.0              # Rate rises with technology

        # Consumption growth ≈ c[t] - c[t-1] (for inflation proxy)
        # We'll compute this in simulation, here just use consumption level
        cg_k_coef = c_k_coef
        cg_a_coef = c_a_coef

        C_obs = np.array([
            [y_k_coef, y_a_coef],   # Output
            [c_k_coef, c_a_coef],   # Consumption
            [l_k_coef, l_a_coef],   # Labor
            [i_k_coef, i_a_coef],   # Investment
            [r_k_coef, r_a_coef],   # Interest rate
            [cg_k_coef, cg_a_coef], # Consumption growth proxy
            [1.0, 0.0],             # Capital (direct observation)
        ], dtype=np.float64)

        return A, B, C_obs

    def _check_saddle_path_stability(
        self, A: npt.NDArray[np.float64]
    ) -> bool:
        """Check saddle-path stability (one eigenvalue inside, one outside unit circle).

        For RBC model with state [k, a]:
        - k (capital) is predetermined (sticky)
        - a (technology) is exogenous

        We need the capital dynamics to be stable (eigenvalue < 1).

        Args:
            A: State transition matrix

        Returns:
            True if system is saddle-path stable
        """
        eigenvalues, _ = eig(A)
        eigenvalues = np.abs(eigenvalues)

        # For simplified RBC: both eigenvalues should be < 1 for stability
        # (This is a departure from full DSGE where we'd have one unstable)
        # Since we're using a reduced-form approximation
        all_stable = np.all(eigenvalues < 1.0)

        # Also check not too persistent (eigenvalues not too close to 1)
        not_explosive = np.all(eigenvalues < 0.999)

        return bool(all_stable and not_explosive)

    def sample_parameters(self, rng: np.random.Generator) -> npt.NDArray[np.float64]:
        """Sample parameters with saddle-path stability filter.

        Args:
            rng: Random number generator

        Returns:
            Parameter vector satisfying stability
        """
        bounds = self.param_manifest.bounds
        max_attempts = 100

        for attempt in range(max_attempts):
            # Sample from uniform distribution within bounds
            theta = rng.uniform(bounds[:, 0], bounds[:, 1])

            # Check bounds
            if not self.validate_parameters(theta):
                continue

            # Compute steady state
            try:
                ss = self._compute_steady_state(theta)

                # Check steady state is reasonable
                if ss["Y"] <= 0 or ss["C"] <= 0 or ss["K"] <= 0 or ss["L"] <= 0:
                    continue
                if ss["L"] > 1.0:  # Labor should be < 1 (fraction of time)
                    continue

                # Linearize
                A, _, _ = self._linearize(theta, ss)

                # Check stability
                if self._check_saddle_path_stability(A):
                    return theta

            except (ValueError, RuntimeWarning, FloatingPointError):
                # Skip if steady state computation fails
                continue

        raise ValueError(
            f"Failed to find stable parameters after {max_attempts} attempts"
        )

    def simulate(
        self,
        theta: npt.NDArray[np.float64],
        eps: npt.NDArray[np.float64],
        T: int,
        x0: npt.NDArray[np.float64] | None = None,
    ) -> SimulatorOutput:
        """Simulate RBC model.

        Args:
            theta: Parameters
            eps: Shock sequence, shape (T, 1) - technology shocks
            T: Number of time steps
            x0: Initial state (optional), shape (2,) - [k, a]

        Returns:
            SimulatorOutput with trajectories
        """
        # Compute steady state and linearization
        ss = self._compute_steady_state(theta)
        A, B, C_obs = self._linearize(theta, ss)

        # Scale shocks by sigma
        sigma_a = theta[6]

        # Initialize state
        if x0 is None:
            x0 = np.zeros(2, dtype=np.float64)  # Start at steady state

        # Pre-allocate
        n_state = 2
        x_state = np.zeros((T, n_state), dtype=np.float64)
        y_all = np.zeros((T, 7), dtype=np.float64)  # All observables
        y_canonical = np.zeros((T, 3), dtype=np.float64)

        # Simulate
        s_t = x0.copy()
        c_prev = 0.0  # For consumption growth calculation

        for t in range(T):
            # Compute observables (log deviations)
            obs = C_obs @ s_t
            y_all[t, :] = obs

            # Extract canonical observables (convert to percent deviations)
            y_canonical[t, 0] = obs[0] * 100.0  # Output (%)

            # Inflation proxy: consumption growth (quarterly rate, annualized)
            c_growth = (obs[1] - c_prev) * 4.0 * 100.0  # Annualized %
            y_canonical[t, 1] = c_growth
            c_prev = obs[1]

            # Interest rate (convert to annualized %)
            # R is in log deviation from steady state
            # r_dev is the log deviation, so actual rate is: R_ss * exp(r_dev)
            # But R_ss is gross return (like 1.01), we want net rate
            r_dev = obs[4]  # Log deviation
            r_ss_quarterly = ss["R"]  # Gross quarterly return (e.g., 1.01)
            r_level = r_ss_quarterly * np.exp(r_dev)  # Gross return at time t
            net_rate_quarterly = r_level - 1.0  # Net return (e.g., 0.01)
            y_canonical[t, 2] = net_rate_quarterly * 4.0 * 100.0  # Annualized %

            # Store state
            x_state[t, :] = s_t

            # Transition
            if t < T - 1:
                s_t = A @ s_t + B @ (eps[t, :] * sigma_a)

        # Extra observables
        y_extra = y_all[:, [6, 1, 2, 3]] * 100.0  # Capital, Consumption, Labor, Investment (%)

        return SimulatorOutput(
            y_canonical=y_canonical,
            y_extra=y_extra,
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
            shock_idx: Index of shock to perturb (0 for technology)
            shock_size: Size of shock in std devs
            H: Horizon length
            x0: Initial state (optional)

        Returns:
            IRF array, shape (H+1, 3)
        """
        if x0 is None:
            x0 = np.zeros(2, dtype=np.float64)  # Steady state

        # Baseline simulation (no shocks)
        eps_baseline = np.zeros((H + 1, 1), dtype=np.float64)
        output_baseline = self.simulate(theta, eps_baseline, H + 1, x0)

        # Shocked simulation (shock at t=0 only)
        eps_shocked = np.zeros((H + 1, 1), dtype=np.float64)
        eps_shocked[0, shock_idx] = shock_size  # Already in std dev units

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
    ) -> npt.NDArray[np.float64] | None:
        """Compute analytic IRF using linearized solution.

        For RBC, we can compute this from the state-space representation.

        Args:
            theta: Parameters
            shock_idx: Index of shock to perturb
            shock_size: Size of shock in std devs
            H: Horizon length

        Returns:
            IRF array shape (H+1, 3)
        """
        # Compute steady state and linearization
        ss = self._compute_steady_state(theta)
        A, B, C_obs = self._linearize(theta, ss)
        sigma_a = theta[6]

        # Shock vector
        e_shock = np.zeros(1, dtype=np.float64)
        e_shock[shock_idx] = shock_size * sigma_a

        # Compute state IRF following simulation timing:
        # y[h] is observed, then x[h+1] = A @ x[h] + B @ eps[h]
        # So shock at t=0 affects state at t=1
        # IRF[0] = 0 (no effect yet)
        # IRF[1] = effect of shock (x[1] includes B @ eps[0])
        irf_state = np.zeros((H + 1, 2), dtype=np.float64)

        # h=0: No effect yet (shock hasn't entered state)
        irf_state[0, :] = 0.0

        # h=1: First effect, state is B @ e_shock
        s_t = (B @ e_shock).ravel()
        irf_state[1, :] = s_t

        # h >= 2: Propagate via A
        for h in range(2, H + 1):
            s_t = A @ s_t
            irf_state[h, :] = s_t

        # Convert to observable IRF
        irf_all = np.zeros((H + 1, 7), dtype=np.float64)
        for h in range(H + 1):
            irf_all[h, :] = C_obs @ irf_state[h, :]

        # Extract canonical observables
        irf = np.zeros((H + 1, 3), dtype=np.float64)

        c_prev = 0.0
        for h in range(H + 1):
            # Output
            irf[h, 0] = irf_all[h, 0] * 100.0

            # Consumption growth (inflation proxy)
            c_growth = (irf_all[h, 1] - c_prev) * 4.0 * 100.0
            irf[h, 1] = c_growth
            c_prev = irf_all[h, 1]

            # Interest rate (approximation for small deviations)
            r_dev = irf_all[h, 4]
            r_ss_quarterly = ss["R"]
            irf[h, 2] = r_dev * r_ss_quarterly * 4.0 * 100.0  # Linearized approximation

        return irf

    def validate_parameters(self, theta: npt.NDArray[np.float64]) -> bool:
        """Check parameter validity.

        Args:
            theta: Parameters to validate

        Returns:
            True if parameters are valid
        """
        # Check bounds
        if not super().validate_parameters(theta):
            return False

        # Additional checks
        beta = theta[0]
        delta = theta[2]

        # Economic validity: beta * (1 - delta) < 1 (ensures steady state exists)
        if beta * (1.0 - delta) >= 1.0:
            return False

        return True
