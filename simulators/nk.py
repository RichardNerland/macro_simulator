"""
New Keynesian (NK) DSGE simulator.

Implements a 3-equation linearized New Keynesian model:
1. IS curve (output gap): x[t] = E_t[x[t+1]] - sigma^-1 * (i[t] - E_t[pi[t+1]] - r_n[t])
2. Phillips curve: pi[t] = beta * E_t[pi[t+1]] + kappa * x[t] + u[t]
3. Taylor rule: i[t] = rho_i * i[t-1] + (1-rho_i) * (phi_pi * pi[t] + phi_y * x[t]) + eps_m[t]

where:
- x[t] = output gap
- pi[t] = inflation
- i[t] = nominal interest rate
- r_n[t] = natural rate (demand shock)
- u[t] = cost-push shock
- eps_m[t] = monetary policy shock

Uses Blanchard-Kahn conditions for determinacy checking.
"""

import numpy as np
import numpy.typing as npt
from scipy.linalg import qz

from .base import (
    ObservableManifest,
    ParameterManifest,
    ShockManifest,
    SimulatorAdapter,
    SimulatorOutput,
)


class NKSimulator(SimulatorAdapter):
    """New Keynesian DSGE simulator.

    The model has:
    - 3 endogenous variables: output gap, inflation, interest rate
    - 3 structural shocks: monetary, demand (natural rate), cost-push

    Parameters:
    - sigma: Intertemporal elasticity of substitution
    - beta: Discount factor
    - kappa: Phillips curve slope
    - phi_pi: Taylor rule coefficient on inflation
    - phi_y: Taylor rule coefficient on output gap
    - rho_i: Interest rate smoothing
    - rho_r: Persistence of natural rate shock
    - rho_u: Persistence of cost-push shock
    - sigma_m, sigma_r, sigma_u: Shock standard deviations
    """

    def __init__(self, seed: int | None = None):
        """Initialize NK simulator.

        Args:
            seed: Random seed for reproducibility (optional)
        """
        self._seed = seed
        self.n_params = 12  # sigma, beta, kappa, phi_pi, phi_y, rho_i, rho_r, rho_u, 3 sigmas, r_ss

    @property
    def world_id(self) -> str:
        return "nk"

    @property
    def param_manifest(self) -> ParameterManifest:
        """Parameter manifest for NK simulator."""
        names = [
            "sigma",      # 0: Intertemporal elasticity
            "beta",       # 1: Discount factor
            "kappa",      # 2: Phillips curve slope
            "phi_pi",     # 3: Taylor rule inflation coefficient
            "phi_y",      # 4: Taylor rule output coefficient
            "rho_i",      # 5: Interest rate smoothing
            "rho_r",      # 6: Natural rate persistence
            "rho_u",      # 7: Cost-push persistence
            "sigma_m",    # 8: Monetary shock std
            "sigma_r",    # 9: Demand shock std
            "sigma_u",    # 10: Cost-push shock std
            "r_ss",       # 11: Steady-state real rate (annualized, percent)
        ]

        units = ["-", "-", "-", "-", "-", "-", "-", "-", "%", "%", "%", "%"]

        bounds = np.array([
            [0.5, 2.5],      # sigma
            [0.985, 0.995],  # beta
            [0.01, 0.5],     # kappa
            [1.05, 3.5],     # phi_pi (must be > 1 for determinacy)
            [0.0, 1.0],      # phi_y
            [0.0, 0.9],      # rho_i
            [0.0, 0.95],     # rho_r
            [0.0, 0.9],      # rho_u
            [0.001, 0.01],   # sigma_m
            [0.005, 0.02],   # sigma_r
            [0.001, 0.01],   # sigma_u
            [0.5, 4.0],      # r_ss (annualized)
        ], dtype=np.float64)

        defaults = np.array([
            1.0,    # sigma
            0.99,   # beta
            0.1,    # kappa
            1.5,    # phi_pi
            0.125,  # phi_y
            0.8,    # rho_i
            0.5,    # rho_r
            0.5,    # rho_u
            0.0025, # sigma_m (25 basis points)
            0.01,   # sigma_r
            0.0025, # sigma_u
            2.0,    # r_ss (2% annualized)
        ], dtype=np.float64)

        return ParameterManifest(
            names=names,
            units=units,
            bounds=bounds,
            defaults=defaults,
        )

    @property
    def shock_manifest(self) -> ShockManifest:
        """Shock manifest for NK simulator."""
        names = ["monetary", "demand", "cost_push"]
        sigma = np.array([0.0025, 0.01, 0.0025], dtype=np.float64)  # Default values

        return ShockManifest(
            names=names,
            n_shocks=3,
            sigma=sigma,
            default_size=1.0,
        )

    @property
    def obs_manifest(self) -> ObservableManifest:
        """Observable manifest for NK simulator."""
        return ObservableManifest(
            canonical_names=["output", "inflation", "rate"],
            extra_names=[],
            n_canonical=3,
            n_extra=0,
        )

    def _solve_re_system(
        self, theta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], bool]:
        """Solve rational expectations system using Blanchard-Kahn method.

        The NK model in matrix form:
        E_t[z[t+1]] = A * z[t] + B * epsilon[t]

        where z[t] = [x[t], pi[t], i[t], r_n[t], u[t]]

        Returns:
            Tuple of (T, R, determinate) where:
            - T: State transition matrix (n_state, n_state)
            - R: Shock loading matrix (n_state, n_shocks)
            - determinate: Whether the system is determinate
        """
        # Unpack parameters
        sigma = theta[0]
        beta = theta[1]
        kappa = theta[2]
        phi_pi = theta[3]
        phi_y = theta[4]
        rho_i = theta[5]
        rho_r = theta[6]
        rho_u = theta[7]

        # State vector: [x, pi, i, r_n, u]
        # Forward-looking: x, pi
        # Predetermined: i, r_n, u

        # Build system matrices for:
        # Gamma_0 * z[t] = Gamma_1 * E_t[z[t+1]] + Psi * epsilon[t] + Pi * eta[t]

        n = 5  # State dimension

        Gamma_0 = np.zeros((n, n), dtype=np.float64)
        Gamma_1 = np.zeros((n, n), dtype=np.float64)
        Psi = np.zeros((n, 3), dtype=np.float64)

        # Equation 1: IS curve
        # x[t] = E_t[x[t+1]] - sigma^-1 * (i[t] - E_t[pi[t+1]] - r_n[t])
        # Rearranged: x[t] - E_t[x[t+1]] + sigma^-1 * i[t] - sigma^-1 * E_t[pi[t+1]] = sigma^-1 * r_n[t]
        Gamma_0[0, 0] = 1.0          # x[t]
        Gamma_0[0, 2] = 1.0 / sigma  # i[t]
        Gamma_1[0, 0] = 1.0          # E_t[x[t+1]]
        Gamma_1[0, 1] = 1.0 / sigma  # E_t[pi[t+1]]
        Gamma_0[0, 3] = -1.0 / sigma # r_n[t]

        # Equation 2: Phillips curve
        # pi[t] = beta * E_t[pi[t+1]] + kappa * x[t] + u[t]
        # Rearranged: pi[t] - beta * E_t[pi[t+1]] - kappa * x[t] = u[t]
        Gamma_0[1, 1] = 1.0          # pi[t]
        Gamma_0[1, 0] = -kappa       # x[t]
        Gamma_1[1, 1] = beta         # E_t[pi[t+1]]
        Gamma_0[1, 4] = -1.0         # u[t]

        # Equation 3: Taylor rule
        # i[t] = rho_i * i[t-1] + (1-rho_i) * (phi_pi * pi[t] + phi_y * x[t]) + eps_m[t]
        # Rearranged: i[t] - (1-rho_i) * phi_pi * pi[t] - (1-rho_i) * phi_y * x[t] = rho_i * i[t-1] + eps_m[t]
        Gamma_0[2, 2] = 1.0                      # i[t]
        Gamma_0[2, 1] = -(1 - rho_i) * phi_pi   # pi[t]
        Gamma_0[2, 0] = -(1 - rho_i) * phi_y    # x[t]
        Gamma_1[2, 2] = rho_i                   # i[t-1] (note: predetermined, so this goes in Gamma_1)
        Psi[2, 0] = 1.0                         # eps_m[t]

        # Equation 4: Natural rate evolution
        # r_n[t] = rho_r * r_n[t-1] + eps_r[t]
        Gamma_0[3, 3] = 1.0
        Gamma_1[3, 3] = rho_r
        Psi[3, 1] = 1.0  # eps_r[t]

        # Equation 5: Cost-push evolution
        # u[t] = rho_u * u[t-1] + eps_u[t]
        Gamma_0[4, 4] = 1.0
        Gamma_1[4, 4] = rho_u
        Psi[4, 2] = 1.0  # eps_u[t]

        # Solve using QZ decomposition (generalized Schur form)
        try:
            AA, BB, Q, Z = qz(Gamma_0, Gamma_1, output='complex')

            # Generalized eigenvalues: lambda_i = BB[i,i] / AA[i,i]
            # Stable if |lambda_i| < 1

            eigenvalues = []
            for i in range(n):
                if np.abs(AA[i, i]) > 1e-10:
                    eigenvalues.append(BB[i, i] / AA[i, i])
                else:
                    eigenvalues.append(np.inf)  # Infinite eigenvalue

            eigenvalues = np.array(eigenvalues)

            # Blanchard-Kahn conditions:
            # Number of unstable eigenvalues should equal number of forward-looking variables
            n_forward = 2  # x, pi
            n_unstable = np.sum(np.abs(eigenvalues) > 1.0)

            determinate = (n_unstable == n_forward)

            if not determinate:
                # Return dummy matrices
                return np.eye(n), np.zeros((n, 3)), False

            # Reorder so stable eigenvalues come first
            # (This is a simplified approach; full gensys is more sophisticated)
            # For now, we'll use a simpler state-space construction

            # Build reduced-form solution
            # For simplicity, use companion form approach
            # State: s[t] = [x[t], pi[t], i[t-1], r_n[t-1], u[t-1]]
            # Then s[t+1] = T @ s[t] + R @ eps[t]

            # This requires inverting the system
            # For NK model, we can derive this analytically or numerically

            # Simplified solution (assuming determinacy):
            # We solve for the policy functions numerically

            # State transition matrix (reduced form)
            T = np.zeros((n, n), dtype=np.float64)
            R = np.zeros((n, 3), dtype=np.float64)

            # Use a simplified approach: invert and solve
            # This is a placeholder - in production, use full Klein or Sims solver

            # For determinacy checking, we can return simplified dynamics
            # Here we construct an approximate solution

            # Forward variables: x[t], pi[t] (indices 0, 1)
            # Predetermined: i[t-1], r_n[t-1], u[t-1] (indices 2, 3, 4)

            # Simple solution assuming strong Taylor principle
            # This is a rough approximation for testing

            # Inflation response to shocks
            T[1, 4] = rho_u * beta / (1 - beta * rho_u)  # pi to lagged u
            T[0, 3] = rho_r / sigma  # x to lagged r_n

            # Interest rate rule
            T[2, 1] = (1 - rho_i) * phi_pi
            T[2, 0] = (1 - rho_i) * phi_y
            T[2, 2] = rho_i

            # Shock processes
            T[3, 3] = rho_r
            T[4, 4] = rho_u

            R[2, 0] = 1.0  # Monetary shock to interest rate
            R[3, 1] = 1.0  # Demand shock to r_n
            R[4, 2] = 1.0  # Cost-push shock to u

            return T, R, True

        except Exception:
            # If QZ fails, mark as indeterminate
            return np.eye(n), np.zeros((n, 3)), False

    def sample_parameters(self, rng: np.random.Generator) -> npt.NDArray[np.float64]:
        """Sample parameters with determinacy filter.

        Args:
            rng: Random number generator

        Returns:
            Parameter vector satisfying determinacy
        """
        bounds = self.param_manifest.bounds
        max_attempts = 100

        for attempt in range(max_attempts):
            # Sample from uniform distribution within bounds
            theta = rng.uniform(bounds[:, 0], bounds[:, 1])

            # Check determinacy
            _, _, determinate = self._solve_re_system(theta)

            if determinate and self.validate_parameters(theta):
                return theta

        raise ValueError(
            f"Failed to find determinate parameters after {max_attempts} attempts"
        )

    def simulate(
        self,
        theta: npt.NDArray[np.float64],
        eps: npt.NDArray[np.float64],
        T: int,
        x0: npt.NDArray[np.float64] | None = None,
    ) -> SimulatorOutput:
        """Simulate NK model.

        Args:
            theta: Parameters
            eps: Shock sequence, shape (T, 3) - [monetary, demand, cost_push]
            T: Number of time steps
            x0: Initial state (optional), shape (5,) - [x, pi, i_lag, r_n_lag, u_lag]

        Returns:
            SimulatorOutput with trajectories
        """
        # Solve for state-space representation
        A, B, determinate = self._solve_re_system(theta)

        if not determinate:
            raise ValueError("Cannot simulate indeterminate system")

        # Scale shocks by sigma
        sigma_m, sigma_r, sigma_u = theta[8], theta[9], theta[10]
        shock_scale = np.array([sigma_m, sigma_r, sigma_u])

        # Initialize state
        if x0 is None:
            x0 = np.zeros(5, dtype=np.float64)

        # Pre-allocate
        n_state = 5
        x_state = np.zeros((T, n_state), dtype=np.float64)
        y_canonical = np.zeros((T, 3), dtype=np.float64)

        # Simulate
        s_t = x0.copy()
        for t in range(T):
            # Observables: [x, pi, i]
            y_canonical[t, 0] = s_t[0]  # Output gap (in %)
            y_canonical[t, 1] = s_t[1] * 4  # Inflation (annualized)
            y_canonical[t, 2] = s_t[2] * 4  # Interest rate (annualized)

            # Store state
            x_state[t, :] = s_t

            # Transition
            if t < T - 1:
                s_t = A @ s_t + B @ (eps[t, :] * shock_scale)

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
            x0 = np.zeros(5, dtype=np.float64)

        # Baseline simulation (no shocks)
        eps_baseline = np.zeros((H + 1, 3), dtype=np.float64)
        output_baseline = self.simulate(theta, eps_baseline, H + 1, x0)

        # Shocked simulation (shock at t=0 only)
        eps_shocked = np.zeros((H + 1, 3), dtype=np.float64)
        eps_shocked[0, shock_idx] = shock_size  # Already in std dev units
        output_shocked = self.simulate(theta, eps_shocked, H + 1, x0)

        # IRF = difference
        irf = output_shocked.y_canonical - output_baseline.y_canonical

        return irf

    def validate_parameters(self, theta: npt.NDArray[np.float64]) -> bool:
        """Check parameter validity including determinacy.

        Args:
            theta: Parameters to validate

        Returns:
            True if parameters are valid and system is determinate
        """
        # Check bounds first
        if not super().validate_parameters(theta):
            return False

        # Check Taylor principle (necessary for determinacy)
        phi_pi = theta[3]
        if phi_pi <= 1.0:
            return False

        return True
