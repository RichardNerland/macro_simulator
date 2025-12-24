"""
Zero Lower Bound (ZLB) Toy simulator.

Extends the New Keynesian model with an occasionally-binding zero lower bound constraint
on the nominal interest rate. The constraint creates nonlinearity:

    i[t] = max(i_unconstrained[t], 0)

where i_unconstrained comes from the standard NK Taylor rule.

This creates state-dependent dynamics: when the ZLB binds, the model behaves differently
than when it's unconstrained.

Metadata tracking: The simulator tracks `binding_fraction`, the fraction of periods
where the ZLB constraint binds.
"""

import numpy as np
import numpy.typing as npt

from .base import SimulatorOutput
from .nk import NKSimulator


class ZLBSimulator(NKSimulator):
    """Zero Lower Bound Toy simulator.

    Inherits from NKSimulator and applies a zero lower bound constraint
    to the interest rate. All other dynamics follow the standard NK model.

    The ZLB creates nonlinearity:
    - When i_unconstrained > 0: standard NK dynamics
    - When i_unconstrained <= 0: ZLB binds, rate clamped at 0

    The binding_fraction (fraction of periods where ZLB binds) is stored
    in SimulatorOutput.metadata.
    """

    def __init__(self, seed: int | None = None):
        """Initialize ZLB simulator.

        Args:
            seed: Random seed for reproducibility (optional)
        """
        super().__init__(seed)

    @property
    def world_id(self) -> str:
        return "zlb"

    def simulate(
        self,
        theta: npt.NDArray[np.float64],
        eps: npt.NDArray[np.float64],
        T: int,
        x0: npt.NDArray[np.float64] | None = None,
    ) -> SimulatorOutput:
        """Simulate ZLB model with constraint.

        Runs standard NK simulation but applies ZLB constraint to interest rate
        at each time step. The constraint is applied after computing the
        unconstrained rate from the Taylor rule.

        Args:
            theta: Parameters
            eps: Shock sequence, shape (T, 3) - [monetary, demand, cost_push]
            T: Number of time steps
            x0: Initial state (optional), shape (5,) - [x, pi, i_lag, r_n_lag, u_lag]

        Returns:
            SimulatorOutput with trajectories and binding_fraction in metadata
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

        # Track ZLB binding
        binding_count = 0

        # Get steady-state nominal rate (annualized, in percent)
        # In a proper NK model, steady-state nominal rate = steady-state real rate + steady-state inflation
        # Here we use r_ss as the steady-state nominal rate for simplicity
        r_ss = theta[11]

        # Simulate with ZLB constraint
        s_t = x0.copy()
        for t in range(T):
            # Observables: [x, pi, i]
            y_canonical[t, 0] = s_t[0]  # Output gap (in %)
            y_canonical[t, 1] = s_t[1] * 4  # Inflation (annualized)

            # Compute interest rate (annualized)
            # State variable i[t] = s_t[2] is in quarterly deviation from steady state
            i_deviation_quarterly = s_t[2]  # Quarterly deviation
            i_deviation_annual = i_deviation_quarterly * 4  # Annualized deviation

            # Compute absolute rate
            # For ZLB model, we output absolute rate (deviation + steady state)
            # This differs from NK which outputs deviations, but makes economic sense for ZLB
            i_absolute = r_ss + i_deviation_annual

            # Apply ZLB constraint to absolute rate
            i_constrained = max(i_absolute, 0.0)  # ZLB: i >= 0
            y_canonical[t, 2] = i_constrained  # Output absolute rate

            # Track binding
            if i_absolute < 0.0:
                binding_count += 1

            # Compute constrained deviation for next period state
            i_deviation_annual_constrained = i_constrained - r_ss
            i_deviation_quarterly_constrained = i_deviation_annual_constrained / 4.0

            # Update state with constrained rate
            s_t_constrained = s_t.copy()
            s_t_constrained[2] = i_deviation_quarterly_constrained

            x_state[t, :] = s_t_constrained

            # Transition
            if t < T - 1:
                # Use constrained interest rate for state evolution
                # This creates the nonlinearity: when ZLB binds, dynamics differ
                s_t = A @ s_t_constrained + B @ (eps[t, :] * shock_scale)

        # Compute binding fraction
        binding_fraction = float(binding_count) / T

        return SimulatorOutput(
            y_canonical=y_canonical,
            x_state=x_state,
            metadata={"binding_fraction": binding_fraction},
        )

    def compute_irf(
        self,
        theta: npt.NDArray[np.float64],
        shock_idx: int,
        shock_size: float,
        H: int,
        x0: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute IRF via baseline subtraction with ZLB constraint.

        The ZLB constraint creates nonlinearity, so IRFs may differ from
        the linear NK model, especially for large negative shocks or
        when starting near the ZLB.

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

        # Baseline simulation (no shocks) with ZLB
        eps_baseline = np.zeros((H + 1, 3), dtype=np.float64)
        output_baseline = self.simulate(theta, eps_baseline, H + 1, x0)

        # Shocked simulation (shock at t=0 only) with ZLB
        eps_shocked = np.zeros((H + 1, 3), dtype=np.float64)
        eps_shocked[0, shock_idx] = shock_size  # Already in std dev units
        output_shocked = self.simulate(theta, eps_shocked, H + 1, x0)

        # IRF = difference
        irf = output_shocked.y_canonical - output_baseline.y_canonical

        return irf

    def validate_parameters(self, theta: npt.NDArray[np.float64]) -> bool:
        """Check parameter validity including determinacy.

        Same as NK model - ZLB is a constraint, not a parameter change.

        Args:
            theta: Parameters to validate

        Returns:
            True if parameters are valid and system is determinate
        """
        return super().validate_parameters(theta)
