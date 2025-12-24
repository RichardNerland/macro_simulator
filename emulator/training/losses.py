"""Loss functions for IRF prediction.

This module implements loss functions for training the Universal Macro Emulator:

1. MultiHorizonLoss: Weighted MSE over all horizons with configurable weighting schemes
2. SmoothnessRegularization: Penalty for oscillatory predictions

All losses operate on IRF tensors with shape:
- (batch_size, H+1, n_obs) for single-shock predictions
- (batch_size, n_shocks, H+1, n_obs) for multi-shock predictions
"""

from typing import Literal

import torch
import torch.nn as nn

from emulator.eval.metrics import (
    exponential_weights,
    impact_weighted,
    uniform_weights,
)


class MultiHorizonLoss(nn.Module):
    """Loss function for IRF prediction.

    Computes weighted MSE over horizons with optional per-variable weighting.

    The loss function is:
        L = Σ_h w[h] * Σ_v λ_v * (ŷ[h,v] - y[h,v])²

    where:
        - w[h] are horizon weights (exponential decay, impact-weighted, or uniform)
        - λ_v are per-variable weights (optional)

    Args:
        H: Horizon length (default: 40)
        weight_scheme: Horizon weighting scheme - "uniform", "exponential", or "impact"
        per_variable_weights: Optional weights for [output, inflation, rate]. If None, all equal.
        tau: Decay parameter for exponential weighting (default: 20.0)
        impact_length: Length of impact period for impact weighting (default: 5)

    Example:
        >>> loss_fn = MultiHorizonLoss(H=40, weight_scheme="exponential", tau=20.0)
        >>> y_pred = torch.randn(32, 41, 3)  # batch_size=32, H+1=41, n_obs=3
        >>> y_true = torch.randn(32, 41, 3)
        >>> loss = loss_fn(y_pred, y_true)
    """

    def __init__(
        self,
        H: int = 40,
        weight_scheme: Literal["uniform", "exponential", "impact"] = "uniform",
        per_variable_weights: list[float] | None = None,
        tau: float = 20.0,
        impact_length: int = 5,
    ):
        super().__init__()

        self.H = H
        self.weight_scheme = weight_scheme
        self.tau = tau
        self.impact_length = impact_length

        # Create horizon weights (convert numpy to torch)
        horizon_weights = self._create_horizon_weights()
        self.register_buffer("horizon_weights", horizon_weights)

        # Create per-variable weights
        if per_variable_weights is not None:
            per_var_weights = torch.tensor(per_variable_weights, dtype=torch.float32)
        else:
            # Default: equal weights for all variables
            per_var_weights = torch.ones(3, dtype=torch.float32)

        # Normalize per-variable weights to sum to number of variables
        per_var_weights = per_var_weights / per_var_weights.sum() * 3.0
        self.register_buffer("per_variable_weights", per_var_weights)

    def _create_horizon_weights(self) -> torch.Tensor:
        """Create horizon weights based on specified scheme.

        Returns:
            Tensor of shape (H+1,) with normalized weights
        """
        if self.weight_scheme == "uniform":
            weights = uniform_weights(self.H)
        elif self.weight_scheme == "exponential":
            weights = exponential_weights(self.H, tau=self.tau)
        elif self.weight_scheme == "impact":
            weights = impact_weighted(self.H, impact_length=self.impact_length)
        else:
            raise ValueError(f"Unknown weight scheme: {self.weight_scheme}")

        return torch.from_numpy(weights).float()

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute weighted MSE loss.

        Args:
            y_pred: Predictions, shape (batch, H+1, n_obs) or (batch, n_shocks, H+1, n_obs)
            y_true: Ground truth, same shape as y_pred
            mask: Optional mask for valid predictions, shape (batch,) or (batch, n_shocks)
                  If provided, only compute loss on masked elements (1.0 = valid, 0.0 = invalid)

        Returns:
            Scalar loss tensor
        """
        # Handle both single-shock and multi-shock cases
        if y_pred.dim() == 3:
            # Single shock: (batch, H+1, n_obs)
            batch_size, H_plus_1, n_obs = y_pred.shape
            n_shocks = 1
            # Add shock dimension
            y_pred = y_pred.unsqueeze(1)
            y_true = y_true.unsqueeze(1)
            if mask is not None:
                mask = mask.unsqueeze(1)
        elif y_pred.dim() == 4:
            # Multi-shock: (batch, n_shocks, H+1, n_obs)
            batch_size, n_shocks, H_plus_1, n_obs = y_pred.shape
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {y_pred.shape}")

        # Verify horizon dimension
        if H_plus_1 != self.H + 1:
            raise ValueError(f"Expected H+1={self.H+1} horizons, got {H_plus_1}")

        # Verify observable dimension
        if n_obs != len(self.per_variable_weights):
            raise ValueError(
                f"Expected {len(self.per_variable_weights)} observables, got {n_obs}"
            )

        # Compute squared errors: (batch, n_shocks, H+1, n_obs)
        sq_errors = (y_pred - y_true) ** 2

        # Apply per-variable weights: (1, 1, 1, n_obs)
        var_weights = self.per_variable_weights.view(1, 1, 1, -1)
        sq_errors = sq_errors * var_weights

        # Apply horizon weights: (1, 1, H+1, 1)
        horiz_weights = self.horizon_weights.view(1, 1, -1, 1)
        sq_errors = sq_errors * horiz_weights

        # Apply mask if provided
        if mask is not None:
            # Expand mask to match all dimensions
            mask = mask.view(batch_size, n_shocks, 1, 1)
            sq_errors = sq_errors * mask
            # Compute mean only over valid elements
            n_valid = mask.sum()
            if n_valid > 0:
                loss = sq_errors.sum() / n_valid
            else:
                loss = sq_errors.sum()  # Shouldn't happen, but avoid NaN
        else:
            # Mean over all dimensions
            loss = sq_errors.mean()

        return loss


class SmoothnessRegularization(nn.Module):
    """Penalize oscillations in predicted IRFs.

    Computes a smoothness penalty based on second-order finite differences.
    This encourages smooth predictions without excessive high-frequency oscillations.

    The penalty is:
        L_smooth = λ_smooth * mean((Δ²ŷ)²)

    where Δ²ŷ[h] = ŷ[h+1] - 2*ŷ[h] + ŷ[h-1] is the second difference.

    Args:
        lambda_smooth: Smoothness penalty coefficient (default: 0.01)

    Example:
        >>> smooth_loss = SmoothnessRegularization(lambda_smooth=0.01)
        >>> y_pred = torch.randn(32, 41, 3)
        >>> loss = smooth_loss(y_pred)
    """

    def __init__(self, lambda_smooth: float = 0.01):
        super().__init__()
        self.lambda_smooth = lambda_smooth

    def forward(
        self,
        y_pred: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute smoothness penalty.

        Args:
            y_pred: Predictions, shape (batch, H+1, n_obs) or (batch, n_shocks, H+1, n_obs)
            mask: Optional mask for valid predictions, shape (batch,) or (batch, n_shocks)

        Returns:
            Scalar smoothness loss
        """
        # Handle both single-shock and multi-shock cases
        if y_pred.dim() == 3:
            # Single shock: (batch, H+1, n_obs)
            # Add shock dimension for uniform handling
            y_pred = y_pred.unsqueeze(1)
            if mask is not None:
                mask = mask.unsqueeze(1)
        elif y_pred.dim() != 4:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {y_pred.shape}")

        # Compute second differences along horizon dimension
        # Δ²y[h] = y[h+1] - 2*y[h] + y[h-1] for h=1,...,H-1
        # Shape: (batch, n_shocks, H-1, n_obs)
        second_diff = y_pred[:, :, 2:, :] - 2 * y_pred[:, :, 1:-1, :] + y_pred[:, :, :-2, :]

        # Squared second differences
        sq_second_diff = second_diff ** 2

        # Apply mask if provided
        if mask is not None:
            # Expand mask to match all dimensions
            batch_size, n_shocks = mask.shape
            mask = mask.view(batch_size, n_shocks, 1, 1)
            sq_second_diff = sq_second_diff * mask
            # Mean over valid elements
            n_valid = mask.sum()
            if n_valid > 0:
                smoothness_loss = sq_second_diff.sum() / n_valid
            else:
                smoothness_loss = sq_second_diff.sum()
        else:
            # Mean over all dimensions
            smoothness_loss = sq_second_diff.mean()

        return self.lambda_smooth * smoothness_loss


class CombinedLoss(nn.Module):
    """Combined loss for IRF prediction with optional smoothness regularization.

    Combines MultiHorizonLoss with optional SmoothnessRegularization:
        L_total = L_horizon + L_smooth

    This is a convenience wrapper for the common case of combining the two losses.

    Args:
        H: Horizon length (default: 40)
        weight_scheme: Horizon weighting scheme
        per_variable_weights: Optional per-variable weights
        tau: Decay parameter for exponential weighting
        impact_length: Length of impact period for impact weighting
        lambda_smooth: Smoothness penalty coefficient (0.0 = no smoothness penalty)

    Example:
        >>> loss_fn = CombinedLoss(
        ...     H=40,
        ...     weight_scheme="exponential",
        ...     tau=20.0,
        ...     lambda_smooth=0.01,
        ... )
        >>> y_pred = torch.randn(32, 41, 3)
        >>> y_true = torch.randn(32, 41, 3)
        >>> loss = loss_fn(y_pred, y_true)
    """

    def __init__(
        self,
        H: int = 40,
        weight_scheme: Literal["uniform", "exponential", "impact"] = "uniform",
        per_variable_weights: list[float] | None = None,
        tau: float = 20.0,
        impact_length: int = 5,
        lambda_smooth: float = 0.0,
    ):
        super().__init__()

        self.horizon_loss = MultiHorizonLoss(
            H=H,
            weight_scheme=weight_scheme,
            per_variable_weights=per_variable_weights,
            tau=tau,
            impact_length=impact_length,
        )

        self.lambda_smooth = lambda_smooth
        if lambda_smooth > 0:
            self.smoothness_loss = SmoothnessRegularization(lambda_smooth=lambda_smooth)
        else:
            self.smoothness_loss = None

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            y_pred: Predictions
            y_true: Ground truth
            mask: Optional mask for valid predictions

        Returns:
            Dictionary with keys:
                - "loss": Total loss (sum of components)
                - "horizon_loss": Multi-horizon component
                - "smoothness_loss": Smoothness component (if lambda_smooth > 0)
        """
        # Compute main loss
        horizon_loss = self.horizon_loss(y_pred, y_true, mask=mask)

        result = {
            "horizon_loss": horizon_loss,
            "loss": horizon_loss,
        }

        # Add smoothness penalty if enabled
        if self.smoothness_loss is not None:
            smoothness_loss = self.smoothness_loss(y_pred, mask=mask)
            result["smoothness_loss"] = smoothness_loss
            result["loss"] = result["loss"] + smoothness_loss

        return result
