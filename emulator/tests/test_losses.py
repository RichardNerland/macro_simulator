"""Tests for loss functions.

Tests for MultiHorizonLoss, SmoothnessRegularization, and CombinedLoss.
"""

import numpy as np
import pytest
import torch

from emulator.training.losses import (
    CombinedLoss,
    MultiHorizonLoss,
    SmoothnessRegularization,
)


@pytest.mark.fast
class TestMultiHorizonLoss:
    """Tests for MultiHorizonLoss."""

    def test_uniform_weights_shape(self):
        """Test that uniform weighting produces correct output shape."""
        loss_fn = MultiHorizonLoss(H=40, weight_scheme="uniform")

        y_pred = torch.randn(32, 41, 3)
        y_true = torch.randn(32, 41, 3)

        loss = loss_fn(y_pred, y_true)

        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Non-negative

    def test_exponential_weights_shape(self):
        """Test exponential weighting scheme."""
        loss_fn = MultiHorizonLoss(H=40, weight_scheme="exponential", tau=20.0)

        y_pred = torch.randn(32, 41, 3)
        y_true = torch.randn(32, 41, 3)

        loss = loss_fn(y_pred, y_true)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_impact_weights_shape(self):
        """Test impact weighting scheme."""
        loss_fn = MultiHorizonLoss(H=40, weight_scheme="impact", impact_length=5)

        y_pred = torch.randn(32, 41, 3)
        y_true = torch.randn(32, 41, 3)

        loss = loss_fn(y_pred, y_true)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_different_weights_produce_different_losses(self):
        """Test that different weighting schemes produce different loss values."""
        torch.manual_seed(42)

        y_pred = torch.randn(32, 41, 3)
        y_true = torch.randn(32, 41, 3)

        loss_uniform = MultiHorizonLoss(H=40, weight_scheme="uniform")(y_pred, y_true)
        loss_exp = MultiHorizonLoss(H=40, weight_scheme="exponential", tau=20.0)(
            y_pred, y_true
        )
        loss_impact = MultiHorizonLoss(H=40, weight_scheme="impact")(y_pred, y_true)

        # Different weighting schemes should produce different losses
        assert not torch.isclose(loss_uniform, loss_exp, rtol=1e-3)
        assert not torch.isclose(loss_uniform, loss_impact, rtol=1e-3)

    def test_per_variable_weights(self):
        """Test per-variable weighting."""
        torch.manual_seed(123)

        # Equal weights (should be same as default)
        loss_fn_equal = MultiHorizonLoss(
            H=40, weight_scheme="uniform", per_variable_weights=[1.0, 1.0, 1.0]
        )

        # Unequal weights - heavily weight first variable
        loss_fn_unequal = MultiHorizonLoss(
            H=40, weight_scheme="uniform", per_variable_weights=[10.0, 1.0, 1.0]
        )

        # Create data where first variable has different error than others
        y_pred = torch.randn(32, 41, 3)
        y_true = torch.randn(32, 41, 3)
        # Make first variable have larger errors
        y_pred[:, :, 0] = y_pred[:, :, 0] * 2.0

        loss_equal = loss_fn_equal(y_pred, y_true)
        loss_unequal = loss_fn_unequal(y_pred, y_true)

        # Unequal weights should produce higher loss due to upweighting first variable
        assert loss_unequal > loss_equal

    def test_multi_shock_input(self):
        """Test that multi-shock inputs work correctly."""
        loss_fn = MultiHorizonLoss(H=40, weight_scheme="uniform")

        # Multi-shock input: (batch, n_shocks, H+1, n_obs)
        y_pred = torch.randn(32, 5, 41, 3)
        y_true = torch.randn(32, 5, 41, 3)

        loss = loss_fn(y_pred, y_true)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_single_shock_input(self):
        """Test that single-shock inputs work correctly."""
        loss_fn = MultiHorizonLoss(H=40, weight_scheme="uniform")

        # Single-shock input: (batch, H+1, n_obs)
        y_pred = torch.randn(32, 41, 3)
        y_true = torch.randn(32, 41, 3)

        loss = loss_fn(y_pred, y_true)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_mask_functionality(self):
        """Test that mask properly filters losses."""
        loss_fn = MultiHorizonLoss(H=40, weight_scheme="uniform")

        # Multi-shock input
        y_pred = torch.randn(32, 5, 41, 3)
        y_true = torch.randn(32, 5, 41, 3)

        # Create mask that only includes first 2 shocks for each sample
        mask = torch.zeros(32, 5)
        mask[:, :2] = 1.0

        loss_masked = loss_fn(y_pred, y_true, mask=mask)
        loss_unmasked = loss_fn(y_pred, y_true)

        # Masked loss should be different (unless by chance)
        assert not torch.isclose(loss_masked, loss_unmasked, rtol=1e-3)

    def test_zero_loss_on_perfect_prediction(self):
        """Test that loss is zero when prediction matches target."""
        loss_fn = MultiHorizonLoss(H=40, weight_scheme="uniform")

        y = torch.randn(32, 41, 3)

        loss = loss_fn(y, y)

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_invalid_weight_scheme_raises(self):
        """Test that invalid weight scheme raises error."""
        with pytest.raises(ValueError, match="Unknown weight scheme"):
            MultiHorizonLoss(H=40, weight_scheme="invalid")

    def test_wrong_horizon_raises(self):
        """Test that wrong horizon dimension raises error."""
        loss_fn = MultiHorizonLoss(H=40)

        y_pred = torch.randn(32, 30, 3)  # Wrong H+1 dimension
        y_true = torch.randn(32, 30, 3)

        with pytest.raises(ValueError, match="Expected H\\+1=41 horizons"):
            loss_fn(y_pred, y_true)

    def test_wrong_n_obs_raises(self):
        """Test that wrong number of observables raises error."""
        loss_fn = MultiHorizonLoss(H=40, per_variable_weights=[1.0, 1.0, 1.0])

        y_pred = torch.randn(32, 41, 5)  # Wrong n_obs
        y_true = torch.randn(32, 41, 5)

        with pytest.raises(ValueError, match="Expected 3 observables"):
            loss_fn(y_pred, y_true)


@pytest.mark.fast
class TestSmoothnessRegularization:
    """Tests for SmoothnessRegularization."""

    def test_smoothness_penalty_shape(self):
        """Test that smoothness penalty produces scalar output."""
        smooth_loss = SmoothnessRegularization(lambda_smooth=0.01)

        y_pred = torch.randn(32, 41, 3)

        loss = smooth_loss(y_pred)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_oscillatory_input_higher_penalty(self):
        """Test that oscillatory inputs have higher smoothness penalty."""
        smooth_loss = SmoothnessRegularization(lambda_smooth=1.0)

        # Smooth trajectory (linear)
        t = torch.linspace(0, 1, 41).unsqueeze(-1).unsqueeze(0).expand(1, 41, 3)
        y_smooth = t

        # Oscillatory trajectory (sine wave)
        y_oscillatory = torch.sin(torch.linspace(0, 10 * np.pi, 41)).unsqueeze(-1).unsqueeze(0).expand(1, 41, 3)

        loss_smooth = smooth_loss(y_smooth)
        loss_oscillatory = smooth_loss(y_oscillatory)

        # Oscillatory should have higher penalty
        assert loss_oscillatory > loss_smooth

    def test_constant_input_zero_penalty(self):
        """Test that constant inputs have zero smoothness penalty."""
        smooth_loss = SmoothnessRegularization(lambda_smooth=1.0)

        # Constant trajectory
        y_constant = torch.ones(32, 41, 3)

        loss = smooth_loss(y_constant)

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_linear_input_zero_penalty(self):
        """Test that linear inputs have zero smoothness penalty."""
        smooth_loss = SmoothnessRegularization(lambda_smooth=1.0)

        # Linear trajectory
        t = torch.linspace(0, 1, 41).unsqueeze(-1).unsqueeze(0).expand(1, 41, 3)
        y_linear = 2.0 * t + 1.0

        loss = smooth_loss(y_linear)

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_multi_shock_input(self):
        """Test smoothness penalty with multi-shock input."""
        smooth_loss = SmoothnessRegularization(lambda_smooth=0.01)

        y_pred = torch.randn(32, 5, 41, 3)

        loss = smooth_loss(y_pred)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_mask_functionality(self):
        """Test that mask properly filters smoothness penalty."""
        smooth_loss = SmoothnessRegularization(lambda_smooth=1.0)

        y_pred = torch.randn(32, 5, 41, 3)

        # Mask only first 2 shocks
        mask = torch.zeros(32, 5)
        mask[:, :2] = 1.0

        loss_masked = smooth_loss(y_pred, mask=mask)
        loss_unmasked = smooth_loss(y_pred)

        # Masked loss should be different
        assert not torch.isclose(loss_masked, loss_unmasked, rtol=1e-3)

    def test_lambda_scaling(self):
        """Test that lambda_smooth scales the penalty correctly."""
        y_pred = torch.randn(32, 41, 3)

        loss_1 = SmoothnessRegularization(lambda_smooth=1.0)(y_pred)
        loss_2 = SmoothnessRegularization(lambda_smooth=2.0)(y_pred)

        # Loss should scale linearly with lambda
        assert torch.isclose(loss_2, 2.0 * loss_1, rtol=1e-5)


@pytest.mark.fast
class TestCombinedLoss:
    """Tests for CombinedLoss."""

    def test_combined_loss_without_smoothness(self):
        """Test combined loss with lambda_smooth=0."""
        loss_fn = CombinedLoss(H=40, weight_scheme="uniform", lambda_smooth=0.0)

        y_pred = torch.randn(32, 41, 3)
        y_true = torch.randn(32, 41, 3)

        result = loss_fn(y_pred, y_true)

        assert "loss" in result
        assert "horizon_loss" in result
        assert "smoothness_loss" not in result
        assert result["loss"] == result["horizon_loss"]

    def test_combined_loss_with_smoothness(self):
        """Test combined loss with smoothness penalty."""
        loss_fn = CombinedLoss(
            H=40, weight_scheme="uniform", lambda_smooth=0.01
        )

        y_pred = torch.randn(32, 41, 3)
        y_true = torch.randn(32, 41, 3)

        result = loss_fn(y_pred, y_true)

        assert "loss" in result
        assert "horizon_loss" in result
        assert "smoothness_loss" in result

        # Total loss should be sum of components
        expected_total = result["horizon_loss"] + result["smoothness_loss"]
        assert torch.isclose(result["loss"], expected_total, rtol=1e-5)

    def test_combined_loss_components_positive(self):
        """Test that all loss components are non-negative."""
        loss_fn = CombinedLoss(
            H=40, weight_scheme="exponential", tau=20.0, lambda_smooth=0.01
        )

        y_pred = torch.randn(32, 41, 3)
        y_true = torch.randn(32, 41, 3)

        result = loss_fn(y_pred, y_true)

        assert result["loss"].item() >= 0
        assert result["horizon_loss"].item() >= 0
        assert result["smoothness_loss"].item() >= 0

    def test_combined_loss_multi_shock(self):
        """Test combined loss with multi-shock input."""
        loss_fn = CombinedLoss(H=40, weight_scheme="uniform", lambda_smooth=0.01)

        y_pred = torch.randn(32, 5, 41, 3)
        y_true = torch.randn(32, 5, 41, 3)

        result = loss_fn(y_pred, y_true)

        assert result["loss"].item() >= 0

    def test_combined_loss_with_mask(self):
        """Test combined loss with mask."""
        loss_fn = CombinedLoss(H=40, weight_scheme="uniform", lambda_smooth=0.01)

        y_pred = torch.randn(32, 5, 41, 3)
        y_true = torch.randn(32, 5, 41, 3)

        mask = torch.zeros(32, 5)
        mask[:, :2] = 1.0

        result = loss_fn(y_pred, y_true, mask=mask)

        assert result["loss"].item() >= 0
        assert "horizon_loss" in result
        assert "smoothness_loss" in result


@pytest.mark.fast
class TestLossGradients:
    """Test that losses support gradient computation."""

    def test_multihorizon_loss_gradients(self):
        """Test that MultiHorizonLoss produces gradients."""
        loss_fn = MultiHorizonLoss(H=40, weight_scheme="uniform")

        y_pred = torch.randn(32, 41, 3, requires_grad=True)
        y_true = torch.randn(32, 41, 3)

        loss = loss_fn(y_pred, y_true)
        loss.backward()

        assert y_pred.grad is not None
        assert y_pred.grad.shape == y_pred.shape

    def test_smoothness_loss_gradients(self):
        """Test that SmoothnessRegularization produces gradients."""
        smooth_loss = SmoothnessRegularization(lambda_smooth=0.01)

        y_pred = torch.randn(32, 41, 3, requires_grad=True)

        loss = smooth_loss(y_pred)
        loss.backward()

        assert y_pred.grad is not None
        assert y_pred.grad.shape == y_pred.shape

    def test_combined_loss_gradients(self):
        """Test that CombinedLoss produces gradients."""
        loss_fn = CombinedLoss(H=40, weight_scheme="uniform", lambda_smooth=0.01)

        y_pred = torch.randn(32, 41, 3, requires_grad=True)
        y_true = torch.randn(32, 41, 3)

        result = loss_fn(y_pred, y_true)
        result["loss"].backward()

        assert y_pred.grad is not None
        assert y_pred.grad.shape == y_pred.shape


@pytest.mark.fast
class TestLossNumericalStability:
    """Test numerical stability of loss functions."""

    def test_multihorizon_loss_no_nan(self):
        """Test that MultiHorizonLoss doesn't produce NaN."""
        loss_fn = MultiHorizonLoss(H=40, weight_scheme="exponential", tau=20.0)

        # Extreme values
        y_pred = torch.randn(32, 41, 3) * 1000
        y_true = torch.randn(32, 41, 3) * 1000

        loss = loss_fn(y_pred, y_true)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_smoothness_loss_no_nan(self):
        """Test that SmoothnessRegularization doesn't produce NaN."""
        smooth_loss = SmoothnessRegularization(lambda_smooth=0.01)

        # Extreme values
        y_pred = torch.randn(32, 41, 3) * 1000

        loss = smooth_loss(y_pred)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_loss_with_zeros(self):
        """Test losses handle zero inputs correctly."""
        loss_fn = MultiHorizonLoss(H=40, weight_scheme="uniform")
        smooth_loss = SmoothnessRegularization(lambda_smooth=0.01)

        y_zero = torch.zeros(32, 41, 3)

        horizon_loss = loss_fn(y_zero, y_zero)
        smoothness = smooth_loss(y_zero)

        assert torch.isclose(horizon_loss, torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(smoothness, torch.tensor(0.0), atol=1e-6)
