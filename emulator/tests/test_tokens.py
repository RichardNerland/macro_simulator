"""Tests for ShockToken and InformationRegime.

Tests cover:
- ShockToken creation, validation, and serialization
- Batch conversion utilities
- InformationRegime properties
- Regime input validation
"""

import pytest
import torch

from emulator.models.tokens import (
    InformationRegime,
    ShockToken,
    batch_shock_tokens,
    validate_regime_inputs,
)


class TestShockToken:
    """Tests for ShockToken dataclass."""

    @pytest.mark.fast
    def test_creation_defaults(self):
        """Test ShockToken creation with default values."""
        token = ShockToken(shock_idx=0)
        assert token.shock_idx == 0
        assert token.shock_size == 1.0
        assert token.shock_timing == 0

    @pytest.mark.fast
    def test_creation_custom(self):
        """Test ShockToken creation with custom values."""
        token = ShockToken(shock_idx=2, shock_size=2.5, shock_timing=5)
        assert token.shock_idx == 2
        assert token.shock_size == 2.5
        assert token.shock_timing == 5

    @pytest.mark.fast
    def test_validation_shock_idx_negative(self):
        """Test that negative shock_idx raises ValueError."""
        with pytest.raises(ValueError, match="shock_idx must be in"):
            ShockToken(shock_idx=-1)

    @pytest.mark.fast
    def test_validation_shock_idx_too_large(self):
        """Test that shock_idx >= MAX_SHOCK_IDX raises ValueError."""
        with pytest.raises(ValueError, match="shock_idx must be in"):
            ShockToken(shock_idx=10)  # MAX_SHOCK_IDX is 10

    @pytest.mark.fast
    def test_validation_negative_size(self):
        """Test that negative shock_size raises ValueError."""
        with pytest.raises(ValueError, match="shock_size must be non-negative"):
            ShockToken(shock_idx=0, shock_size=-1.0)

    @pytest.mark.fast
    def test_validation_negative_timing(self):
        """Test that negative shock_timing raises ValueError."""
        with pytest.raises(ValueError, match="shock_timing must be in"):
            ShockToken(shock_idx=0, shock_timing=-1)

    @pytest.mark.fast
    def test_validation_timing_too_large(self):
        """Test that shock_timing >= MAX_TIMING raises ValueError."""
        with pytest.raises(ValueError, match="shock_timing must be in"):
            ShockToken(shock_idx=0, shock_timing=200)  # MAX_TIMING is 200

    @pytest.mark.fast
    def test_to_tensor(self):
        """Test conversion to tensor."""
        token = ShockToken(shock_idx=1, shock_size=2.0, shock_timing=3)
        tensor = token.to_tensor()

        assert tensor.shape == (3,)
        assert tensor.dtype == torch.float32
        assert tensor[0].item() == 1.0
        assert tensor[1].item() == 2.0
        assert tensor[2].item() == 3.0

    @pytest.mark.fast
    def test_to_tensor_device(self):
        """Test conversion to tensor with device specification."""
        token = ShockToken(shock_idx=0)
        device = torch.device("cpu")
        tensor = token.to_tensor(device=device)

        assert tensor.device == device

    @pytest.mark.fast
    def test_from_tensor(self):
        """Test creation from tensor."""
        tensor = torch.tensor([2.0, 1.5, 4.0])
        token = ShockToken.from_tensor(tensor)

        assert token.shock_idx == 2
        assert token.shock_size == 1.5
        assert token.shock_timing == 4

    @pytest.mark.fast
    def test_from_tensor_wrong_shape(self):
        """Test that from_tensor raises ValueError for wrong shape."""
        tensor = torch.tensor([1.0, 2.0])  # Shape (2,) instead of (3,)
        with pytest.raises(ValueError, match="Expected tensor of shape"):
            ShockToken.from_tensor(tensor)

    @pytest.mark.fast
    def test_roundtrip_serialization(self):
        """Test that to_tensor -> from_tensor is identity."""
        original = ShockToken(shock_idx=1, shock_size=2.0, shock_timing=3)
        tensor = original.to_tensor()
        recovered = ShockToken.from_tensor(tensor)

        assert recovered.shock_idx == original.shock_idx
        assert recovered.shock_size == original.shock_size
        assert recovered.shock_timing == original.shock_timing

    @pytest.mark.fast
    def test_repr(self):
        """Test string representation."""
        token = ShockToken(shock_idx=1, shock_size=2.0, shock_timing=3)
        repr_str = repr(token)
        assert "idx=1" in repr_str
        assert "size=2.00" in repr_str
        assert "t=3" in repr_str


class TestBatchShockTokens:
    """Tests for batch_shock_tokens utility."""

    @pytest.mark.fast
    def test_batch_single(self):
        """Test batching a single token."""
        tokens = [ShockToken(shock_idx=0)]
        batch = batch_shock_tokens(tokens)

        assert batch.shape == (1, 3)
        assert batch[0, 0].item() == 0.0
        assert batch[0, 1].item() == 1.0
        assert batch[0, 2].item() == 0.0

    @pytest.mark.fast
    def test_batch_multiple(self):
        """Test batching multiple tokens."""
        tokens = [
            ShockToken(shock_idx=0, shock_size=1.0, shock_timing=0),
            ShockToken(shock_idx=1, shock_size=2.0, shock_timing=5),
            ShockToken(shock_idx=2, shock_size=1.5, shock_timing=10),
        ]
        batch = batch_shock_tokens(tokens)

        assert batch.shape == (3, 3)
        # First token
        assert batch[0, 0].item() == 0.0
        assert batch[0, 1].item() == 1.0
        assert batch[0, 2].item() == 0.0
        # Second token
        assert batch[1, 0].item() == 1.0
        assert batch[1, 1].item() == 2.0
        assert batch[1, 2].item() == 5.0
        # Third token
        assert batch[2, 0].item() == 2.0
        assert batch[2, 1].item() == 1.5
        assert batch[2, 2].item() == 10.0

    @pytest.mark.fast
    def test_batch_empty(self):
        """Test that batching empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot batch empty list"):
            batch_shock_tokens([])

    @pytest.mark.fast
    def test_batch_device(self):
        """Test batching with device specification."""
        tokens = [ShockToken(shock_idx=0), ShockToken(shock_idx=1)]
        device = torch.device("cpu")
        batch = batch_shock_tokens(tokens, device=device)

        assert batch.device == device


class TestInformationRegime:
    """Tests for InformationRegime enum."""

    @pytest.mark.fast
    def test_regime_values(self):
        """Test that all expected regimes exist."""
        assert InformationRegime.A.value == "A"
        assert InformationRegime.B1.value == "B1"
        assert InformationRegime.C.value == "C"

    @pytest.mark.fast
    def test_regime_from_string(self):
        """Test creating regime from string."""
        regime = InformationRegime("A")
        assert regime == InformationRegime.A

        regime = InformationRegime("B1")
        assert regime == InformationRegime.B1

        regime = InformationRegime("C")
        assert regime == InformationRegime.C

    @pytest.mark.fast
    def test_regime_invalid_string(self):
        """Test that invalid string raises ValueError."""
        with pytest.raises(ValueError):
            InformationRegime("invalid")

    @pytest.mark.fast
    def test_uses_theta(self):
        """Test uses_theta property."""
        assert InformationRegime.A.uses_theta is True
        assert InformationRegime.B1.uses_theta is False
        assert InformationRegime.C.uses_theta is True

    @pytest.mark.fast
    def test_uses_eps(self):
        """Test uses_eps property."""
        assert InformationRegime.A.uses_eps is True
        assert InformationRegime.B1.uses_eps is False
        assert InformationRegime.C.uses_eps is False

    @pytest.mark.fast
    def test_uses_history(self):
        """Test uses_history property."""
        assert InformationRegime.A.uses_history is False
        assert InformationRegime.B1.uses_history is True
        assert InformationRegime.C.uses_history is True

    @pytest.mark.fast
    def test_uses_shock_token(self):
        """Test uses_shock_token property (always True)."""
        assert InformationRegime.A.uses_shock_token is True
        assert InformationRegime.B1.uses_shock_token is True
        assert InformationRegime.C.uses_shock_token is True

    @pytest.mark.fast
    def test_required_inputs_regime_a(self):
        """Test required inputs for Regime A."""
        inputs = InformationRegime.A.required_inputs()
        assert inputs == {"world_id", "theta", "shock_token", "eps_sequence"}

    @pytest.mark.fast
    def test_required_inputs_regime_b1(self):
        """Test required inputs for Regime B1."""
        inputs = InformationRegime.B1.required_inputs()
        assert inputs == {"world_id", "shock_token", "history"}

    @pytest.mark.fast
    def test_required_inputs_regime_c(self):
        """Test required inputs for Regime C."""
        inputs = InformationRegime.C.required_inputs()
        assert inputs == {"world_id", "theta", "shock_token", "history"}

    @pytest.mark.fast
    def test_repr(self):
        """Test string representation."""
        assert repr(InformationRegime.A) == "Regime.A"
        assert repr(InformationRegime.B1) == "Regime.B1"

    @pytest.mark.fast
    def test_str(self):
        """Test string conversion."""
        assert str(InformationRegime.A) == "A"
        assert str(InformationRegime.B1) == "B1"


class TestValidateRegimeInputs:
    """Tests for validate_regime_inputs function."""

    @pytest.mark.fast
    def test_valid_regime_a(self):
        """Test validation passes for valid Regime A inputs."""
        # Should not raise
        validate_regime_inputs(
            regime=InformationRegime.A,
            world_id=0,
            theta=torch.randn(5),
            shock_token=ShockToken(shock_idx=0),
            eps_sequence=torch.randn(40, 3),
        )

    @pytest.mark.fast
    def test_valid_regime_b1(self):
        """Test validation passes for valid Regime B1 inputs."""
        # Should not raise
        validate_regime_inputs(
            regime=InformationRegime.B1,
            world_id=0,
            shock_token=ShockToken(shock_idx=0),
            history=torch.randn(20, 3),
        )

    @pytest.mark.fast
    def test_valid_regime_c(self):
        """Test validation passes for valid Regime C inputs."""
        # Should not raise
        validate_regime_inputs(
            regime=InformationRegime.C,
            world_id=0,
            theta=torch.randn(5),
            shock_token=ShockToken(shock_idx=0),
            history=torch.randn(20, 3),
        )

    @pytest.mark.fast
    def test_missing_world_id(self):
        """Test validation fails when world_id is missing."""
        with pytest.raises(ValueError, match="world_id is required"):
            validate_regime_inputs(
                regime=InformationRegime.A,
                world_id=None,
                theta=torch.randn(5),
                shock_token=ShockToken(shock_idx=0),
                eps_sequence=torch.randn(40, 3),
            )

    @pytest.mark.fast
    def test_missing_shock_token(self):
        """Test validation fails when shock_token is missing."""
        with pytest.raises(ValueError, match="shock_token is required"):
            validate_regime_inputs(
                regime=InformationRegime.A,
                world_id=0,
                theta=torch.randn(5),
                shock_token=None,
                eps_sequence=torch.randn(40, 3),
            )

    @pytest.mark.fast
    def test_missing_theta_regime_a(self):
        """Test validation fails when theta is missing for Regime A."""
        with pytest.raises(ValueError, match="theta is required for regime A"):
            validate_regime_inputs(
                regime=InformationRegime.A,
                world_id=0,
                theta=None,
                shock_token=ShockToken(shock_idx=0),
                eps_sequence=torch.randn(40, 3),
            )

    @pytest.mark.fast
    def test_missing_eps_regime_a(self):
        """Test validation fails when eps_sequence is missing for Regime A."""
        with pytest.raises(ValueError, match="eps_sequence is required for regime A"):
            validate_regime_inputs(
                regime=InformationRegime.A,
                world_id=0,
                theta=torch.randn(5),
                shock_token=ShockToken(shock_idx=0),
                eps_sequence=None,
            )

    @pytest.mark.fast
    def test_missing_history_regime_b1(self):
        """Test validation fails when history is missing for Regime B1."""
        with pytest.raises(ValueError, match="history is required for regime B1"):
            validate_regime_inputs(
                regime=InformationRegime.B1,
                world_id=0,
                shock_token=ShockToken(shock_idx=0),
                history=None,
            )

    @pytest.mark.fast
    def test_missing_theta_regime_c(self):
        """Test validation fails when theta is missing for Regime C."""
        with pytest.raises(ValueError, match="theta is required for regime C"):
            validate_regime_inputs(
                regime=InformationRegime.C,
                world_id=0,
                theta=None,
                shock_token=ShockToken(shock_idx=0),
                history=torch.randn(20, 3),
            )

    @pytest.mark.fast
    def test_theta_not_required_regime_b1(self):
        """Test that theta is not required for Regime B1."""
        # Should not raise even though theta is None
        validate_regime_inputs(
            regime=InformationRegime.B1,
            world_id=0,
            theta=None,
            shock_token=ShockToken(shock_idx=0),
            history=torch.randn(20, 3),
        )

    @pytest.mark.fast
    def test_eps_not_required_regime_c(self):
        """Test that eps_sequence is not required for Regime C."""
        # Should not raise even though eps_sequence is None
        validate_regime_inputs(
            regime=InformationRegime.C,
            world_id=0,
            theta=torch.randn(5),
            shock_token=ShockToken(shock_idx=0),
            eps_sequence=None,
            history=torch.randn(20, 3),
        )

    @pytest.mark.fast
    def test_multiple_missing_inputs(self):
        """Test validation reports all missing inputs."""
        with pytest.raises(ValueError) as exc_info:
            validate_regime_inputs(
                regime=InformationRegime.A,
                world_id=None,
                theta=None,
                shock_token=None,
                eps_sequence=None,
            )

        error_msg = str(exc_info.value)
        assert "world_id is required" in error_msg
        assert "shock_token is required" in error_msg
        assert "theta is required" in error_msg
        assert "eps_sequence is required" in error_msg
