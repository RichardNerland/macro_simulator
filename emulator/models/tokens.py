"""Token representations for the Universal Macro Emulator.

This module defines:
- ShockToken: Identifies which IRF to compute (shock index, size, timing)
- InformationRegime: Enum defining what inputs are available
- Batch utilities for converting tokens to tensors
"""

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import torch


@dataclass
class ShockToken:
    """Token representing which IRF to compute.

    This is distinct from eps_sequence which is the full shock path.
    ShockToken tells the model: "compute the IRF for shock k hitting at time t with size s"

    Key distinction from spec section 6.1:
    - shock_token is always provided for IRF tasks (tells model which IRF is requested)
    - eps_sequence is regime-dependent (full innovation path, optional)

    Attributes:
        shock_idx: Index of shock variable (0, 1, or 2 for 3-shock systems)
        shock_size: Size of shock in std dev units (typically 1.0)
        shock_timing: When shock hits (typically 0 for IRF computation)
    """
    shock_idx: int
    shock_size: float = 1.0
    shock_timing: int = 0

    # Class-level constants for validation
    MAX_SHOCK_IDX: ClassVar[int] = 10  # Support up to 10 shock types
    MAX_TIMING: ClassVar[int] = 200  # Max timing for shock

    def __post_init__(self):
        """Validate token values."""
        if not (0 <= self.shock_idx < self.MAX_SHOCK_IDX):
            raise ValueError(
                f"shock_idx must be in [0, {self.MAX_SHOCK_IDX}), got {self.shock_idx}"
            )
        if self.shock_size < 0:
            raise ValueError(f"shock_size must be non-negative, got {self.shock_size}")
        if not (0 <= self.shock_timing < self.MAX_TIMING):
            raise ValueError(
                f"shock_timing must be in [0, {self.MAX_TIMING}), got {self.shock_timing}"
            )

    def to_tensor(self, device: torch.device | None = None) -> torch.Tensor:
        """Convert to tensor for model input.

        Args:
            device: Target device for tensor

        Returns:
            Tensor of shape (3,) with [shock_idx, shock_size, shock_timing]
        """
        t = torch.tensor(
            [self.shock_idx, self.shock_size, self.shock_timing],
            dtype=torch.float32
        )
        if device is not None:
            t = t.to(device)
        return t

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "ShockToken":
        """Create ShockToken from tensor.

        Args:
            tensor: Tensor of shape (3,) with [shock_idx, shock_size, shock_timing]

        Returns:
            ShockToken instance
        """
        if tensor.shape != (3,):
            raise ValueError(f"Expected tensor of shape (3,), got {tensor.shape}")
        return cls(
            shock_idx=int(tensor[0].item()),
            shock_size=float(tensor[1].item()),
            shock_timing=int(tensor[2].item()),
        )

    def __repr__(self) -> str:
        return f"ShockToken(idx={self.shock_idx}, size={self.shock_size:.2f}, t={self.shock_timing})"


def batch_shock_tokens(
    tokens: list[ShockToken],
    device: torch.device | None = None
) -> torch.Tensor:
    """Convert list of ShockTokens to batched tensor.

    Args:
        tokens: List of ShockToken instances
        device: Target device

    Returns:
        Tensor of shape (batch, 3)

    Example:
        >>> tokens = [ShockToken(0, 1.0, 0), ShockToken(1, 2.0, 0)]
        >>> batch = batch_shock_tokens(tokens)
        >>> batch.shape
        torch.Size([2, 3])
    """
    if not tokens:
        raise ValueError("Cannot batch empty list of tokens")
    return torch.stack([t.to_tensor(device) for t in tokens])


class InformationRegime(str, Enum):
    """Information regimes for the emulator.

    Defines what inputs are available during training/inference.
    Based on spec section 6.2.

    Attributes:
        A: Full structural assist (world_id, theta, eps_sequence, shock_token)
        B1: Observables + world known (world_id, shock_token, history, no theta/eps)
        C: Partial (world_id, theta, shock_token, history, no eps)
    """
    A = "A"    # Full: world_id, theta, eps_sequence, shock_token
    B1 = "B1"  # Observables: world_id, shock_token, history (no theta, no eps)
    C = "C"    # Partial: world_id, theta, shock_token, history (no eps)

    @property
    def uses_theta(self) -> bool:
        """Whether this regime uses theta (parameter vector)."""
        return self in (InformationRegime.A, InformationRegime.C)

    @property
    def uses_eps(self) -> bool:
        """Whether this regime uses eps_sequence (full shock path)."""
        return self == InformationRegime.A

    @property
    def uses_history(self) -> bool:
        """Whether this regime uses observable history."""
        return self in (InformationRegime.B1, InformationRegime.C)

    @property
    def uses_shock_token(self) -> bool:
        """Whether this regime uses shock_token (always True for IRF tasks)."""
        return True  # All regimes use shock_token for IRF prediction

    def required_inputs(self) -> set[str]:
        """Get set of required input names for this regime.

        Returns:
            Set of input names like {"world_id", "theta", "shock_token", ...}
        """
        inputs = {"world_id", "shock_token"}  # Always required
        if self.uses_theta:
            inputs.add("theta")
        if self.uses_eps:
            inputs.add("eps_sequence")
        if self.uses_history:
            inputs.add("history")
        return inputs

    def __repr__(self) -> str:
        return f"Regime.{self.value}"

    def __str__(self) -> str:
        return self.value


# For backward compatibility and convenience
Regime = InformationRegime


def validate_regime_inputs(
    regime: InformationRegime,
    world_id: int | None = None,
    theta: torch.Tensor | None = None,
    shock_token: ShockToken | None = None,
    eps_sequence: torch.Tensor | None = None,
    history: torch.Tensor | None = None,
) -> None:
    """Validate that required inputs are provided for a given regime.

    Args:
        regime: Information regime
        world_id: World identifier
        theta: Parameter vector
        shock_token: Shock token
        eps_sequence: Full shock sequence
        history: Observable history

    Raises:
        ValueError: If required inputs are missing
    """
    errors = []

    if world_id is None:
        errors.append("world_id is required for all regimes")

    if shock_token is None:
        errors.append("shock_token is required for IRF prediction")

    if regime.uses_theta and theta is None:
        errors.append(f"theta is required for regime {regime.value}")

    if regime.uses_eps and eps_sequence is None:
        errors.append(f"eps_sequence is required for regime {regime.value}")

    if regime.uses_history and history is None:
        errors.append(f"history is required for regime {regime.value}")

    if errors:
        raise ValueError(f"Invalid inputs for regime {regime.value}:\n" + "\n".join(f"  - {e}" for e in errors))
