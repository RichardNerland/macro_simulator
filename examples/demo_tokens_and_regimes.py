"""Demonstration of ShockToken and InformationRegime usage.

This script demonstrates:
1. Creating and using ShockTokens
2. Working with different InformationRegimes
3. Loading and using training configs
4. Validating regime-specific inputs
"""

import torch

from emulator.models.tokens import (
    InformationRegime,
    ShockToken,
    batch_shock_tokens,
    validate_regime_inputs,
)
from emulator.training.config import UniversalTrainingConfig


def demo_shock_tokens():
    """Demonstrate ShockToken creation and usage."""
    print("=" * 60)
    print("SHOCK TOKEN DEMONSTRATION")
    print("=" * 60)

    # Create shock tokens for different scenarios
    print("\n1. Creating ShockTokens:")

    # Standard IRF: 1 std dev shock to first variable at t=0
    token1 = ShockToken(shock_idx=0, shock_size=1.0, shock_timing=0)
    print(f"  Standard IRF: {token1}")

    # Larger shock to second variable
    token2 = ShockToken(shock_idx=1, shock_size=2.0, shock_timing=0)
    print(f"  Large shock: {token2}")

    # Delayed shock
    token3 = ShockToken(shock_idx=2, shock_size=1.0, shock_timing=5)
    print(f"  Delayed shock: {token3}")

    # Convert to tensors
    print("\n2. Converting to tensors:")
    tensor1 = token1.to_tensor()
    print(f"  Token1 as tensor: {tensor1}")
    print(f"  Shape: {tensor1.shape}, dtype: {tensor1.dtype}")

    # Batch multiple tokens
    print("\n3. Batching tokens:")
    tokens = [token1, token2, token3]
    batch = batch_shock_tokens(tokens)
    print(f"  Batch shape: {batch.shape}")
    print(f"  Batch:\n{batch}")

    # Roundtrip test
    print("\n4. Roundtrip serialization:")
    recovered = ShockToken.from_tensor(tensor1)
    print(f"  Original: {token1}")
    print(f"  Recovered: {recovered}")
    print(f"  Match: {token1.shock_idx == recovered.shock_idx and token1.shock_size == recovered.shock_size}")


def demo_information_regimes():
    """Demonstrate InformationRegime properties."""
    print("\n" + "=" * 60)
    print("INFORMATION REGIME DEMONSTRATION")
    print("=" * 60)

    regimes = [InformationRegime.A, InformationRegime.B1, InformationRegime.C]

    print("\n1. Regime Properties:")
    print(f"{'Regime':<10} {'uses_theta':<12} {'uses_eps':<12} {'uses_history':<15}")
    print("-" * 60)
    for regime in regimes:
        print(f"{regime.value:<10} {str(regime.uses_theta):<12} "
              f"{str(regime.uses_eps):<12} {str(regime.uses_history):<15}")

    print("\n2. Required Inputs:")
    for regime in regimes:
        inputs = regime.required_inputs()
        print(f"  {regime.value}: {sorted(inputs)}")

    print("\n3. Regime-specific use cases:")
    print("  Regime A: Full structural model, known shocks")
    print("           -> Upper bound performance, ideal for training")
    print("  Regime B1: Only observables, world known")
    print("           -> Econometrician within model class")
    print("  Regime C: Calibrated model, unknown shocks")
    print("           -> Must infer shocks from history")


def demo_input_validation():
    """Demonstrate regime input validation."""
    print("\n" + "=" * 60)
    print("INPUT VALIDATION DEMONSTRATION")
    print("=" * 60)

    # Valid Regime A inputs
    print("\n1. Valid Regime A inputs:")
    try:
        validate_regime_inputs(
            regime=InformationRegime.A,
            world_id=0,
            theta=torch.randn(5),
            shock_token=ShockToken(shock_idx=0),
            eps_sequence=torch.randn(40, 3),
        )
        print("  ✓ Validation passed")
    except ValueError as e:
        print(f"  ✗ Validation failed: {e}")

    # Invalid Regime A: missing eps_sequence
    print("\n2. Invalid Regime A (missing eps_sequence):")
    try:
        validate_regime_inputs(
            regime=InformationRegime.A,
            world_id=0,
            theta=torch.randn(5),
            shock_token=ShockToken(shock_idx=0),
            eps_sequence=None,  # Missing!
        )
        print("  ✓ Validation passed")
    except ValueError as e:
        print(f"  ✗ Validation failed (expected): Missing eps_sequence")

    # Valid Regime B1: no theta or eps needed
    print("\n3. Valid Regime B1 (no theta/eps):")
    try:
        validate_regime_inputs(
            regime=InformationRegime.B1,
            world_id=0,
            shock_token=ShockToken(shock_idx=0),
            history=torch.randn(20, 3),
        )
        print("  ✓ Validation passed")
    except ValueError as e:
        print(f"  ✗ Validation failed: {e}")


def demo_training_configs():
    """Demonstrate loading training configs."""
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION DEMONSTRATION")
    print("=" * 60)

    print("\n1. Loading Regime A config:")
    config_a = UniversalTrainingConfig.from_yaml("configs/examples/universal_regime_A.yaml")
    print(f"  Regime: {config_a.regime}")
    print(f"  Batch size: {config_a.batch_size}")
    print(f"  Learning rate: {config_a.lr}")
    print(f"  Worlds: {config_a.worlds}")
    print(f"  Required inputs: {config_a.get_regime_inputs()}")

    print("\n2. Loading Regime B1 config:")
    config_b1 = UniversalTrainingConfig.from_yaml("configs/examples/universal_regime_B1.yaml")
    print(f"  Regime: {config_b1.regime}")
    print(f"  Required inputs: {config_b1.get_regime_inputs()}")
    print(f"  Note: No theta or eps needed!")

    print("\n3. Creating custom config:")
    config_custom = UniversalTrainingConfig(
        regime=InformationRegime.C,
        batch_size=64,
        lr=5e-4,
        worlds=["lss", "var"],
        checkpoint_dir="runs/demo_regime_C",
    )
    print(f"  Regime: {config_custom.regime}")
    print(f"  Batch size: {config_custom.batch_size}")
    print(f"  Learning rate: {config_custom.lr}")
    print(f"  Worlds: {config_custom.worlds}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("UNIVERSAL MACRO EMULATOR: TOKENS & REGIMES DEMO")
    print("=" * 60)

    demo_shock_tokens()
    demo_information_regimes()
    demo_input_validation()
    demo_training_configs()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nKey takeaways:")
    print("1. ShockToken identifies WHICH IRF to compute (shock_idx, size, timing)")
    print("2. eps_sequence is the full shock path (regime-dependent)")
    print("3. Three information regimes control what inputs are available")
    print("4. Configs enforce regime-specific input requirements")
    print("5. All configs are YAML-serializable for reproducibility")


if __name__ == "__main__":
    main()
