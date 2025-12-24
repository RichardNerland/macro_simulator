# ShockToken and InformationRegime

This document explains the ShockToken dataclass and InformationRegime enum that are central to the Universal Macro Emulator's design.

## Overview

The emulator operates across different **information regimes** that control what inputs are available during training and inference. The **ShockToken** is a critical component that identifies which impulse response function (IRF) to compute.

## ShockToken

### Purpose

A `ShockToken` specifies **which** IRF to compute, distinct from `eps_sequence` (the full shock path).

From spec section 6.1:
- **shock_token**: Identifies which shock IRF is queried (always provided for IRF tasks)
- **eps_sequence**: Full innovation sequence (only available in Regime A)

### Structure

```python
@dataclass
class ShockToken:
    shock_idx: int      # Which shock variable (0, 1, 2, ...)
    shock_size: float   # Size in std dev units (typically 1.0)
    shock_timing: int   # When shock hits (typically 0 for IRFs)
```

### Usage

```python
from emulator.models.tokens import ShockToken, batch_shock_tokens

# Create a token for shock to first variable
token = ShockToken(shock_idx=0, shock_size=1.0, shock_timing=0)

# Convert to tensor for model input
tensor = token.to_tensor()  # shape: (3,)

# Batch multiple tokens
tokens = [ShockToken(i, 1.0, 0) for i in range(3)]
batch = batch_shock_tokens(tokens)  # shape: (3, 3)

# Roundtrip serialization
recovered = ShockToken.from_tensor(tensor)
```

## InformationRegime

### Regime Definitions

From spec section 6.2:

| Regime | world_id | theta | shock_token | eps_sequence | history | Use Case |
|--------|----------|-------|-------------|--------------|---------|----------|
| **A** | ✓ | ✓ | ✓ | ✓ | Optional | Full structural assist, upper bound |
| **B1** | ✓ | ✗ | ✓ | ✗ | ✓ | Econometrician within model class |
| **C** | ✓ | ✓ | ✓ | ✗ | ✓ | Calibrated model, unknown shocks |

### Properties

```python
from emulator.models.tokens import InformationRegime

# Regime A: Full structural information
regime_a = InformationRegime.A
assert regime_a.uses_theta is True
assert regime_a.uses_eps is True
assert regime_a.uses_history is False

# Regime B1: Observables only
regime_b1 = InformationRegime.B1
assert regime_b1.uses_theta is False
assert regime_b1.uses_eps is False
assert regime_b1.uses_history is True

# Regime C: Partial structural information
regime_c = InformationRegime.C
assert regime_c.uses_theta is True
assert regime_c.uses_eps is False
assert regime_c.uses_history is True
```

### Required Inputs

```python
# Get required inputs for a regime
regime = InformationRegime.A
inputs = regime.required_inputs()
# {'world_id', 'theta', 'shock_token', 'eps_sequence'}

# Validate inputs
from emulator.models.tokens import validate_regime_inputs
import torch

validate_regime_inputs(
    regime=InformationRegime.A,
    world_id=0,
    theta=torch.randn(5),
    shock_token=ShockToken(shock_idx=0),
    eps_sequence=torch.randn(40, 3),
)  # Passes

validate_regime_inputs(
    regime=InformationRegime.B1,
    world_id=0,
    shock_token=ShockToken(shock_idx=0),
    history=torch.randn(20, 3),
)  # Passes (no theta/eps needed)
```

## Training Configuration

The `UniversalTrainingConfig` includes regime settings:

```python
from emulator.training.config import UniversalTrainingConfig

# Load regime-specific config
config = UniversalTrainingConfig.from_yaml("configs/examples/universal_regime_A.yaml")
print(config.regime)  # InformationRegime.A
print(config.get_regime_inputs())  # {'world_id', 'theta', 'shock_token', 'eps_sequence'}

# Create custom config
config = UniversalTrainingConfig(
    regime=InformationRegime.C,
    batch_size=128,
    lr=1e-4,
    worlds=["lss", "var", "nk"],
)
```

## Example Configs

Pre-configured example configs are provided in `configs/examples/`:
- `universal_regime_A.yaml` - Full structural assist
- `universal_regime_B1.yaml` - Observables only
- `universal_regime_C.yaml` - Partial structural
- `baseline_mlp.yaml` - Per-world MLP baseline

Generate these with:
```python
from emulator.training.config import create_example_configs
create_example_configs("configs/examples")
```

## Key Distinctions

### ShockToken vs eps_sequence

- **ShockToken**: Which IRF is being requested (always provided)
  - Example: "Compute IRF for monetary policy shock (idx=1) of size 1 std dev"

- **eps_sequence**: Full path of all shocks over time (regime-dependent)
  - Example: `eps[t, k]` for `t=0..T` and all shock types `k`
  - Only available in Regime A

### Regime Selection Guide

- **Regime A**: Use when you have full model specification and shock realizations
  - Training the universal emulator with maximum information
  - Provides upper bound on performance

- **Regime B1**: Use when you only have observable data
  - Real-world forecasting scenario
  - Must infer structure from history alone

- **Regime C**: Use when you have model calibration but not shock realizations
  - Common in policy analysis
  - Must infer shocks from observable history

## Testing

All functionality is tested in:
- `emulator/tests/test_tokens.py` - ShockToken, InformationRegime, validation
- `emulator/tests/test_config.py` - Config loading, YAML serialization

Run tests:
```bash
pytest emulator/tests/test_tokens.py -v
pytest emulator/tests/test_config.py -v
```

## Demonstration

Run the demo script to see everything in action:
```bash
python -m examples.demo_tokens_and_regimes
```

## References

- Spec section 6.1: Shock Input Decomposition
- Spec section 6.2: Regime Definitions
- Spec section 6.3: Task-Specific Input Signatures
