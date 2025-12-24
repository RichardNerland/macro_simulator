# Information Regime Quick Reference

## Regime Comparison

| Feature | Regime A | Regime B1 | Regime C |
|---------|----------|-----------|----------|
| **Name** | Full Structural Assist | Observables + World Known | Partial Structural |
| **world_id** | ✓ | ✓ | ✓ |
| **theta** | ✓ | ✗ | ✓ |
| **shock_token** | ✓ | ✓ | ✓ |
| **eps_sequence** | ✓ | ✗ | ✗ |
| **history** | Optional | ✓ | ✓ |
| **Use Case** | Upper bound, training | Real-world forecasting | Policy analysis |

## When to Use Each Regime

### Regime A: Full Structural Assist
**Use when:** You have complete model specification and shock realizations

**Typical scenarios:**
- Training universal emulator with maximum information
- Establishing upper bound on performance
- Controlled experiments with known shocks

**Input requirements:**
```python
{
    'world_id': int,           # Simulator identifier (0-5)
    'theta': Tensor,           # Parameter vector (n_params,)
    'shock_token': ShockToken, # Which IRF to compute
    'eps_sequence': Tensor,    # Full shock path (T, n_shocks)
}
```

**Example:**
```python
config = UniversalTrainingConfig(
    regime=InformationRegime.A,
    worlds=["lss", "var", "nk", "rbc", "switching", "zlb"],
)
```

---

### Regime B1: Observables + World Known
**Use when:** You only have observable data (no structural parameters or shocks)

**Typical scenarios:**
- Real-world forecasting with historical data
- Econometrician within a known model class
- Data-driven policy analysis

**Input requirements:**
```python
{
    'world_id': int,           # Simulator identifier (0-5)
    'shock_token': ShockToken, # Which IRF to compute
    'history': Tensor,         # Observable history (k, n_obs)
}
```

**Example:**
```python
config = UniversalTrainingConfig(
    regime=InformationRegime.B1,
    worlds=["lss", "var", "nk"],
)
```

---

### Regime C: Partial Structural
**Use when:** You have model calibration but not shock realizations

**Typical scenarios:**
- Policy counterfactuals with calibrated models
- Model-based forecasting with parameter estimates
- Inference of shocks from observables

**Input requirements:**
```python
{
    'world_id': int,           # Simulator identifier (0-5)
    'theta': Tensor,           # Parameter vector (n_params,)
    'shock_token': ShockToken, # Which IRF to compute
    'history': Tensor,         # Observable history (k, n_obs)
}
```

**Example:**
```python
config = UniversalTrainingConfig(
    regime=InformationRegime.C,
    worlds=["nk", "rbc", "zlb"],
)
```

## ShockToken Structure

All regimes use `ShockToken` to specify which IRF to compute:

```python
ShockToken(
    shock_idx=0,     # Which shock (0=output, 1=inflation, 2=rate, etc.)
    shock_size=1.0,  # Size in std dev units
    shock_timing=0,  # When shock hits (0 for standard IRF)
)
```

**Key distinction:** ShockToken tells the model **which** IRF to compute, while `eps_sequence` (Regime A only) provides the **full shock path**.

## Code Examples

### Regime A: Full Information
```python
from emulator.models import ShockToken, InformationRegime
from emulator.training import UniversalTrainingConfig

# Create config
config = UniversalTrainingConfig.from_yaml("configs/examples/universal_regime_A.yaml")

# Prepare inputs
world_id = 0  # LSS simulator
theta = torch.randn(10)  # Parameters
shock_token = ShockToken(shock_idx=1, shock_size=1.0)  # Monetary shock
eps_sequence = torch.randn(40, 3)  # Full shock path

# Model forward pass (pseudo-code)
irf = model(
    world_id=world_id,
    theta=theta,
    shock_token=shock_token,
    eps_sequence=eps_sequence,
)
```

### Regime B1: Observables Only
```python
# Create config
config = UniversalTrainingConfig.from_yaml("configs/examples/universal_regime_B1.yaml")

# Prepare inputs
world_id = 0  # LSS simulator
shock_token = ShockToken(shock_idx=1, shock_size=1.0)  # Monetary shock
history = torch.randn(20, 3)  # Last 20 periods of observables

# Model forward pass (pseudo-code)
irf = model(
    world_id=world_id,
    shock_token=shock_token,
    history=history,
)
# Note: No theta or eps_sequence!
```

### Regime C: Partial Information
```python
# Create config
config = UniversalTrainingConfig.from_yaml("configs/examples/universal_regime_C.yaml")

# Prepare inputs
world_id = 2  # NK simulator
theta = torch.randn(15)  # Calibrated parameters
shock_token = ShockToken(shock_idx=0, shock_size=1.0)  # Technology shock
history = torch.randn(20, 3)  # Last 20 periods of observables

# Model forward pass (pseudo-code)
irf = model(
    world_id=world_id,
    theta=theta,
    shock_token=shock_token,
    history=history,
)
# Note: No eps_sequence (must infer shocks from history)
```

## Validation

Use `validate_regime_inputs()` to check inputs:

```python
from emulator.models.tokens import validate_regime_inputs

# This will raise ValueError if inputs are invalid
validate_regime_inputs(
    regime=InformationRegime.B1,
    world_id=0,
    shock_token=ShockToken(shock_idx=0),
    history=torch.randn(20, 3),
)
```

## Configuration Files

Load pre-configured examples:
```bash
configs/examples/
├── universal_regime_A.yaml    # Full structural
├── universal_regime_B1.yaml   # Observables only
├── universal_regime_C.yaml    # Partial structural
└── baseline_mlp.yaml          # Baseline model
```

## Training Commands

```bash
# Train in Regime A (full information)
python -m emulator.training.trainer --config configs/examples/universal_regime_A.yaml

# Train in Regime B1 (observables only)
python -m emulator.training.trainer --config configs/examples/universal_regime_B1.yaml

# Train in Regime C (partial information)
python -m emulator.training.trainer --config configs/examples/universal_regime_C.yaml
```

## Expected Performance

From spec section 7.2:

| Metric | Target |
|--------|--------|
| Universal vs Baselines | Beat on all worlds |
| Universal vs Specialists | Mean gap ≤ 20%, max gap ≤ 35% |
| Shape preservation | HF-ratio ≤ 1.1× specialist |

**Regime ordering** (expected performance):
- Regime A > Regime C > Regime B1
- Regime A provides upper bound (most information)
- Regime B1 is most challenging (least information)

## Further Reading

- Full spec: `spec/spec.md` section 6 (Information Regimes)
- Implementation: `emulator/models/tokens.py`
- Tests: `emulator/tests/test_tokens.py`, `emulator/tests/test_config.py`
- Demo: `python -m examples.demo_tokens_and_regimes`
