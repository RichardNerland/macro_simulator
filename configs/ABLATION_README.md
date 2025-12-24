# Ablation Study Configurations

This directory contains configurations for ablation studies that measure the contribution of different input components to the Universal Emulator's performance.

## Overview

Each ablation config removes one input component by setting its embedding dimension to 0. This allows us to quantify how much each component contributes to the model's ability to predict IRFs.

## Ablation Variants

### 1. `ablation_no_world_id.yaml` - Remove World Embedding

**What it tests**: The contribution of world_id (simulator family identification)

**Configuration**:
- `world_embed_dim: 0` (disabled)
- All other inputs enabled: theta, shock_token, eps_sequence
- Uses Regime A

**Hypothesis**: Without world_id, the model must infer the simulator family from parameters alone. Performance should degrade, especially when parameters don't uniquely identify the world.

**Usage**:
```bash
python train_universal.py --config configs/ablation_no_world_id.yaml --epochs 10
```

---

### 2. `ablation_no_theta.yaml` - Remove Parameter Encoding

**What it tests**: The contribution of parameter (theta) encoding

**Configuration**:
- `theta_embed_dim: 0` (disabled)
- All other inputs enabled: world_id, shock_token, eps_sequence
- Uses Regime A

**Hypothesis**: Without parameters, the model must rely on world_id to provide "average" IRFs for each simulator family. Performance should degrade significantly, especially for worlds with high parameter heterogeneity.

**Usage**:
```bash
python train_universal.py --config configs/ablation_no_theta.yaml --epochs 10
```

---

### 3. `ablation_no_eps.yaml` - Remove Shock Sequence

**What it tests**: The contribution of eps_sequence (full shock path)

**Configuration**:
- Uses Regime C (which doesn't provide eps_sequence)
- `eps_embed_dim: 0` (not used in Regime C)
- All other inputs enabled: world_id, theta, shock_token, history
- Requires `history_embed_dim: 64` (Regime C needs history encoder)

**Hypothesis**: Without the full shock sequence, the model must predict IRFs from parameters and observable history alone. This tests whether knowing the exact shock path is crucial or if structural parameters are sufficient.

**Usage**:
```bash
python train_universal.py --config configs/ablation_no_eps.yaml --epochs 10
```

---

## Implementation Details

### Model Modifications

The `UniversalEmulator` model was modified to support ablations by:

1. **Conditional component creation**: Components with `embed_dim=0` are not instantiated (set to `None`)
2. **Conditional forward pass**: Only enabled components are used in forward pass
3. **Trunk input dimension**: Automatically adjusted based on enabled components

### Code Changes

```python
# In UniversalEmulator.__init__():
if world_embed_dim > 0:
    self.world_embedding = WorldEmbedding(world_ids, world_embed_dim)
else:
    self.world_embedding = None

# In forward():
embeddings = []
if self.world_embedding is not None:
    embeddings.append(self.world_embedding(world_id))
# ... similar for other components
combined = torch.cat(embeddings, dim=-1)
```

## Running Ablation Studies

### Quick Test (10 epochs)

All ablation configs are configured for quick testing with 10 epochs:

```bash
# Run all ablations
for config in configs/ablation_*.yaml; do
    python train_universal.py --config $config --epochs 10
done
```

### Full Training (100 epochs)

To run full ablation training, override the epochs parameter:

```bash
python train_universal.py --config configs/ablation_no_world_id.yaml --epochs 100
```

## Evaluation

After training, compare ablation models against the full model:

```bash
# Evaluate full model (baseline)
python -m emulator.eval.evaluate \
    --checkpoint runs/smoke_test_regime_A/best_model.pt \
    --dataset datasets/v1.0-dev/ \
    --output results/full_model.json

# Evaluate ablation models
python -m emulator.eval.evaluate \
    --checkpoint runs/ablation_no_world_id/best_model.pt \
    --dataset datasets/v1.0-dev/ \
    --output results/ablation_no_world_id.json

python -m emulator.eval.evaluate \
    --checkpoint runs/ablation_no_theta/best_model.pt \
    --dataset datasets/v1.0-dev/ \
    --output results/ablation_no_theta.json

python -m emulator.eval.evaluate \
    --checkpoint runs/ablation_no_eps/best_model.pt \
    --dataset datasets/v1.0-dev/ \
    --output results/ablation_no_eps.json
```

## Expected Results

### Metrics to Compare

1. **IRF MSE**: Primary metric - mean squared error on IRF predictions
2. **Per-world performance**: How does ablation affect different simulator families?
3. **Parameter count**: Models with disabled components have fewer parameters

### Example Analysis

```python
import json

# Load results
with open('results/full_model.json') as f:
    full = json.load(f)

with open('results/ablation_no_world_id.json') as f:
    no_world = json.load(f)

# Compare performance degradation
degradation = (no_world['irf_mse'] - full['irf_mse']) / full['irf_mse']
print(f"Performance degradation without world_id: {degradation:.2%}")
```

## Dataset Requirements

All ablation configs use the same dataset:
- Path: `datasets/v1.0-dev/`
- Worlds: lss, var, nk
- Splits: train, val, test_interpolation

## Configuration Parameters

All ablation configs share these settings (based on smoke test config):

```yaml
training:
  batch_size: 32
  lr: 1.0e-4
  epochs: 10  # Quick test, increase for full training
  warmup_epochs: 1

model:
  trunk_dim: 128
  trunk_layers: 2
  dropout: 0.1
  H: 40
  n_obs: 3

loss:
  type: multi_horizon
  weight_scheme: exponential
  tau: 20.0
  smoothness_lambda: 0.01

seed: 42
device: cpu
```

## Testing Ablation Configs

To verify all ablation configs are valid:

```bash
python test_ablation_configs.py
```

This tests:
1. Config loading
2. Model instantiation with disabled components
3. Forward pass with correct regime inputs
4. Output shape validation

## Notes

### Regime Selection for Ablations

- **No world_id, no theta**: Use Regime A (has all other inputs)
- **No eps**: Must use Regime C (Regime A requires eps, but C doesn't)

### Shock Token vs Eps Sequence

Important distinction:
- **shock_token**: Specifies WHICH IRF to compute (shock index, size, timing)
- **eps_sequence**: Full shock path for the trajectory (Regime A only)

The `ablation_no_eps` study removes eps_sequence (not shock_token), testing whether the model needs the full shock path or can predict IRFs from parameters alone.

### Model Parameter Counts

Approximate parameter counts (based on quick test):
- Full model (smoke test): ~125k parameters
- No world_id: ~109k parameters (-13%)
- No theta: ~86k parameters (-31%)
- No eps: N/A (Regime C uses history encoder, different architecture)

## Future Extensions

Potential additional ablations:
1. **No shock_token**: Test if model can predict all IRFs jointly
2. **No history (Regime C)**: Compare Regime C vs Regime A
3. **Combined ablations**: Remove multiple components simultaneously
4. **Architecture ablations**: Vary trunk_dim, trunk_layers, etc.
