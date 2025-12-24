# Universal Emulator - Quick Reference

## Overview

The `UniversalEmulator` is a neural network that predicts impulse response functions (IRFs) across all 6 simulator worlds: LSS, VAR, NK, RBC, regime-switching, and ZLB.

**Key Features:**
- Single model for all worlds (no per-world specialists)
- Supports 3 information regimes (A, B1, C)
- Handles variable-length parameter vectors
- Optional history conditioning
- ~609k parameters in default configuration

## Quick Start

```python
from emulator.models import UniversalEmulator
import torch

# 1. Define world configuration
world_ids = ["lss", "var", "nk", "rbc", "switching", "zlb"]
param_dims = {
    "lss": 15,
    "var": 12,
    "nk": 10,
    "rbc": 8,
    "switching": 25,
    "zlb": 12,
}

# 2. Create model
model = UniversalEmulator(
    world_ids=world_ids,
    param_dims=param_dims,
    H=40,  # IRF horizon
    use_history_encoder=True,  # For regimes B1/C
)

# 3. Forward pass - Regime A (structural assist)
theta = torch.randn(8, 10)  # Batch of 8 NK parameter vectors
shock_idx = torch.zeros(8, dtype=torch.long)  # First shock

irf = model(
    world_id="nk",
    theta=theta,
    shock_idx=shock_idx,
    regime="A",
)
# Output: (8, 41, 3) - batch, H+1 horizons, 3 observables
```

## Information Regimes

### Regime A: Full Structural Assist
**Inputs:** world_id, theta, shock_token
**Use case:** Known model with known parameters

```python
irf = model(
    world_id="nk",
    theta=theta,          # (batch, n_params)
    shock_idx=shock_idx,  # (batch,)
    regime="A",
)
```

### Regime B1: Observables Only
**Inputs:** world_id, shock_token, history
**Use case:** Econometrician within model class (no parameters)

```python
history = torch.randn(8, 20, 3)  # (batch, k, n_obs)

irf = model(
    world_id="nk",
    shock_idx=shock_idx,
    history=history,
    regime="B1",
)
```

### Regime C: Partial Information
**Inputs:** world_id, theta, shock_token, history
**Use case:** Calibrated parameters but unknown shock history

```python
irf = model(
    world_id="nk",
    theta=theta,
    shock_idx=shock_idx,
    history=history,
    regime="C",
)
```

## Mixed-World Batches

```python
# Different worlds in same batch
world_ids = ["nk", "nk", "var", "var"]
theta_padded = torch.randn(4, 15)  # Padded to max
theta_mask = torch.zeros(4, 15, dtype=torch.bool)
theta_mask[:2, :10] = True   # NK: 10 params
theta_mask[2:, :12] = True   # VAR: 12 params

irf = model(
    world_id=world_ids,
    theta=theta_padded,
    theta_mask=theta_mask,
    shock_idx=torch.zeros(4, dtype=torch.long),
    regime="A",
)
```

## Architecture Components

### 1. World Embedding
- Maps discrete world_id to continuous embedding
- Dimension: 32 (default)

### 2. Parameter Encoder
- Encodes variable-length theta vectors
- MLP with masked pooling
- Output dimension: 64 (default)

### 3. Shock Encoder
- Encodes (shock_idx, shock_size, shock_timing)
- Learnable embedding + MLP
- Output dimension: 16 (default)

### 4. History Encoder (Optional)
- GRU or Transformer architecture
- Encodes y[0:k] trajectory
- Output dimension: 64 (default)
- Required for regimes B1 and C

### 5. Trunk Network
- Combines all embeddings
- MLP (default) or Transformer
- Hidden dimension: 256, 4 layers (default)

### 6. IRF Head
- Predicts (H+1, 3) IRF
- Direct prediction (non-autoregressive)

### 7. Trajectory Head (Optional)
- For trajectory forecasting (not primary for Phase 1)

## Configuration Options

```python
model = UniversalEmulator(
    world_ids=["lss", "var", "nk", "rbc", "switching", "zlb"],
    param_dims={"lss": 15, "var": 12, ...},

    # Embedding dimensions
    world_embed_dim=32,
    theta_embed_dim=64,
    shock_embed_dim=16,
    history_embed_dim=64,

    # Trunk network
    trunk_dim=256,
    trunk_layers=4,
    trunk_architecture="mlp",  # or "transformer"

    # History encoder
    history_architecture="gru",  # or "transformer"
    use_history_encoder=True,

    # Output configuration
    H=40,  # IRF horizon (40 or 80)
    n_obs=3,  # Number of observables
    max_shocks=3,  # Max shocks across worlds

    # Trajectory prediction (optional)
    use_trajectory_head=False,

    # Regularization
    dropout=0.1,
)
```

## Training Integration

### With IRFDataset

```python
from emulator.training.dataset import IRFDataset, collate_mixed_worlds
from torch.utils.data import DataLoader

# Load dataset
dataset = IRFDataset(
    zarr_root="datasets/v1.0-dev",
    world_ids=["nk", "var"],
    split="train",
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_mixed_worlds,
)

# Training loop
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for batch in dataloader:
    # Extract shock index (first shock)
    shock_idx = torch.zeros(batch['theta'].shape[0], dtype=torch.long)

    # Forward pass
    predictions = model(
        world_id=batch['world_ids'],
        theta=batch['theta'],
        theta_mask=batch['theta_mask'],
        shock_idx=shock_idx,
        regime="A",
    )

    # Compute loss
    targets = batch['irf'][:, 0, :, :]  # First shock
    loss = torch.nn.functional.mse_loss(predictions, targets)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### With Trainer

```python
from emulator.training import Trainer, TrainingConfig

config = TrainingConfig(
    lr=1e-4,
    batch_size=64,
    epochs=100,
    checkpoint_dir="runs/universal_regime_A",
    seed=42,
)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
)

trainer.fit()
```

## Model Size

Default configuration (6 worlds):

| Component | Parameters |
|-----------|-----------|
| World embeddings | 192 |
| Parameter encoder | ~13k |
| Shock encoder | ~300 |
| History encoder (GRU) | ~50k |
| Trunk (4-layer MLP) | ~460k |
| IRF head | ~85k |
| **Total** | **~609k** |

## Performance

- **Forward pass:** ~5-10ms for batch_size=16 (CPU)
- **Memory:** ~2GB for training with batch_size=64
- **Scaling:** Tested up to batch_size=128
- **Optimization:** Compatible with torch.compile, mixed precision

## Testing

Run comprehensive test suite:
```bash
pytest emulator/tests/test_universal.py -v
```

53 tests covering:
- Individual components
- Full model forward pass
- All regimes (A, B1, C)
- Gradient flow
- Multi-world batches
- Edge cases

## Common Patterns

### Evaluation Mode
```python
model.eval()
with torch.no_grad():
    irf = model(world_id="nk", theta=theta, shock_idx=shock_idx, regime="A")
```

### Parameter Counting
```python
n_params = model.get_num_parameters()
print(f"Model has {n_params:,} parameters")
```

### Shock Size and Timing
```python
# 2 standard deviation shock at t=0
shock_size = torch.ones(batch_size) * 2.0
shock_timing = torch.zeros(batch_size)

irf = model(
    world_id="nk",
    theta=theta,
    shock_idx=shock_idx,
    shock_size=shock_size,
    shock_timing=shock_timing,
    regime="A",
)
```

### With Trajectory Prediction
```python
# Enable trajectory head
model = UniversalEmulator(..., use_trajectory_head=True)

# Get both IRF and trajectory
irf, trajectory = model(
    world_id="nk",
    theta=theta,
    shock_idx=shock_idx,
    regime="A",
    return_trajectory=True,
)
```

## Integration Example

See `examples/test_universal_integration.py` for a complete working example with real data.

## See Also

- Specification: `spec/spec.md` (Section 5: Emulator Architecture, Section 6: Information Regimes)
- Baselines: `emulator/models/baselines.py`
- Training: `emulator/training/trainer.py`
- Dataset: `emulator/training/dataset.py`
- Tests: `emulator/tests/test_universal.py`
