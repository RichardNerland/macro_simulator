# Training Infrastructure

This module provides the training infrastructure for the Universal Macro Emulator, including dataset loading, training loops, and configuration management.

## Components

### 1. `dataset.py` - IRFDataset

PyTorch Dataset for loading IRF data from Zarr arrays.

**Key Features:**
- Loads theta (parameters) and IRFs from per-world Zarr directories
- Supports train/val/test splits via `splits.json`
- Handles mixed-world batches with automatic padding
- Efficient chunked loading from Zarr

**Usage:**

```python
from emulator.training import IRFDataset, collate_mixed_worlds
from torch.utils.data import DataLoader

# Create dataset
dataset = IRFDataset(
    zarr_root="datasets/v1.0-dev",
    world_ids=["nk", "var", "lss"],
    split="train",
)

# Create dataloader with mixed-world collation
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_mixed_worlds,
)

# Iterate
for batch in loader:
    theta = batch["theta"]  # (batch_size, max_n_params) - padded
    irf = batch["irf"]      # (batch_size, max_n_shocks, H+1, 3) - padded
    world_ids = batch["world_ids"]  # list of world identifiers
    # ... training code
```

**Data Format:**

Each sample is a dictionary:
```python
{
    "theta": torch.Tensor,      # (n_params,) parameter vector
    "irf": torch.Tensor,        # (n_shocks, H+1, 3) IRF array
    "world_id": str,            # simulator identifier
    "sample_idx": int,          # original sample index
}
```

**Collate Functions:**

- `collate_mixed_worlds`: Pads theta and IRFs to handle variable dimensions across worlds
- `collate_single_world`: Stacks directly (no padding) when all samples from same world

### 2. `trainer.py` - Trainer

Main training class with full training loop, validation, checkpointing, and logging.

**Key Features:**
- Config-driven training via `TrainingConfig` dataclass
- Early stopping with patience
- Checkpoint saving (best and periodic)
- Learning rate scheduling (cosine annealing with warmup)
- Optional W&B integration
- Deterministic seeding for reproducibility

**Usage:**

```python
from emulator.training import Trainer, TrainingConfig
from emulator.models.baselines import PerWorldMLPBaseline
from torch.utils.data import DataLoader

# Create model
model = PerWorldMLPBaseline(
    param_dims={"nk": 12, "var": 27, "lss": 36},
    hidden_dims=[256, 128, 64],
    H=40,
)

# Create dataloaders (see dataset.py)
train_loader = DataLoader(...)
val_loader = DataLoader(...)

# Training configuration
config = TrainingConfig(
    lr=1e-4,
    batch_size=64,
    epochs=100,
    warmup_epochs=5,
    patience=10,
    checkpoint_dir="runs/baseline_mlp",
    log_every_n_steps=50,
    use_wandb=False,
    seed=42,
)

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
)

# Train
history = trainer.fit()

# Results saved to:
# - runs/baseline_mlp/best.pt (best checkpoint)
# - runs/baseline_mlp/final.pt (final checkpoint)
# - runs/baseline_mlp/history.json (training history)
# - runs/baseline_mlp/config.json (configuration)
```

**Configuration Options:**

```python
@dataclass
class TrainingConfig:
    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    epochs: int = 100
    grad_clip: float = 1.0

    # Learning rate schedule
    warmup_epochs: int = 5
    scheduler: str = "cosine"  # "cosine" or "none"

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # Checkpointing
    checkpoint_dir: str = "runs/default"
    save_every_n_epochs: int = 10
    save_best: bool = True

    # Logging
    log_every_n_steps: int = 50
    use_wandb: bool = False
    wandb_project: str = "macro-emulator"

    # Reproducibility
    seed: int = 42

    # Loss function
    loss_fn: str = "mse"  # "mse" or "mae"
    horizon_weights: str = "uniform"  # "uniform", "exponential", or "impact"
    lambda_smooth: float = 0.01  # Smoothness penalty
```

**Loading from Config File:**

```python
import yaml

with open("configs/baseline_mlp.yaml", "r") as f:
    config_dict = yaml.safe_load(f)

config = TrainingConfig(**config_dict)
```

## Training Loop Details

### Loss Computation

The trainer computes multi-horizon MSE loss with optional weighting:

```
L_irf = (1/N) Σ_i Σ_s Σ_h Σ_v w[h] * (ŷ[i,s,h,v] - y[i,s,h,v])²
```

Where:
- `i`: sample index
- `s`: shock index
- `h`: horizon index
- `v`: variable index (0=output, 1=inflation, 2=rate)
- `w[h]`: horizon weight

**Horizon Weighting Schemes:**

1. **Uniform** (default): `w[h] = 1/(H+1)` for all horizons
2. **Exponential**: `w[h] = exp(-h/τ) / Z` with `τ=20`
3. **Impact-weighted**: `w[0] = 0.3`, `w[h>0] = 0.7/H`

**Optional Regularization:**

- Smoothness penalty: `L_smooth = λ_smooth * Σ ||Δ²ŷ[h]||²`
  - Penalizes high-frequency oscillations in IRF predictions

### Learning Rate Schedule

1. **Warmup**: Linear warmup for first `warmup_epochs` epochs
2. **Cosine Annealing**: Cosine decay from peak LR to `0.01 * lr`

### Early Stopping

Training stops if validation loss doesn't improve for `patience` epochs (default: 10).

Improvement is defined as: `val_loss < best_val_loss - min_delta`

## Model Interface

Models must implement:

```python
def forward(
    theta: torch.Tensor,        # (batch_size, n_params)
    world_id: str | list[str],  # Single world or list for mixed batches
) -> torch.Tensor:              # (batch_size, n_shocks, H+1, 3) or (batch_size, H+1, 3)
    ...
```

For single-shock models, output can be `(batch_size, H+1, 3)` - trainer handles this.

## File Structure

```
emulator/training/
├── __init__.py          # Exports
├── dataset.py           # IRFDataset and collate functions
├── trainer.py           # Trainer and TrainingConfig
├── README.md            # This file
└── configs/             # Example YAML configs (optional)
    ├── baseline_mlp.yaml
    └── universal_A.yaml
```

## Example Training Script

See `test_training_infrastructure.py` for a complete example.

Minimal training script:

```python
#!/usr/bin/env python
"""Train a baseline MLP model."""

from emulator.training import IRFDataset, Trainer, TrainingConfig, collate_mixed_worlds
from emulator.models.baselines import PerWorldMLPBaseline
from torch.utils.data import DataLoader

# Datasets
train_dataset = IRFDataset("datasets/v1.0-dev", ["nk", "var", "lss"], "train")
val_dataset = IRFDataset("datasets/v1.0-dev", ["nk", "var", "lss"], "val")

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_mixed_worlds)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_mixed_worlds)

# Model
param_dims = {wid: train_dataset.get_world_info(wid)["n_params"] for wid in train_dataset.world_ids}
model = PerWorldMLPBaseline(param_dims=param_dims, hidden_dims=[256, 128, 64], H=40)

# Config
config = TrainingConfig(
    lr=1e-4,
    batch_size=64,
    epochs=100,
    checkpoint_dir="runs/baseline_mlp",
    seed=42,
)

# Train
trainer = Trainer(model, train_loader, val_loader, config)
history = trainer.fit()

print(f"Best validation loss: {trainer.best_val_loss:.6f}")
```

## Testing

Unit tests in `emulator/tests/test_training.py`:

```bash
# Run fast tests only
pytest emulator/tests/test_training.py -m fast -v

# Run all tests
pytest emulator/tests/test_training.py -v
```

## Integration with Existing Code

- **Dataset**: Compatible with `data/scripts/generate_dataset.py` output
- **Models**: Works with baselines in `emulator/models/baselines.py`
- **Evaluation**: Checkpoints can be loaded in `emulator/eval/evaluate.py`

## Reproducibility

All training runs are deterministic given:
1. `config.seed` - controls PyTorch random state
2. Dataset with fixed splits (from `splits.json`)
3. Same model initialization

Re-running with same config + seed produces identical results.

## Next Steps

1. Create YAML config files in `configs/`
2. Add command-line interface: `python -m emulator.training.trainer --config configs/baseline.yaml`
3. Integrate with evaluation pipeline
4. Add mixed-precision training support (AMP)
5. Add distributed training support (DDP)
