# Configuration Guide - Universal Macro Emulator

This directory contains YAML configuration files for training the Universal Macro Emulator across different information regimes and experimental conditions.

## Quick Start

```bash
# Train full Regime A (full structural assist)
python -m emulator.training.trainer --config configs/universal_regime_A.yaml

# Quick smoke test (5-10 min)
python -m emulator.training.trainer --config configs/smoke_test_regime_A.yaml

# Train Regime B1 (observables only)
python -m emulator.training.trainer --config configs/universal_regime_B1.yaml

# Run ablation study (no shock sequence)
python -m emulator.training.trainer --config configs/ablation_no_eps.yaml
```

## All Available Configs

| Config | Purpose | Runtime | Use When |
|--------|---------|---------|----------|
| **universal_regime_A.yaml** | Full structural assist | 30-60 min | Production, baseline comparisons |
| **universal_regime_B1.yaml** | Observables only | 40-80 min | Realistic empirical scenario |
| **universal_regime_C.yaml** | Parameters + history | 35-70 min | When params known, shocks inferred |
| **smoke_test_regime_A.yaml** | Quick validation | 5-10 min | Code testing, CI/CD pipelines |
| **ablation_no_eps.yaml** | Remove shock sequence | 1-2 min | Measure eps importance |
| **ablation_no_theta.yaml** | Remove parameters | 1-2 min | Measure param importance |
| **ablation_no_world_id.yaml** | Remove world identifier | 1-2 min | Measure world importance |
| **lowo_exclude_lss.yaml** | Hold out LSS | 30-60 min | Generalization to unseen simulators |
| **lowo_exclude_var.yaml** | Hold out VAR | 30-60 min | ... |
| **lowo_exclude_nk.yaml** | Hold out NK | 30-60 min | ... |
| **lowo_exclude_rbc.yaml** | Hold out RBC | 30-60 min | ... |
| **lowo_exclude_switching.yaml** | Hold out Switching | 30-60 min | ... |
| **lowo_exclude_zlb.yaml** | Hold out ZLB | 30-60 min | ... |

## Config Files Overview

### Production Configs (Full Training)

#### `universal_regime_A.yaml` - Full Structural Assist (Baseline)
- **Information available**: world_id, theta, eps_sequence, shock_token
- **Best for**: Establishing upper bound performance
- **Runtime**: ~30-60 minutes (GPU) on full dataset
- **Expected error**: Lowest of all regimes
- **Use case**: Production model for when all structural info is available

**Key settings**:
- `model.trunk_dim: 256` (full capacity)
- `model.trunk_layers: 4`
- `training.epochs: 100`
- `training.batch_size: 128`
- `loss.smoothness_lambda: 0.01` (light regularization)

#### `universal_regime_B1.yaml` - Observable History Only (Realistic)
- **Information available**: world_id, history (observable trajectory), shock_token
- **NOT available**: theta, eps_sequence (must infer from history)
- **Best for**: Realistic scenario where only macro observables are available
- **Runtime**: ~40-80 minutes (GPU) on full dataset
- **Expected error**: ~10-20% worse than Regime A
- **Use case**: Production model for empirical applications

**Key differences**:
- Uses `history_encoder_type: gru` to process observable history
- `history_hidden_dim: 128` (larger than Regime C)
- `loss.smoothness_lambda: 0.02` (stronger regularization, no theta constraint)
- May benefit from higher `grad_clip` (1.5) due to recurrent encoder

#### `universal_regime_C.yaml` - Theta + History (Intermediate)
- **Information available**: world_id, theta, history, shock_token
- **NOT available**: eps_sequence (must infer from history)
- **Best for**: Intermediate difficulty between A and B1
- **Runtime**: ~35-70 minutes (GPU) on full dataset
- **Expected error**: ~5-10% worse than Regime A, ~5-10% better than B1
- **Use case**: Transfer learning, when parameters are known but shocks aren't

**Key differences**:
- Uses `history_encoder_type: gru` but with smaller capacity
- `history_hidden_dim: 96` (smaller than B1 since theta is known)
- `loss.smoothness_lambda: 0.015` (intermediate regularization)

### Testing Configs

#### `smoke_test_regime_A.yaml` - Quick Validation Test
- **Purpose**: Fast end-to-end pipeline verification (5-10 min runtime)
- **Best for**:
  - Testing after code changes
  - Verifying data pipeline
  - CI/CD integration tests
  - Quick debugging
- **NOT for**: Evaluation, production training, or paper results

**Reduced dimensions** (50% of full):
```
trunk_dim: 128        (vs 256)
trunk_layers: 2       (vs 4)
batch_size: 32        (vs 128)
epochs: 20            (vs 100)
```

**Reduced dataset**:
- Only 3 simulator families (LSS, VAR, NK)
- Uses development dataset (v1.0-dev) - much smaller

**Use this to verify changes work before submitting full runs**:
```bash
python -m emulator.training.trainer --config configs/smoke_test_regime_A.yaml
```

### Ablation Studies (Component Importance)

Ablation studies measure the contribution of different input components to model performance. All ablations use:
- Reduced model size (smoke test dimensions)
- 10 epochs (fast convergence assessment)
- Development dataset (v1.0-dev) for speed

#### `ablation_no_eps.yaml` - Remove Shock Sequence
- **Tests**: Importance of full shock realization (eps_sequence)
- **Ablation**: `eps_embed_dim: 0` (was 16)
- **Expected**: Large degradation (30-50% error increase)
- **Interpretation**: If performance is much worse, eps_sequence is critical for accurate IRFs

#### `ablation_no_theta.yaml` - Remove Parameter Encoding
- **Tests**: Importance of structural parameters (theta)
- **Ablation**: `theta_embed_dim: 0` (was 32)
- **Expected**: Moderate-to-large degradation (30-60% error increase)
- **Interpretation**: Parameters are crucial for IRF shape prediction

#### `ablation_no_world_id.yaml` - Remove World Identifier
- **Tests**: Importance of knowing simulator family
- **Ablation**: `world_embed_dim: 0` (was 16)
- **Expected**: Mild-to-moderate degradation (10-25% error increase)
- **Interpretation**: Parameters likely contain most simulator-specific information

**Run all ablations**:
```bash
python -m emulator.training.trainer --config configs/ablation_no_eps.yaml
python -m emulator.training.trainer --config configs/ablation_no_theta.yaml
python -m emulator.training.trainer --config configs/ablation_no_world_id.yaml

# Compare errors to smoke test baseline
```

### Generalization Tests (LOWO - Leave-One-World-Out)

Test out-of-distribution generalization by excluding one simulator family during training:

#### `lowo_exclude_lss.yaml` - Hold out LSS
#### `lowo_exclude_var.yaml` - Hold out VAR
#### `lowo_exclude_nk.yaml` - Hold out NK
#### `lowo_exclude_rbc.yaml` - Hold out RBC
#### `lowo_exclude_switching.yaml` - Hold out Switching
#### `lowo_exclude_zlb.yaml` - Hold out ZLB

**Purpose**: Measure how well the emulator generalizes to unseen simulator families

Each LOWO config:
- Trains on 5 of 6 simulator families
- Evaluates on the held-out family
- Tests transfer learning to OOD simulators

**Expected performance**:
- Worse than full training (baseline)
- Ratio: LOWO error / baseline error ≈ 1.2-1.4
- Shows how much emulator relies on universal vs. simulator-specific knowledge

**Example command**:
```bash
python -m emulator.training.trainer --config configs/lowo_exclude_lss.yaml
```

**Use these to**:
- Measure generalization robustness
- Identify which simulators are easiest/hardest to generalize to
- Compare universal vs. specialist performance gaps

### Example Configs (in `configs/examples/`)

These are simplified configuration templates showing different model types:
- `universal_regime_A.yaml` - Minimal regime A config (flat structure, fewer comments)
- `baseline_mlp.yaml` - Simple MLP baseline (single-world, no universal setting)

**Note**: Use configs in root `configs/` directory for actual training. Example configs are for reference only.

---

## Configuration Structure

Every config file has four main sections:

### 1. Model Configuration

Controls neural network architecture for embedding and processing inputs.

**Embedding dimensions** - Size of learned representations for each input type:
- `world_embed_dim`: Embedding for simulator family (8-64 typical)
- `theta_embed_dim`: Embedding for structural parameters (32-128 typical)
- `shock_embed_dim`: Embedding for shock token (8-32 typical)
- `eps_embed_dim`: Embedding for shock sequence (16-64 typical)
- `history_embed_dim`: For Regimes B1/C - encoding observable history (64-256)

**Trunk architecture** - Main network after embeddings:
- `trunk_dim`: Hidden dimension (256-512 for universal)
- `trunk_layers`: Number of layers (3-5 typical)
- `n_heads`: Attention heads for transformer (2-8)
- `dropout`: Regularization rate (0.0-0.2)

**Output**:
- `H`: IRF horizon in quarters (20, 40, 60, or 80)
- `n_obs`: Number of observables (3: output, inflation, rate)

**Encoder type**:
- `encoder_type`: "transformer" (recommended), "mlp", or "gru"

### 2. Training Configuration

Hyperparameters for the optimization process.

**Optimization**:
- `batch_size`: Training batch size (32-256)
- `lr`: Initial learning rate (1e-5 to 1e-3)
- `weight_decay`: L2 regularization coefficient (0.0-0.1)
- `grad_clip`: Maximum gradient norm (0.5-2.0)

**Learning schedule**:
- `epochs`: Total training epochs (20-150)
- `warmup_epochs`: Linear warmup phase (0-10)
- `scheduler`: "cosine" or "none"

**Early stopping** (validation monitoring):
- `patience`: Epochs without improvement before stopping (5-20)
- `min_delta`: Minimum improvement to count as "better" (1e-5 to 1e-3)

**Checkpointing**:
- `checkpoint_dir`: Where to save models
- `save_every_n_epochs`: Checkpoint frequency (5-20)
- `save_best`: Always save best validation checkpoint (true recommended)

**Logging**:
- `log_every_n_steps`: Log frequency (10-100)
- `use_wandb`: Enable Weights & Biases logging (true/false)
- `wandb_project`: W&B project name
- `wandb_run_name`: Run identifier

### 3. Loss Configuration

How the model is trained to predict accurate IRFs.

**Loss type**:
- `type`: "multi_horizon" (separate per horizon) or "combined"

**Horizon weighting** - Weight IRFs at different horizons:
- `weight_scheme`: "uniform" (equal), "exponential" (decay), or "impact" (h=0-5 emphasized)
- `tau`: Decay parameter for exponential scheme (5-40)
- `impact_length`: Number of impact periods for impact scheme (3-8)

**Per-variable weights** (optional):
- `per_variable_weights`: [weight_output, weight_inflation, weight_rate]
- Example: `[1.0, 1.5, 1.0]` to emphasize inflation prediction

**Regularization**:
- `smoothness_lambda`: Penalty on oscillatory predictions (0.0-0.1)
  - Regime A: 0.01 (light)
  - Regime B1: 0.02 (strong)
  - Regime C: 0.015 (moderate)

### 4. Data Configuration

Dataset and dataloader settings.

**Dataset**:
- `dataset_path`: Root path to dataset
- `worlds`: List of simulator families to include (subset or all 6)

**Splits**:
- `train_split`: Training set name (usually "train")
- `val_split`: Validation set (usually "val")
- `test_split`: Test set (usually "test")

**History** (Regimes B1 and C only):
- `history_length`: Observable history length in quarters (40-160)
  - 80 is standard (20 years)
  - Can reduce to 40 for faster experiments

**Dataloader**:
- `num_workers`: Parallel loading workers (0 for debugging, 4 for production)
- `pin_memory`: Pin to GPU memory (true for GPU, false for CPU)
- `shuffle_train`: Shuffle training data (true recommended)

---

## Common Patterns and Recipes

### 1. Quick Local Testing

Use `smoke_test_regime_A.yaml` to test changes before submitting larger runs:

```bash
# Test data loading and training loop
python -m emulator.training.trainer --config configs/smoke_test_regime_A.yaml

# Should complete in 5-10 minutes on CPU
# Check that:
# - Data loads without errors
# - Training progresses
# - Checkpoints are saved
```

### 2. Creating Custom Configs

To create a new config (e.g., for hyperparameter tuning):

1. **Start from a template**:
   ```bash
   cp configs/universal_regime_A.yaml configs/my_experiment.yaml
   ```

2. **Modify specific parameters**:
   ```yaml
   training:
     lr: 2.0e-4          # Try higher LR
     batch_size: 64      # Try smaller batch
     epochs: 150         # Try longer training

   loss:
     smoothness_lambda: 0.05  # Try more regularization
   ```

3. **Keep documentation**:
   - Add comments explaining why you changed each parameter
   - Update the PURPOSE section at top

4. **Run experiment**:
   ```bash
   python -m emulator.training.trainer --config configs/my_experiment.yaml
   ```

### 3. Comparing Regimes

Run all three regimes on the same dataset to compare:

```bash
python -m emulator.training.trainer --config configs/universal_regime_A.yaml
python -m emulator.training.trainer --config configs/universal_regime_B1.yaml
python -m emulator.training.trainer --config configs/universal_regime_C.yaml

# Compare:
# - Final validation loss in runs/universal_regime_*/logs
# - Model size and training time
# - Checkpoint paths for evaluation
```

### 4. Running Ablation Studies

Ablations help understand which inputs matter most:

```bash
# Run baseline
python -m emulator.training.trainer --config configs/smoke_test_regime_A.yaml

# Run ablations
python -m emulator.training.trainer --config configs/ablation_no_eps.yaml
python -m emulator.training.trainer --config configs/ablation_no_theta.yaml
python -m emulator.training.trainer --config configs/ablation_no_world_id.yaml

# Compare test error to baseline
# Large error increase = component is important
# Small error increase = component is redundant
```

### 5. Hyperparameter Tuning

Common adjustments for different situations:

**Model is underfitting** (training loss not decreasing):
```yaml
model:
  trunk_dim: 512        # Increase capacity
  trunk_layers: 5       # Deeper model

training:
  lr: 5.0e-4           # Higher learning rate
  weight_decay: 0.0    # Reduce regularization

loss:
  smoothness_lambda: 0.0  # Remove smoothness penalty
```

**Model is overfitting** (val loss increases while train decreases):
```yaml
model:
  dropout: 0.2         # Stronger dropout

training:
  weight_decay: 0.05   # Stronger regularization
  epochs: 50           # Early stopping

loss:
  smoothness_lambda: 0.05  # Stronger smoothness penalty
```

**Training is slow** (want faster iteration):
```yaml
training:
  batch_size: 64       # Fewer samples per batch
  epochs: 50           # Fewer total epochs

data:
  num_workers: 0       # Single-threaded (actually can be faster locally)

device: cpu            # If no GPU available
```

**Training is noisy** (loss jumps around):
```yaml
training:
  grad_clip: 0.5       # Tighter clipping
  warmup_epochs: 10    # Longer warmup

loss:
  smoothness_lambda: 0.02  # More regularization
```

---

## Parameter Ranges and Recommendations

### Model Architecture

| Parameter | Typical Range | Smoke Test | Full |
|-----------|---------------|-----------|------|
| world_embed_dim | 16-64 | 16 | 32 |
| theta_embed_dim | 32-128 | 32 | 64 |
| shock_embed_dim | 8-32 | 8 | 16 |
| eps_embed_dim | 16-64 | 16 | 32 |
| trunk_dim | 128-512 | 128 | 256 |
| trunk_layers | 2-6 | 2 | 4 |
| n_heads | 2-8 | 2 | 4 |
| dropout | 0.0-0.2 | 0.1 | 0.1 |

### Training Hyperparameters

| Parameter | Recommended | Min | Max |
|-----------|-------------|-----|-----|
| batch_size | 128 | 32 | 256 |
| learning_rate | 1e-4 | 1e-5 | 1e-3 |
| weight_decay | 0.01 | 0.0 | 0.1 |
| grad_clip | 1.0 | 0.5 | 2.0 |
| epochs | 100 | 20 | 200 |
| warmup_epochs | 5 | 0 | 20 |
| patience | 10 | 5 | 20 |

### Loss Configuration

| Parameter | Regime A | Regime B1 | Regime C |
|-----------|----------|-----------|----------|
| weight_scheme | exponential | exponential | exponential |
| tau | 20.0 | 20.0 | 20.0 |
| smoothness_lambda | 0.01 | 0.02 | 0.015 |

---

## Output Structure

After training with a config, the checkpoint directory contains:

```
runs/universal_regime_A/
├── best_model.pt              # Best validation checkpoint
├── checkpoint_epoch_0010.pt    # Periodic checkpoints
├── checkpoint_epoch_0020.pt
├── config.yaml                 # Config used for training (auto-saved)
├── training_log.csv            # Metrics per epoch (if enabled)
└── loss_log.json               # Loss history
```

Use the best model for inference:
```python
import torch
model = torch.load("runs/universal_regime_A/best_model.pt")
```

---

## Troubleshooting

### "Config not found" error
```bash
# Make sure you're in the right directory
cd /path/to/macro_simulator
python -m emulator.training.trainer --config configs/universal_regime_A.yaml
```

### Out of Memory (OOM) error
```yaml
# Reduce batch size
training:
  batch_size: 64  # was 128

# Or reduce model size
model:
  trunk_dim: 128  # was 256
```

### Training loss doesn't decrease
```yaml
# Try higher learning rate
training:
  lr: 5.0e-4  # was 1.0e-4

# Disable regularization temporarily
loss:
  smoothness_lambda: 0.0  # was 0.01

training:
  weight_decay: 0.0  # was 0.01
```

### Validation loss doesn't improve
```yaml
# May be overfitting - add regularization
model:
  dropout: 0.2  # was 0.1

training:
  weight_decay: 0.05  # was 0.01

loss:
  smoothness_lambda: 0.05  # was 0.01
```

---

## Full Configuration Example

Here's a complete custom configuration for an experiment:

```yaml
# Experiment: Test MLP encoder instead of Transformer
# Goal: Compare encoder architectures on Regime A

model:
  type: universal

  world_embed_dim: 32
  theta_embed_dim: 64
  shock_embed_dim: 16
  eps_embed_dim: 32

  trunk_dim: 256
  trunk_layers: 4
  n_heads: 4          # Unused since encoder_type: mlp
  dropout: 0.1

  H: 40
  n_obs: 3

  encoder_type: mlp   # Changed from transformer

training:
  regime: A
  batch_size: 128
  lr: 1.0e-4
  weight_decay: 0.01
  grad_clip: 1.0

  epochs: 100
  warmup_epochs: 5
  scheduler: cosine

  patience: 10
  min_delta: 1.0e-4

  checkpoint_dir: runs/encoder_comparison_mlp
  save_every_n_epochs: 10
  save_best: true

  log_every_n_steps: 50
  use_wandb: false

loss:
  type: multi_horizon
  weight_scheme: exponential
  tau: 20.0
  impact_length: 5
  per_variable_weights: null
  smoothness_lambda: 0.01

data:
  dataset_path: datasets/v1.0/
  worlds:
    - lss
    - var
    - nk
    - rbc
    - switching
    - zlb

  train_split: train
  val_split: val
  test_split: test

  num_workers: 4
  pin_memory: true
  shuffle_train: true

seed: 42
device: cuda
```

---

## Key Takeaways

1. **Start with existing configs** - Don't write from scratch; modify templates
2. **Use smoke test for quick validation** - Test changes in 5-10 minutes before full runs
3. **Understand information regimes**:
   - Regime A: Easiest (full structural info)
   - Regime C: Medium (theta known, must infer shocks)
   - Regime B1: Hardest (observables only)
4. **Run ablations to understand importance** - Which inputs matter most?
5. **Document your configs** - Add comments explaining why each parameter is set
6. **Experiment systematically** - Change one thing at a time

For more details, see the inline comments in each config file.
