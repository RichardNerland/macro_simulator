# Random Seed Documentation

This document provides a complete reference for all random seeds used in the Universal Macro Emulator project, ensuring full reproducibility of results.

## Global Seed Policy

**All experiments use `seed=42`** as the global random seed for maximum reproducibility.

This seed is propagated through three layers:
1. **Data generation**: Dataset creation and splitting
2. **Model training**: Weight initialization and batch shuffling
3. **Evaluation**: Sampling for visualizations (if applicable)

## Data Generation Seeds

### Dataset Manifests

All generated datasets include a `manifest.json` file documenting the exact seeds used:

```json
{
  "rng_seeds": {
    "global": 42,
    "per_sample": true,
    "derivation_algorithm": "global_seed + sample_index"
  }
}
```

### Seed Derivation

For parallel data generation, per-sample seeds are derived deterministically:

```python
def get_sample_seed(global_seed: int, sample_index: int) -> int:
    """Derive per-sample seed from global seed and sample index."""
    return global_seed + sample_index
```

This ensures:
- **Reproducibility**: Same global seed → same dataset
- **Parallelism**: Each sample can be generated independently
- **Determinism**: Sample order doesn't affect results

### Split Seeds

Train/val/test splits use the global seed for reproducibility:

```json
{
  "splits": {
    "train": {
      "fraction": 0.8,
      "seed": 42,
      "algorithm": "random_interpolation"
    },
    "val": {
      "fraction": 0.1,
      "seed": 42,
      "algorithm": "random_interpolation"
    },
    "test_interpolation": {
      "fraction": 0.05,
      "seed": 42,
      "algorithm": "random_interpolation"
    },
    "test_extrapolation_slice": {
      "fraction": 0.025,
      "seed": 42,
      "algorithm": "slice_predicate"
    },
    "test_extrapolation_corner": {
      "fraction": 0.025,
      "seed": 42,
      "algorithm": "corner_extremes"
    }
  }
}
```

### Dataset-Specific Seeds

| Dataset | Global Seed | Samples | Worlds | Location |
|---------|-------------|---------|--------|----------|
| v1.0-test | 42 | 100/world | lss, var, nk | `datasets/v1.0-test/manifest.json` |
| v1.0-dev | 42 | 1,000/world | all 6 | `datasets/v1.0-dev/manifest.json` |
| v1.0 | 42 | 10,000/world | all 6 | `datasets/v1.0/manifest.json` |

## Training Seeds

### Configuration Files

All training configs specify `seed: 42`:

```yaml
# Universal Emulator - Regime A
model:
  type: universal
  # ... model config ...

training:
  regime: A
  # ... training config ...

# Reproducibility
seed: 42
```

### Seed Application

The seed is applied to multiple RNG sources in the training loop:

```python
def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

### Training-Specific Seeds

| Experiment | Config File | Seed | Purpose |
|------------|-------------|------|---------|
| Universal Regime A | `configs/universal_regime_A.yaml` | 42 | Full structural assist |
| Universal Regime B1 | `configs/universal_regime_B1.yaml` | 42 | Observables only |
| Universal Regime C | `configs/universal_regime_C.yaml` | 42 | Partial info |
| Ablation (no world_id) | `configs/ablation_no_world_id.yaml` | 42 | Remove world embedding |
| Ablation (no theta) | `configs/ablation_no_theta.yaml` | 42 | Remove parameters |
| Ablation (no eps) | `configs/ablation_no_eps.yaml` | 42 | Remove shock sequence |
| Smoke Test | `configs/smoke_test_regime_A.yaml` | 42 | Quick validation |

## GPU Non-Determinism

### PyTorch CUDA Operations

Despite fixed seeds, some PyTorch CUDA operations are non-deterministic by default. To enforce full determinism:

```python
# In training script
import torch

# Enforce deterministic operations (may reduce performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Note**: This is currently **NOT** enabled by default for performance reasons. Results may differ by ~0.1% across runs on different hardware.

### Floating-Point Precision

Different hardware (CPU vs GPU, different GPU models) may produce slightly different floating-point results due to:
- Different instruction sets
- Different accumulation orders
- Different precision modes

**Expected variance**: ±0.1% in final metrics across hardware configurations.

## Seed Verification

### Checking Dataset Seeds

```bash
# Extract seed from manifest
python -c "import json; print(json.load(open('datasets/v1.0/manifest.json'))['rng_seeds'])"
```

Expected output:
```json
{"global": 42, "per_sample": true, "derivation_algorithm": "global_seed + sample_index"}
```

### Checking Training Config Seeds

```bash
# Extract seed from config
python -c "import yaml; print(yaml.safe_load(open('configs/universal_regime_A.yaml'))['seed'])"
```

Expected output:
```
42
```

### Reproducing Exact Results

To reproduce exact results:

1. **Use same dataset**: Verify manifest has `global: 42`
2. **Use same config**: Verify config has `seed: 42`
3. **Use same hardware**: CPU vs GPU differences are expected
4. **Use same PyTorch version**: Different versions may have different RNG implementations

## Advanced: Custom Seeds

For custom experiments, you can override the default seed:

### Data Generation with Custom Seed

```bash
python -m data.scripts.generate_dataset \
    --world all \
    --n_samples 10000 \
    --seed 1234 \
    --output datasets/custom/
```

This will:
- Set `rng_seeds.global = 1234` in manifest
- Use seed 1234 for all random operations
- Ensure reproducibility with the new seed

### Training with Custom Seed

Modify the config file:

```yaml
# Custom seed experiment
seed: 1234  # Changed from default 42
```

Or override via command line (if implemented):

```bash
python -m emulator.training.trainer \
    --config configs/universal_regime_A.yaml \
    --seed 1234
```

## Seed Audit Trail

### Git Hashes

All datasets include the git hash at generation time:

```json
{
  "version": "1.0.0",
  "created_at": "2025-12-24T03:36:42.791035+00:00",
  "git_hash": "12517506bfa23b61b1fdb8268680aa1abc6a915e",
  ...
}
```

This allows tracing exact code version used for generation.

### Config Hashes

Per-world configs are also hashed:

```json
{
  "simulators": {
    "lss": {
      "config_hash": "032ba93c2211ee58",
      ...
    }
  }
}
```

This ensures simulator parameters are tracked.

## Troubleshooting Non-Reproducibility

If results differ from expected:

### 1. Verify Seeds Match

```bash
# Check dataset seed
jq '.rng_seeds.global' datasets/v1.0/manifest.json
# Expected: 42

# Check training seed
yq '.seed' configs/universal_regime_A.yaml
# Expected: 42
```

### 2. Check PyTorch Version

```bash
python -c "import torch; print(torch.__version__)"
# Expected: 2.0.0 or later
```

Different PyTorch versions may have different RNG implementations.

### 3. Check CUDA Determinism

```bash
# Verify if deterministic mode is enabled
python -c "import torch; print(torch.backends.cudnn.deterministic)"
# True = deterministic, False = non-deterministic (faster)
```

### 4. Check Data Loading Order

Ensure `shuffle_train: true` is set consistently in configs. Different shuffle states will affect training trajectory.

### 5. Hardware Differences

Accept small variance (±0.1%) across:
- CPU vs GPU
- Different GPU models (e.g., V100 vs A100)
- Different precision modes (FP32 vs FP16)

## Seed Best Practices

### For Development

Use `seed=42` consistently during development for:
- Easy reproduction of bugs
- Consistent benchmarking
- Comparable results across team members

### For Experiments

When running sensitivity analysis or ablations:
- Keep base seed fixed (`seed=42`)
- Only change experimental variable
- Document any seed changes in experiment notes

### For Production

For production models:
- Train with multiple seeds (e.g., 42, 43, 44, 45, 46)
- Report mean and std dev across seeds
- Use ensemble if variance is high

## Seed Registry

Complete registry of all seeds used in the project:

| Component | Seed | Location | Notes |
|-----------|------|----------|-------|
| Global default | 42 | All configs | Standard seed for all experiments |
| Dataset v1.0 | 42 | `datasets/v1.0/manifest.json` | Full dataset |
| Dataset v1.0-dev | 42 | `datasets/v1.0-dev/manifest.json` | Dev dataset |
| Dataset v1.0-test | 42 | `datasets/v1.0-test/manifest.json` | Test dataset |
| Universal Regime A | 42 | `configs/universal_regime_A.yaml` | Training seed |
| Universal Regime B1 | 42 | `configs/universal_regime_B1.yaml` | Training seed |
| Universal Regime C | 42 | `configs/universal_regime_C.yaml` | Training seed |
| All ablations | 42 | `configs/ablation_*.yaml` | Training seed |
| Smoke tests | 42 | `configs/smoke_test_*.yaml` | Training seed |

## Future: Multi-Seed Ensembles

For Phase 2, we plan to train ensembles with multiple seeds:

```yaml
# Future: Ensemble training config
ensemble:
  n_models: 5
  seeds: [42, 43, 44, 45, 46]
  aggregation: mean  # or "median", "mixture"
```

This will provide:
- Uncertainty quantification
- More robust predictions
- Variance estimates

Stay tuned for updates in Sprint 6+.

## Contact

For questions about reproducibility or seed management:
- Check this documentation first
- Verify your setup matches requirements
- Open an issue with full details if problems persist
