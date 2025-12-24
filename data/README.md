# Data Pipeline

This module implements the dataset infrastructure for the Universal Macro Emulator (S1.24-S1.27).

## Components

### 1. JSON Schema (`schemas/manifest.json`)
Validates the manifest structure per spec §4.2. The schema ensures:
- Version is in semver format
- All required fields are present
- Simulator configurations are complete
- Split fractions are valid

### 2. Zarr Dataset Writer (`writer.py`)
Writes simulator outputs to Zarr format with:
- **Trajectories**: `(n_samples, T, n_obs)` - Time series data
- **IRFs**: `(n_samples, n_shocks, H+1, 3)` - Impulse response functions
- **Shocks**: `(n_samples, T, n_shocks)` - Shock sequences in std dev units
- **Theta**: `(n_samples, n_params)` - Parameters in natural units

**Features**:
- Efficient chunking along sample dimension
- Blosc compression (zstd, level 3)
- float32 storage by default (per spec §4.4)
- Batch writing support
- Metadata storage for regime sequences, binding flags, etc.

### 3. Manifest Generator (`manifest.py`)
Creates and validates `manifest.json` files with:
- Git hash for version control
- Configuration hashes for reproducibility
- Per-simulator metadata (parameters, shocks, observables)
- Split definitions with algorithms
- RNG seed configuration

**Key Principle**: The manifest is the authoritative metadata source. All queries should reference it first.

## Usage

### Basic Dataset Writing

```python
from data import DatasetWriter, ManifestGenerator
import numpy as np

# Initialize writer
writer = DatasetWriter(
    output_dir="datasets/v1.0",
    world_id="nk",
    n_samples=100000,
    T=200,
    H=40,
    n_params=8,
    n_shocks=3,
)

# Write samples
for i in range(100000):
    theta = np.random.randn(8)
    trajectories = np.random.randn(200, 3)  # T x n_obs
    irfs = np.random.randn(3, 41, 3)  # n_shocks x (H+1) x 3
    shocks = np.random.randn(200, 3)  # T x n_shocks

    writer.write_sample(i, theta, trajectories, irfs, shocks)

summary = writer.finalize()
print(f"Compression ratio: {summary['arrays']['trajectories']['compression_ratio']:.2f}x")
```

### Creating a Manifest

```python
from data import ManifestGenerator
from simulators.nk import NKSimulator  # Example

# Initialize generator
gen = ManifestGenerator(
    version="1.0.0",
    output_dir="datasets/v1.0",
    global_seed=42,
    T=200,
    H=40,
)

# Add simulator
simulator = NKSimulator()
gen.add_simulator(
    world_id="nk",
    n_samples=100000,
    param_manifest=simulator.param_manifest,
    shock_manifest=simulator.shock_manifest,
    obs_manifest=simulator.obs_manifest,
)

# Add standard splits
gen.add_standard_splits(seed=42)

# Save and validate
manifest_path = gen.save(validate=True)
print(f"Manifest saved to {manifest_path}")
```

### Loading a Dataset

```python
from data import DatasetWriter

# Load arrays
trajectories, irfs, shocks, theta = DatasetWriter.load_dataset(
    "datasets/v1.0", "nk"
)

print(f"Loaded {trajectories.shape[0]} samples")
print(f"First trajectory shape: {trajectories[0].shape}")
```

## Storage Schema

```
datasets/v1.0/
├── manifest.json           # Metadata (authoritative)
└── nk/                     # Per-world directories
    ├── trajectories.zarr   # (n_samples, T, n_obs)
    ├── irfs.zarr          # (n_samples, n_shocks, H+1, 3)
    ├── shocks.zarr        # (n_samples, T, n_shocks)
    ├── theta.zarr         # (n_samples, n_params)
    └── metadata.zarr      # Optional: regime_seq, binding_flags, etc.
```

## Testing

Run the test suite:

```bash
# Fast tests only (< 1s each)
pytest data/tests/ -m "fast" -v

# All tests
pytest data/tests/ -v
```

## Conventions

### IRF Convention (§3.5.1)
```
IRF[h] = y_shocked[h] - y_baseline[h]
```
- Shock hits at t=0
- IRF[0] shows impact effect
- Horizon: h = 0, 1, ..., H (H+1 total observations)

### Deterministic Seeding (§4.2)
Per-sample seeds are derived from global seed:
```python
sample_seed = global_seed + sample_index
```

This ensures:
- Same global seed → identical dataset
- Parallel generation is safe (no seed conflicts)
- Reproducibility across machines

### Data Types
- **Storage**: float32 (spec §4.4)
- **Simulation**: float64 internally for numerical stability
- **Automatic conversion**: Writer handles dtype conversion

## File Sizes

For 100k samples per simulator (T=200, H=40):
- **Trajectories**: ~240 MB uncompressed → ~50-100 MB compressed
- **IRFs**: ~500 MB uncompressed → ~100-200 MB compressed
- **Shocks**: ~240 MB uncompressed → ~50-100 MB compressed
- **Theta**: ~3 MB (small, parameter count varies)

Total per simulator: ~200-400 MB (compressed)
For 6 simulators: ~1.2-2.4 GB

## Next Steps

- **S2.15-S2.20**: Implement `generate_dataset.py` script
- **S2.21-S2.22**: Implement `validate_dataset.py` script
- Add split construction algorithms (random, slice, corner)
- Add parallel generation support
