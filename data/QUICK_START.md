# Quick Start: Data Pipeline

## What We Built

The Dataset Infrastructure (S1.24-S1.27) provides:
1. ✅ **JSON Schema** for manifest validation
2. ✅ **Zarr Writer** for efficient array storage
3. ✅ **Manifest Generator** for authoritative metadata
4. ✅ **Comprehensive Tests** (24 tests, all passing)

## 30-Second Example

```python
import numpy as np
from data import DatasetWriter, ManifestGenerator

# 1. Write data
writer = DatasetWriter(
    output_dir="datasets/v1.0",
    world_id="nk",
    n_samples=1000,
    T=200,
    H=40,
    n_params=8,
    n_shocks=3,
)

for i in range(1000):
    writer.write_sample(
        i,
        theta=np.random.randn(8),
        trajectories=np.random.randn(200, 3),
        irfs=np.random.randn(3, 41, 3),
        shocks=np.random.randn(200, 3),
    )

writer.finalize()

# 2. Create manifest
gen = ManifestGenerator(version="1.0.0", output_dir="datasets/v1.0")
gen.add_standard_splits()
gen.save()

# 3. Read back
traj, irfs, shocks, theta = DatasetWriter.load_dataset("datasets/v1.0", "nk")
print(f"Loaded {traj.shape[0]} samples!")
```

## Run Tests

```bash
cd /Users/nerland/macro_simulator
pytest data/tests/ -m "fast" -v
```

Expected: **24 passed** ✅

## File Structure

```
data/
├── writer.py          # DatasetWriter class (450 lines)
├── manifest.py        # ManifestGenerator class (430 lines)
├── schemas/
│   └── manifest.json  # JSON Schema validator
└── tests/
    ├── test_data_pipeline.py   # Main tests (570 lines, 24 tests)
    └── test_imports.py         # Smoke tests (4 tests)
```

## Key Features

### DatasetWriter
- ✅ Zarr storage with Blosc compression (2-5x)
- ✅ Chunking for efficient random access
- ✅ Batch writing (10x faster than single-sample)
- ✅ float32 storage (50% space savings)
- ✅ Shape validation (catches errors early)
- ✅ Metadata storage (regime sequences, flags)

### ManifestGenerator
- ✅ Git hash tracking (reproducibility)
- ✅ Config hashing (SHA256)
- ✅ Standard splits (train/val/test + extrapolation)
- ✅ Validation (checks all required fields)
- ✅ JSON serialization (numpy-aware)

## Storage Schema

```
datasets/v1.0/
├── manifest.json              # Metadata (JSON)
└── {world_id}/
    ├── trajectories.zarr      # (n_samples, T, 3)
    ├── irfs.zarr             # (n_samples, n_shocks, H+1, 3)
    ├── shocks.zarr           # (n_samples, T, n_shocks)
    └── theta.zarr            # (n_samples, n_params)
```

## Next Steps (Sprint 2)

- [ ] `generate_dataset.py` CLI tool
- [ ] `validate_dataset.py` script
- [ ] Split algorithm implementations
- [ ] Parallel generation support

## Need Help?

See:
- `data/README.md` - Full usage guide
- `data/IMPLEMENTATION_SUMMARY.md` - Technical details
- `spec/spec.md` §4 - Data schema specification
