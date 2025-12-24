# LOWO (Leave-One-World-Out) Split Usage Guide

## Overview

The LOWO (Leave-One-World-Out) split enables **cross-world generalization testing** by training on all simulator families except one and testing on the held-out family. This is critical for evaluating whether the Universal Macro Emulator truly generalizes across different economic model structures.

## Implementation

### Location
- **Module**: `/Users/nerland/macro_simulator/data/splits.py`
- **Tests**: `/Users/nerland/macro_simulator/data/tests/test_splits.py`
- **Examples**: `/Users/nerland/macro_simulator/examples/lowo_example.py`

### Functions

#### `split_lowo(world_ids, held_out_world)`

Creates a LOWO split by excluding one world from training.

**Arguments:**
- `world_ids` (list[str]): List of all available world identifiers
- `held_out_world` (str): World to hold out for testing

**Returns:**
- Dictionary with keys:
  - `"train_worlds"`: List of world_ids for training (all except held_out_world)
  - `"test_worlds"`: List containing only the held_out_world

**Example:**
```python
from data import split_lowo

all_worlds = ["lss", "var", "nk", "rbc", "switching", "zlb"]
split = split_lowo(all_worlds, "nk")

# Result:
# {
#     "train_worlds": ["lss", "var", "rbc", "switching", "zlb"],
#     "test_worlds": ["nk"]
# }
```

#### `get_lowo_world_lists(all_worlds, held_out_world)`

Convenience function that returns train and test world lists as a tuple.

**Arguments:**
- `all_worlds` (list[str]): List of all available world identifiers
- `held_out_world` (str): World to hold out for testing

**Returns:**
- Tuple of `(train_worlds, test_worlds)`

**Example:**
```python
from data import get_lowo_world_lists

train_worlds, test_worlds = get_lowo_world_lists(
    ["lss", "var", "nk", "rbc", "switching", "zlb"],
    "nk"
)

# train_worlds = ["lss", "var", "rbc", "switching", "zlb"]
# test_worlds = ["nk"]
```

## Usage Patterns

### Pattern 1: Single LOWO Experiment

Train on 5 worlds, test on 1 held-out world:

```python
from pathlib import Path
from data import get_lowo_world_lists
from emulator.training.dataset import IRFDataset

# Configuration
dataset_root = Path("datasets/v1.0")
all_worlds = ["lss", "var", "nk", "rbc", "switching", "zlb"]
held_out_world = "nk"

# Create LOWO split
train_worlds, test_worlds = get_lowo_world_lists(all_worlds, held_out_world)

# Load datasets with world filtering
train_dataset = IRFDataset(
    zarr_root=dataset_root,
    world_ids=train_worlds,  # 5 worlds: lss, var, rbc, switching, zlb
    split="train"
)

test_dataset = IRFDataset(
    zarr_root=dataset_root,
    world_ids=test_worlds,   # 1 world: nk
    split="test_interpolation"
)

# Train model
model = train_model(train_dataset)

# Evaluate on held-out world
metrics = evaluate_model(model, test_dataset)
print(f"LOWO performance on {held_out_world}: {metrics}")
```

### Pattern 2: 6-Fold LOWO Cross-Validation

Run complete LOWO evaluation across all worlds:

```python
from data import get_lowo_world_lists
from emulator.training.dataset import IRFDataset

all_worlds = ["lss", "var", "nk", "rbc", "switching", "zlb"]
dataset_root = Path("datasets/v1.0")

lowo_results = {}

for held_out_world in all_worlds:
    print(f"\n=== LOWO: Held-out world = {held_out_world} ===")

    # Get split
    train_worlds, test_worlds = get_lowo_world_lists(all_worlds, held_out_world)

    # Load data
    train_dataset = IRFDataset(dataset_root, train_worlds, split="train")
    test_dataset = IRFDataset(dataset_root, test_worlds, split="test_interpolation")

    # Train and evaluate
    model = train_model(train_dataset)
    metrics = evaluate_model(model, test_dataset)

    lowo_results[held_out_world] = metrics
    print(f"  MSE: {metrics['mse']:.4f}")

# Aggregate results
avg_mse = sum(r['mse'] for r in lowo_results.values()) / len(lowo_results)
print(f"\nAverage LOWO MSE: {avg_mse:.4f}")
```

### Pattern 3: Training Configuration

Add LOWO support to training config:

```yaml
# configs/lowo_nk.yaml
dataset:
  path: "datasets/v1.0"
  lowo_mode: true
  held_out_world: "nk"
  train_worlds: ["lss", "var", "rbc", "switching", "zlb"]
  test_worlds: ["nk"]

training:
  batch_size: 32
  epochs: 100
  # ...
```

Then in your trainer:

```python
from data import get_lowo_world_lists

if config.dataset.lowo_mode:
    # Use LOWO split
    train_worlds, test_worlds = get_lowo_world_lists(
        config.dataset.all_worlds,
        config.dataset.held_out_world
    )
else:
    # Use all worlds
    train_worlds = config.dataset.all_worlds
    test_worlds = config.dataset.all_worlds

train_dataset = IRFDataset(zarr_root, train_worlds, split="train")
test_dataset = IRFDataset(zarr_root, test_worlds, split="test_interpolation")
```

## Properties

### Determinism
- **No randomness**: LOWO splits are completely deterministic
- **Same input → Same output**: Given the same `world_ids` and `held_out_world`, the split is always identical
- **Reproducible**: No seed required

### Disjointness
- **Train/test separation**: Train worlds and test worlds are completely disjoint
- **No data leakage**: Held-out world samples never seen during training
- **Coverage**: Train + test worlds = all worlds

### World Filtering
- **Pre-filtering**: Worlds are filtered before data loading
- **Efficient**: Only loads data for relevant worlds
- **Compatible**: Works seamlessly with existing `IRFDataset` loader

## Validation

All LOWO splits are validated with comprehensive tests:

```bash
# Run LOWO tests
pytest data/tests/test_splits.py::TestLOWO -v

# Run all split tests
pytest data/tests/test_splits.py -v -m fast
```

### Test Coverage
- Basic functionality
- Train/test world correctness
- Disjointness verification
- All-world holdout coverage
- Invalid world error handling
- Order preservation
- Single world holdout size
- Minimal case (2 worlds)
- Determinism
- Convenience function equivalence

## Integration with Existing Infrastructure

### Dataset Loading
The `IRFDataset` class already supports world filtering via the `world_ids` parameter:

```python
# Load only specific worlds
dataset = IRFDataset(
    zarr_root="datasets/v1.0",
    world_ids=["lss", "var", "rbc"],  # Only these worlds
    split="train"
)
```

LOWO simply provides the correct `world_ids` lists for train/test.

### Splits Per World
LOWO operates at the **world level**, not the sample level:
- Within each world, standard splits (train/val/test) are still used
- LOWO determines **which worlds** are available for training vs testing
- Sample-level splits (interpolation, extrapolation) are orthogonal to LOWO

Example:
```python
# Train on 5 worlds, using their "train" samples
train_dataset = IRFDataset(zarr_root, train_worlds, split="train")

# Test on held-out world, using its "test_interpolation" samples
test_dataset = IRFDataset(zarr_root, test_worlds, split="test_interpolation")
```

## Acceptance Criteria (All Met)

- [x] LOWO split function that excludes specified world from training
- [x] Training set contains samples from 5 worlds
- [x] Test set contains samples only from held-out world
- [x] Unit tests verify disjointness and correct world filtering
- [x] Deterministic (no randomness)
- [x] Integration with existing `IRFDataset` loader
- [x] Documentation and examples

## Files Modified

1. **`/Users/nerland/macro_simulator/data/splits.py`**
   - Added `split_lowo()` function
   - Added `get_lowo_world_lists()` convenience function

2. **`/Users/nerland/macro_simulator/data/__init__.py`**
   - Exported `split_lowo` and `get_lowo_world_lists`

3. **`/Users/nerland/macro_simulator/data/tests/test_splits.py`**
   - Added `TestLOWO` test class with 11 tests
   - All tests pass

4. **`/Users/nerland/macro_simulator/examples/lowo_example.py`**
   - Complete usage examples
   - Demonstrates all patterns

## Performance Considerations

- **Lightweight**: LOWO split is a simple list filtering operation
- **No overhead**: O(n) where n = number of worlds (typically 6)
- **Memory efficient**: Only world IDs stored, not data
- **Lazy loading**: Data loaded only for relevant worlds

## Future Extensions

Possible enhancements for Sprint 6+:

1. **Multi-world holdout**: Hold out multiple worlds simultaneously
2. **World grouping**: Hold out related worlds (e.g., all linear models)
3. **Stratified LOWO**: Balance train set by world characteristics
4. **LOWO metrics**: Specialized evaluation for cross-world performance
5. **Config integration**: Automatic LOWO experiment generation from config

## References

- Specification: `spec/spec.md` §7.3 (Evaluation Splits)
- Task: Sprint 5 Task S5.1
- Related: Sprint 4 (universal emulator), Sprint 6 (LOWO experiments)
