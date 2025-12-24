# Evaluation and Visualization Module

This module provides evaluation metrics and visualization tools for the Universal Macro Emulator.

## Implemented Components (Sprint 1)

### S1.28: IRF Panel Plot Generator (`plots.py`)

The `plot_irf_panel()` function creates publication-quality IRF visualizations with flexible layouts:

**Features:**
- 3×N grid layout (rows: output, inflation, rate; columns: simulators or shocks)
- Support for 2D IRFs (single shock, multiple models) and 3D IRFs (multiple shocks)
- Two layout modes: `obs_rows` (observables as rows) and `shock_rows` (shocks as rows)
- Customizable horizon, labels, and titles
- Consistent color palette across simulators

**Example Usage:**

```python
from simulators import LSSSimulator
import numpy as np
from emulator.eval.plots import plot_irf_panel, save_figure

# Initialize simulator
lss = LSSSimulator()
rng = np.random.default_rng(42)
theta = lss.sample_parameters(rng)

# Compute IRF
irf = lss.compute_irf(theta, shock_idx=0, shock_size=1.0, H=40)

# Create plot
irfs = {"LSS": irf}
fig = plot_irf_panel(irfs, title="LSS Impulse Responses")

# Save to file
save_figure(fig, "lss_irfs.png")
```

### S1.29: Plot Styling and Save Functionality

**Plot Styling:**
- Publication-ready defaults (DPI=300 for saves, DPI=150 for display)
- Consistent font sizes and line widths
- Professional grid styling with 30% alpha
- Color palette optimized for simulator identification

**Save Functionality:**
- PNG output with high resolution (300 DPI)
- Multi-format support (PNG, PDF, SVG, etc.)
- Automatic directory creation
- Format inference from file extension

```python
# Save in multiple formats
save_figure(fig, "output", formats=["png", "pdf"])

# Auto-creates nested directories
save_figure(fig, "results/figures/irf_panel.png")
```

## Command-Line Interface

Generate IRF plots directly from the command line:

```bash
# Basic usage: plot LSS, VAR, NK simulators
python -m emulator.eval.plots --simulators lss,var,nk --output sprint1_irf_panel.png

# Custom horizon and shock size
python -m emulator.eval.plots --simulators lss --horizon 80 --shock-size 2.0 --output lss_large_shock.png

# Generate all shocks per simulator
python -m emulator.eval.plots --simulators lss,var --all-shocks --output multi_shock_panel.png

# Multiple output formats
python -m emulator.eval.plots --simulators lss,var,nk --formats png,pdf --output results/sprint1_irf_panel
```

### CLI Arguments

- `--simulators`: Comma-separated list of simulators (required)
- `--output`: Output file path (default: `irf_panel.png`)
- `--horizon`: IRF horizon length (default: 40)
- `--shock-idx`: Shock index to plot (default: 0, ignored if `--all-shocks`)
- `--shock-size`: Shock size in std devs (default: 1.0)
- `--all-shocks`: Plot all shocks for each simulator
- `--seed`: Random seed for parameter sampling (default: 42)
- `--layout`: Layout mode: `obs_rows` or `shock_rows` (default: `obs_rows`)
- `--formats`: Comma-separated output formats (default: `png`)

## Additional Functions

### `plot_irf_comparison()`

Compare ground truth vs predicted IRFs side-by-side:

```python
from emulator.eval.plots import plot_irf_comparison

fig = plot_irf_comparison(
    irf_true=ground_truth_irf,
    irf_pred=predicted_irf,
    title="Model Comparison"
)
save_figure(fig, "comparison.png")
```

### `setup_plot_style()`

Apply consistent matplotlib styling:

```python
from emulator.eval.plots import setup_plot_style

setup_plot_style()
# Now all matplotlib plots will use the consistent style
```

## Testing

Run unit tests:

```bash
# Fast tests only
pytest emulator/tests/test_plots.py -v -m fast

# All tests
pytest emulator/tests/test_plots.py -v
```

Test coverage includes:
- Plot generation with 2D and 3D IRFs
- Layout modes and custom labels
- Horizon truncation
- Shape validation
- Save functionality (PNG, PDF, multi-format)
- Directory creation
- Edge cases (empty data, single observable, zero horizon)

## Design Rationale

### Why This Layout?

The 3×N grid with observables as rows follows standard macro practice:
- Readers can scan horizontally to compare models for a given variable
- Vertical scanning shows the system-wide response (output-inflation-rate)
- Aligns with how economists think about policy transmission

### Color Palette

Colors chosen for:
- **LSS**: Blue (stable, linear)
- **VAR**: Orange (empirical, data-driven)
- **NK**: Green (microfounded, structural)
- **RBC**: Red (real shocks)
- **Switching**: Purple (regime changes)
- **ZLB**: Brown (constrained)
- **Ground Truth**: Black (authoritative)
- **Prediction**: Pink (estimated)

### Performance

- Plots render efficiently even with H=80 and multiple shocks
- Minimal dependencies (matplotlib, numpy)
- No heavy computation in plotting functions (pre-computed IRFs)

## Future Enhancements (Later Sprints)

- **Sprint 3**: Shape diagnostics (HF-ratio, overshoot, spectral plots)
- **Sprint 3**: Metric overlays (NRMSE annotations)
- **Sprint 4**: Regime comparison panels
- **Sprint 5**: LOWO transfer visualizations

## Acceptance Criteria (Sprint 1)

- [x] S1.28: IRF panel plot generator implemented
  - [x] 3×N grid layout
  - [x] Works with any simulator's IRF output
  - [x] Support for 2D (single shock) and 3D (multiple shocks)
  - [x] Customizable labels and titles

- [x] S1.29: Plot styling and save functionality
  - [x] Consistent matplotlib styling
  - [x] PNG output capability
  - [x] Clear axis labels, legends, titles
  - [x] Professional appearance

- [x] CLI support: `python -m emulator.eval.plots --simulators lss,var,nk --output sprint1_irf_panel.png`

- [x] Unit tests with >80% coverage

- [x] Documentation and examples

---

## Sprint 3: Evaluation Harness and Leaderboard

### Components Added

#### 1. Evaluation CLI (`evaluate.py`)

Main evaluation script for running model assessments on dataset splits.

**Usage:**
```bash
# Evaluate on test set (oracle mode - uses ground truth)
python -m emulator.eval.evaluate \
    --dataset datasets/v1.0/ \
    --worlds lss,var,nk \
    --splits test_interpolation \
    --output results.json

# Evaluate trained model checkpoint
python -m emulator.eval.evaluate \
    --checkpoint runs/universal_A/checkpoint.pt \
    --dataset datasets/v1.0/ \
    --worlds lss,var,nk,rbc,switching,zlb \
    --splits test_interpolation,test_extrapolation_slice \
    --weight-scheme exponential \
    --output eval_results.json
```

**Key Features:**
- Load model checkpoints or run in oracle mode (ground truth)
- Compute all metrics per world, per split, and aggregated
- Support for multiple dataset splits (interpolation, extrapolation)
- Flexible horizon weighting (uniform, exponential)
- JSON output with structured results

**Output Structure:**
```json
{
  "aggregate": {
    "nrmse": 0.15,
    "iae": 2.3,
    "sign_at_impact": 0.95,
    "hf_ratio": 0.12,
    "overshoot_ratio": 1.5,
    "sign_flip_count": 3.2,
    "n_samples": 5000
  },
  "per_world": {
    "lss": {"nrmse": 0.12, ...},
    "var": {"nrmse": 0.18, ...}
  },
  "per_split": {
    "test_interpolation": {"nrmse": 0.14, ...},
    "test_extrapolation_slice": {"nrmse": 0.20, ...}
  }
}
```

#### 2. Leaderboard Generator (`leaderboard.py`)

Compare multiple models and generate formatted tables for papers and posts.

**CLI Usage:**
```bash
# Compare all models in a directory
python -m emulator.eval.leaderboard compare \
    --results-dir experiments/sweep/ \
    --output-csv leaderboard.csv \
    --output-markdown leaderboard.md

# Custom leaderboard from specific files
python -m emulator.eval.leaderboard custom \
    --models "Universal:results/universal.json,MLP:results/mlp.json,Oracle:results/oracle.json" \
    --oracle "Oracle" \
    --output-csv leaderboard.csv
```

**Programmatic Usage:**
```python
from emulator.eval import LeaderboardGenerator

leaderboard = LeaderboardGenerator()
leaderboard.add_model("Universal (Regime A)", "results/universal_A.json")
leaderboard.add_model("MLP Baseline", "results/mlp.json")
leaderboard.add_model("Oracle", "results/oracle.json", is_oracle=True)

# Generate table sorted by NRMSE
df = leaderboard.generate_table(sort_by="NRMSE", ascending=True)
leaderboard.save_csv("leaderboard.csv")
leaderboard.save_markdown("leaderboard.md")
leaderboard.print_summary()
```

**Success Criteria Checking:**
```python
from emulator.eval import compute_success_criteria

criteria = compute_success_criteria(
    results_json_path="results/universal_A.json",
    specialist_results={
        "lss": "results/specialist_lss.json",
        "var": "results/specialist_var.json",
    },
    baseline_results={
        "mlp": "results/mlp_baseline.json",
    }
)

# Automatic checks per spec §7.2.3
print(f"Beat baselines: {criteria['beat_all_baselines']}")
print(f"Mean gap: {criteria['mean_gap']:.1f}% (≤20%: {criteria['mean_gap_pass']})")
print(f"Max gap: {criteria['max_gap']:.1f}% (≤35%: {criteria['max_gap_pass']})")
print(f"Shape preservation: {criteria['mean_hf_ratio']:.2f} (≤1.1: {criteria['shape_preservation_pass']})")
```

#### 3. Metrics Module (`metrics.py`)

Core evaluation metrics as specified in §7:

**Accuracy Metrics:**
- `nrmse()`: Normalized RMSE with horizon weighting
- `iae()`: Integrated Absolute Error
- `sign_at_impact()`: Direction correctness

**Shape Metrics:**
- `hf_ratio()`: High-frequency energy (detects oscillations)
- `overshoot_ratio()`: Peak-to-impact ratio
- `sign_flip_count()`: Sign change counting

**Weighting Schemes:**
- `uniform_weights()`: Equal weight to all horizons
- `exponential_weights()`: Exponential decay (tau=20)
- `impact_weighted()`: Emphasize first few horizons

### Evaluation Workflow

**1. Train Models:**
```bash
python -m emulator.training.trainer --config configs/universal_regime_A.yaml
python -m emulator.training.trainer --config configs/mlp_baseline.yaml
```

**2. Evaluate on Splits:**
```bash
for model in universal_A mlp_baseline oracle; do
    python -m emulator.eval.evaluate \
        --checkpoint runs/${model}/checkpoint.pt \
        --dataset datasets/v1.0/ \
        --splits test_interpolation,test_extrapolation_slice \
        --output eval/${model}.json
done
```

**3. Generate Leaderboard:**
```bash
python -m emulator.eval.leaderboard compare \
    --results-dir eval/ \
    --output-csv leaderboard.csv \
    --output-markdown leaderboard.md
```

### Success Criteria (Spec §7.2.3)

The universal emulator must meet:

1. **Beat baselines**: NRMSE_universal < NRMSE_baseline for all worlds
2. **Mean gap**: mean_w(Gap(w)) ≤ 20% vs specialists
3. **Max gap**: max_w(Gap(w)) ≤ 35% vs specialists
4. **Shape preservation**: mean HF_ratio ≤ 1.1× specialist

Use `compute_success_criteria()` to automatically verify these conditions.

### Testing

Comprehensive unit tests for all components:

```bash
# Run evaluation harness tests
pytest emulator/tests/test_evaluate.py -m fast -v

# Run leaderboard tests
pytest emulator/tests/test_leaderboard.py -m fast -v

# All evaluation tests
pytest emulator/tests/test_evaluate.py emulator/tests/test_leaderboard.py -v
```

**Test Coverage:**
- Metric computation with toy data
- Perfect prediction edge case (zero error)
- Multi-shock IRF handling
- Weighted aggregation
- Leaderboard sorting and formatting
- Gap computation vs oracle
- Success criteria validation
- CSV/Markdown export
- Per-world breakdown tables

### Adding New Metrics

To extend the evaluation framework:

1. **Implement in `metrics.py`:**
```python
def my_metric(
    y_pred: npt.NDArray,
    y_true: npt.NDArray,
) -> float:
    """New metric description.

    Rationale: Detects X failure mode.

    Formula: metric = ...
    """
    # Implementation
    return value
```

2. **Add to `compute_all_metrics()` in `evaluate.py`:**
```python
metrics["my_metric"] = float(my_metric(y_pred_avg, y_true_avg))
```

3. **Update leaderboard columns in `leaderboard.py`:**
```python
row = {
    ...
    "MyMetric": metrics.get("my_metric", float("nan")),
}
```

4. **Write unit tests:**
```python
@pytest.mark.fast
def test_my_metric():
    y_pred = np.random.randn(10, 41, 3)
    y_true = np.random.randn(10, 41, 3)
    value = my_metric(y_pred, y_true)
    assert value >= 0.0  # or other expected properties
```

### Dependencies

- `numpy`: Core numerical operations
- `pandas`: Leaderboard table generation
- `zarr`: Dataset loading
- `torch` (optional): Required only for loading model checkpoints

Oracle mode (ground truth evaluation) works without PyTorch.

### Acceptance Criteria (Sprint 3)

- [x] Evaluation CLI (`evaluate.py`) with JSON output
  - [x] Load checkpoints or run oracle mode
  - [x] Per-world, per-split, and aggregate metrics
  - [x] Support for all 6 worlds
  - [x] Configurable weighting schemes

- [x] Leaderboard generator (`leaderboard.py`)
  - [x] Compare multiple models
  - [x] CSV and Markdown export
  - [x] Per-world breakdown tables
  - [x] Gap computation vs oracle
  - [x] Success criteria checking

- [x] Comprehensive unit tests (24 tests, all passing)
  - [x] Metric computation
  - [x] Aggregation logic
  - [x] Leaderboard sorting
  - [x] File I/O

- [x] Updated `__init__.py` with new exports

### References

- Spec §7: Evaluation Framework
- Spec §7.2.3: Success Criteria
- `CLAUDE.md`: Project conventions
