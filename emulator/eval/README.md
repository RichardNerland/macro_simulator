# Evaluation System for Universal Macro Emulator

This directory contains the complete evaluation pipeline for the Universal Macro Emulator project.

## Overview

The evaluation system provides:
- Model loading and inference (with automatic architecture detection)
- Comprehensive metric computation (accuracy and shape metrics)
- Publication-quality visualization
- Leaderboard generation for model comparison
- Support for multiple information regimes (A, B1, C)

## Key Modules

### 1. `evaluate.py` - Main Evaluation Harness

**Purpose**: End-to-end evaluation of trained models on dataset splits

**Key Functions**:
- `load_universal_model(checkpoint_path)`: Load model with automatic architecture inference
- `compute_predictions(model, theta, world_id, ...)`: Batch inference
- `evaluate_on_split(model, dataset_path, world_id, split_name)`: Evaluate on single split
- `evaluate_all_worlds(model, dataset_path, world_ids, split_names)`: Full evaluation

**Usage**:
```bash
# Evaluate model on test set
python -m emulator.eval.evaluate \
    --checkpoint runs/smoke_test_regime_A/best.pt \
    --dataset datasets/v1.0-dev/ \
    --worlds lss,var,nk \
    --splits test_interpolation \
    --output results/universal_A.json
```

**Key Features**:
- **Automatic architecture inference**: Reads model structure from state_dict, no config file needed
- **Oracle mode**: Run without checkpoint for ground-truth validation
- **Regime detection**: Automatically detects information regime from checkpoint
- **Robust dataset loading**: Supports multiple dataset structures

### 2. `metrics.py` - Metric Computation

**Purpose**: Compute accuracy and shape preservation metrics

**Metrics**:
- **Accuracy Metrics**:
  - `nrmse`: Normalized Root Mean Square Error (weighted by horizon)
  - `iae`: Integrated Absolute Error
  - `sign_at_impact`: Sign-at-impact accuracy (first 3 horizons)

- **Shape Metrics**:
  - `hf_ratio`: High-frequency ratio (detects spurious oscillations)
  - `overshoot_ratio`: Fraction of IRFs with overshooting
  - `sign_flip_count`: Number of spurious sign reversals

**Weighting Schemes**:
- `uniform_weights`: Equal weight to all horizons
- `exponential_weights`: Exponentially decay weights (emphasize near-term)

### 3. `figures.py` - Publication-Quality Visualization

**Purpose**: Generate standardized figures for papers and blog posts

**Functions**:
1. `plot_nrmse_bar_chart(results, output_path, baseline_results)`:
   - Per-world NRMSE comparison
   - Universal vs baseline models
   - Grouped bar chart

2. `plot_shape_comparison(results, output_path, baseline_results)`:
   - 3-panel figure: HF-ratio, Overshoot, Sign-flips
   - Compares universal vs baselines across all worlds

3. `plot_regime_comparison(results_A, results_B1, results_C, output_path)`:
   - Compare Regime A vs B1 vs C performance
   - Shows information loss across regimes

4. `plot_irf_panel(y_pred, y_true, world_id, output_path)`:
   - 3-row panel (output, inflation, rate)
   - Prediction vs ground truth overlay
   - Optional confidence bands

**Example**:
```python
from emulator.eval.figures import plot_nrmse_bar_chart, plot_irf_panel

# NRMSE comparison
results = {...}  # From evaluate.py
baselines = {"MLP": {...}, "VAR": {...}}
plot_nrmse_bar_chart(results, "nrmse_comparison.png", baselines)

# IRF panel
plot_irf_panel(y_pred, y_true, "lss", "lss_irf_panel.png")
```

### 4. `leaderboard.py` - Model Comparison

**Purpose**: Aggregate and compare evaluation results across multiple models

**Key Class**: `LeaderboardGenerator`

**Usage**:
```python
from emulator.eval.leaderboard import LeaderboardGenerator

leaderboard = LeaderboardGenerator()
leaderboard.add_model("Universal (Regime A)", "results/universal_A.json")
leaderboard.add_model("MLP Baseline", "results/mlp_baseline.json", is_oracle=False)
leaderboard.add_model("Oracle", "results/oracle.json", is_oracle=True)

# Generate table
df = leaderboard.generate_table()
leaderboard.save_csv("leaderboard.csv")
leaderboard.save_markdown("leaderboard.md")
```

**CLI**:
```bash
# Compare all models in directory
python -m emulator.eval.leaderboard compare \
    --results-dir results/ \
    --output-csv leaderboard.csv \
    --output-markdown leaderboard.md
```

### 5. `plots.py` - Core Plotting Utilities

**Purpose**: Low-level plotting functions for IRFs and dataset diagnostics

**Key Functions**:
- `plot_irf_panel`: Multi-simulator/multi-shock IRF grids
- `plot_irf_comparison`: Ground truth vs prediction comparison
- `plot_parameter_histograms`: Parameter distribution diagnostics
- `plot_sample_irfs`: Random sample visualization

Used by `figures.py` for higher-level visualizations.

## Workflow

### 1. Train Model
```bash
python -m emulator.training.trainer --config configs/universal_regime_A.yaml
```

### 2. Evaluate Model
```bash
python -m emulator.eval.evaluate \
    --checkpoint runs/universal_A/best.pt \
    --dataset datasets/v1.0/ \
    --worlds lss,var,nk,rbc,switching,zlb \
    --splits test_interpolation,test_extrapolation_slice \
    --output results/universal_A.json
```

### 3. Generate Figures
```bash
python examples/evaluate_model.py \
    --checkpoint runs/universal_A/best.pt \
    --dataset datasets/v1.0/ \
    --output-dir results/figures/ \
    --figures
```

### 4. Create Leaderboard
```bash
python -m emulator.eval.leaderboard compare \
    --results-dir results/ \
    --output-csv leaderboard.csv \
    --output-markdown leaderboard.md
```

## Output Structure

### Evaluation Results JSON
```json
{
  "aggregate": {
    "nrmse": 0.15,
    "iae": 1.2,
    "sign_at_impact": 0.95,
    "hf_ratio": 1.1,
    "overshoot_ratio": 0.3,
    "sign_flip_count": 1.5,
    "n_samples": 1000
  },
  "per_world": {
    "lss": {"nrmse": 0.12, ...},
    "var": {"nrmse": 0.14, ...},
    ...
  },
  "per_split": {
    "test_interpolation": {...},
    "test_extrapolation_slice": {...}
  },
  "per_world_per_split": {
    "lss": {
      "test_interpolation": {...},
      ...
    }
  }
}
```

### Leaderboard CSV/Markdown
```
Model                  | NRMSE  | IAE   | Sign@Impact | HF-ratio | Gap (%)
-----------------------|--------|-------|-------------|----------|--------
Universal (Regime A)   | 0.150  | 1.20  | 0.95        | 1.10     | 15.0
MLP Baseline           | 0.200  | 1.80  | 0.88        | 1.25     | 53.8
Oracle                 | 0.130  | 1.00  | 1.00        | 1.00     | --
```

## Advanced Features

### Automatic Architecture Inference

The `load_universal_model` function automatically infers model architecture from the state_dict:
- No need to save architecture config in checkpoint
- Works with any checkpoint structure
- Detects MLP vs Transformer trunk
- Infers all embedding dimensions
- Determines horizon H and n_obs

This makes the evaluation pipeline robust to changes in model architecture.

### Multi-Regime Support

The evaluation system supports all three information regimes:
- **Regime A**: Full structural (world_id, theta, shock_token)
- **Regime B1**: Observable-only (world_id, shock_token, history)
- **Regime C**: Partial (world_id, theta, shock_token, history)

Regime is automatically detected from checkpoint config.

### Dataset Compatibility

The evaluation system works with both:
- World-specific splits (`lss/splits.json`)
- Manifest-based splits (`manifest.json`)

This ensures compatibility with different dataset versions.

## Testing

Run tests:
```bash
# Fast unit tests
pytest emulator/tests/test_evaluate_integration.py -m fast -v
pytest emulator/tests/test_figures.py -m fast -v

# Integration tests (slower)
pytest emulator/tests/test_evaluate_integration.py -m slow -v
```

## Notes

- All metrics are computed per-sample, then aggregated with sample-size weighting
- NRMSE is normalized by std(y_true) computed from data
- Shape metrics help detect qualitative failures (oscillations, overshooting)
- Figures are publication-ready (300 DPI, consistent styling)

## Related Documentation

- `EVALUATION_INTEGRATION_SUMMARY.md` - Implementation summary (S4.23-S4.26)
- `spec/spec.md` - Full technical specification
- `examples/evaluate_model.py` - Example evaluation script
