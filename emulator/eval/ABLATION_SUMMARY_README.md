# Ablation Summary Module

## Overview

The `ablation_summary.py` module provides tools for comparing ablation study results and generating summary tables showing the impact of removing each input component from a model.

## Purpose

In ablation studies, we systematically remove components from a model to measure their importance. This module:

1. Loads evaluation results for a full model and various ablations
2. Computes delta metrics (percentage change when component is removed)
3. Generates formatted comparison tables
4. Interprets results to identify most important components

## Key Concepts

### Delta Metrics

Delta metrics show the percentage change in performance when a component is removed:

```
Δ NRMSE % = (NRMSE_ablation - NRMSE_full) / NRMSE_full × 100%
```

**Interpretation:**
- **Positive Δ %**: Removing the component made performance worse → component helps
- **Negative Δ %**: Removing the component improved performance → component may hurt
- **Larger absolute Δ %**: More important component

### Example

If the full model has NRMSE = 0.15 and removing theta gives NRMSE = 0.22:
- Δ NRMSE % = (0.22 - 0.15) / 0.15 × 100% = +46.7%
- This means theta is very important (removing it degraded performance by 46.7%)

## Usage

### Python API

```python
from emulator.eval.ablation_summary import generate_ablation_summary

# Define result paths
full_results_path = "results/universal_A.json"
ablation_results = {
    "No World ID": "results/ablation_no_world.json",
    "No Theta": "results/ablation_no_theta.json",
    "No Eps": "results/ablation_no_eps.json",
}

# Generate summary
summary = generate_ablation_summary(full_results_path, ablation_results)

# Display
summary.print_summary()

# Save
summary.save_csv("ablation_summary.csv")
summary.save_markdown("ablation_summary.md")
```

### Command Line Interface

#### Specify individual ablations

```bash
python -m emulator.eval.ablation_summary \
    --full results/universal_A.json \
    --ablations "No World:results/ablation_no_world.json,No Theta:results/ablation_no_theta.json" \
    --output ablation_summary.md
```

#### Auto-discover ablations in a directory

```bash
python -m emulator.eval.ablation_summary \
    --full results/universal_A.json \
    --ablations-dir results/ \
    --output ablation_summary.md
```

This will find all `ablation_*.json` files in the directory.

#### Options

- `--full`: Path to full model results (required)
- `--ablations`: Comma-separated name:path pairs (mutually exclusive with --ablations-dir)
- `--ablations-dir`: Directory with ablation_*.json files (mutually exclusive with --ablations)
- `--output`: Output file path (.md or .csv, default: ablation_summary.md)
- `--sort-by`: Metric to sort by (default: nrmse)

## Input Format

Evaluation results must be in the standard JSON format produced by `evaluate.py`:

```json
{
  "aggregate": {
    "nrmse": 0.15,
    "iae": 2.3,
    "sign_at_impact": 0.92,
    "hf_ratio": 0.08,
    "overshoot_ratio": 1.15,
    "sign_flip_count": 2.1,
    "n_samples": 1000
  },
  "per_world": {...},
  "per_split": {...}
}
```

Only the `aggregate` section is used for the summary table.

## Output Format

### Console Output

```
====================================================================================================
ABLATION STUDY SUMMARY
====================================================================================================
                Model  NRMSE  Δ NRMSE %    IAE  Δ IAE %  Sign@Impact  HF-ratio  Samples
           Full Model 0.1500     0.0000 2.3000   0.0000       0.9200    0.0800     1000
      No Eps (Shocks) 0.1600     6.6667 2.4000   4.3478       0.9100    0.0800     1000
           No History 0.1700    13.3333 2.6000  13.0435       0.8900    0.0900     1000
          No World ID 0.1800    20.0000 2.8000  21.7391       0.8800    0.0900     1000
No Theta (Parameters) 0.2200    46.6667 3.1000  34.7826       0.8500    0.1000     1000
====================================================================================================

Interpretation:
  - Positive Δ % = Removing component hurt performance (component helps)
  - Negative Δ % = Removing component improved performance (component may hurt)
  - Larger absolute Δ % = More important component
====================================================================================================
```

### CSV Output

Simple CSV with all columns, suitable for further analysis in Excel/R/Python.

### Markdown Output

Nicely formatted Markdown with:
- Explanatory header
- Formatted table (using +/- signs for deltas)
- Metric definitions
- Interpretation guide

Example markdown output:

```markdown
# Ablation Study Summary

## Interpretation

This table shows the impact of removing each input component from the full model.

- **Positive Δ %**: Removing the component made performance worse → component helps
- **Negative Δ %**: Removing the component improved performance → component may hurt
- **Larger absolute Δ %**: More important component

## Results

| Model                 | NRMSE | Δ NRMSE % | IAE  | Δ IAE % | Sign@Impact | HF-ratio | Samples |
|:----------------------|------:|:----------|-----:|:--------|------------:|---------:|--------:|
| Full Model            |  0.15 | baseline  | 2.3  | baseline|        0.92 |     0.08 |    1000 |
| No Eps (Shocks)       |  0.16 | +6.7%     | 2.4  | +4.3%   |        0.91 |     0.08 |    1000 |
| No World ID           |  0.18 | +20.0%    | 2.8  | +21.7%  |        0.88 |     0.09 |    1000 |
| No Theta (Parameters) |  0.22 | +46.7%    | 3.1  | +34.8%  |        0.85 |     0.10 |    1000 |

## Metrics

- **NRMSE**: Normalized Root Mean Square Error (lower is better)
- **IAE**: Integrated Absolute Error (lower is better)
- **Sign@Impact**: Fraction of correct signs in first 3 horizons (higher is better)
- **HF-ratio**: High-frequency energy ratio for oscillation detection (lower is better)
```

## Common Workflows

### 1. Full Ablation Study

Run evaluation on full model and all ablations, then summarize:

```bash
# Evaluate full model
python -m emulator.eval.evaluate \
    --checkpoint runs/universal_A/checkpoint.pt \
    --dataset datasets/v1.0/ \
    --output results/full_model.json

# Evaluate ablations
for ablation in no_world no_theta no_eps; do
    python -m emulator.eval.evaluate \
        --checkpoint runs/ablation_${ablation}/checkpoint.pt \
        --dataset datasets/v1.0/ \
        --output results/ablation_${ablation}.json
done

# Generate summary
python -m emulator.eval.ablation_summary \
    --full results/full_model.json \
    --ablations-dir results/ \
    --output ablation_summary.md
```

### 2. Component Importance Ranking

Sort by delta percentage to rank components by importance:

```python
summary = generate_ablation_summary(full_path, ablation_paths)
df = summary.generate_table(sort_by="nrmse", ascending=True)

# Components with largest positive delta are most important
print(df[["Model", "Δ NRMSE %"]].iloc[1:])  # Skip full model
```

### 3. Multi-Metric Analysis

Compare multiple metrics to understand different aspects:

```python
df = summary.generate_table()

# Show all delta columns
print(df[["Model", "Δ NRMSE %", "Δ IAE %"]])
```

## Implementation Details

### Class: `AblationSummary`

Main class for managing ablation comparisons.

**Methods:**
- `add_full_model(path)`: Load full model results
- `add_ablation(name, path)`: Add an ablation
- `generate_table(sort_by, ascending)`: Generate comparison DataFrame
- `save_csv(path)`: Save to CSV
- `save_markdown(path)`: Save to Markdown with formatting
- `print_summary()`: Print to console

### Function: `generate_ablation_summary()`

High-level convenience function that creates an `AblationSummary` and loads all results.

### Function: `load_ablation_results()`

Auto-discover ablation result files in a directory using glob patterns.

## Testing

Comprehensive test suite in `emulator/tests/test_ablation_summary.py`:

```bash
# Run all ablation summary tests
pytest emulator/tests/test_ablation_summary.py -v

# Run specific test
pytest emulator/tests/test_ablation_summary.py::test_generate_table -v
```

Tests cover:
- Loading from JSON files
- Delta computation
- Table generation and sorting
- CSV/Markdown output
- Error handling
- Edge cases (negative deltas, missing files)

## Examples

See `examples/ablation_summary_demo.py` for a complete working example.

Run the demo:

```bash
PYTHONPATH=/Users/nerland/macro_simulator python examples/ablation_summary_demo.py
```

## Integration with Evaluation Pipeline

This module integrates seamlessly with the evaluation pipeline:

1. `evaluate.py` generates JSON results
2. `ablation_summary.py` compares results across models
3. `leaderboard.py` ranks different model architectures

All three use the same JSON format and metric definitions from `metrics.py`.

## Best Practices

1. **Consistent Seeds**: Use the same random seed for all evaluations to ensure fair comparison
2. **Same Dataset**: Evaluate all models on the same test split
3. **Naming Convention**: Use descriptive ablation names that clearly indicate what was removed
4. **Multiple Metrics**: Look at both NRMSE and IAE; sometimes components affect them differently
5. **Sign@Impact**: Check if ablations preserve directional accuracy even if magnitude is off

## Troubleshooting

### "Full model metrics not set"

You must call `add_full_model()` before calling `generate_table()`.

### "Metrics file not found"

Check that the path is correct and the file exists. Use absolute paths if needed.

### Sorting not working

The module handles case-insensitive column matching. Valid sort keys include:
- `"nrmse"` or `"NRMSE"`
- `"iae"` or `"IAE"`
- Any column name in the DataFrame

### Negative deltas

Negative deltas mean the ablation performed BETTER than the full model. This could indicate:
- The removed component was hurting performance (overfitting, noise)
- Statistical variance (run multiple seeds to confirm)
- Bug in the full model implementation

## Future Enhancements

Potential additions:
- Per-world ablation analysis
- Statistical significance testing (bootstrap confidence intervals)
- Visualization (bar charts of delta percentages)
- Multi-model comparison (compare ablations across different architectures)
