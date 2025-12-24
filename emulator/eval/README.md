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
