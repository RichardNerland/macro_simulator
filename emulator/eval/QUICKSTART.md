# IRF Visualization - Quick Start Guide

## 5-Minute Tutorial

### 1. Command Line (Easiest)

Generate IRF plots directly from the terminal:

```bash
# Compare LSS, VAR, and NK simulators (shock 0)
python -m emulator.eval.plots --simulators lss,var,nk --output sprint1_irf_panel.png

# Show all shocks from LSS model
python -m emulator.eval.plots --simulators lss --all-shocks --output lss_all_shocks.png

# Custom settings
python -m emulator.eval.plots \
    --simulators lss,var,nk \
    --horizon 80 \
    --shock-size 2.0 \
    --seed 123 \
    --output custom_panel.png
```

### 2. Python Script (Flexible)

```python
from simulators import LSSSimulator, VARSimulator, NKSimulator
from emulator.eval.plots import plot_irf_panel, save_figure
import numpy as np

# Initialize
rng = np.random.default_rng(42)
lss = LSSSimulator()
var = VARSimulator()

# Generate IRFs
theta_lss = lss.sample_parameters(rng)
theta_var = var.sample_parameters(rng)

irf_lss = lss.compute_irf(theta_lss, shock_idx=0, shock_size=1.0, H=40)
irf_var = var.compute_irf(theta_var, shock_idx=0, shock_size=1.0, H=40)

# Plot
irfs = {"LSS": irf_lss, "VAR": irf_var}
fig = plot_irf_panel(irfs, title="LSS vs VAR")
save_figure(fig, "comparison.png")
```

### 3. Programmatic (Full Control)

```python
from emulator.eval.plots import plot_irf_panel, save_figure

# Your IRF data (H+1, 3) arrays
irf_model1 = ...  # shape: (41, 3)
irf_model2 = ...  # shape: (41, 3)

# Create plot
irfs = {
    "Model 1": irf_model1,
    "Model 2": irf_model2,
}

fig = plot_irf_panel(
    irfs=irfs,
    horizon=40,
    obs_names=["Output", "Inflation", "Rate"],
    title="Model Comparison",
)

# Save
save_figure(fig, "my_plot.png", dpi=300)
```

## Common Use Cases

### Compare Multiple Simulators (Single Shock)

```bash
python -m emulator.eval.plots \
    --simulators lss,var,nk \
    --shock-idx 0 \
    --output simulators_shock0.png
```

### All Shocks from One Simulator

```bash
python -m emulator.eval.plots \
    --simulators lss \
    --all-shocks \
    --output lss_all_shocks.png
```

### Large Shock Response

```bash
python -m emulator.eval.plots \
    --simulators nk \
    --shock-size 3.0 \
    --horizon 80 \
    --output nk_large_shock.png
```

### Multi-Format Output

```bash
python -m emulator.eval.plots \
    --simulators lss,var,nk \
    --formats png,pdf \
    --output results/panel
# Creates: results/panel.png and results/panel.pdf
```

### Custom Layout (Shocks as Rows)

```python
from emulator.eval.plots import plot_irf_panel

# irf_data has shape (n_shocks, H+1, 3)
fig = plot_irf_panel(
    irfs={"Model": irf_data},
    layout="shock_rows",  # Rows=shocks, Cols=observables
    shock_labels=["Monetary", "Technology"],
)
```

## CLI Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--simulators` | (required) | Comma-separated: `lss,var,nk` |
| `--output` | `irf_panel.png` | Output file path |
| `--horizon` | `40` | IRF horizon (0 to H) |
| `--shock-idx` | `0` | Shock index (ignored if --all-shocks) |
| `--shock-size` | `1.0` | Shock size (std devs) |
| `--all-shocks` | `False` | Plot all shocks per simulator |
| `--seed` | `42` | Random seed for sampling |
| `--layout` | `obs_rows` | Layout: `obs_rows` or `shock_rows` |
| `--formats` | `png` | Formats: `png,pdf,svg` |

## Tips

1. **Use `--all-shocks` for comprehensive view**: Shows how each shock affects all observables
2. **Use `obs_rows` layout (default)**: Best for comparing models
3. **Use `shock_rows` layout**: Best for analyzing shock transmission
4. **Set high DPI for presentations**: Default is 300, suitable for publication
5. **Use deterministic seeds**: Ensures reproducible plots

## Output Format

All plots follow this structure:

**Default Layout (obs_rows):**
```
           Shock 0    Shock 1    Shock 2
Output     [plot]     [plot]     [plot]
Inflation  [plot]     [plot]     [plot]
Rate       [plot]     [plot]     [plot]
```

Each subplot shows:
- X-axis: Horizon (0 to H)
- Y-axis: Response (in percent for canonical observables)
- Zero line (dashed gray)
- Grid for readability
- Legend (first subplot only)

## Troubleshooting

**Import Error:**
```bash
# Make sure you're in the project root
cd /Users/nerland/macro_simulator
python -m emulator.eval.plots ...
```

**Unknown Simulator:**
```bash
# Available simulators: lss, var, nk
# (rbc, switching, zlb coming in Sprint 2)
python -m emulator.eval.plots --simulators lss,var,nk --output panel.png
```

**Shape Mismatch:**
```python
# All IRFs must have same shape
# For 2D: all (H+1, n_obs)
# For 3D: all (n_shocks, H+1, n_obs)
irfs = {
    "Model A": irf_a,  # (41, 3)
    "Model B": irf_b,  # (41, 3) ✓ Same shape
}
```

## Next Steps

- See `emulator/eval/README.md` for detailed documentation
- Run `python test_visualization_demo.py` for examples
- Check `emulator/tests/test_plots.py` for usage patterns
- Explore `emulator/eval/plots.py` source for advanced customization

## Quick Demo

Try this now:
```bash
python -m emulator.eval.plots \
    --simulators lss,var,nk \
    --all-shocks \
    --seed 42 \
    --output demo_panel.png

# Open demo_panel.png to see the result
```

This creates a comprehensive panel showing all shocks from all three simulators!
