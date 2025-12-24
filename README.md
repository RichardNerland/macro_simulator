# Universal Macro Emulator

[![CI](https://github.com/your-repo/macro_simulator/workflows/CI/badge.svg)](https://github.com/your-repo/macro_simulator)
[![Tests](https://github.com/your-repo/macro_simulator/workflows/Tests/badge.svg)](https://github.com/your-repo/macro_simulator)

A neural network emulator that predicts impulse response functions (IRFs) and trajectories across diverse macroeconomic simulators. Train a single model that generalizes across 6+ simulator families including LSS, VAR, New Keynesian, RBC, regime-switching, and ZLB models.

## Overview

The Universal Macro Emulator is designed for macroeconomic researchers and practitioners who need fast, accurate predictions of how economic systems respond to shocks. Instead of running computationally expensive simulations repeatedly, use a neural network that learns the input-output patterns of multiple macro models.

**Key Features:**
- 🧠 **One model, many simulators:** Train once, use for LSS, VAR, NK, RBC, switching, ZLB
- ⚡ **Fast inference:** Predict IRFs in milliseconds (vs minutes for direct simulation)
- 📊 **Multiple prediction tasks:** IRFs (impulse responses) and full trajectories
- 🎯 **Information regime flexibility:** Handle different levels of information availability
- 🧪 **Research-ready:** Fully tested, reproducible, with evaluation pipelines
- 📈 **Extensible:** Easy to add new simulators via the `SimulatorAdapter` protocol

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy, SciPy, Zarr, Pandas

## Installation

### Basic Installation

Clone the repository and install in development mode:

```bash
git clone https://github.com/your-repo/macro_simulator.git
cd macro_simulator
pip install -e ".[dev]"
```

This installs:
- Core dependencies: torch, numpy, scipy, zarr, pandas, matplotlib, pyyaml, tqdm
- Development tools: pytest, ruff (linting), mypy (type checking), pre-commit

### Verify Installation

```bash
python -c "from simulators import get_simulator; print('Installation successful!')"
pytest -m "fast" -v  # Run quick tests
```

## Quick Start

### 1. Generate a Dataset

Generate synthetic data from multiple simulators:

```bash
# Small dataset for testing (10k samples per world)
python -m data.scripts.generate_dataset \
  --world all \
  --n_samples 10000 \
  --seed 42 \
  --output datasets/v1.0/

# Validate the dataset
python -m data.scripts.validate_dataset --path datasets/v1.0/
```

This creates:
```
datasets/v1.0/
├── manifest.json          # Metadata, random seeds, train/test splits
├── lss/
│   ├── trajectories.zarr  # Simulated time series
│   ├── irfs.zarr          # Impulse response functions
│   ├── theta.zarr         # Structural parameters
│   └── shocks.zarr        # Random shocks used
├── var/
├── nk/
├── rbc/
├── switching/
└── zlb/
```

### 2. Train a Universal Model

Train a single neural network to predict IRFs across all simulators:

```bash
# Regime A (full structural information)
python -m emulator.training.trainer --config configs/universal_regime_A.yaml

# View results
ls runs/universal_regime_A/
```

Training outputs:
- `checkpoint.pt` - Trained model weights
- `best.pt` - Best validation checkpoint
- `config.yaml` - Exact configuration used
- `training_history.csv` - Loss curves (can plot with matplotlib)

### 3. Evaluate the Model

Compare your model to baselines and ground truth:

```bash
python examples/evaluate_model.py \
  --checkpoint runs/universal_regime_A/best.pt \
  --dataset datasets/v1.0/ \
  --worlds lss,var,nk \
  --splits test_interpolation \
  --output-dir results/

# Generate figures
python examples/evaluate_model.py \
  --checkpoint runs/universal_regime_A/best.pt \
  --dataset datasets/v1.0/ \
  --figures \
  --output-dir results/
```

Evaluation produces:
- `results.json` - Quantitative metrics (NRMSE, IAE, sign accuracy, shape measures)
- `leaderboard.csv` - Comparison across multiple models
- `*.png` - Visualizations (IRF panels, bar charts, comparisons)

### 4. Visualize Simulator Outputs

Generate publication-ready IRF plots from any simulator:

```bash
# IRF panel for LSS, VAR, NK simulators
python -m emulator.eval.plots \
  --simulators lss,var,nk \
  --output sprint1_irf_panel.png

# All shocks for one simulator
python -m emulator.eval.plots \
  --simulators lss \
  --all-shocks \
  --output lss_all_shocks.png
```

## Project Structure

```
macro_simulator/
├── README.md                          # This file
├── CLAUDE.md                          # Developer guide (detailed commands)
├── pyproject.toml                     # Package configuration, dependencies
├── spec/
│   ├── spec.md                        # Full technical specification
│   └── sprint-plan.md                 # Sprint breakdown & acceptance criteria
│
├── simulators/                        # Macroeconomic model implementations
│   ├── base.py                        # SimulatorAdapter protocol
│   ├── lss.py                         # Linear State Space (LSS)
│   ├── var.py                         # Vector Autoregression (VAR)
│   ├── nk.py                          # New Keynesian (NK)
│   ├── rbc.py                         # Real Business Cycle (RBC)
│   ├── switching.py                   # Regime-switching models
│   ├── zlb.py                         # Zero Lower Bound constraints
│   └── tests/                         # Simulator tests
│
├── data/                              # Data pipeline
│   ├── manifest.py                    # Dataset metadata & split definitions
│   ├── writer.py                      # Zarr-based data writing
│   ├── scripts/
│   │   ├── generate_dataset.py        # Create synthetic training data
│   │   └── validate_dataset.py        # Verify dataset integrity
│   └── tests/                         # Data pipeline tests
│
├── emulator/                          # Neural network emulator
│   ├── models/                        # PyTorch model architectures
│   │   ├── universal.py               # Main UniversalEmulator class
│   │   ├── embedders.py               # Input encoding modules
│   │   ├── backbone.py                # Transformer/MLP backbone
│   │   └── decoders.py                # Output decoding heads
│   ├── training/
│   │   └── trainer.py                 # Training loop & checkpointing
│   ├── eval/
│   │   ├── evaluate.py                # Evaluation pipeline
│   │   ├── metrics.py                 # NRMSE, IAE, shape metrics
│   │   ├── plots.py                   # Simulator visualization
│   │   ├── figures.py                 # Result visualization
│   │   └── leaderboard.py             # Model comparison
│   └── tests/                         # Model & training tests
│
├── configs/                           # Training configuration templates
│   ├── universal_regime_A.yaml        # Full information regime
│   ├── universal_regime_B1.yaml       # Observable-based regime
│   └── universal_regime_C.yaml        # Partial information regime
│
├── examples/                          # Example scripts
│   ├── evaluate_model.py              # Single model evaluation
│   ├── demo_tokens_and_regimes.py     # Information regime demo
│   ├── ablation_summary_demo.py       # Ablation analysis
│   └── test_universal_integration.py  # End-to-end example
│
└── datasets/                          # Generated datasets (not in repo)
    └── v1.0/                          # Dataset version 1.0
        ├── lss/
        ├── var/
        └── ...
```

## Usage Examples

### Example 1: Train and Evaluate a Model

```python
# See examples/test_universal_integration.py
# This shows how to:
# 1. Load a trained model
# 2. Get predictions on a batch
# 3. Compare to ground truth

python examples/test_universal_integration.py
```

### Example 2: Understand Information Regimes

The emulator supports three information regimes:

```python
# See examples/demo_tokens_and_regimes.py
python examples/demo_tokens_and_regimes.py
```

**Regime A (Full Structural):** Know `world_id`, parameters `theta`, all shocks `eps[0:T]`, and shock token
- Best performance, maximum information
- Typical in controlled simulation studies

**Regime B1 (Observable-based):** Know `world_id`, observable history `y[0:k]`, shock token
- More realistic, limited to what's observable
- Typical in real-world forecasting

**Regime C (Partial):** Know `world_id`, `theta`, shock token, history, but NO shock sequence
- Medium information, unknown exogenous shocks
- Typical in scenario analysis

### Example 3: Add a New Simulator

1. Create a new simulator class implementing `SimulatorAdapter`:

```python
# simulators/my_model.py
from simulators.base import SimulatorAdapter, SimulatorOutput
import numpy as np

class MyModelSimulator(SimulatorAdapter):
    def sample_parameters(self, rng: np.random.Generator):
        """Sample valid parameters."""
        theta = rng.uniform(0.1, 0.9, size=10)
        return theta

    def simulate(self, theta, eps, T, x0=None):
        """Run the simulation."""
        # Your simulation logic here
        y = np.zeros((T, 3))  # output, inflation, rate
        return SimulatorOutput(trajectory=y, x_final=y[-1])

    def compute_irf(self, theta, shock_idx, shock_size, H, x0):
        """Compute impulse response."""
        # Your IRF logic here
        irf = np.zeros((H+1, 3))
        return irf
```

2. Register it:

```python
# simulators/__init__.py
from simulators.my_model import MyModelSimulator

SIMULATORS = {
    'mymodel': MyModelSimulator(),
    # ... existing simulators
}
```

3. Generate data for it:

```bash
python -m data.scripts.generate_dataset --world mymodel --n_samples 10000
```

## Testing

### Run All Tests

```bash
# Fast tests only (< 1 second each, suitable for CI)
pytest -m "fast" -v

# All tests except slow statistical tests
pytest -m "not slow" -v

# Specific test suites
pytest simulators/tests/ -v    # Simulator tests only
pytest emulator/tests/ -v      # Model & training tests only
pytest data/tests/ -v          # Data pipeline tests only

# Coverage report
pytest --cov=simulators --cov=data --cov=emulator -v
```

### Test Markers

Tests use markers to categorize runtime:
- `@pytest.mark.fast` - Unit tests (< 1s), run on every PR
- `@pytest.mark.slow` - Statistical tests (10-60s), run nightly
- `@pytest.mark.integration` - End-to-end tests (1-5 min), run before release

## Code Quality

### Linting

```bash
# Check code style
ruff check .

# Auto-fix style issues
ruff check . --fix
```

### Type Checking

```bash
mypy simulators/ emulator/ --ignore-missing-imports
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

## Documentation

For detailed technical information, see:

- **[CLAUDE.md](CLAUDE.md)** - Developer guide with all available commands
- **[spec/spec.md](spec/spec.md)** - Full technical specification
  - Simulator interfaces and protocols
  - Data schema and split construction
  - Emulator architecture and training
  - Information regimes definition
  - Evaluation metrics and success criteria
- **[spec/sprint-plan.md](spec/sprint-plan.md)** - Sprint breakdowns with acceptance criteria

## Key Concepts

### Impulse Response Functions (IRFs)

An IRF shows how a variable responds to a one-time shock over time:

```
IRF[h] = y_shocked[h] - y_baseline[h]

where:
  - h = 0..H (horizon, typically 0..40)
  - y = [output, inflation, rate] (three canonical observables)
  - Shock hits at t=0, IRF[0] is the impact effect
```

### Parameter and Shock Normalization

- **Parameters:** Stored in natural units, normalized via z-score for model input
- **Shocks:** Always in standard deviation units for consistency

### Dataset Splits

- **Train:** For model training (70% of samples)
- **Validation:** For hyperparameter tuning (10% of samples)
- **Test Interpolation:** Within-sample range test (10% of samples)
- **Test Extrapolation:** Out-of-sample range test (10% of samples)

## Performance Targets

The Universal Emulator aims to achieve:

- **Accuracy:** Within 20% of per-world specialist models (mean gap)
- **Robustness:** Beats VAR and MLP baselines on all worlds
- **Speed:** Inference < 100ms per 32-sample batch
- **Scalability:** Handles 100k+ samples per simulator world

Current performance metrics are tracked in evaluation results under the `runs/` directory.

## Contributing

To contribute:

1. Create a feature branch
2. Make changes and add tests
3. Run `pytest -m "fast"` to verify
4. Submit a pull request

See [CLAUDE.md](CLAUDE.md) for detailed developer workflows.

## Citation

If you use this emulator in research, please cite:

```bibtex
@software{macro_emulator_2025,
  title = {Universal Macro Emulator: Neural Network for Impulse Response Prediction},
  author = {Macro Emulator Team},
  year = {2025},
  url = {https://github.com/your-repo/macro_simulator}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This project implements research on universal emulation for macroeconomic models, combining:
- State-space simulation (LSS, VAR)
- Dynamic stochastic general equilibrium (DSGE) models (NK, RBC)
- Regime-switching dynamics
- Nonlinear constraints (ZLB)

Built with PyTorch, Zarr, and modern ML practices.

## Support

For questions and issues:

- Check [CLAUDE.md](CLAUDE.md) for common commands
- Review [spec/spec.md](spec/spec.md) for technical details
- See `examples/` for usage demonstrations
- Open an issue on GitHub

---

**Last Updated:** December 2025
**Version:** 0.1.0
