# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Universal Macro Emulator: A neural network that predicts impulse response functions (IRFs) and trajectories across diverse macroeconomic simulators. The goal is a single model that generalizes across 6 simulator families (LSS, VAR, NK, RBC, regime-switching, ZLB).

## Commands

```bash
# Install
pip install -e ".[dev]"

# Lint and type check
ruff check .
mypy simulators/ emulator/ --ignore-missing-imports

# Tests
pytest -m "fast" -v                    # Fast tests only (CI)
pytest -m "not slow" -v                # Skip slow statistical tests
pytest simulators/tests/ -v            # Simulator tests only
pytest emulator/tests/ -v              # Emulator tests only

# Visualization (Sprint 1)
python -m emulator.eval.plots --simulators lss,var,nk --output sprint1_irf_panel.png
python -m emulator.eval.plots --simulators lss --all-shocks --output lss_all_shocks.png
python test_visualization_demo.py      # Run demonstration script

# Dataset generation
python -m data.scripts.generate_dataset --world all --n_samples 10000 --seed 42 --output datasets/v1.0/
python -m data.scripts.validate_dataset --path datasets/v1.0/

# Training
python train_universal.py --config configs/universal_regime_A.yaml

# LOWO (Leave-One-World-Out) Training
for world in lss var nk rbc switching zlb; do
    python train_universal.py --config configs/lowo_exclude_${world}.yaml
done

# Evaluation
python -m emulator.eval.evaluate --checkpoint runs/universal_A/checkpoint.pt --dataset datasets/v1.0/

# Final Results Generation (Sprint 5)
python -m emulator.eval.final_results \
    --output-dir results/final/ \
    --universal-results results/universal_A.json \
    --baseline-results "MLP:results/mlp.json,VAR:results/var.json" \
    --specialist-results "lss:results/specialist_lss.json,var:results/specialist_var.json"
```

## Architecture

### Three-Layer Design

1. **Simulators** (`simulators/`): Macroeconomic model implementations conforming to `SimulatorAdapter` protocol
2. **Data Pipeline** (`data/`): Zarr-based dataset generation with deterministic seeding
3. **Emulator** (`emulator/`): PyTorch models for IRF/trajectory prediction

### Simulator Adapter Protocol

All simulators implement `SimulatorAdapter` from `simulators/base.py`:
- `sample_parameters(rng)` → sample valid (stable/determinate) parameters
- `simulate(theta, eps, T, x0)` → run simulation, return `SimulatorOutput`
- `compute_irf(theta, shock_idx, shock_size, H, x0)` → compute impulse response
- Must output 3 canonical observables: output, inflation, rate (in percent units)

### Information Regimes

The emulator operates in different information regimes:
- **Regime A**: Full structural assist (world_id, theta, eps_sequence, shock_token)
- **Regime B1**: Observables + world known (world_id, shock_token, history)
- **Regime C**: Partial (world_id, theta, shock_token, history, no eps)

Key distinction: `shock_token` (which IRF to compute) is always provided; `eps_sequence` (full shock path) is regime-dependent.

### Dataset Structure

```
datasets/v1.0/
├── manifest.json           # Metadata, seeds, splits
└── {world_id}/
    ├── trajectories.zarr   # (n_samples, T, n_obs)
    ├── irfs.zarr           # (n_samples, n_shocks, H+1, 3)
    ├── shocks.zarr         # (n_samples, T, n_shocks)
    └── theta.zarr          # (n_samples, n_params)
```

## Key Conventions

### IRF Computation
- IRFs are differences: `IRF[h] = y_shocked[h] - y_baseline[h]`
- Shock hits at t=0, IRF[0] shows impact effect
- Default horizon H=40 (configurable to 80)
- Shocks in standard deviation units

### Parameter Handling
- Natural units stored as ground truth
- Normalized via z-score with bounds-derived scale for model input
- Use `normalize_bounded()` (logit transform) for probability parameters

### Test Markers
- `@pytest.mark.fast`: Unit tests (<1s each), run on every PR
- `@pytest.mark.slow`: Statistical tests (10-60s), nightly only
- `@pytest.mark.integration`: End-to-end pipeline (1-5 min)

## Specifications

Detailed specifications in `spec/`:
- `spec.md`: Full technical specification (interfaces, data schema, metrics, success criteria)
- `sprint-plan.md`: Task breakdowns with acceptance criteria

## Success Criteria

- Universal emulator beats baselines (VAR, MLP) on all worlds
- Mean gap to specialists ≤ 20%, max gap ≤ 35%
- Shape preservation: HF-ratio ≤ 1.1× specialist (no excess oscillation)
