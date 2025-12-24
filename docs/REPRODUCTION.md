# Reproduction Guide

This document provides detailed instructions for reproducing all experimental results from the Universal Macro Emulator project.

## Quick Start

From a clean clone of the repository:

```bash
# One-command full reproduction
./run.sh reproduce
```

This will:
1. Install all dependencies
2. Generate datasets (10k samples/world)
3. Train all models (universal, ablations, baselines)
4. Run all evaluations
5. Generate figures and metrics

**Estimated Time**: 8-12 hours (depending on hardware)

## Step-by-Step Reproduction

For more control, run each phase separately:

### 1. Setup Environment

```bash
./run.sh setup
```

**What it does:**
- Verifies Python 3.10+ is installed
- Installs package with `pip install -e ".[dev]"`
- Verifies all dependencies (numpy, scipy, zarr, torch, etc.)
- Runs linters (ruff, mypy)
- Runs fast test suite to verify installation

**Expected output:**
- All dependencies installed successfully
- Linters pass (or minor warnings)
- Fast tests pass (100% success rate)

**Common issues:**
- **Python version too old**: Upgrade to Python 3.10 or later
- **CUDA not found**: Install PyTorch with CUDA support, or modify configs to use CPU
- **Import errors**: Ensure all dependencies installed with `pip install -e ".[dev]"`

### 2. Generate Datasets

```bash
./run.sh data
```

**What it does:**
- Creates `datasets/v1.0-dev/` (1k samples/world, for smoke tests)
- Creates `datasets/v1.0/` (10k samples/world, for full training)
- Validates dataset integrity (Zarr format, manifest structure)

**Expected output:**
- `datasets/v1.0-dev/manifest.json` with 6 worlds (lss, var, nk, rbc, switching, zlb)
- `datasets/v1.0/manifest.json` with 60k total samples
- Validation passes for both datasets

**Expected time:**
- Dev dataset: ~2-5 minutes
- Full dataset: ~20-30 minutes

**Common issues:**
- **Insufficient disk space**: Datasets require ~5-10GB total
- **Memory errors**: Reduce `--n_samples` or close other programs
- **Validation failures**: Check for corrupt Zarr files, regenerate if needed

### 3. Train Models

```bash
./run.sh train
```

**What it does:**
- Trains universal emulator in 3 information regimes (A, B1, C)
- Trains 3 ablation models (no world_id, no theta, no eps)
- Trains baselines if configs exist (MLP, GRU)
- Trains LOWO models if configs exist

**Expected output:**
- Checkpoints in `runs/universal_regime_A/`, `runs/universal_regime_B1/`, etc.
- Training logs with loss curves
- Best checkpoints saved as `checkpoint_best.pt`

**Expected time:**
- Per model: 1-3 hours (depending on hardware)
- Total: 6-12 hours for all models

**Common issues:**
- **CUDA out of memory**: Reduce `batch_size` in configs
- **Training divergence**: Check learning rate, gradient clipping
- **Slow training on CPU**: Training on CPU is 10-50x slower; use GPU if possible

### 4. Run Evaluations

```bash
./run.sh eval
```

**What it does:**
- Evaluates all trained models on test set
- Computes metrics (NRMSE, IAE, Gap, HF-ratio, etc.)
- Generates figures (IRF panels, accuracy bars, shape diagnostics)
- Creates leaderboard table

**Expected output:**
- Metrics in `results/metrics/`
- Figures in `results/figures/`
- Leaderboard in `results/leaderboard.md`
- Ablation summary in `results/ablation_summary.md`

**Expected time:**
- Per model: 5-15 minutes
- Total: ~1 hour for all evaluations

**Common issues:**
- **Missing checkpoints**: Ensure training completed successfully
- **Evaluation errors**: Check dataset paths in config match generated datasets
- **Figure generation fails**: Ensure matplotlib backend configured correctly

### 5. Verify Tests

```bash
./run.sh verify
```

**What it does:**
- Runs full test suite (fast, simulator, emulator, data tests)
- Validates all components work correctly

**Expected output:**
- All tests pass (100% success rate)
- Coverage report (if pytest-cov installed)

**Expected time:**
- Fast tests: ~30 seconds
- Full test suite: ~2-5 minutes

## Hardware Requirements

### Minimum Requirements

- **CPU**: Modern multi-core processor (4+ cores)
- **RAM**: 16GB
- **GPU**: None (CPU training supported, but slow)
- **Storage**: 50GB free space

### Recommended Requirements

- **CPU**: 8+ cores
- **RAM**: 32GB
- **GPU**: CUDA-capable GPU with 16GB+ VRAM (e.g., RTX 3090, A100)
- **Storage**: 100GB free space (SSD preferred)

### Hardware-Specific Notes

**CPU-only training:**
- Modify `device: cuda` to `device: cpu` in all configs
- Training will be 10-50x slower
- Consider reducing dataset size or model complexity

**Multi-GPU training:**
- Not currently implemented
- Can be added by modifying `emulator/training/trainer.py` to use DataParallel

**Apple Silicon (M1/M2/M3):**
- Use `device: mps` for Metal acceleration
- May require PyTorch 2.0+ with MPS support
- Performance between CPU and CUDA

## Reproducibility Verification

### Deterministic Results

All experiments use fixed random seeds:
- **Data generation**: `seed=42` in dataset manifest
- **Model training**: `seed=42` in config files
- **Data splitting**: `seed=42` for train/val/test splits

### Verifying Exact Reproduction

To verify results match expected:

```bash
# Generate checksums of key outputs
md5sum results/metrics/*.json > checksums.txt

# Compare to expected checksums (if provided)
diff checksums.txt expected_checksums.txt
```

**Note**: Due to GPU non-determinism and floating-point precision, results may differ slightly (±0.1% in metrics) across hardware configurations.

### Expected Metric Ranges

| Model | Mean NRMSE | Max NRMSE | HF-ratio |
|-------|-----------|-----------|----------|
| Universal Regime A | 0.10-0.15 | 0.20-0.30 | 1.05-1.15 |
| Universal Regime B1 | 0.15-0.25 | 0.30-0.45 | 1.10-1.20 |
| Universal Regime C | 0.12-0.18 | 0.25-0.35 | 1.08-1.18 |
| Ablation (no world_id) | 0.15-0.20 | 0.30-0.40 | 1.10-1.20 |
| Ablation (no theta) | 0.20-0.30 | 0.40-0.60 | 1.15-1.30 |
| Ablation (no eps) | 0.12-0.18 | 0.25-0.35 | 1.08-1.18 |

If your results fall outside these ranges, check:
- Training converged properly (loss plateaued)
- Dataset generated correctly (validate checksums)
- Correct random seeds used
- Hardware differences (GPU vs CPU, precision)

## Environment Dependencies

### Python Version

- **Required**: Python 3.10 or later
- **Tested on**: Python 3.10, 3.11, 3.12

### Key Dependencies

```
numpy>=1.24.0
scipy>=1.10.0
zarr>=2.16.0
torch>=2.0.0
matplotlib>=3.7.0
pyyaml>=6.0
tqdm>=4.65.0
```

See `pyproject.toml` for complete dependency list.

### Platform Support

- **Linux**: Fully supported (Ubuntu 20.04+, Debian 11+, etc.)
- **macOS**: Supported (Intel and Apple Silicon)
- **Windows**: Supported with WSL2 (native Windows not tested)

### Docker (Optional)

A Dockerfile is provided for containerized reproduction:

```bash
# Build Docker image
docker build -t macro-emulator .

# Run full reproduction in container
docker run -v $(pwd)/results:/results macro-emulator ./run.sh reproduce
```

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution:**
1. Reduce batch size in config files (e.g., `batch_size: 64` → `batch_size: 32`)
2. Use gradient accumulation (not currently implemented)
3. Switch to CPU training (modify `device: cuda` → `device: cpu`)

### Issue: "Dataset validation failed"

**Solution:**
1. Delete corrupted dataset: `rm -rf datasets/v1.0/`
2. Regenerate: `./run.sh data`
3. Check disk space and permissions

### Issue: "Training diverges (loss → NaN)"

**Solution:**
1. Check learning rate (try reducing by 10x)
2. Enable gradient clipping (verify `grad_clip: 1.0` in config)
3. Check input normalization (parameters should be normalized)

### Issue: "Tests fail"

**Solution:**
1. Ensure fresh install: `pip install -e ".[dev]" --force-reinstall`
2. Check Python version: `python --version` (must be 3.10+)
3. Run specific failing test for details: `pytest <test_file>::<test_name> -v`

### Issue: "Evaluation script crashes"

**Solution:**
1. Verify checkpoint exists: `ls runs/*/checkpoint_best.pt`
2. Check dataset path matches config
3. Ensure model and data compatible (same horizon, observables)

## Clean Environment Test

To verify reproduction from scratch:

```bash
# Create fresh conda environment
conda create -n macro-emulator python=3.10
conda activate macro-emulator

# Clone repository
git clone <repo-url>
cd macro_simulator

# Run full reproduction
./run.sh reproduce
```

Expected result: All steps complete successfully, results match expected ranges.

## Advanced Usage

### Custom Dataset Sizes

For faster iteration or testing:

```bash
# Generate smaller dataset (100 samples/world)
python -m data.scripts.generate_dataset \
    --world all \
    --n_samples 100 \
    --seed 42 \
    --output datasets/v1.0-tiny/
```

### Training Single Models

```bash
# Train only regime A
python -m emulator.training.trainer --config configs/universal_regime_A.yaml

# Train with custom config
python -m emulator.training.trainer --config my_custom_config.yaml
```

### Evaluating Single Models

```bash
# Evaluate specific checkpoint
python -m emulator.eval.evaluate \
    --checkpoint runs/universal_regime_A/checkpoint_best.pt \
    --dataset datasets/v1.0/ \
    --output results/my_eval/
```

### Partial Reproduction

To reproduce only specific experiments:

```bash
# Only universal models (skip ablations)
python -m emulator.training.trainer --config configs/universal_regime_A.yaml
python -m emulator.training.trainer --config configs/universal_regime_B1.yaml
python -m emulator.training.trainer --config configs/universal_regime_C.yaml
```

## Contact & Support

For issues with reproduction:
1. Check this guide for troubleshooting steps
2. Verify environment matches requirements
3. Open an issue on GitHub with:
   - Full error message and stack trace
   - System info (OS, Python version, GPU)
   - Steps to reproduce

## Appendix: File Locations

After successful reproduction, expect these files:

```
macro_simulator/
├── datasets/
│   ├── v1.0-dev/          # Development dataset (1k/world)
│   ├── v1.0/              # Full dataset (10k/world)
│   └── */manifest.json    # Dataset metadata with seeds
├── runs/
│   ├── universal_regime_A/
│   │   ├── checkpoint_best.pt
│   │   ├── checkpoint_last.pt
│   │   └── training_log.json
│   ├── universal_regime_B1/
│   ├── universal_regime_C/
│   ├── ablation_no_world_id/
│   ├── ablation_no_theta/
│   └── ablation_no_eps/
├── results/
│   ├── metrics/           # JSON metric outputs
│   ├── figures/           # PNG/PDF figures
│   ├── leaderboard.md     # Model comparison table
│   └── ablation_summary.md
└── experiments/configs/archive/
    └── sprint4/           # Archived configs used
```

Total disk usage: ~30-50GB (depending on checkpointing frequency)
