#!/bin/bash

# Universal Macro Emulator - Master Reproduction Script
#
# This script orchestrates the complete reproduction of all experimental results
# from a clean clone of the repository.
#
# Usage:
#   ./run.sh setup      - Install dependencies and setup environment
#   ./run.sh data       - Generate all datasets
#   ./run.sh train      - Train all models (baselines, universal, ablations)
#   ./run.sh eval       - Run all evaluations
#   ./run.sh reproduce  - Run full pipeline end-to-end
#   ./run.sh verify     - Verify test suite passes
#
# For reproducibility, all random seeds are documented in configs and manifests.
# See docs/SEEDS.md for a complete reference.

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Progress tracking
TOTAL_STEPS=0
CURRENT_STEP=0

start_step() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    log_info "Step ${CURRENT_STEP}/${TOTAL_STEPS}: $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# ============================================================================
# SETUP: Install dependencies and prepare environment
# ============================================================================
setup() {
    log_info "Setting up Universal Macro Emulator environment..."
    TOTAL_STEPS=5
    CURRENT_STEP=0

    start_step "Checking Python version"
    if ! command_exists python3; then
        log_error "Python 3 not found. Please install Python 3.10 or later."
    fi
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    log_info "Found Python ${PYTHON_VERSION}"

    start_step "Installing package in editable mode"
    python3 -m pip install -e ".[dev]" || log_error "Failed to install package"
    log_success "Package installed"

    start_step "Verifying core dependencies"
    python3 -c "import numpy, scipy, zarr, torch, matplotlib, yaml, tqdm" || \
        log_error "Failed to import required packages"
    log_success "Dependencies verified"

    start_step "Running linters"
    ruff check . || log_warning "Ruff linting issues found"
    mypy simulators/ emulator/ --ignore-missing-imports || log_warning "MyPy type checking issues found"

    start_step "Running fast test suite"
    pytest -m "fast" -v || log_error "Fast tests failed"
    log_success "Fast tests passed"

    log_success "Setup complete!"
}

# ============================================================================
# DATA: Generate all datasets
# ============================================================================
generate_data() {
    log_info "Generating datasets..."
    TOTAL_STEPS=4
    CURRENT_STEP=0

    start_step "Creating dataset directories"
    mkdir -p datasets/v1.0 datasets/v1.0-dev datasets/v1.0-test
    log_success "Directories created"

    start_step "Generating development dataset (1000 samples/world for testing)"
    python3 -m data.scripts.generate_dataset \
        --world all \
        --n_samples 1000 \
        --seed 42 \
        --output datasets/v1.0-dev/ || log_error "Failed to generate dev dataset"
    log_success "Dev dataset generated"

    start_step "Generating full dataset (10000 samples/world)"
    python3 -m data.scripts.generate_dataset \
        --world all \
        --n_samples 10000 \
        --seed 42 \
        --output datasets/v1.0/ || log_error "Failed to generate full dataset"
    log_success "Full dataset generated"

    start_step "Validating datasets"
    python3 -m data.scripts.validate_dataset --path datasets/v1.0-dev/ || \
        log_error "Dev dataset validation failed"
    python3 -m data.scripts.validate_dataset --path datasets/v1.0/ || \
        log_error "Full dataset validation failed"
    log_success "Datasets validated"

    log_success "Data generation complete!"
}

# ============================================================================
# TRAIN: Train all models
# ============================================================================
train_models() {
    log_info "Training all models..."
    TOTAL_STEPS=10
    CURRENT_STEP=0

    # Universal emulator - all regimes
    start_step "Training universal emulator (Regime A - Full Structural Assist)"
    python3 -m emulator.training.trainer \
        --config configs/universal_regime_A.yaml || \
        log_error "Failed to train universal regime A"
    log_success "Universal regime A trained"

    start_step "Training universal emulator (Regime B1 - Observables + World Known)"
    python3 -m emulator.training.trainer \
        --config configs/universal_regime_B1.yaml || \
        log_error "Failed to train universal regime B1"
    log_success "Universal regime B1 trained"

    start_step "Training universal emulator (Regime C - Partial Info)"
    python3 -m emulator.training.trainer \
        --config configs/universal_regime_C.yaml || \
        log_error "Failed to train universal regime C"
    log_success "Universal regime C trained"

    # Ablations
    start_step "Training ablation: no world_id"
    python3 -m emulator.training.trainer \
        --config configs/ablation_no_world_id.yaml || \
        log_error "Failed to train ablation no_world_id"
    log_success "Ablation no_world_id trained"

    start_step "Training ablation: no theta"
    python3 -m emulator.training.trainer \
        --config configs/ablation_no_theta.yaml || \
        log_error "Failed to train ablation no_theta"
    log_success "Ablation no_theta trained"

    start_step "Training ablation: no eps"
    python3 -m emulator.training.trainer \
        --config configs/ablation_no_eps.yaml || \
        log_error "Failed to train ablation no_eps"
    log_success "Ablation no_eps trained"

    # Baselines (if configs exist)
    if [ -f "configs/examples/baseline_mlp.yaml" ]; then
        start_step "Training baseline: MLP"
        python3 -m emulator.training.trainer \
            --config configs/examples/baseline_mlp.yaml || \
            log_warning "Failed to train baseline MLP"
        log_success "Baseline MLP trained"
    fi

    # LOWO experiments (if configs exist)
    start_step "Checking for LOWO configs"
    LOWO_TRAINED=0
    for world in lss var nk rbc switching zlb; do
        if [ -f "configs/lowo_exclude_${world}.yaml" ]; then
            log_info "Training LOWO model excluding ${world}"
            python3 -m emulator.training.trainer \
                --config "configs/lowo_exclude_${world}.yaml" || \
                log_warning "Failed to train LOWO excluding ${world}"
            LOWO_TRAINED=$((LOWO_TRAINED + 1))
        fi
    done
    if [ $LOWO_TRAINED -gt 0 ]; then
        log_success "Trained ${LOWO_TRAINED} LOWO models"
    else
        log_warning "No LOWO configs found, skipping"
    fi

    log_success "Model training complete!"
}

# ============================================================================
# EVAL: Run all evaluations
# ============================================================================
run_evaluations() {
    log_info "Running evaluations..."
    TOTAL_STEPS=7
    CURRENT_STEP=0

    # Create output directories
    start_step "Creating output directories"
    mkdir -p results/metrics results/figures
    log_success "Output directories created"

    # Evaluate universal models
    start_step "Evaluating universal regime A"
    python3 -m emulator.eval.evaluate \
        --checkpoint runs/universal_regime_A/checkpoint_best.pt \
        --dataset datasets/v1.0/ \
        --output results/universal_regime_A/ || \
        log_error "Failed to evaluate universal regime A"
    log_success "Universal regime A evaluated"

    start_step "Evaluating universal regime B1"
    python3 -m emulator.eval.evaluate \
        --checkpoint runs/universal_regime_B1/checkpoint_best.pt \
        --dataset datasets/v1.0/ \
        --output results/universal_regime_B1/ || \
        log_error "Failed to evaluate universal regime B1"
    log_success "Universal regime B1 evaluated"

    start_step "Evaluating universal regime C"
    python3 -m emulator.eval.evaluate \
        --checkpoint runs/universal_regime_C/checkpoint_best.pt \
        --dataset datasets/v1.0/ \
        --output results/universal_regime_C/ || \
        log_error "Failed to evaluate universal regime C"
    log_success "Universal regime C evaluated"

    # Generate ablation summary
    start_step "Generating ablation summary"
    python3 -m emulator.eval.ablation_summary \
        --runs runs/ \
        --output results/ablation_summary.md || \
        log_warning "Failed to generate ablation summary"
    log_success "Ablation summary generated"

    # Generate figures
    start_step "Generating figures"
    python3 -m emulator.eval.figures \
        --runs runs/ \
        --output results/figures/ || \
        log_warning "Failed to generate figures"
    log_success "Figures generated"

    # Generate leaderboard
    start_step "Generating leaderboard"
    python3 -m emulator.eval.leaderboard \
        --runs runs/ \
        --output results/leaderboard.md || \
        log_warning "Failed to generate leaderboard"
    log_success "Leaderboard generated"

    log_success "Evaluations complete!"
}

# ============================================================================
# VERIFY: Run test suite
# ============================================================================
verify() {
    log_info "Verifying test suite..."
    TOTAL_STEPS=4
    CURRENT_STEP=0

    start_step "Running fast tests"
    pytest -m "fast" -v || log_error "Fast tests failed"
    log_success "Fast tests passed"

    start_step "Running simulator tests"
    pytest simulators/tests/ -v || log_error "Simulator tests failed"
    log_success "Simulator tests passed"

    start_step "Running emulator tests"
    pytest emulator/tests/ -v || log_error "Emulator tests failed"
    log_success "Emulator tests passed"

    start_step "Running data tests"
    pytest data/tests/ -v || log_error "Data tests failed"
    log_success "Data tests passed"

    log_success "All tests passed!"
}

# ============================================================================
# REPRODUCE: Full end-to-end reproduction
# ============================================================================
reproduce() {
    log_info "Running full reproduction pipeline..."
    log_info "This will take several hours depending on hardware."
    log_info ""

    # Run all steps
    setup
    echo ""
    generate_data
    echo ""
    train_models
    echo ""
    run_evaluations
    echo ""
    verify
    echo ""

    log_success "=============================================="
    log_success "FULL REPRODUCTION COMPLETE!"
    log_success "=============================================="
    log_info ""
    log_info "Results are available in:"
    log_info "  - Trained models: runs/"
    log_info "  - Metrics: results/metrics/"
    log_info "  - Figures: results/figures/"
    log_info "  - Leaderboard: results/leaderboard.md"
    log_info ""
    log_info "Random seeds used:"
    log_info "  - Data generation: seed=42 (see datasets/*/manifest.json)"
    log_info "  - Training: seed=42 (see configs/*.yaml)"
    log_info ""
    log_info "For detailed seed documentation, see docs/SEEDS.md"
}

# ============================================================================
# MAIN
# ============================================================================
usage() {
    echo "Usage: $0 {setup|data|train|eval|verify|reproduce}"
    echo ""
    echo "Commands:"
    echo "  setup      - Install dependencies and setup environment"
    echo "  data       - Generate all datasets"
    echo "  train      - Train all models (baselines, universal, ablations)"
    echo "  eval       - Run all evaluations"
    echo "  verify     - Verify test suite passes"
    echo "  reproduce  - Run full pipeline end-to-end"
    echo ""
    exit 1
}

# Check if no arguments provided
if [ $# -eq 0 ]; then
    usage
fi

# Parse command
case "$1" in
    setup)
        setup
        ;;
    data)
        generate_data
        ;;
    train)
        train_models
        ;;
    eval)
        run_evaluations
        ;;
    verify)
        verify
        ;;
    reproduce)
        reproduce
        ;;
    *)
        log_error "Unknown command: $1"
        usage
        ;;
esac
