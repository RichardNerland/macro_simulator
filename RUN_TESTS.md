# How to Run Sprint 1 Tests

## Prerequisites

Ensure dependencies are installed:
```bash
cd /Users/nerland/macro_simulator
pip install -e ".[dev]"
```

## Quick Verification

### 1. Test Imports
```bash
python -c "from simulators import LSSSimulator, VARSimulator, NKSimulator; print('✓ All imports successful')"
```

### 2. Quick Functionality Test
```bash
python test_imports.py
```

### 3. Comprehensive Verification
```bash
python verify_implementation.py
```

## Running Unit Tests

### All Fast Tests (Recommended)
```bash
pytest simulators/tests/ -m "fast" -v
```

Expected output:
```
simulators/tests/test_lss.py::test_lss_initialization PASSED
simulators/tests/test_lss.py::test_lss_manifests PASSED
... (51 tests total)
========================= 51 passed in X.XXs =========================
```

### Individual Simulator Tests

**LSS Simulator (17 tests)**:
```bash
pytest simulators/tests/test_lss.py -v
```

**VAR Simulator (18 tests)**:
```bash
pytest simulators/tests/test_var.py -v
```

**NK Simulator (16 tests)**:
```bash
pytest simulators/tests/test_nk.py -v
```

### Specific Test
```bash
pytest simulators/tests/test_lss.py::test_lss_analytic_irf_matches_simulated -v
```

## Code Quality Checks

### Linting (Ruff)
```bash
ruff check simulators/
```

Expected: No errors

### Type Checking (MyPy)
```bash
mypy simulators/ --ignore-missing-imports
```

Expected: Success (no type errors)

### Combined Pre-Commit Check
```bash
ruff check . && mypy simulators/ --ignore-missing-imports && pytest simulators/tests/ -m "fast" -v
```

## Test Markers

All Sprint 1 tests are marked as `@pytest.mark.fast`:
- Fast tests: `pytest -m "fast"` (runs all Sprint 1 tests)
- Skip slow tests: `pytest -m "not slow"` (same as above for Sprint 1)
- All tests: `pytest` (no marker filter)

## Debugging Failed Tests

If a test fails:

1. **Run with verbose output**:
   ```bash
   pytest simulators/tests/test_lss.py::test_name -vv
   ```

2. **Show stdout/stderr**:
   ```bash
   pytest simulators/tests/test_lss.py::test_name -s
   ```

3. **Drop into debugger on failure**:
   ```bash
   pytest simulators/tests/test_lss.py::test_name --pdb
   ```

## Common Issues and Solutions

### ImportError: No module named 'simulators'

**Solution**: Install package in editable mode:
```bash
pip install -e .
```

### ModuleNotFoundError: No module named 'pytest'

**Solution**: Install dev dependencies:
```bash
pip install -e ".[dev]"
```

### Tests pass but warnings appear

**Solution**: This is normal. Common warnings:
- NumPy deprecation warnings
- SciPy QZ convergence warnings (expected for some NK parameter combinations)

To suppress warnings:
```bash
pytest simulators/tests/ -v -W ignore::DeprecationWarning
```

### NK determinacy sampling takes long

**Solution**: This is expected. The NK simulator uses rejection sampling for determinacy. If it takes > 30s, the parameter ranges might need adjustment.

## Expected Test Duration

On typical hardware:
- LSS tests: ~2-3 seconds
- VAR tests: ~3-4 seconds
- NK tests: ~5-10 seconds (determinacy sampling)
- **Total: ~10-15 seconds**

## Continuous Integration

The fast test suite is designed for CI:
```yaml
# .github/workflows/ci.yml (example)
- name: Run fast tests
  run: pytest -m "fast" -v --tb=short
```

## Manual Verification Examples

### LSS Example
```python
import numpy as np
from simulators import LSSSimulator

# Create simulator
lss = LSSSimulator(n_state=4, n_shocks=2)

# Sample parameters
rng = np.random.default_rng(42)
theta = lss.sample_parameters(rng)

# Simulate
eps = rng.standard_normal((100, 2)) * 0.01
output = lss.simulate(theta, eps, 100)
print(f"Simulation shape: {output.y_canonical.shape}")  # (100, 3)

# Compute IRF
irf = lss.compute_irf(theta, shock_idx=0, shock_size=1.0, H=40)
print(f"IRF shape: {irf.shape}")  # (41, 3)

# Get analytic IRF
irf_analytic = lss.get_analytic_irf(theta, shock_idx=0, shock_size=1.0, H=40)
print(f"IRF matches analytic: {np.allclose(irf, irf_analytic)}")  # True
```

### VAR Example
```python
import numpy as np
from simulators import VARSimulator

var = VARSimulator(n_vars=3, lag_order=2)
rng = np.random.default_rng(42)
theta = var.sample_parameters(rng)

# Check stationarity
_, A_matrices, _ = var._unpack_theta(theta)
F = var._build_companion_matrix(A_matrices)
eigenvalues = np.linalg.eigvals(F)
print(f"Max eigenvalue: {np.max(np.abs(eigenvalues)):.4f}")  # < 0.95
```

### NK Example
```python
import numpy as np
from simulators import NKSimulator

nk = NKSimulator()
rng = np.random.default_rng(42)
theta = nk.sample_parameters(rng)

# Check determinacy
assert nk.validate_parameters(theta)
_, _, determinate = nk._solve_re_system(theta)
print(f"System determinate: {determinate}")  # True

# Check Taylor principle
phi_pi = theta[3]
print(f"phi_pi = {phi_pi:.4f}")  # > 1.0
```

## Performance Benchmarking

To benchmark test execution time:
```bash
pytest simulators/tests/ -m "fast" --durations=10
```

This shows the 10 slowest tests.

## Test Coverage Report

To generate a coverage report:
```bash
pip install pytest-cov
pytest simulators/tests/ --cov=simulators --cov-report=html
open htmlcov/index.html
```

Expected coverage: >95% for simulator code

## Summary

**Quickest verification** (5 seconds):
```bash
python test_imports.py
```

**Standard test run** (15 seconds):
```bash
pytest simulators/tests/ -m "fast" -v
```

**Full verification** (30 seconds):
```bash
python verify_implementation.py && pytest simulators/tests/ -v
```
