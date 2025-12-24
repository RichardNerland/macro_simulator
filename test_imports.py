"""Quick import test for simulators."""
import sys
sys.path.insert(0, '/Users/nerland/macro_simulator')

try:
    from simulators.lss import LSSSimulator
    print("✓ LSS import successful")

    from simulators.var import VARSimulator
    print("✓ VAR import successful")

    from simulators.nk import NKSimulator
    print("✓ NK import successful")

    # Quick functionality test
    import numpy as np

    # Test LSS
    lss = LSSSimulator(n_state=3, n_shocks=2)
    rng = np.random.default_rng(42)
    theta_lss = lss.sample_parameters(rng)
    print(f"✓ LSS sample_parameters works: {theta_lss.shape}")

    # Test VAR
    var = VARSimulator(n_vars=3, lag_order=2)
    rng = np.random.default_rng(42)
    theta_var = var.sample_parameters(rng)
    print(f"✓ VAR sample_parameters works: {theta_var.shape}")

    # Test NK
    nk = NKSimulator()
    rng = np.random.default_rng(42)
    theta_nk = nk.sample_parameters(rng)
    print(f"✓ NK sample_parameters works: {theta_nk.shape}")

    print("\n✓✓✓ All imports and basic functionality tests passed!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
