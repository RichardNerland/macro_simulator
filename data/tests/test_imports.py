"""Smoke tests to verify imports work correctly.

These tests ensure that the data module can be imported and basic
functionality works without requiring the full simulator infrastructure.
"""

import pytest


@pytest.mark.fast
def test_import_writer():
    """Test that DatasetWriter can be imported."""
    from data.writer import DatasetWriter

    assert DatasetWriter is not None


@pytest.mark.fast
def test_import_manifest():
    """Test that ManifestGenerator can be imported."""
    from data.manifest import ManifestGenerator

    assert ManifestGenerator is not None


@pytest.mark.fast
def test_import_from_data():
    """Test that main classes can be imported from data package."""
    from data import DatasetWriter, ManifestGenerator

    assert DatasetWriter is not None
    assert ManifestGenerator is not None


@pytest.mark.fast
def test_manifest_stubs():
    """Test that manifest stub types work."""
    from data.manifest import ParameterManifest, ShockManifest, ObservableManifest
    import numpy as np

    # Create stub manifests (should work without simulators.base)
    param_manifest = ParameterManifest(
        names=["beta", "sigma"],
        units=["-", "-"],
        bounds=np.array([[0.9, 1.0], [0.5, 2.5]]),
        defaults=np.array([0.99, 1.0]),
        priors=[],
    )
    assert len(param_manifest.names) == 2

    shock_manifest = ShockManifest(
        names=["monetary"],
        n_shocks=1,
        sigma=np.array([0.001]),
    )
    assert shock_manifest.n_shocks == 1

    obs_manifest = ObservableManifest(
        canonical_names=["output", "inflation", "rate"],
        extra_names=[],
        n_canonical=3,
        n_extra=0,
    )
    assert obs_manifest.n_canonical == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
