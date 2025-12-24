"""Data pipeline for Universal Macro Emulator.

This module provides functionality for:
- Zarr-based dataset storage with efficient chunking
- Manifest generation and validation
- Deterministic dataset generation with reproducible seeding
"""

from data.manifest import ManifestGenerator
from data.writer import DatasetWriter

__all__ = ["DatasetWriter", "ManifestGenerator"]
