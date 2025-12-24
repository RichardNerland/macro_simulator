"""Macroeconomic simulator bank for the Universal Macro Emulator."""

from simulators.base import (
    ObservableManifest,
    ParameterManifest,
    ShockManifest,
    SimulatorAdapter,
    SimulatorOutput,
    check_bounds,
    check_spectral_radius,
    normalize_bounded,
    normalize_param,
)
from simulators.lss import LSSSimulator
from simulators.nk import NKSimulator
from simulators.rbc import RBCSimulator
from simulators.switching import SwitchingSimulator
from simulators.var import VARSimulator
from simulators.zlb import ZLBSimulator

__all__ = [
    "SimulatorAdapter",
    "SimulatorOutput",
    "ParameterManifest",
    "ShockManifest",
    "ObservableManifest",
    "normalize_param",
    "normalize_bounded",
    "check_bounds",
    "check_spectral_radius",
    "LSSSimulator",
    "VARSimulator",
    "NKSimulator",
    "RBCSimulator",
    "SwitchingSimulator",
    "ZLBSimulator",
]
