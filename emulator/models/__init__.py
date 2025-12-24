"""Neural network models for the Universal Macro Emulator."""

from emulator.models.baselines import (
    LinearBaseline,
    OracleBaseline,
    PerWorldGRUBaseline,
    PerWorldMLPBaseline,
    PooledMLPBaseline,
)
from emulator.models.tokens import (
    InformationRegime,
    Regime,
    ShockToken,
    batch_shock_tokens,
    validate_regime_inputs,
)
from emulator.models.universal import (
    HistoryEncoder,
    IRFHead,
    ParameterEncoder,
    ShockEncoder,
    TrajectoryHead,
    TrunkNetwork,
    UniversalEmulator,
    WorldEmbedding,
)

__all__ = [
    # Baselines
    "OracleBaseline",
    "LinearBaseline",
    "PerWorldMLPBaseline",
    "PerWorldGRUBaseline",
    "PooledMLPBaseline",
    # Universal emulator
    "UniversalEmulator",
    # Components (for testing and ablations)
    "WorldEmbedding",
    "ParameterEncoder",
    "ShockEncoder",
    "HistoryEncoder",
    "TrunkNetwork",
    "IRFHead",
    "TrajectoryHead",
    # Tokens and regimes
    "InformationRegime",
    "Regime",
    "ShockToken",
    "batch_shock_tokens",
    "validate_regime_inputs",
]
