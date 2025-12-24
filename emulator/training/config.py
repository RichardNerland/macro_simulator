"""Configuration dataclasses for the Universal Macro Emulator.

This module defines configuration for:
- Universal model training (regime-aware)
- Baseline model training
- Shared training hyperparameters
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from emulator.models.tokens import InformationRegime


@dataclass
class UniversalModelConfig:
    """Configuration for universal emulator architecture.

    Defines the model structure for the universal emulator that generalizes
    across all simulator families.
    """
    # Embedding dimensions
    world_embed_dim: int = 32
    theta_embed_dim: int = 64
    shock_embed_dim: int = 16
    history_embed_dim: int = 128

    # Trunk architecture
    trunk_dim: int = 256
    trunk_layers: int = 4
    trunk_activation: str = "gelu"
    trunk_dropout: float = 0.1
    trunk_use_layer_norm: bool = True

    # Output heads
    irf_head_layers: int = 2
    irf_head_dim: int = 128
    trajectory_head_layers: int = 2
    trajectory_head_dim: int = 128

    # IRF prediction
    max_horizon: int = 40
    n_observables: int = 3  # output, inflation, rate


@dataclass
class BaselineModelConfig:
    """Configuration for baseline models (Linear, MLP, GRU)."""
    model_type: str = "mlp"  # "linear", "mlp", "gru"
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.1
    max_horizon: int = 40
    n_observables: int = 3


@dataclass
class UniversalTrainingConfig:
    """Configuration for universal emulator training.

    This is the main config class that combines model architecture,
    training hyperparameters, and regime settings.
    """
    # Model architecture
    model: UniversalModelConfig = field(default_factory=UniversalModelConfig)

    # Information regime
    regime: InformationRegime = InformationRegime.A

    # Training hyperparameters
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 100
    warmup_epochs: int = 5
    grad_clip: float = 1.0

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # Loss function
    loss_fn: str = "mse"  # "mse" or "mae"
    weight_scheme: str = "exponential"  # "uniform", "exponential", "impact"
    weight_decay_tau: float = 20.0  # For exponential weighting

    # Multi-task weights
    lambda_traj: float = 0.0  # Trajectory loss weight (0.0 = IRF only)
    lambda_reg: float = 0.0  # Regularization weight
    lambda_smooth: float = 0.01  # Smoothness penalty

    # Data
    dataset_path: str = "datasets/v1.0/"
    worlds: list[str] | None = None  # Default: all worlds
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    # test_fraction is implicit: 1 - train - val

    # Checkpointing
    checkpoint_dir: str = "runs/universal"
    save_every_n_epochs: int = 10
    save_best: bool = True

    # Logging
    log_every_n_steps: int = 50
    use_wandb: bool = False
    wandb_project: str = "macro-emulator"
    wandb_run_name: str | None = None

    # Reproducibility
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Set default worlds if not specified
        if self.worlds is None:
            self.worlds = ["lss", "var", "nk", "rbc", "switching", "zlb"]

        # Convert regime string to enum if needed
        if isinstance(self.regime, str):
            self.regime = InformationRegime(self.regime)

        # Convert lr from string if needed (YAML scientific notation)
        if isinstance(self.lr, str):
            self.lr = float(self.lr)

        # Validate fractions
        if not (0 < self.train_fraction < 1):
            raise ValueError(f"train_fraction must be in (0, 1), got {self.train_fraction}")
        if not (0 < self.val_fraction < 1):
            raise ValueError(f"val_fraction must be in (0, 1), got {self.val_fraction}")
        if self.train_fraction + self.val_fraction >= 1.0:
            raise ValueError(
                f"train_fraction + val_fraction must be < 1.0, "
                f"got {self.train_fraction} + {self.val_fraction} = {self.train_fraction + self.val_fraction}"
            )

        # Validate model config
        if isinstance(self.model, dict):
            self.model = UniversalModelConfig(**self.model)

        # Create checkpoint directory
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "UniversalTrainingConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            UniversalTrainingConfig instance

        Example YAML:
            ```yaml
            regime: "A"
            batch_size: 256
            lr: 1e-4
            epochs: 100

            model:
              world_embed_dim: 32
              theta_embed_dim: 64
              trunk_dim: 256

            dataset_path: "datasets/v1.0/"
            worlds: ["lss", "var", "nk"]
            seed: 42
            ```
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Handle nested model config
        if "model" in config_dict and isinstance(config_dict["model"], dict):
            config_dict["model"] = UniversalModelConfig(**config_dict["model"])

        return cls(**config_dict)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML config
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict for serialization
        config_dict = self.to_dict()

        with open(path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation suitable for serialization
        """
        d = {}
        for key, value in self.__dict__.items():
            if isinstance(value, InformationRegime):
                d[key] = value.value
            elif isinstance(value, UniversalModelConfig):
                d[key] = {k: v for k, v in value.__dict__.items()}
            elif isinstance(value, Path):
                d[key] = str(value)
            else:
                d[key] = value
        return d

    @property
    def test_fraction(self) -> float:
        """Compute test fraction from train and val fractions."""
        return 1.0 - self.train_fraction - self.val_fraction

    def get_regime_inputs(self) -> set[str]:
        """Get required inputs for the configured regime.

        Returns:
            Set of required input names
        """
        return self.regime.required_inputs()


@dataclass
class BaselineTrainingConfig:
    """Configuration for baseline model training.

    Similar to UniversalTrainingConfig but for simpler baseline models.
    """
    # Model
    model: BaselineModelConfig = field(default_factory=BaselineModelConfig)

    # Training
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 100
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    patience: int = 10

    # Loss
    loss_fn: str = "mse"
    weight_scheme: str = "uniform"

    # Data
    dataset_path: str = "datasets/v1.0/"
    world: str = "lss"  # Baselines train on single world
    train_fraction: float = 0.7
    val_fraction: float = 0.15

    # Checkpointing
    checkpoint_dir: str = "runs/baseline"
    save_every_n_epochs: int = 10

    # Logging
    log_every_n_steps: int = 50
    use_wandb: bool = False

    # Reproducibility
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        """Post-initialization validation."""
        if isinstance(self.model, dict):
            self.model = BaselineModelConfig(**self.model)
        # Convert lr from string if needed (YAML scientific notation)
        if isinstance(self.lr, str):
            self.lr = float(self.lr)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BaselineTrainingConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        if "model" in config_dict and isinstance(config_dict["model"], dict):
            config_dict["model"] = BaselineModelConfig(**config_dict["model"])
        return cls(**config_dict)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML config
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict for serialization
        config_dict = self.to_dict()

        with open(path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation suitable for serialization
        """
        d = {}
        for key, value in self.__dict__.items():
            if isinstance(value, BaselineModelConfig):
                d[key] = {k: v for k, v in value.__dict__.items()}
            elif isinstance(value, Path):
                d[key] = str(value)
            else:
                d[key] = value
        return d


def create_example_configs(output_dir: str | Path = "configs/examples") -> None:
    """Create example configuration files for reference.

    Args:
        output_dir: Directory to save example configs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Example: Universal Regime A
    config_a = UniversalTrainingConfig(
        regime=InformationRegime.A,
        batch_size=256,
        lr=1e-4,
        epochs=100,
        dataset_path="datasets/v1.0/",
        checkpoint_dir="runs/universal_regime_A",
        seed=42,
    )
    config_a.to_yaml(output_dir / "universal_regime_A.yaml")

    # Example: Universal Regime B1
    config_b1 = UniversalTrainingConfig(
        regime=InformationRegime.B1,
        batch_size=256,
        lr=1e-4,
        epochs=100,
        dataset_path="datasets/v1.0/",
        checkpoint_dir="runs/universal_regime_B1",
        seed=42,
    )
    config_b1.to_yaml(output_dir / "universal_regime_B1.yaml")

    # Example: Universal Regime C
    config_c = UniversalTrainingConfig(
        regime=InformationRegime.C,
        batch_size=256,
        lr=1e-4,
        epochs=100,
        dataset_path="datasets/v1.0/",
        checkpoint_dir="runs/universal_regime_C",
        seed=42,
    )
    config_c.to_yaml(output_dir / "universal_regime_C.yaml")

    # Example: Baseline MLP
    baseline_config = BaselineTrainingConfig(
        model=BaselineModelConfig(model_type="mlp", hidden_dim=128, n_layers=2),
        world="lss",
        checkpoint_dir="runs/baseline_lss_mlp",
        seed=42,
    )
    baseline_config.to_yaml(output_dir / "baseline_mlp.yaml")

    print(f"Created example configs in {output_dir}/")
