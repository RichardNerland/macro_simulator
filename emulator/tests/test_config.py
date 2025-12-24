"""Tests for training configuration.

Tests cover:
- UniversalTrainingConfig creation and validation
- BaselineTrainingConfig creation
- YAML loading/saving
- Regime-aware configuration
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from emulator.models.tokens import InformationRegime
from emulator.training.config import (
    BaselineModelConfig,
    BaselineTrainingConfig,
    UniversalModelConfig,
    UniversalTrainingConfig,
)


class TestUniversalModelConfig:
    """Tests for UniversalModelConfig."""

    @pytest.mark.fast
    def test_creation_defaults(self):
        """Test creation with default values."""
        config = UniversalModelConfig()
        assert config.world_embed_dim == 32
        assert config.theta_embed_dim == 64
        assert config.shock_embed_dim == 16
        assert config.trunk_dim == 256
        assert config.trunk_layers == 4
        assert config.max_horizon == 40
        assert config.n_observables == 3

    @pytest.mark.fast
    def test_creation_custom(self):
        """Test creation with custom values."""
        config = UniversalModelConfig(
            world_embed_dim=64,
            trunk_dim=512,
            trunk_layers=6,
            max_horizon=80,
        )
        assert config.world_embed_dim == 64
        assert config.trunk_dim == 512
        assert config.trunk_layers == 6
        assert config.max_horizon == 80


class TestBaselineModelConfig:
    """Tests for BaselineModelConfig."""

    @pytest.mark.fast
    def test_creation_defaults(self):
        """Test creation with default values."""
        config = BaselineModelConfig()
        assert config.model_type == "mlp"
        assert config.hidden_dim == 128
        assert config.n_layers == 2
        assert config.max_horizon == 40

    @pytest.mark.fast
    def test_creation_custom(self):
        """Test creation with custom model type."""
        config = BaselineModelConfig(model_type="gru", hidden_dim=256)
        assert config.model_type == "gru"
        assert config.hidden_dim == 256


class TestUniversalTrainingConfig:
    """Tests for UniversalTrainingConfig."""

    @pytest.mark.fast
    def test_creation_defaults(self):
        """Test creation with default values."""
        config = UniversalTrainingConfig()
        assert config.regime == InformationRegime.A
        assert config.batch_size == 128
        assert config.lr == 1e-4
        assert config.epochs == 100
        assert config.seed == 42
        assert config.worlds == ["lss", "var", "nk", "rbc", "switching", "zlb"]

    @pytest.mark.fast
    def test_creation_custom(self):
        """Test creation with custom values."""
        config = UniversalTrainingConfig(
            regime=InformationRegime.B1,
            batch_size=256,
            lr=5e-4,
            worlds=["lss", "var"],
        )
        assert config.regime == InformationRegime.B1
        assert config.batch_size == 256
        assert config.lr == 5e-4
        assert config.worlds == ["lss", "var"]

    @pytest.mark.fast
    def test_regime_string_conversion(self):
        """Test that regime string is converted to enum."""
        config = UniversalTrainingConfig(regime="B1")
        assert isinstance(config.regime, InformationRegime)
        assert config.regime == InformationRegime.B1

    @pytest.mark.fast
    def test_invalid_train_fraction(self):
        """Test validation fails for invalid train_fraction."""
        with pytest.raises(ValueError, match="train_fraction must be in"):
            UniversalTrainingConfig(train_fraction=0.0)

        with pytest.raises(ValueError, match="train_fraction must be in"):
            UniversalTrainingConfig(train_fraction=1.1)

    @pytest.mark.fast
    def test_invalid_val_fraction(self):
        """Test validation fails for invalid val_fraction."""
        with pytest.raises(ValueError, match="val_fraction must be in"):
            UniversalTrainingConfig(val_fraction=0.0)

    @pytest.mark.fast
    def test_invalid_fraction_sum(self):
        """Test validation fails when train + val >= 1.0."""
        with pytest.raises(ValueError, match="train_fraction \\+ val_fraction must be"):
            UniversalTrainingConfig(train_fraction=0.7, val_fraction=0.4)

    @pytest.mark.fast
    def test_test_fraction_property(self):
        """Test test_fraction is computed correctly."""
        config = UniversalTrainingConfig(train_fraction=0.7, val_fraction=0.15)
        assert config.test_fraction == pytest.approx(0.15)

    @pytest.mark.fast
    def test_get_regime_inputs(self):
        """Test get_regime_inputs returns correct set."""
        config = UniversalTrainingConfig(regime=InformationRegime.A)
        inputs = config.get_regime_inputs()
        assert inputs == {"world_id", "theta", "shock_token", "eps_sequence"}

        config = UniversalTrainingConfig(regime=InformationRegime.B1)
        inputs = config.get_regime_inputs()
        assert inputs == {"world_id", "shock_token", "history"}

    @pytest.mark.fast
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = UniversalTrainingConfig(
            regime=InformationRegime.B1,
            batch_size=256,
            worlds=["lss", "var"],
        )
        d = config.to_dict()

        assert d["regime"] == "B1"  # Should be string
        assert d["batch_size"] == 256
        assert d["worlds"] == ["lss", "var"]
        assert isinstance(d["model"], dict)

    @pytest.mark.fast
    def test_model_dict_conversion(self):
        """Test that model dict is converted to UniversalModelConfig."""
        config = UniversalTrainingConfig(
            model={"world_embed_dim": 64, "trunk_dim": 512}
        )
        assert isinstance(config.model, UniversalModelConfig)
        assert config.model.world_embed_dim == 64
        assert config.model.trunk_dim == 512

    @pytest.mark.fast
    def test_yaml_roundtrip(self):
        """Test saving and loading from YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"

            # Create and save config
            original = UniversalTrainingConfig(
                regime=InformationRegime.B1,
                batch_size=256,
                lr=5e-4,
                epochs=50,
                worlds=["lss", "var"],
                seed=123,
            )
            original.to_yaml(path)

            # Load config
            loaded = UniversalTrainingConfig.from_yaml(path)

            # Check values match
            assert loaded.regime == original.regime
            assert loaded.batch_size == original.batch_size
            assert loaded.lr == original.lr
            assert loaded.epochs == original.epochs
            assert loaded.worlds == original.worlds
            assert loaded.seed == original.seed

    @pytest.mark.fast
    def test_yaml_loading_with_model_config(self):
        """Test loading YAML with nested model config."""
        yaml_content = """
regime: "A"
batch_size: 256
lr: 1e-4
epochs: 100

model:
  world_embed_dim: 64
  trunk_dim: 512
  trunk_layers: 6

dataset_path: "datasets/v1.0/"
worlds: ["lss", "var", "nk"]
seed: 42
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            with open(path, "w") as f:
                f.write(yaml_content)

            config = UniversalTrainingConfig.from_yaml(path)

            assert config.regime == InformationRegime.A
            assert config.batch_size == 256
            assert config.lr == 1e-4
            assert config.epochs == 100
            assert isinstance(config.model, UniversalModelConfig)
            assert config.model.world_embed_dim == 64
            assert config.model.trunk_dim == 512
            assert config.model.trunk_layers == 6
            assert config.worlds == ["lss", "var", "nk"]

    @pytest.mark.fast
    def test_from_yaml_missing_file(self):
        """Test that loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            UniversalTrainingConfig.from_yaml("nonexistent.yaml")

    @pytest.mark.fast
    def test_checkpoint_dir_creation(self):
        """Test that checkpoint directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "runs" / "test"
            config = UniversalTrainingConfig(checkpoint_dir=str(checkpoint_dir))
            assert checkpoint_dir.exists()
            assert checkpoint_dir.is_dir()


class TestBaselineTrainingConfig:
    """Tests for BaselineTrainingConfig."""

    @pytest.mark.fast
    def test_creation_defaults(self):
        """Test creation with default values."""
        config = BaselineTrainingConfig()
        assert config.world == "lss"
        assert config.batch_size == 128
        assert config.lr == 1e-4
        assert config.epochs == 100
        assert isinstance(config.model, BaselineModelConfig)

    @pytest.mark.fast
    def test_creation_custom(self):
        """Test creation with custom values."""
        config = BaselineTrainingConfig(
            world="var",
            batch_size=64,
            lr=5e-5,
        )
        assert config.world == "var"
        assert config.batch_size == 64
        assert config.lr == 5e-5

    @pytest.mark.fast
    def test_model_dict_conversion(self):
        """Test that model dict is converted to BaselineModelConfig."""
        config = BaselineTrainingConfig(
            model={"model_type": "gru", "hidden_dim": 256}
        )
        assert isinstance(config.model, BaselineModelConfig)
        assert config.model.model_type == "gru"
        assert config.model.hidden_dim == 256

    @pytest.mark.fast
    def test_yaml_loading(self):
        """Test loading baseline config from YAML."""
        yaml_content = """
world: "nk"
batch_size: 64
lr: 5e-5
epochs: 50

model:
  model_type: "gru"
  hidden_dim: 256
  n_layers: 3

checkpoint_dir: "runs/baseline_nk"
seed: 123
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baseline_config.yaml"
            with open(path, "w") as f:
                f.write(yaml_content)

            config = BaselineTrainingConfig.from_yaml(path)

            assert config.world == "nk"
            assert config.batch_size == 64
            assert config.lr == 5e-5
            assert config.epochs == 50
            assert isinstance(config.model, BaselineModelConfig)
            assert config.model.model_type == "gru"
            assert config.model.hidden_dim == 256
            assert config.model.n_layers == 3
            assert config.seed == 123


class TestConfigExamples:
    """Tests for example config generation."""

    @pytest.mark.fast
    def test_example_configs_structure(self):
        """Test that example configs have valid structure."""
        from emulator.training.config import create_example_configs

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "examples"
            create_example_configs(output_dir)

            # Check all expected files exist
            assert (output_dir / "universal_regime_A.yaml").exists()
            assert (output_dir / "universal_regime_B1.yaml").exists()
            assert (output_dir / "universal_regime_C.yaml").exists()
            assert (output_dir / "baseline_mlp.yaml").exists()

            # Load and validate each config
            config_a = UniversalTrainingConfig.from_yaml(output_dir / "universal_regime_A.yaml")
            assert config_a.regime == InformationRegime.A

            config_b1 = UniversalTrainingConfig.from_yaml(output_dir / "universal_regime_B1.yaml")
            assert config_b1.regime == InformationRegime.B1

            config_c = UniversalTrainingConfig.from_yaml(output_dir / "universal_regime_C.yaml")
            assert config_c.regime == InformationRegime.C

            baseline = BaselineTrainingConfig.from_yaml(output_dir / "baseline_mlp.yaml")
            assert baseline.model.model_type == "mlp"

    @pytest.mark.fast
    def test_regime_specific_configs(self):
        """Test that regime-specific configs have correct settings."""
        from emulator.training.config import create_example_configs

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "examples"
            create_example_configs(output_dir)

            # Regime A should use theta and eps
            config_a = UniversalTrainingConfig.from_yaml(output_dir / "universal_regime_A.yaml")
            assert config_a.regime.uses_theta is True
            assert config_a.regime.uses_eps is True
            assert config_a.regime.uses_history is False

            # Regime B1 should use history but not theta/eps
            config_b1 = UniversalTrainingConfig.from_yaml(output_dir / "universal_regime_B1.yaml")
            assert config_b1.regime.uses_theta is False
            assert config_b1.regime.uses_eps is False
            assert config_b1.regime.uses_history is True

            # Regime C should use theta and history but not eps
            config_c = UniversalTrainingConfig.from_yaml(output_dir / "universal_regime_C.yaml")
            assert config_c.regime.uses_theta is True
            assert config_c.regime.uses_eps is False
            assert config_c.regime.uses_history is True
