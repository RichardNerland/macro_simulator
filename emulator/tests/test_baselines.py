"""
Unit tests for baseline models.

Tests verify output shapes, gradient flow, batch handling, and device compatibility
for all baseline IRF prediction models.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from emulator.models.baselines import (
    OracleBaseline,
    LinearBaseline,
    PerWorldMLPBaseline,
    PerWorldGRUBaseline,
    PooledMLPBaseline,
)


# Test configuration
PARAM_DIMS = {"lss": 5, "var": 8, "nk": 10}
WORLD_IDS = ["lss", "var", "nk"]
H = 40
N_OBS = 3


@pytest.mark.fast
class TestLinearBaseline:
    """Test suite for LinearBaseline."""

    @pytest.fixture
    def model(self):
        """Create a LinearBaseline model."""
        torch.manual_seed(42)
        return LinearBaseline(n_params=8, H=H)

    def test_output_shape_single_sample(self, model):
        """Test output shape for single sample."""
        torch.manual_seed(42)
        theta = torch.randn(1, 8)
        irf = model(theta)
        assert irf.shape == (1, H + 1, N_OBS)

    def test_output_shape_batch_8(self, model):
        """Test output shape for batch size 8."""
        torch.manual_seed(42)
        theta = torch.randn(8, 8)
        irf = model(theta)
        assert irf.shape == (8, H + 1, N_OBS)

    def test_output_shape_batch_32(self, model):
        """Test output shape for batch size 32."""
        torch.manual_seed(42)
        theta = torch.randn(32, 8)
        irf = model(theta)
        assert irf.shape == (32, H + 1, N_OBS)

    def test_forward_pass_deterministic(self, model):
        """Test forward pass is deterministic."""
        torch.manual_seed(42)
        theta = torch.randn(4, 8)

        torch.manual_seed(42)
        theta1 = theta.clone()
        irf1 = model(theta1)

        torch.manual_seed(42)
        theta2 = theta.clone()
        irf2 = model(theta2)

        torch.testing.assert_close(irf1, irf2)

    def test_gradient_flow(self, model):
        """Test gradients flow through the model."""
        torch.manual_seed(42)
        theta = torch.randn(4, 8, requires_grad=True)

        irf = model(theta)
        loss = irf.sum()
        loss.backward()

        # Check gradients exist
        assert theta.grad is not None
        assert not torch.all(theta.grad == 0)

        # Check model parameter gradients
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.all(param.grad == 0)

    def test_different_param_dims(self):
        """Test with different parameter dimensions."""
        torch.manual_seed(42)
        for n_params in [5, 8, 10]:
            model = LinearBaseline(n_params=n_params, H=H)
            theta = torch.randn(4, n_params)
            irf = model(theta)
            assert irf.shape == (4, H + 1, N_OBS)

    def test_world_id_ignored(self, model):
        """Test that world_id parameter is ignored."""
        torch.manual_seed(42)
        theta = torch.randn(4, 8)

        irf1 = model(theta, world_id="lss")
        irf2 = model(theta, world_id="var")
        irf3 = model(theta, world_id=None)

        torch.testing.assert_close(irf1, irf2)
        torch.testing.assert_close(irf1, irf3)

    def test_shock_idx_ignored(self, model):
        """Test that shock_idx parameter is ignored."""
        torch.manual_seed(42)
        theta = torch.randn(4, 8)

        irf1 = model(theta, shock_idx=0)
        irf2 = model(theta, shock_idx=1)
        irf3 = model(theta, shock_idx=2)

        torch.testing.assert_close(irf1, irf2)
        torch.testing.assert_close(irf1, irf3)

    def test_cpu_device(self, model):
        """Test model works on CPU."""
        theta = torch.randn(4, 8, device='cpu')
        irf = model(theta)
        assert irf.device.type == 'cpu'
        assert irf.shape == (4, H + 1, N_OBS)


@pytest.mark.fast
class TestPerWorldMLPBaseline:
    """Test suite for PerWorldMLPBaseline."""

    @pytest.fixture
    def model(self):
        """Create a PerWorldMLPBaseline model."""
        torch.manual_seed(42)
        return PerWorldMLPBaseline(
            param_dims=PARAM_DIMS,
            hidden_dims=[256, 128, 64],
            H=H,
        )

    def test_output_shape_single_sample(self, model):
        """Test output shape for single sample across all worlds."""
        torch.manual_seed(42)
        for world_id, n_params in PARAM_DIMS.items():
            theta = torch.randn(1, n_params)
            irf = model(theta, world_id=world_id)
            assert irf.shape == (1, H + 1, N_OBS)

    def test_output_shape_batch_8(self, model):
        """Test output shape for batch size 8 across all worlds."""
        torch.manual_seed(42)
        for world_id, n_params in PARAM_DIMS.items():
            theta = torch.randn(8, n_params)
            irf = model(theta, world_id=world_id)
            assert irf.shape == (8, H + 1, N_OBS)

    def test_output_shape_batch_32(self, model):
        """Test output shape for batch size 32 across all worlds."""
        torch.manual_seed(42)
        for world_id, n_params in PARAM_DIMS.items():
            theta = torch.randn(32, n_params)
            irf = model(theta, world_id=world_id)
            assert irf.shape == (32, H + 1, N_OBS)

    def test_different_worlds_different_outputs(self, model):
        """Test that different worlds produce different outputs."""
        torch.manual_seed(42)
        # Use different parameter counts
        theta_lss = torch.randn(4, PARAM_DIMS["lss"])
        theta_var = torch.randn(4, PARAM_DIMS["var"])
        theta_nk = torch.randn(4, PARAM_DIMS["nk"])

        irf_lss = model(theta_lss, world_id="lss")
        irf_var = model(theta_var, world_id="var")
        irf_nk = model(theta_nk, world_id="nk")

        # Different models should give different outputs
        assert not torch.allclose(irf_lss, irf_var, atol=1e-5)
        assert not torch.allclose(irf_lss, irf_nk, atol=1e-5)

    def test_gradient_flow(self, model):
        """Test gradients flow through the model for each world."""
        torch.manual_seed(42)
        for world_id, n_params in PARAM_DIMS.items():
            theta = torch.randn(4, n_params, requires_grad=True)

            irf = model(theta, world_id=world_id)
            loss = irf.sum()
            loss.backward()

            # Check input gradients
            assert theta.grad is not None
            assert not torch.all(theta.grad == 0)

            # Reset gradients for next world
            model.zero_grad()

    def test_per_world_mlp_isolation(self, model):
        """Test that each world has its own MLP."""
        assert len(model.mlps) == len(PARAM_DIMS)
        for world_id in PARAM_DIMS:
            assert world_id in model.mlps
            assert isinstance(model.mlps[world_id], nn.Sequential)

    def test_custom_hidden_dims(self):
        """Test with custom hidden dimensions."""
        torch.manual_seed(42)
        custom_dims = [128, 64]
        model = PerWorldMLPBaseline(
            param_dims=PARAM_DIMS,
            hidden_dims=custom_dims,
            H=H,
        )

        for world_id, n_params in PARAM_DIMS.items():
            theta = torch.randn(4, n_params)
            irf = model(theta, world_id=world_id)
            assert irf.shape == (4, H + 1, N_OBS)

    def test_cpu_device(self, model):
        """Test model works on CPU."""
        for world_id, n_params in PARAM_DIMS.items():
            theta = torch.randn(4, n_params, device='cpu')
            irf = model(theta, world_id=world_id)
            assert irf.device.type == 'cpu'
            assert irf.shape == (4, H + 1, N_OBS)


@pytest.mark.fast
class TestPerWorldGRUBaseline:
    """Test suite for PerWorldGRUBaseline."""

    @pytest.fixture
    def model(self):
        """Create a PerWorldGRUBaseline model."""
        torch.manual_seed(42)
        return PerWorldGRUBaseline(
            param_dims=PARAM_DIMS,
            hidden_dim=128,
            H=H,
        )

    def test_output_shape_single_sample(self, model):
        """Test output shape for single sample across all worlds."""
        torch.manual_seed(42)
        for world_id, n_params in PARAM_DIMS.items():
            theta = torch.randn(1, n_params)
            irf = model(theta, world_id=world_id)
            assert irf.shape == (1, H + 1, N_OBS)

    def test_output_shape_batch_8(self, model):
        """Test output shape for batch size 8 across all worlds."""
        torch.manual_seed(42)
        for world_id, n_params in PARAM_DIMS.items():
            theta = torch.randn(8, n_params)
            irf = model(theta, world_id=world_id)
            assert irf.shape == (8, H + 1, N_OBS)

    def test_output_shape_batch_32(self, model):
        """Test output shape for batch size 32 across all worlds."""
        torch.manual_seed(42)
        for world_id, n_params in PARAM_DIMS.items():
            theta = torch.randn(32, n_params)
            irf = model(theta, world_id=world_id)
            assert irf.shape == (32, H + 1, N_OBS)

    def test_different_worlds_different_outputs(self, model):
        """Test that different worlds produce different outputs."""
        torch.manual_seed(42)
        theta_lss = torch.randn(4, PARAM_DIMS["lss"])
        theta_var = torch.randn(4, PARAM_DIMS["var"])
        theta_nk = torch.randn(4, PARAM_DIMS["nk"])

        irf_lss = model(theta_lss, world_id="lss")
        irf_var = model(theta_var, world_id="var")
        irf_nk = model(theta_nk, world_id="nk")

        # Different models should give different outputs
        assert not torch.allclose(irf_lss, irf_var, atol=1e-5)
        assert not torch.allclose(irf_lss, irf_nk, atol=1e-5)

    def test_gradient_flow(self, model):
        """Test gradients flow through the model for each world."""
        torch.manual_seed(42)
        for world_id, n_params in PARAM_DIMS.items():
            theta = torch.randn(4, n_params, requires_grad=True)

            irf = model(theta, world_id=world_id)
            loss = irf.sum()
            loss.backward()

            # Check input gradients
            assert theta.grad is not None
            assert not torch.all(theta.grad == 0)

            # Reset gradients for next world
            model.zero_grad()

    def test_per_world_components(self, model):
        """Test that each world has its own encoder, GRU, and output layer."""
        assert len(model.theta_encoders) == len(PARAM_DIMS)
        assert len(model.grus) == len(PARAM_DIMS)
        assert len(model.output_layers) == len(PARAM_DIMS)

        for world_id in PARAM_DIMS:
            assert world_id in model.theta_encoders
            assert world_id in model.grus
            assert world_id in model.output_layers
            assert isinstance(model.grus[world_id], nn.GRUCell)

    def test_custom_hidden_dim(self):
        """Test with custom hidden dimension."""
        torch.manual_seed(42)
        custom_hidden = 64
        model = PerWorldGRUBaseline(
            param_dims=PARAM_DIMS,
            hidden_dim=custom_hidden,
            H=H,
        )

        for world_id, n_params in PARAM_DIMS.items():
            theta = torch.randn(4, n_params)
            irf = model(theta, world_id=world_id)
            assert irf.shape == (4, H + 1, N_OBS)

    def test_sequential_generation(self, model):
        """Test that GRU generates sequence properly (H+1 timesteps)."""
        torch.manual_seed(42)
        theta = torch.randn(2, PARAM_DIMS["lss"])
        irf = model(theta, world_id="lss")

        # Should have exactly H+1 timesteps
        assert irf.shape[1] == H + 1

        # Each timestep should have 3 observables
        for t in range(H + 1):
            assert irf[:, t, :].shape == (2, N_OBS)

    def test_cpu_device(self, model):
        """Test model works on CPU."""
        for world_id, n_params in PARAM_DIMS.items():
            theta = torch.randn(4, n_params, device='cpu')
            irf = model(theta, world_id=world_id)
            assert irf.device.type == 'cpu'
            assert irf.shape == (4, H + 1, N_OBS)


@pytest.mark.fast
class TestPooledMLPBaseline:
    """Test suite for PooledMLPBaseline."""

    @pytest.fixture
    def model(self):
        """Create a PooledMLPBaseline model."""
        torch.manual_seed(42)
        return PooledMLPBaseline(
            world_ids=WORLD_IDS,
            param_dims=PARAM_DIMS,
            world_embed_dim=16,
            hidden_dims=[512, 256, 128],
            H=H,
        )

    def test_output_shape_single_sample(self, model):
        """Test output shape for single sample across all worlds."""
        torch.manual_seed(42)
        for world_id, n_params in PARAM_DIMS.items():
            theta = torch.randn(1, n_params)
            irf = model(theta, world_id=world_id)
            assert irf.shape == (1, H + 1, N_OBS)

    def test_output_shape_batch_8(self, model):
        """Test output shape for batch size 8 across all worlds."""
        torch.manual_seed(42)
        for world_id, n_params in PARAM_DIMS.items():
            theta = torch.randn(8, n_params)
            irf = model(theta, world_id=world_id)
            assert irf.shape == (8, H + 1, N_OBS)

    def test_output_shape_batch_32(self, model):
        """Test output shape for batch size 32 across all worlds."""
        torch.manual_seed(42)
        for world_id, n_params in PARAM_DIMS.items():
            theta = torch.randn(32, n_params)
            irf = model(theta, world_id=world_id)
            assert irf.shape == (32, H + 1, N_OBS)

    def test_different_worlds_different_embeddings(self, model):
        """Test that different worlds use different embeddings."""
        torch.manual_seed(42)
        # Same theta dimension for comparison
        theta = torch.randn(4, 5)

        irf_lss = model(theta, world_id="lss")
        irf_var = model(theta, world_id="var")
        irf_nk = model(theta, world_id="nk")

        # Different world embeddings should give different outputs
        assert not torch.allclose(irf_lss, irf_var, atol=1e-5)
        assert not torch.allclose(irf_lss, irf_nk, atol=1e-5)

    def test_parameter_padding(self, model):
        """Test that parameters are correctly padded to max_params."""
        assert model.max_params == max(PARAM_DIMS.values())  # Should be 10

        # Test with world that needs padding (lss has 5 params, max is 10)
        torch.manual_seed(42)
        theta_lss = torch.randn(4, PARAM_DIMS["lss"])  # 5 params
        irf = model(theta_lss, world_id="lss")
        assert irf.shape == (4, H + 1, N_OBS)

        # Test with world at max params (nk has 10 params)
        theta_nk = torch.randn(4, PARAM_DIMS["nk"])  # 10 params
        irf = model(theta_nk, world_id="nk")
        assert irf.shape == (4, H + 1, N_OBS)

    def test_gradient_flow(self, model):
        """Test gradients flow through the model for each world."""
        torch.manual_seed(42)
        for world_id, n_params in PARAM_DIMS.items():
            theta = torch.randn(4, n_params, requires_grad=True)

            irf = model(theta, world_id=world_id)
            loss = irf.sum()
            loss.backward()

            # Check input gradients
            assert theta.grad is not None
            assert not torch.all(theta.grad == 0)

            # Reset gradients for next world
            model.zero_grad()

    def test_world_embedding_layer(self, model):
        """Test world embedding layer properties."""
        assert isinstance(model.world_embeddings, nn.Embedding)
        assert model.world_embeddings.num_embeddings == len(WORLD_IDS)
        assert model.world_embeddings.embedding_dim == 16

    def test_world_to_idx_mapping(self, model):
        """Test world_id to index mapping is correct."""
        assert len(model.world_to_idx) == len(WORLD_IDS)
        for idx, world_id in enumerate(WORLD_IDS):
            assert model.world_to_idx[world_id] == idx

    def test_custom_world_embed_dim(self):
        """Test with custom world embedding dimension."""
        torch.manual_seed(42)
        custom_embed_dim = 32
        model = PooledMLPBaseline(
            world_ids=WORLD_IDS,
            param_dims=PARAM_DIMS,
            world_embed_dim=custom_embed_dim,
            H=H,
        )

        assert model.world_embed_dim == custom_embed_dim
        assert model.world_embeddings.embedding_dim == custom_embed_dim

        for world_id, n_params in PARAM_DIMS.items():
            theta = torch.randn(4, n_params)
            irf = model(theta, world_id=world_id)
            assert irf.shape == (4, H + 1, N_OBS)

    def test_custom_hidden_dims(self):
        """Test with custom hidden dimensions."""
        torch.manual_seed(42)
        custom_dims = [256, 128]
        model = PooledMLPBaseline(
            world_ids=WORLD_IDS,
            param_dims=PARAM_DIMS,
            hidden_dims=custom_dims,
            H=H,
        )

        for world_id, n_params in PARAM_DIMS.items():
            theta = torch.randn(4, n_params)
            irf = model(theta, world_id=world_id)
            assert irf.shape == (4, H + 1, N_OBS)

    def test_cpu_device(self, model):
        """Test model works on CPU."""
        for world_id, n_params in PARAM_DIMS.items():
            theta = torch.randn(4, n_params, device='cpu')
            irf = model(theta, world_id=world_id)
            assert irf.device.type == 'cpu'
            assert irf.shape == (4, H + 1, N_OBS)


@pytest.mark.fast
class TestOracleBaseline:
    """Test suite for OracleBaseline (using mocks)."""

    @pytest.fixture
    def mock_simulator(self):
        """Create a mock simulator."""
        sim = MagicMock()

        # Mock get_analytic_irf to return proper shape
        def mock_get_analytic_irf(theta, shock_idx, shock_size, H):
            import numpy as np
            return np.random.randn(H + 1, 3)

        sim.get_analytic_irf = mock_get_analytic_irf
        return sim

    @pytest.fixture
    def model(self, mock_simulator):
        """Create an OracleBaseline model with mock simulators."""
        simulators = {
            "lss": mock_simulator,
            "var": mock_simulator,
            "nk": mock_simulator,
        }
        return OracleBaseline(simulators=simulators)

    def test_output_shape_single_sample(self, model):
        """Test output shape for single sample."""
        torch.manual_seed(42)
        theta = torch.randn(1, 5)
        irf = model(theta, world_id="lss", H=H)
        assert irf.shape == (1, H + 1, N_OBS)

    def test_output_shape_batch_8(self, model):
        """Test output shape for batch size 8."""
        torch.manual_seed(42)
        theta = torch.randn(8, 5)
        irf = model(theta, world_id="lss", H=H)
        assert irf.shape == (8, H + 1, N_OBS)

    def test_output_shape_batch_32(self, model):
        """Test output shape for batch size 32."""
        torch.manual_seed(42)
        theta = torch.randn(32, 5)
        irf = model(theta, world_id="lss", H=H)
        assert irf.shape == (32, H + 1, N_OBS)

    def test_different_worlds(self, model):
        """Test that different world_ids access different simulators."""
        torch.manual_seed(42)
        theta = torch.randn(4, 5)

        # Should work for all world_ids
        irf_lss = model(theta, world_id="lss", H=H)
        irf_var = model(theta, world_id="var", H=H)
        irf_nk = model(theta, world_id="nk", H=H)

        assert irf_lss.shape == (4, H + 1, N_OBS)
        assert irf_var.shape == (4, H + 1, N_OBS)
        assert irf_nk.shape == (4, H + 1, N_OBS)

    def test_theta_conversion_to_numpy(self):
        """Test that theta is properly converted to numpy for simulator."""
        import numpy as np

        # Create simulator with MagicMock to track calls
        sim = MagicMock()
        sim.get_analytic_irf = MagicMock(
            side_effect=lambda theta, shock_idx, shock_size, H: np.random.randn(H + 1, 3)
        )

        model = OracleBaseline(simulators={"lss": sim})

        torch.manual_seed(42)
        theta = torch.randn(2, 5)

        # Call model
        irf = model(theta, world_id="lss", H=H)

        # Verify simulator was called
        assert sim.get_analytic_irf.called
        # Verify it was called with numpy arrays (not torch tensors)
        call_args = sim.get_analytic_irf.call_args_list[0]
        assert isinstance(call_args[0][0], np.ndarray)

    def test_output_device_matches_input(self, model):
        """Test that output device matches input device."""
        theta_cpu = torch.randn(4, 5, device='cpu')
        irf = model(theta_cpu, world_id="lss", H=H)
        assert irf.device.type == 'cpu'

    def test_fallback_to_compute_irf(self):
        """Test fallback to compute_irf when get_analytic_irf returns None."""
        import numpy as np

        # Create simulator that returns None for get_analytic_irf
        sim = MagicMock()
        sim.get_analytic_irf = MagicMock(return_value=None)

        def mock_compute_irf(theta, shock_idx, shock_size, H, x0):
            return np.random.randn(H + 1, 3)

        sim.compute_irf = mock_compute_irf

        model = OracleBaseline(simulators={"test": sim})

        theta = torch.randn(2, 5)
        irf = model(theta, world_id="test", H=H)

        # Should have called both methods
        assert sim.get_analytic_irf.called

        # Output should still be correct shape
        assert irf.shape == (2, H + 1, N_OBS)


@pytest.mark.fast
class TestBaselineIntegration:
    """Integration tests comparing baseline models."""

    def test_all_baselines_same_interface(self):
        """Test that all baselines have compatible interfaces."""
        torch.manual_seed(42)

        # Create all models
        linear = LinearBaseline(n_params=8, H=H)
        per_world_mlp = PerWorldMLPBaseline(param_dims=PARAM_DIMS, H=H)
        per_world_gru = PerWorldGRUBaseline(param_dims=PARAM_DIMS, H=H)
        pooled = PooledMLPBaseline(
            world_ids=WORLD_IDS,
            param_dims=PARAM_DIMS,
            H=H,
        )

        # Test same input works for all
        theta = torch.randn(4, 8)
        world_id = "var"

        irf_linear = linear(theta, world_id=world_id, shock_idx=0)
        irf_per_mlp = per_world_mlp(theta, world_id=world_id, shock_idx=0)
        irf_per_gru = per_world_gru(theta, world_id=world_id, shock_idx=0)
        irf_pooled = pooled(theta, world_id=world_id, shock_idx=0)

        # All should produce same shape
        assert irf_linear.shape == (4, H + 1, N_OBS)
        assert irf_per_mlp.shape == (4, H + 1, N_OBS)
        assert irf_per_gru.shape == (4, H + 1, N_OBS)
        assert irf_pooled.shape == (4, H + 1, N_OBS)

    def test_different_horizons(self):
        """Test that all baselines work with different horizons."""
        torch.manual_seed(42)

        for H_test in [20, 40, 80]:
            linear = LinearBaseline(n_params=8, H=H_test)
            per_world_mlp = PerWorldMLPBaseline(param_dims=PARAM_DIMS, H=H_test)
            per_world_gru = PerWorldGRUBaseline(param_dims=PARAM_DIMS, H=H_test)
            pooled = PooledMLPBaseline(
                world_ids=WORLD_IDS,
                param_dims=PARAM_DIMS,
                H=H_test,
            )

            theta = torch.randn(4, 8)

            assert linear(theta).shape == (4, H_test + 1, N_OBS)
            assert per_world_mlp(theta, "var").shape == (4, H_test + 1, N_OBS)
            assert per_world_gru(theta, "var").shape == (4, H_test + 1, N_OBS)
            assert pooled(theta, "var").shape == (4, H_test + 1, N_OBS)

    def test_parameter_count(self):
        """Test that models have reasonable parameter counts."""
        linear = LinearBaseline(n_params=8, H=40)
        per_world_mlp = PerWorldMLPBaseline(param_dims=PARAM_DIMS, H=40)
        per_world_gru = PerWorldGRUBaseline(param_dims=PARAM_DIMS, H=40)
        pooled = PooledMLPBaseline(
            world_ids=WORLD_IDS,
            param_dims=PARAM_DIMS,
            H=40,
        )

        def count_params(model):
            return sum(p.numel() for p in model.parameters())

        # Linear should have fewest parameters
        linear_params = count_params(linear)
        assert linear_params > 0

        # Per-world models should have more (separate networks)
        per_mlp_params = count_params(per_world_mlp)
        per_gru_params = count_params(per_world_gru)
        assert per_mlp_params > linear_params
        assert per_gru_params > 0

        # Pooled should be in between (shared network)
        pooled_params = count_params(pooled)
        assert pooled_params > linear_params
