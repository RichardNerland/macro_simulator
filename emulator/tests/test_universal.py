"""
Tests for the Universal Emulator architecture.

This module tests all components of the universal emulator:
- Individual components (WorldEmbedding, ParameterEncoder, etc.)
- Full model forward pass
- Different information regimes (A, B1, C)
- Gradient flow
- Shape correctness
- Multi-world batches
"""

import pytest
import torch

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


@pytest.mark.fast
class TestWorldEmbedding:
    """Test world embedding layer."""

    def test_init(self):
        """Test initialization."""
        world_ids = ["lss", "var", "nk"]
        embed = WorldEmbedding(world_ids, embed_dim=32)

        assert len(embed.world_ids) == 3
        assert embed.embed_dim == 32
        assert embed.embeddings.num_embeddings == 3

    def test_forward_single_string(self):
        """Test forward with single world_id string."""
        world_ids = ["lss", "var", "nk"]
        embed = WorldEmbedding(world_ids, embed_dim=32)

        output = embed("nk")
        assert output.shape == (1, 32)

    def test_forward_list(self):
        """Test forward with list of world_ids."""
        world_ids = ["lss", "var", "nk"]
        embed = WorldEmbedding(world_ids, embed_dim=32)

        output = embed(["nk", "var", "lss"])
        assert output.shape == (3, 32)

    def test_forward_tensor(self):
        """Test forward with tensor of indices."""
        world_ids = ["lss", "var", "nk"]
        embed = WorldEmbedding(world_ids, embed_dim=32)

        indices = torch.tensor([0, 1, 2, 1])
        output = embed(indices)
        assert output.shape == (4, 32)

    def test_different_worlds_different_embeddings(self):
        """Test that different worlds get different embeddings."""
        world_ids = ["lss", "var", "nk"]
        embed = WorldEmbedding(world_ids, embed_dim=32)

        emb_lss = embed("lss")
        emb_var = embed("var")

        # Should be different
        assert not torch.allclose(emb_lss, emb_var)


@pytest.mark.fast
class TestParameterEncoder:
    """Test parameter encoder."""

    def test_init(self):
        """Test initialization."""
        encoder = ParameterEncoder(max_params=50, embed_dim=64)
        assert encoder.max_params == 50
        assert encoder.embed_dim == 64

    def test_forward_no_mask(self):
        """Test forward without mask."""
        encoder = ParameterEncoder(max_params=50, embed_dim=64)
        batch_size = 8
        n_params = 12

        theta = torch.randn(batch_size, n_params)
        output = encoder(theta)

        assert output.shape == (batch_size, 64)

    def test_forward_with_mask(self):
        """Test forward with mask."""
        encoder = ParameterEncoder(max_params=50, embed_dim=64)
        batch_size = 8
        max_params = 20

        # Create theta with variable lengths (padded)
        theta = torch.randn(batch_size, max_params)
        mask = torch.ones(batch_size, max_params, dtype=torch.bool)
        # Mask out last 5 parameters for some samples
        mask[0, 15:] = False
        mask[1, 10:] = False

        output = encoder(theta, mask)

        assert output.shape == (batch_size, 64)

    def test_gradient_flow(self):
        """Test that gradients flow through encoder."""
        encoder = ParameterEncoder(max_params=50, embed_dim=64)
        theta = torch.randn(4, 12, requires_grad=True)

        output = encoder(theta)
        loss = output.sum()
        loss.backward()

        assert theta.grad is not None
        assert not torch.allclose(theta.grad, torch.zeros_like(theta.grad))


@pytest.mark.fast
class TestShockEncoder:
    """Test shock encoder."""

    def test_init(self):
        """Test initialization."""
        encoder = ShockEncoder(max_shocks=3, embed_dim=16)
        assert encoder.max_shocks == 3
        assert encoder.embed_dim == 16

    def test_forward_defaults(self):
        """Test forward with default shock_size and shock_timing."""
        encoder = ShockEncoder(max_shocks=3, embed_dim=16)
        batch_size = 8

        shock_idx = torch.randint(0, 3, (batch_size,))
        output = encoder(shock_idx)

        assert output.shape == (batch_size, 16)

    def test_forward_all_args(self):
        """Test forward with all arguments."""
        encoder = ShockEncoder(max_shocks=3, embed_dim=16)
        batch_size = 8

        shock_idx = torch.randint(0, 3, (batch_size,))
        shock_size = torch.ones(batch_size) * 2.0
        shock_timing = torch.zeros(batch_size)

        output = encoder(shock_idx, shock_size, shock_timing)

        assert output.shape == (batch_size, 16)

    def test_different_shocks_different_embeddings(self):
        """Test that different shock indices produce different embeddings."""
        encoder = ShockEncoder(max_shocks=3, embed_dim=16)

        shock_0 = encoder(torch.tensor([0]))
        shock_1 = encoder(torch.tensor([1]))

        assert not torch.allclose(shock_0, shock_1)

    def test_gradient_flow(self):
        """Test gradient flow."""
        encoder = ShockEncoder(max_shocks=3, embed_dim=16)
        shock_idx = torch.tensor([0, 1, 2])

        output = encoder(shock_idx)
        loss = output.sum()
        loss.backward()

        # Check gradients exist for embedding layer
        assert encoder.shock_idx_embed.weight.grad is not None


@pytest.mark.fast
class TestHistoryEncoder:
    """Test history encoder."""

    def test_init_gru(self):
        """Test initialization with GRU."""
        encoder = HistoryEncoder(n_obs=3, embed_dim=64, architecture="gru")
        assert encoder.architecture == "gru"
        assert encoder.embed_dim == 64

    def test_init_transformer(self):
        """Test initialization with Transformer."""
        encoder = HistoryEncoder(n_obs=3, embed_dim=64, architecture="transformer")
        assert encoder.architecture == "transformer"

    def test_forward_gru(self):
        """Test forward with GRU."""
        encoder = HistoryEncoder(n_obs=3, embed_dim=64, architecture="gru")
        batch_size = 8
        k = 20  # history length

        history = torch.randn(batch_size, k, 3)
        output = encoder(history)

        assert output.shape == (batch_size, 64)

    def test_forward_transformer(self):
        """Test forward with Transformer."""
        encoder = HistoryEncoder(n_obs=3, embed_dim=64, architecture="transformer")
        batch_size = 8
        k = 20

        history = torch.randn(batch_size, k, 3)
        output = encoder(history)

        assert output.shape == (batch_size, 64)

    def test_forward_with_mask(self):
        """Test forward with padding mask."""
        encoder = HistoryEncoder(n_obs=3, embed_dim=64, architecture="gru")
        batch_size = 8
        k = 20

        history = torch.randn(batch_size, k, 3)
        mask = torch.ones(batch_size, k, dtype=torch.bool)
        # Mask out last 5 steps for some samples
        mask[0, 15:] = False

        output = encoder(history, mask)

        assert output.shape == (batch_size, 64)

    def test_gradient_flow(self):
        """Test gradient flow."""
        encoder = HistoryEncoder(n_obs=3, embed_dim=64, architecture="gru")
        history = torch.randn(4, 10, 3, requires_grad=True)

        output = encoder(history)
        loss = output.sum()
        loss.backward()

        assert history.grad is not None


@pytest.mark.fast
class TestTrunkNetwork:
    """Test trunk network."""

    def test_init_mlp(self):
        """Test initialization with MLP."""
        trunk = TrunkNetwork(input_dim=128, hidden_dim=256, n_layers=4, architecture="mlp")
        assert trunk.architecture == "mlp"
        assert trunk.hidden_dim == 256

    def test_init_transformer(self):
        """Test initialization with Transformer."""
        trunk = TrunkNetwork(input_dim=128, hidden_dim=256, n_layers=4, architecture="transformer")
        assert trunk.architecture == "transformer"

    def test_forward_mlp(self):
        """Test forward with MLP."""
        trunk = TrunkNetwork(input_dim=128, hidden_dim=256, n_layers=4, architecture="mlp")
        batch_size = 8

        x = torch.randn(batch_size, 128)
        output = trunk(x)

        assert output.shape == (batch_size, 256)

    def test_forward_transformer(self):
        """Test forward with Transformer."""
        trunk = TrunkNetwork(input_dim=128, hidden_dim=256, n_layers=4, architecture="transformer")
        batch_size = 8

        x = torch.randn(batch_size, 128)
        output = trunk(x)

        assert output.shape == (batch_size, 256)

    def test_gradient_flow(self):
        """Test gradient flow."""
        trunk = TrunkNetwork(input_dim=128, hidden_dim=256, n_layers=2, architecture="mlp")
        x = torch.randn(4, 128, requires_grad=True)

        output = trunk(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


@pytest.mark.fast
class TestIRFHead:
    """Test IRF prediction head."""

    def test_init(self):
        """Test initialization."""
        head = IRFHead(input_dim=256, H=40, n_obs=3)
        assert head.H == 40
        assert head.n_obs == 3

    def test_forward(self):
        """Test forward pass."""
        head = IRFHead(input_dim=256, H=40, n_obs=3)
        batch_size = 8

        x = torch.randn(batch_size, 256)
        irf = head(x)

        assert irf.shape == (batch_size, 41, 3)  # H+1 = 41

    def test_different_horizons(self):
        """Test with different horizon lengths."""
        for H in [20, 40, 80]:
            head = IRFHead(input_dim=256, H=H, n_obs=3)
            x = torch.randn(4, 256)
            irf = head(x)
            assert irf.shape == (4, H + 1, 3)

    def test_gradient_flow(self):
        """Test gradient flow."""
        head = IRFHead(input_dim=256, H=40, n_obs=3)
        x = torch.randn(4, 256, requires_grad=True)

        irf = head(x)
        loss = irf.sum()
        loss.backward()

        assert x.grad is not None


@pytest.mark.fast
class TestTrajectoryHead:
    """Test trajectory prediction head."""

    def test_init(self):
        """Test initialization."""
        head = TrajectoryHead(input_dim=256, T=40, n_obs=3)
        assert head.T == 40
        assert head.n_obs == 3

    def test_forward(self):
        """Test forward pass."""
        head = TrajectoryHead(input_dim=256, T=40, n_obs=3)
        batch_size = 8

        x = torch.randn(batch_size, 256)
        traj = head(x)

        assert traj.shape == (batch_size, 40, 3)

    def test_gradient_flow(self):
        """Test gradient flow."""
        head = TrajectoryHead(input_dim=256, T=40, n_obs=3)
        x = torch.randn(4, 256, requires_grad=True)

        traj = head(x)
        loss = traj.sum()
        loss.backward()

        assert x.grad is not None


@pytest.mark.fast
class TestUniversalEmulator:
    """Test full universal emulator model."""

    @pytest.fixture
    def world_ids(self):
        """World identifiers for testing."""
        return ["lss", "var", "nk", "rbc", "switching", "zlb"]

    @pytest.fixture
    def param_dims(self):
        """Parameter dimensions per world."""
        return {
            "lss": 15,
            "var": 12,
            "nk": 10,
            "rbc": 8,
            "switching": 25,
            "zlb": 12,
        }

    @pytest.fixture
    def model(self, world_ids, param_dims):
        """Create test model."""
        return UniversalEmulator(
            world_ids=world_ids,
            param_dims=param_dims,
            world_embed_dim=32,
            theta_embed_dim=64,
            shock_embed_dim=16,
            history_embed_dim=64,
            trunk_dim=256,
            trunk_layers=2,  # Smaller for faster tests
            H=40,
            n_obs=3,
            max_shocks=3,
            use_history_encoder=True,
            use_trajectory_head=False,
            dropout=0.1,
        )

    def test_init(self, model, world_ids, param_dims):
        """Test initialization."""
        assert model.world_ids == world_ids
        assert model.param_dims == param_dims
        assert model.H == 40
        assert model.n_obs == 3
        assert model.use_history_encoder is True

    def test_regime_a_single_world(self, model):
        """Test Regime A: world_id, theta, shock_token."""
        batch_size = 8
        n_params = 10  # nk has 10 params

        theta = torch.randn(batch_size, n_params)
        shock_idx = torch.randint(0, 3, (batch_size,))

        irf = model(
            world_id="nk",
            theta=theta,
            shock_idx=shock_idx,
            regime="A",
        )

        assert irf.shape == (batch_size, 41, 3)  # (batch, H+1, n_obs)

    def test_regime_a_mixed_worlds(self, model):
        """Test Regime A with mixed worlds in batch."""
        batch_size = 8

        # Mixed batch: different worlds with different param dims
        world_ids = ["nk", "nk", "var", "var", "lss", "lss", "rbc", "rbc"]

        # Create padded theta (pad to max in batch)
        max_params_in_batch = 15  # lss has 15
        theta = torch.randn(batch_size, max_params_in_batch)

        # Create mask
        theta_mask = torch.zeros(batch_size, max_params_in_batch, dtype=torch.bool)
        theta_mask[:2, :10] = True  # nk: 10 params
        theta_mask[2:4, :12] = True  # var: 12 params
        theta_mask[4:6, :15] = True  # lss: 15 params
        theta_mask[6:8, :8] = True  # rbc: 8 params

        shock_idx = torch.randint(0, 3, (batch_size,))

        irf = model(
            world_id=world_ids,
            theta=theta,
            theta_mask=theta_mask,
            shock_idx=shock_idx,
            regime="A",
        )

        assert irf.shape == (batch_size, 41, 3)

    def test_regime_b1_with_history(self, model):
        """Test Regime B1: world_id, shock_token, history (no theta)."""
        batch_size = 8
        k = 20  # history length

        history = torch.randn(batch_size, k, 3)
        shock_idx = torch.randint(0, 3, (batch_size,))

        irf = model(
            world_id="nk",
            shock_idx=shock_idx,
            history=history,
            regime="B1",
        )

        assert irf.shape == (batch_size, 41, 3)

    def test_regime_c_theta_and_history(self, model):
        """Test Regime C: world_id, theta, shock_token, history."""
        batch_size = 8
        n_params = 10
        k = 20

        theta = torch.randn(batch_size, n_params)
        history = torch.randn(batch_size, k, 3)
        shock_idx = torch.randint(0, 3, (batch_size,))

        irf = model(
            world_id="nk",
            theta=theta,
            shock_idx=shock_idx,
            history=history,
            regime="C",
        )

        assert irf.shape == (batch_size, 41, 3)

    def test_regime_a_missing_theta_raises(self, model):
        """Test that Regime A without theta raises error."""
        shock_idx = torch.randint(0, 3, (4,))

        with pytest.raises(ValueError, match="Regime A requires theta"):
            model(
                world_id="nk",
                shock_idx=shock_idx,
                regime="A",
            )

    def test_regime_b1_missing_history_raises(self, model):
        """Test that Regime B1 without history raises error."""
        shock_idx = torch.randint(0, 3, (4,))

        with pytest.raises(ValueError, match="Regime B1 requires history"):
            model(
                world_id="nk",
                shock_idx=shock_idx,
                regime="B1",
            )

    def test_regime_c_missing_inputs_raises(self, model):
        """Test that Regime C without required inputs raises error."""
        shock_idx = torch.randint(0, 3, (4,))

        # Missing theta
        with pytest.raises(ValueError, match="Regime C requires theta"):
            model(
                world_id="nk",
                shock_idx=shock_idx,
                history=torch.randn(4, 10, 3),
                regime="C",
            )

        # Missing history
        with pytest.raises(ValueError, match="Regime C requires history"):
            model(
                world_id="nk",
                theta=torch.randn(4, 10),
                shock_idx=shock_idx,
                regime="C",
            )

    def test_gradient_flow_regime_a(self, model):
        """Test gradient flow in Regime A."""
        batch_size = 4
        n_params = 10

        theta = torch.randn(batch_size, n_params, requires_grad=True)
        shock_idx = torch.randint(0, 3, (batch_size,))

        irf = model(
            world_id="nk",
            theta=theta,
            shock_idx=shock_idx,
            regime="A",
        )

        loss = irf.sum()
        loss.backward()

        # Check gradients exist
        assert theta.grad is not None
        assert not torch.allclose(theta.grad, torch.zeros_like(theta.grad))

    def test_gradient_flow_regime_b1(self, model):
        """Test gradient flow in Regime B1."""
        batch_size = 4
        k = 10

        history = torch.randn(batch_size, k, 3, requires_grad=True)
        shock_idx = torch.randint(0, 3, (batch_size,))

        irf = model(
            world_id="nk",
            shock_idx=shock_idx,
            history=history,
            regime="B1",
        )

        loss = irf.sum()
        loss.backward()

        assert history.grad is not None

    def test_trajectory_head(self, world_ids, param_dims):
        """Test model with trajectory head."""
        model = UniversalEmulator(
            world_ids=world_ids,
            param_dims=param_dims,
            use_trajectory_head=True,
            trunk_layers=2,
        )

        batch_size = 4
        n_params = 10
        theta = torch.randn(batch_size, n_params)
        shock_idx = torch.randint(0, 3, (batch_size,))

        irf, traj = model(
            world_id="nk",
            theta=theta,
            shock_idx=shock_idx,
            regime="A",
            return_trajectory=True,
        )

        assert irf.shape == (batch_size, 41, 3)
        assert traj.shape == (batch_size, 40, 3)  # Same as H

    def test_trajectory_head_without_flag_raises(self, model):
        """Test that requesting trajectory without head raises error."""
        theta = torch.randn(4, 10)
        shock_idx = torch.randint(0, 3, (4,))

        with pytest.raises(ValueError, match="return_trajectory=True but use_trajectory_head=False"):
            model(
                world_id="nk",
                theta=theta,
                shock_idx=shock_idx,
                regime="A",
                return_trajectory=True,
            )

    def test_get_num_parameters(self, model):
        """Test parameter counting."""
        n_params = model.get_num_parameters()
        assert n_params > 0
        assert isinstance(n_params, int)

    def test_different_worlds_produce_different_outputs(self, model):
        """Test that different worlds produce different outputs (sanity check)."""
        batch_size = 4
        n_params = 10
        theta = torch.randn(batch_size, n_params)
        shock_idx = torch.randint(0, 3, (batch_size,))

        # Same inputs, different worlds
        irf_nk = model(world_id="nk", theta=theta, shock_idx=shock_idx, regime="A")
        irf_var = model(world_id="var", theta=theta[:, :12], shock_idx=shock_idx, regime="A")

        # Outputs should be different due to world embedding
        assert not torch.allclose(irf_nk, irf_var[:, :, :])

    def test_shock_size_and_timing(self, model):
        """Test with non-default shock size and timing."""
        batch_size = 4
        n_params = 10
        theta = torch.randn(batch_size, n_params)
        shock_idx = torch.randint(0, 3, (batch_size,))
        shock_size = torch.ones(batch_size) * 2.0  # 2 std dev shock
        shock_timing = torch.zeros(batch_size)  # Impact at t=0

        irf = model(
            world_id="nk",
            theta=theta,
            shock_idx=shock_idx,
            shock_size=shock_size,
            shock_timing=shock_timing,
            regime="A",
        )

        assert irf.shape == (batch_size, 41, 3)

    def test_model_eval_mode(self, model):
        """Test model in eval mode (no dropout)."""
        model.eval()

        theta = torch.randn(4, 10)
        shock_idx = torch.randint(0, 3, (4,))

        with torch.no_grad():
            irf = model(
                world_id="nk",
                theta=theta,
                shock_idx=shock_idx,
                regime="A",
            )

        assert irf.shape == (4, 41, 3)

    def test_model_train_mode(self, model):
        """Test model in train mode (with dropout)."""
        model.train()

        theta = torch.randn(4, 10)
        shock_idx = torch.randint(0, 3, (4,))

        irf = model(
            world_id="nk",
            theta=theta,
            shock_idx=shock_idx,
            regime="A",
        )

        assert irf.shape == (4, 41, 3)


@pytest.mark.fast
class TestUniversalEmulatorEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def world_ids(self):
        return ["lss", "var", "nk"]

    @pytest.fixture
    def param_dims(self):
        return {"lss": 15, "var": 12, "nk": 10}

    @pytest.fixture
    def model(self, world_ids, param_dims):
        return UniversalEmulator(
            world_ids=world_ids,
            param_dims=param_dims,
            trunk_layers=2,
        )

    def test_invalid_regime_raises(self, model):
        """Test that invalid regime raises error."""
        theta = torch.randn(4, 10)
        shock_idx = torch.randint(0, 3, (4,))

        with pytest.raises(ValueError, match="Unknown regime"):
            model(
                world_id="nk",
                theta=theta,
                shock_idx=shock_idx,
                regime="INVALID",
            )

    def test_batch_size_one(self, model):
        """Test with batch size 1."""
        theta = torch.randn(1, 10)
        shock_idx = torch.randint(0, 3, (1,))

        irf = model(
            world_id="nk",
            theta=theta,
            shock_idx=shock_idx,
            regime="A",
        )

        assert irf.shape == (1, 41, 3)

    def test_large_batch(self, model):
        """Test with large batch size."""
        batch_size = 128
        theta = torch.randn(batch_size, 10)
        shock_idx = torch.randint(0, 3, (batch_size,))

        irf = model(
            world_id="nk",
            theta=theta,
            shock_idx=shock_idx,
            regime="A",
        )

        assert irf.shape == (batch_size, 41, 3)

    def test_no_history_encoder_regime_b1_raises(self, world_ids, param_dims):
        """Test that Regime B1 without history encoder raises error."""
        model = UniversalEmulator(
            world_ids=world_ids,
            param_dims=param_dims,
            use_history_encoder=False,
            trunk_layers=2,
        )

        shock_idx = torch.randint(0, 3, (4,))
        history = torch.randn(4, 10, 3)

        with pytest.raises(ValueError, match="Regime B1 requires use_history_encoder=True"):
            model(
                world_id="nk",
                shock_idx=shock_idx,
                history=history,
                regime="B1",
            )
