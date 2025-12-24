"""
Baseline models for IRF prediction.

This module implements simple baseline models to compare against the universal emulator:
- OracleBaseline: Uses analytic IRF when available, otherwise simulation
- LinearBaseline: Simple linear regression from theta to IRF
- PerWorldMLPBaseline: One MLP per world_id
- PerWorldGRUBaseline: One GRU per world_id for sequence modeling
- PooledMLPBaseline: Single MLP for all worlds with world embeddings
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn


class OracleBaseline(nn.Module):
    """Oracle baseline using analytic IRFs or ground-truth simulation.

    This baseline uses the simulator's analytic IRF when available (LSS, VAR, RBC),
    otherwise falls back to the simulator's compute_irf method. This represents
    the best possible baseline - it uses ground truth parameters and dynamics.

    Note: This is not a trainable model, it's purely for evaluation.
    """

    def __init__(self, simulators: dict[str, Any]):
        """Initialize oracle baseline.

        Args:
            simulators: Dict mapping world_id to simulator instance
        """
        super().__init__()
        self.simulators = simulators

    def forward(
        self,
        theta: torch.Tensor,
        world_id: str,
        shock_idx: int = 0,
        H: int = 40,
    ) -> torch.Tensor:
        """Compute IRF using ground-truth simulator.

        Args:
            theta: Parameters, shape (batch, n_params)
            world_id: Simulator identifier
            shock_idx: Index of shock to compute IRF for
            H: Horizon length (returns H+1 points)

        Returns:
            IRF predictions, shape (batch, H+1, 3)
        """
        simulator = self.simulators[world_id]
        batch_size = theta.shape[0]

        # Convert to numpy for simulator
        theta_np = theta.detach().cpu().numpy()

        irfs = []
        for i in range(batch_size):
            # Try analytic IRF first
            irf = simulator.get_analytic_irf(theta_np[i], shock_idx, shock_size=1.0, H=H)

            # Fall back to simulation if no analytic solution
            if irf is None:
                irf = simulator.compute_irf(
                    theta_np[i], shock_idx, shock_size=1.0, H=H, x0=None
                )

            irfs.append(irf)

        # Convert back to torch
        irfs = np.stack(irfs, axis=0)  # (batch, H+1, 3)
        return torch.from_numpy(irfs).to(theta.device)


class LinearBaseline(nn.Module):
    """Simple linear regression baseline: theta -> IRF.

    Maps parameter vector directly to flattened IRF via a learned weight matrix.
    """

    def __init__(self, n_params: int, H: int = 40):
        """Initialize linear baseline.

        Args:
            n_params: Number of input parameters
            H: Horizon length (returns H+1 points)
        """
        super().__init__()
        self.n_params = n_params
        self.H = H
        self.n_obs = 3  # output, inflation, rate

        # Linear layer: theta -> flattened IRF
        self.linear = nn.Linear(n_params, (H + 1) * self.n_obs)

    def forward(
        self,
        theta: torch.Tensor,
        world_id: str | None = None,
        shock_idx: int = 0,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            theta: Parameters, shape (batch, n_params)
            world_id: Ignored (for API compatibility)
            shock_idx: Ignored (single-shock version)

        Returns:
            IRF predictions, shape (batch, H+1, 3)
        """
        batch_size = theta.shape[0]

        # Linear transformation
        irf_flat = self.linear(theta)  # (batch, (H+1)*3)

        # Reshape to IRF format
        irf = irf_flat.view(batch_size, self.H + 1, self.n_obs)

        return irf


class PerWorldMLPBaseline(nn.Module):
    """Per-world MLP baseline.

    Maintains one separate MLP for each world_id. Each MLP learns the mapping
    from that world's parameters to IRFs.
    """

    def __init__(
        self,
        param_dims: dict[str, int],
        hidden_dims: list[int] | None = None,
        H: int = 40,
    ):
        """Initialize per-world MLP baseline.

        Args:
            param_dims: Dict mapping world_id to number of parameters
            hidden_dims: Hidden layer dimensions (default [256, 128, 64])
            H: Horizon length (returns H+1 points)
        """
        super().__init__()
        self.param_dims = param_dims
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.H = H
        self.n_obs = 3

        # Create one MLP per world
        self.mlps = nn.ModuleDict()
        for world_id, n_params in param_dims.items():
            layers: list[nn.Module] = []

            # Input layer
            in_dim = n_params
            for hidden_dim in self.hidden_dims:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                in_dim = hidden_dim

            # Output layer
            layers.append(nn.Linear(in_dim, (H + 1) * self.n_obs))

            self.mlps[world_id] = nn.Sequential(*layers)

    def forward(
        self,
        theta: torch.Tensor,
        world_id: str,
        shock_idx: int = 0,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            theta: Parameters, shape (batch, n_params)
            world_id: Simulator identifier
            shock_idx: Ignored (single-shock version)

        Returns:
            IRF predictions, shape (batch, H+1, 3)
        """
        batch_size = theta.shape[0]

        # Select MLP for this world
        mlp = self.mlps[world_id]

        # Forward pass
        irf_flat = mlp(theta)  # (batch, (H+1)*3)

        # Reshape to IRF format
        irf = irf_flat.view(batch_size, self.H + 1, self.n_obs)

        return irf


class PerWorldGRUBaseline(nn.Module):
    """Per-world GRU baseline for sequence modeling.

    Uses GRU to produce IRF sequence horizon-by-horizon. Parameters are
    encoded to initial hidden state, then GRU generates H+1 outputs.
    """

    def __init__(
        self,
        param_dims: dict[str, int],
        hidden_dim: int = 128,
        H: int = 40,
    ):
        """Initialize per-world GRU baseline.

        Args:
            param_dims: Dict mapping world_id to number of parameters
            hidden_dim: GRU hidden dimension
            H: Horizon length (returns H+1 points)
        """
        super().__init__()
        self.param_dims = param_dims
        self.hidden_dim = hidden_dim
        self.H = H
        self.n_obs = 3

        # Create theta encoder and GRU for each world
        self.theta_encoders = nn.ModuleDict()
        self.grus = nn.ModuleDict()
        self.output_layers = nn.ModuleDict()

        for world_id, n_params in param_dims.items():
            # Encoder: theta -> initial hidden state
            self.theta_encoders[world_id] = nn.Sequential(
                nn.Linear(n_params, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            # GRU cell
            self.grus[world_id] = nn.GRUCell(self.n_obs, hidden_dim)

            # Output layer: hidden -> observables
            self.output_layers[world_id] = nn.Linear(hidden_dim, self.n_obs)

    def forward(
        self,
        theta: torch.Tensor,
        world_id: str,
        shock_idx: int = 0,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            theta: Parameters, shape (batch, n_params)
            world_id: Simulator identifier
            shock_idx: Ignored (single-shock version)

        Returns:
            IRF predictions, shape (batch, H+1, 3)
        """
        batch_size = theta.shape[0]
        device = theta.device

        # Encode theta to initial hidden state
        h = self.theta_encoders[world_id](theta)  # (batch, hidden_dim)

        # Generate sequence
        outputs = []
        y_prev = torch.zeros(batch_size, self.n_obs, device=device)

        gru = self.grus[world_id]
        output_layer = self.output_layers[world_id]

        for t in range(self.H + 1):
            # Update hidden state
            h = gru(y_prev, h)  # (batch, hidden_dim)

            # Generate output
            y_t = output_layer(h)  # (batch, 3)
            outputs.append(y_t)

            # Feed output back as input
            y_prev = y_t

        # Stack outputs
        irf = torch.stack(outputs, dim=1)  # (batch, H+1, 3)

        return irf


class PooledMLPBaseline(nn.Module):
    """Pooled MLP baseline for all worlds.

    Single MLP that handles all worlds. Uses learnable world embeddings and
    pads theta to maximum parameter dimension across all worlds.
    """

    def __init__(
        self,
        world_ids: list[str],
        param_dims: dict[str, int],
        world_embed_dim: int = 16,
        hidden_dims: list[int] | None = None,
        H: int = 40,
    ):
        """Initialize pooled MLP baseline.

        Args:
            world_ids: List of world identifiers
            param_dims: Dict mapping world_id to number of parameters
            world_embed_dim: Dimension of world embeddings
            hidden_dims: Hidden layer dimensions
            H: Horizon length (returns H+1 points)
        """
        super().__init__()
        self.world_ids = world_ids
        self.param_dims = param_dims
        self.world_embed_dim = world_embed_dim
        self.hidden_dims = hidden_dims or [512, 256, 128]
        self.H = H
        self.n_obs = 3

        # Maximum parameter dimension across all worlds
        self.max_params = max(param_dims.values())

        # World embeddings
        self.world_embeddings = nn.Embedding(len(world_ids), world_embed_dim)

        # World ID to index mapping
        self.world_to_idx = {wid: idx for idx, wid in enumerate(world_ids)}

        # MLP
        layers: list[nn.Module] = []
        in_dim = world_embed_dim + self.max_params

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, (H + 1) * self.n_obs))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        theta: torch.Tensor,
        world_id: str,
        shock_idx: int = 0,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            theta: Parameters, shape (batch, n_params) - may vary by world
            world_id: Simulator identifier
            shock_idx: Ignored (single-shock version)

        Returns:
            IRF predictions, shape (batch, H+1, 3)
        """
        batch_size = theta.shape[0]
        device = theta.device

        # Get world embedding
        world_idx = self.world_to_idx[world_id]
        world_embed = self.world_embeddings(
            torch.tensor([world_idx], device=device)
        )  # (1, world_embed_dim)
        world_embed = world_embed.expand(batch_size, -1)  # (batch, world_embed_dim)

        # Pad theta to max_params
        n_params = theta.shape[1]
        if n_params < self.max_params:
            padding = torch.zeros(batch_size, self.max_params - n_params, device=device)
            theta_padded = torch.cat([theta, padding], dim=1)
        else:
            theta_padded = theta

        # Concatenate world embedding and padded theta
        x = torch.cat(
            [world_embed, theta_padded], dim=1
        )  # (batch, world_embed_dim + max_params)

        # Forward pass through MLP
        irf_flat = self.mlp(x)  # (batch, (H+1)*3)

        # Reshape to IRF format
        irf = irf_flat.view(batch_size, self.H + 1, self.n_obs)

        return irf
