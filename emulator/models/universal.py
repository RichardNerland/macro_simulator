"""
Universal Macro Emulator: Main neural network architecture for IRF prediction across all worlds.

This module implements the universal emulator that predicts impulse response functions (IRFs)
across 6 simulator families: LSS, VAR, NK, RBC, regime-switching, and ZLB.

Architecture components:
- World embeddings: Learnable representation per simulator family
- Parameter encoder: Handles variable-length parameter vectors
- Shock encoder: Encodes (shock_idx, shock_size, shock_timing)
- History encoder: Optional trajectory conditioning for regimes B1/C
- Trunk network: Transformer or MLP combining all embeddings
- IRF head: Multi-horizon prediction (H+1, 3) observables
- Trajectory head: Optional long-horizon trajectory prediction

Information Regimes:
- Regime A: world_id, theta, eps_sequence, shock_token
- Regime B1: world_id, shock_token, history (no theta, no eps)
- Regime C: world_id, theta, shock_token, history (no eps)
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldEmbedding(nn.Module):
    """Learnable embedding layer for world_id (simulator family).

    Maps discrete world_id (e.g., "nk", "var", "lss") to continuous embedding space.
    """

    def __init__(self, world_ids: list[str], embed_dim: int = 32):
        """Initialize world embedding layer.

        Args:
            world_ids: List of world identifiers (e.g., ["lss", "var", "nk", ...])
            embed_dim: Embedding dimension (default 32)
        """
        super().__init__()
        self.world_ids = world_ids
        self.embed_dim = embed_dim

        # Create mapping from world_id to index
        self.world_to_idx = {wid: idx for idx, wid in enumerate(world_ids)}

        # Learnable embeddings
        self.embeddings = nn.Embedding(len(world_ids), embed_dim)

    def forward(self, world_id: str | list[str] | torch.Tensor) -> torch.Tensor:
        """Embed world_id(s).

        Args:
            world_id: Either:
                - Single string: "nk"
                - List of strings: ["nk", "var", ...]
                - Tensor of indices: shape (batch,)

        Returns:
            Embeddings, shape (batch, embed_dim)
        """
        if isinstance(world_id, str):
            # Single world_id
            idx = torch.tensor([self.world_to_idx[world_id]], dtype=torch.long)
            return self.embeddings(idx)  # (1, embed_dim)

        elif isinstance(world_id, list):
            # List of world_ids
            indices = torch.tensor([self.world_to_idx[wid] for wid in world_id], dtype=torch.long)
            return self.embeddings(indices)  # (batch, embed_dim)

        else:
            # Already tensor of indices
            return self.embeddings(world_id)  # (batch, embed_dim)


class ParameterEncoder(nn.Module):
    """Encodes variable-length parameter vectors to fixed-dimension representation.

    Handles the fact that different simulators have different numbers of parameters
    by padding to max_params and using an MLP with masking.

    Architecture options:
    - MLP with masked pooling (default)
    - Set transformer (for better permutation invariance, future)
    """

    def __init__(
        self,
        max_params: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        """Initialize parameter encoder.

        Args:
            max_params: Maximum number of parameters across all worlds
            embed_dim: Output embedding dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.max_params = max_params
        self.embed_dim = embed_dim

        # MLP to encode each parameter value
        self.param_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Pooling: mean over valid parameters
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, theta: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode parameter vector.

        Args:
            theta: Parameter values, shape (batch, n_params) where n_params <= max_params
            mask: Boolean mask for valid parameters, shape (batch, n_params)
                  If None, assumes all are valid

        Returns:
            Parameter embeddings, shape (batch, embed_dim)
        """
        batch_size, n_params = theta.shape

        # Encode each parameter value independently
        theta_expanded = theta.unsqueeze(-1)  # (batch, n_params, 1)
        param_embeds = self.param_mlp(theta_expanded)  # (batch, n_params, hidden_dim)

        # Masked mean pooling
        if mask is None:
            # No mask, use all parameters
            pooled = param_embeds.mean(dim=1)  # (batch, hidden_dim)
        else:
            # Apply mask
            mask_expanded = mask.unsqueeze(-1).float()  # (batch, n_params, 1)
            masked_embeds = param_embeds * mask_expanded
            pooled = masked_embeds.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)  # (batch, hidden_dim)

        # Project to output dimension
        output = self.output_proj(pooled)  # (batch, embed_dim)

        return output


class ShockEncoder(nn.Module):
    """Encodes shock token: (shock_idx, shock_size, shock_timing).

    The shock_token specifies WHICH IRF to compute:
    - shock_idx: Which shock type (0 to n_shocks-1)
    - shock_size: Magnitude in std dev units (typically 1.0)
    - shock_timing: When shock hits (typically 0 for impact at t=0)

    Note: This is distinct from eps_sequence (full shock path), which is regime-dependent.
    """

    def __init__(
        self,
        max_shocks: int = 3,
        embed_dim: int = 16,
    ):
        """Initialize shock encoder.

        Args:
            max_shocks: Maximum number of shocks across all worlds
            embed_dim: Output embedding dimension
        """
        super().__init__()
        self.max_shocks = max_shocks
        self.embed_dim = embed_dim

        # Learnable embedding for shock index
        self.shock_idx_embed = nn.Embedding(max_shocks, embed_dim // 2)

        # MLP for continuous features (size, timing)
        self.continuous_mlp = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.ReLU(),
        )

        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        shock_idx: torch.Tensor,
        shock_size: torch.Tensor | None = None,
        shock_timing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode shock token.

        Args:
            shock_idx: Shock index, shape (batch,) with values in [0, max_shocks-1]
            shock_size: Shock magnitude in std devs, shape (batch,), default 1.0
            shock_timing: Shock timing in periods, shape (batch,), default 0

        Returns:
            Shock embeddings, shape (batch, embed_dim)
        """
        batch_size = shock_idx.shape[0]
        device = shock_idx.device

        # Default values
        if shock_size is None:
            shock_size = torch.ones(batch_size, device=device)
        if shock_timing is None:
            shock_timing = torch.zeros(batch_size, device=device)

        # Embed shock index
        idx_embed = self.shock_idx_embed(shock_idx)  # (batch, embed_dim // 2)

        # Encode continuous features
        continuous = torch.stack([shock_size, shock_timing], dim=-1)  # (batch, 2)
        cont_embed = self.continuous_mlp(continuous)  # (batch, embed_dim // 2)

        # Concatenate
        shock_embed = torch.cat([idx_embed, cont_embed], dim=-1)  # (batch, embed_dim)

        # Project
        output = self.output_proj(shock_embed)  # (batch, embed_dim)

        return output


class HistoryEncoder(nn.Module):
    """Encodes observable history y[0:k] for regimes B1 and C.

    Uses a GRU or Transformer to encode variable-length trajectory history.
    For Regime B1: Only observables available (no theta, no eps)
    For Regime C: Observables + theta available (no eps)
    """

    def __init__(
        self,
        n_obs: int = 3,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        architecture: Literal["gru", "transformer"] = "gru",
    ):
        """Initialize history encoder.

        Args:
            n_obs: Number of observables (default 3: output, inflation, rate)
            embed_dim: Output embedding dimension
            hidden_dim: Hidden dimension for GRU/Transformer
            n_layers: Number of layers
            dropout: Dropout rate
            architecture: "gru" or "transformer"
        """
        super().__init__()
        self.n_obs = n_obs
        self.embed_dim = embed_dim
        self.architecture = architecture

        # Input projection
        self.input_proj = nn.Linear(n_obs, hidden_dim)

        if architecture == "gru":
            self.encoder = nn.GRU(
                hidden_dim,
                hidden_dim,
                n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
        elif architecture == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, history: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode observable history.

        Args:
            history: Observable trajectory, shape (batch, k, n_obs)
                     where k is the history length
            mask: Optional padding mask, shape (batch, k)

        Returns:
            History embeddings, shape (batch, embed_dim)
        """
        # Input projection
        x = self.input_proj(history)  # (batch, k, hidden_dim)

        if self.architecture == "gru":
            # GRU encoding
            _, h_n = self.encoder(x)  # h_n: (n_layers, batch, hidden_dim)
            # Take final layer's hidden state
            encoded = h_n[-1]  # (batch, hidden_dim)

        else:  # transformer
            # Transformer encoding
            if mask is not None:
                # Transformer expects mask as (batch, seq_len) with True for padding
                src_key_padding_mask = ~mask
            else:
                src_key_padding_mask = None

            encoded_seq = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # (batch, k, hidden_dim)
            # Mean pooling over sequence
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                encoded = (encoded_seq * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
            else:
                encoded = encoded_seq.mean(dim=1)  # (batch, hidden_dim)

        # Output projection
        output = self.output_proj(encoded)  # (batch, embed_dim)

        return output


class TrunkNetwork(nn.Module):
    """Trunk network that combines all embeddings and produces latent representation.

    Takes concatenated embeddings (world, theta, shock, optional history) and
    produces a rich latent representation for the prediction heads.

    Architecture: Transformer blocks or MLP layers with residual connections.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
        architecture: Literal["mlp", "transformer"] = "mlp",
    ):
        """Initialize trunk network.

        Args:
            input_dim: Input dimension (sum of all embedding dims)
            hidden_dim: Hidden dimension
            n_layers: Number of layers
            dropout: Dropout rate
            architecture: "mlp" or "transformer"
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.architecture = architecture

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if architecture == "mlp":
            # MLP with residual connections
            layers = []
            for _ in range(n_layers):
                layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ))
            self.layers = nn.ModuleList(layers)

        elif architecture == "transformer":
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process combined embeddings.

        Args:
            x: Combined embeddings, shape (batch, input_dim)

        Returns:
            Latent representation, shape (batch, hidden_dim)
        """
        # Input projection
        h = self.input_proj(x)  # (batch, hidden_dim)

        if self.architecture == "mlp":
            # MLP with residual connections
            for layer in self.layers:
                h = h + layer(h)  # Residual connection

        else:  # transformer
            # Transformer expects (batch, seq_len, hidden_dim)
            # We have single token, so add seq dimension
            h = h.unsqueeze(1)  # (batch, 1, hidden_dim)
            h = self.encoder(h)  # (batch, 1, hidden_dim)
            h = h.squeeze(1)  # (batch, hidden_dim)

        return h


class IRFHead(nn.Module):
    """Prediction head for impulse response functions.

    Takes trunk output and predicts IRF: (H+1, n_obs) per sample.
    Uses MLP to generate all horizons jointly.
    """

    def __init__(
        self,
        input_dim: int,
        H: int = 40,
        n_obs: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        """Initialize IRF head.

        Args:
            input_dim: Input dimension from trunk
            H: Horizon length (predicts H+1 points: h=0..H)
            n_obs: Number of observables (default 3)
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.H = H
        self.n_obs = n_obs

        # MLP to generate IRF
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, (H + 1) * n_obs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict IRF.

        Args:
            x: Trunk output, shape (batch, input_dim)

        Returns:
            IRF predictions, shape (batch, H+1, n_obs)
        """
        batch_size = x.shape[0]

        # Generate flattened IRF
        irf_flat = self.mlp(x)  # (batch, (H+1)*n_obs)

        # Reshape to IRF format
        irf = irf_flat.view(batch_size, self.H + 1, self.n_obs)  # (batch, H+1, n_obs)

        return irf


class TrajectoryHead(nn.Module):
    """Prediction head for trajectory forecasting (optional).

    For trajectory prediction tasks (not primary for Phase 1).
    Predicts (T, n_obs) trajectory using autoregressive or direct prediction.
    """

    def __init__(
        self,
        input_dim: int,
        T: int = 40,
        n_obs: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        """Initialize trajectory head.

        Args:
            input_dim: Input dimension from trunk
            T: Trajectory length
            n_obs: Number of observables
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.T = T
        self.n_obs = n_obs

        # Direct prediction (non-autoregressive for now)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, T * n_obs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict trajectory.

        Args:
            x: Trunk output, shape (batch, input_dim)

        Returns:
            Trajectory predictions, shape (batch, T, n_obs)
        """
        batch_size = x.shape[0]

        # Generate flattened trajectory
        traj_flat = self.mlp(x)  # (batch, T*n_obs)

        # Reshape to trajectory format
        traj = traj_flat.view(batch_size, self.T, self.n_obs)  # (batch, T, n_obs)

        return traj


class UniversalEmulator(nn.Module):
    """Universal Macro Emulator for IRF prediction across all simulator worlds.

    This is the main model that predicts impulse response functions across
    6 simulator families: LSS, VAR, NK, RBC, regime-switching, ZLB.

    Supports information regimes:
    - Regime A: world_id, theta, eps_sequence, shock_token
    - Regime B1: world_id, shock_token, history (no theta, no eps)
    - Regime C: world_id, theta, shock_token, history (no eps)

    Architecture:
        1. World embedding: Learnable embedding per world_id
        2. Parameter encoder: Encodes variable-length theta vectors
        3. Shock encoder: Encodes (shock_idx, shock_size, shock_timing)
        4. History encoder (optional): Encodes y[0:k] for regime B1/C
        5. Trunk network: Combines all embeddings
        6. IRF head: Predicts (H+1, 3) IRF
        7. Trajectory head (optional): Predicts (T, 3) trajectory
    """

    def __init__(
        self,
        world_ids: list[str],
        param_dims: dict[str, int],
        world_embed_dim: int = 32,
        theta_embed_dim: int = 64,
        shock_embed_dim: int = 16,
        history_embed_dim: int = 64,
        trunk_dim: int = 256,
        trunk_layers: int = 4,
        trunk_architecture: Literal["mlp", "transformer"] = "mlp",
        history_architecture: Literal["gru", "transformer"] = "gru",
        H: int = 40,
        n_obs: int = 3,
        max_shocks: int = 3,
        use_history_encoder: bool = True,
        use_trajectory_head: bool = False,
        dropout: float = 0.1,
    ):
        """Initialize universal emulator.

        Args:
            world_ids: List of world identifiers (e.g., ["lss", "var", "nk", ...])
            param_dims: Dict mapping world_id to number of parameters
            world_embed_dim: World embedding dimension
            theta_embed_dim: Parameter embedding dimension
            shock_embed_dim: Shock embedding dimension
            history_embed_dim: History embedding dimension
            trunk_dim: Trunk network hidden dimension
            trunk_layers: Number of trunk layers
            trunk_architecture: "mlp" or "transformer"
            history_architecture: "gru" or "transformer"
            H: IRF horizon length (returns H+1 points)
            n_obs: Number of observables (default 3)
            max_shocks: Maximum number of shocks across worlds
            use_history_encoder: Whether to include history encoder (for regimes B1/C)
            use_trajectory_head: Whether to include trajectory prediction head
            dropout: Dropout rate
        """
        super().__init__()

        # Store config
        self.world_ids = world_ids
        self.param_dims = param_dims
        self.max_params = max(param_dims.values())
        self.H = H
        self.n_obs = n_obs
        self.max_shocks = max_shocks
        self.use_history_encoder = use_history_encoder
        self.use_trajectory_head = use_trajectory_head

        # Component dimensions
        self.world_embed_dim = world_embed_dim
        self.theta_embed_dim = theta_embed_dim
        self.shock_embed_dim = shock_embed_dim
        self.history_embed_dim = history_embed_dim
        self.trunk_dim = trunk_dim

        # 1. World embedding (if enabled)
        if world_embed_dim > 0:
            self.world_embedding = WorldEmbedding(world_ids, world_embed_dim)
        else:
            self.world_embedding = None

        # 2. Parameter encoder (if enabled)
        if theta_embed_dim > 0:
            self.param_encoder = ParameterEncoder(
                max_params=self.max_params,
                embed_dim=theta_embed_dim,
                dropout=dropout,
            )
        else:
            self.param_encoder = None

        # 3. Shock encoder (if enabled)
        if shock_embed_dim > 0:
            self.shock_encoder = ShockEncoder(
                max_shocks=max_shocks,
                embed_dim=shock_embed_dim,
            )
        else:
            self.shock_encoder = None

        # 4. History encoder (optional)
        if use_history_encoder and history_embed_dim > 0:
            self.history_encoder = HistoryEncoder(
                n_obs=n_obs,
                embed_dim=history_embed_dim,
                architecture=history_architecture,
                dropout=dropout,
            )
        else:
            self.history_encoder = None

        # 5. Trunk network
        # Input dimension depends on which embeddings are concatenated
        trunk_input_dim = world_embed_dim + shock_embed_dim
        trunk_input_dim += theta_embed_dim  # theta may be masked out in regime B1
        if use_history_encoder and history_embed_dim > 0:
            trunk_input_dim += history_embed_dim  # history may be masked out in regime A

        # Ensure trunk has at least some input
        if trunk_input_dim == 0:
            raise ValueError("At least one embedding dimension must be > 0")

        self.trunk = TrunkNetwork(
            input_dim=trunk_input_dim,
            hidden_dim=trunk_dim,
            n_layers=trunk_layers,
            dropout=dropout,
            architecture=trunk_architecture,
        )

        # 6. IRF head
        self.irf_head = IRFHead(
            input_dim=trunk_dim,
            H=H,
            n_obs=n_obs,
            dropout=dropout,
        )

        # 7. Trajectory head (optional)
        if use_trajectory_head:
            self.trajectory_head = TrajectoryHead(
                input_dim=trunk_dim,
                T=H,  # Use same horizon as IRF for now
                n_obs=n_obs,
                dropout=dropout,
            )

    def forward(
        self,
        world_id: str | list[str],
        shock_idx: torch.Tensor,
        theta: torch.Tensor | None = None,
        theta_mask: torch.Tensor | None = None,
        shock_size: torch.Tensor | None = None,
        shock_timing: torch.Tensor | None = None,
        history: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        regime: Literal["A", "B1", "C"] = "A",
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            world_id: World identifier(s) - string or list of strings
            shock_idx: Shock index, shape (batch,)
            theta: Parameters, shape (batch, n_params) - required for regimes A, C
            theta_mask: Boolean mask for valid parameters, shape (batch, n_params)
            shock_size: Shock magnitude, shape (batch,), default 1.0
            shock_timing: Shock timing, shape (batch,), default 0
            history: Observable history, shape (batch, k, n_obs) - required for regimes B1, C
            history_mask: Boolean mask for valid history steps, shape (batch, k)
            regime: Information regime ("A", "B1", or "C")
            return_trajectory: Whether to also return trajectory prediction (if head exists)

        Returns:
            IRF predictions, shape (batch, H+1, n_obs)
            If return_trajectory=True: (irf, trajectory) tuple

        Raises:
            ValueError: If required inputs for regime are missing
        """
        # Validate inputs based on regime
        if regime == "A":
            # Regime A: world_id, theta, shock_token (history optional)
            if theta is None:
                raise ValueError("Regime A requires theta")

        elif regime == "B1":
            # Regime B1: world_id, shock_token, history (no theta)
            if history is None:
                raise ValueError("Regime B1 requires history")
            if not self.use_history_encoder:
                raise ValueError("Regime B1 requires use_history_encoder=True")

        elif regime == "C":
            # Regime C: world_id, theta, shock_token, history (no eps)
            if theta is None:
                raise ValueError("Regime C requires theta")
            if history is None:
                raise ValueError("Regime C requires history")
            if not self.use_history_encoder:
                raise ValueError("Regime C requires use_history_encoder=True")

        else:
            raise ValueError(f"Unknown regime: {regime}")

        # Get batch size and device
        batch_size = shock_idx.shape[0]
        device = shock_idx.device

        # Collect embeddings (only include non-zero-dim components)
        embeddings = []

        # 1. Embed world_id (if enabled)
        if self.world_embedding is not None:
            world_embed = self.world_embedding(world_id)  # (batch, world_embed_dim)
            if world_embed.shape[0] == 1 and batch_size > 1:
                # Single world_id for whole batch
                world_embed = world_embed.expand(batch_size, -1)
            embeddings.append(world_embed)

        # 2. Encode parameters (if enabled and available)
        if self.param_encoder is not None:
            if theta is not None and regime != "B1":
                theta_embed = self.param_encoder(theta, theta_mask)  # (batch, theta_embed_dim)
            else:
                # Regime B1: no theta available, use zero embedding
                theta_embed = torch.zeros(batch_size, self.theta_embed_dim, device=device)
            embeddings.append(theta_embed)

        # 3. Encode shock token (if enabled)
        if self.shock_encoder is not None:
            shock_embed = self.shock_encoder(shock_idx, shock_size, shock_timing)  # (batch, shock_embed_dim)
            embeddings.append(shock_embed)

        # 4. Encode history (if enabled and available)
        if self.history_encoder is not None:
            if history is not None:
                history_embed = self.history_encoder(history, history_mask)  # (batch, history_embed_dim)
            else:
                # No history available, use zero embedding
                history_embed = torch.zeros(batch_size, self.history_embed_dim, device=device)
            embeddings.append(history_embed)

        # 5. Concatenate all embeddings
        if len(embeddings) == 0:
            raise ValueError("No embeddings available - at least one component must be enabled")

        combined = torch.cat(embeddings, dim=-1)  # (batch, trunk_input_dim)

        # 6. Trunk network
        trunk_output = self.trunk(combined)  # (batch, trunk_dim)

        # 7. IRF head
        irf = self.irf_head(trunk_output)  # (batch, H+1, n_obs)

        # 8. Trajectory head (optional)
        if return_trajectory:
            if not self.use_trajectory_head:
                raise ValueError("return_trajectory=True but use_trajectory_head=False")
            trajectory = self.trajectory_head(trunk_output)  # (batch, T, n_obs)
            return irf, trajectory

        return irf

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
