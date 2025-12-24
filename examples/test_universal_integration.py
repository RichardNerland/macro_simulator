"""
Integration example for UniversalEmulator with the training infrastructure.

This script demonstrates:
1. Creating a UniversalEmulator with realistic config
2. Loading data from the dataset
3. Running a forward pass
4. Computing loss
5. Training for a few steps

This is NOT a formal test, just a demonstration of end-to-end usage.
"""

import torch
from pathlib import Path

from emulator.models import UniversalEmulator
from emulator.training.dataset import IRFDataset, collate_mixed_worlds
from torch.utils.data import DataLoader


def main():
    """Run integration demo."""
    print("=" * 80)
    print("Universal Emulator Integration Demo")
    print("=" * 80)

    # 1. Check if dataset exists
    dataset_path = Path("datasets/v1.0-dev")
    if not dataset_path.exists():
        print(f"\nDataset not found at {dataset_path}")
        print("Skipping integration demo - run dataset generation first:")
        print("  python -m data.scripts.generate_dataset --world all --n_samples 1000 --seed 42 --output datasets/v1.0-dev")
        return

    # 2. Setup world configuration
    world_ids = ["lss", "var", "nk", "rbc", "switching", "zlb"]
    param_dims = {
        "lss": 15,
        "var": 12,
        "nk": 10,
        "rbc": 8,
        "switching": 25,
        "zlb": 12,
    }

    print("\n1. Creating UniversalEmulator...")
    model = UniversalEmulator(
        world_ids=world_ids,
        param_dims=param_dims,
        world_embed_dim=32,
        theta_embed_dim=64,
        shock_embed_dim=16,
        history_embed_dim=64,
        trunk_dim=256,
        trunk_layers=4,
        H=40,
        n_obs=3,
        max_shocks=3,
        use_history_encoder=True,
        use_trajectory_head=False,
        dropout=0.1,
    )

    n_params = model.get_num_parameters()
    print(f"   Model created with {n_params:,} parameters")

    # 3. Load dataset
    print("\n2. Loading dataset...")
    try:
        dataset = IRFDataset(
            zarr_root=dataset_path,
            world_ids=["nk"],  # Start with single world for simplicity
            split="train",
        )
        print(f"   Loaded {len(dataset)} samples")

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            collate_fn=collate_mixed_worlds,
        )

        # 4. Get a batch
        print("\n3. Getting a batch...")
        batch = next(iter(dataloader))
        print(f"   Batch keys: {batch.keys()}")
        print(f"   theta shape: {batch['theta'].shape}")
        print(f"   irf shape: {batch['irf'].shape}")
        print(f"   world_ids: {set(batch['world_ids'])}")

        # 5. Forward pass (Regime A)
        print("\n4. Running forward pass (Regime A)...")
        model.eval()

        with torch.no_grad():
            # Extract first shock for single-shock prediction
            shock_idx = torch.zeros(batch['theta'].shape[0], dtype=torch.long)

            predictions = model(
                world_id=batch['world_ids'],
                theta=batch['theta'],
                theta_mask=batch['theta_mask'],
                shock_idx=shock_idx,
                regime="A",
            )

        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Expected shape: (batch={batch['theta'].shape[0]}, H+1=41, n_obs=3)")

        # 6. Compute loss
        print("\n5. Computing loss...")
        targets = batch['irf'][:, 0, :, :]  # Take first shock
        loss = torch.nn.functional.mse_loss(predictions, targets)
        print(f"   MSE loss (untrained): {loss.item():.6f}")

        # 7. Test gradient flow
        print("\n6. Testing gradient flow...")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Single training step
        optimizer.zero_grad()
        predictions = model(
            world_id=batch['world_ids'],
            theta=batch['theta'],
            theta_mask=batch['theta_mask'],
            shock_idx=shock_idx,
            regime="A",
        )
        loss = torch.nn.functional.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()

        print(f"   Backward pass successful")
        print(f"   Loss after 1 step: {loss.item():.6f}")

        # 8. Test Regime B1 (history-based)
        print("\n7. Testing Regime B1 (history-based)...")
        # Create dummy history
        history = torch.randn(batch['theta'].shape[0], 20, 3)

        with torch.no_grad():
            predictions_b1 = model(
                world_id=batch['world_ids'],
                shock_idx=shock_idx,
                history=history,
                regime="B1",
            )

        print(f"   Regime B1 predictions shape: {predictions_b1.shape}")

        # 9. Test Regime C (theta + history)
        print("\n8. Testing Regime C (theta + history)...")
        with torch.no_grad():
            predictions_c = model(
                world_id=batch['world_ids'],
                theta=batch['theta'],
                theta_mask=batch['theta_mask'],
                shock_idx=shock_idx,
                history=history,
                regime="C",
            )

        print(f"   Regime C predictions shape: {predictions_c.shape}")

        print("\n" + "=" * 80)
        print("Integration demo completed successfully!")
        print("=" * 80)

    except FileNotFoundError as e:
        print(f"\n   Error: {e}")
        print("   Dataset files not found. Please generate dataset first.")
        return


if __name__ == "__main__":
    main()
