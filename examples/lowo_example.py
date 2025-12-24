#!/usr/bin/env python
"""Example of using LOWO (Leave-One-World-Out) splits for cross-world generalization.

This script demonstrates how to:
1. Create LOWO splits for training and testing
2. Load datasets using LOWO splits
3. Train/evaluate with held-out worlds

Usage:
    python examples/lowo_example.py
"""


from data import get_lowo_world_lists, split_lowo

# Example 1: Basic LOWO split
print("=" * 80)
print("Example 1: Basic LOWO Split")
print("=" * 80)

all_worlds = ["lss", "var", "nk", "rbc", "switching", "zlb"]
held_out_world = "nk"

# Create LOWO split
lowo_split = split_lowo(all_worlds, held_out_world)

print(f"\nHeld-out world: {held_out_world}")
print(f"Training worlds ({len(lowo_split['train_worlds'])}): {lowo_split['train_worlds']}")
print(f"Test worlds ({len(lowo_split['test_worlds'])}): {lowo_split['test_worlds']}")

# Example 2: Using convenience function
print("\n" + "=" * 80)
print("Example 2: Convenience Function")
print("=" * 80)

train_worlds, test_worlds = get_lowo_world_lists(all_worlds, "rbc")

print("\nHeld-out world: rbc")
print(f"Training worlds: {train_worlds}")
print(f"Test worlds: {test_worlds}")

# Example 3: LOWO for all worlds (6-fold cross-validation)
print("\n" + "=" * 80)
print("Example 3: LOWO for All Worlds (6-fold)")
print("=" * 80)

for held_out in all_worlds:
    train, test = get_lowo_world_lists(all_worlds, held_out)
    print(f"\nFold {held_out}:")
    print(f"  Train on: {', '.join(train)}")
    print(f"  Test on: {test[0]}")

# Example 4: Using LOWO with Dataset Loader (pseudocode)
print("\n" + "=" * 80)
print("Example 4: LOWO with Dataset Loader (Pseudocode)")
print("=" * 80)

print("""
# In your training script:
from data import get_lowo_world_lists
from emulator.training.dataset import IRFDataset

# Set up LOWO split
all_worlds = ["lss", "var", "nk", "rbc", "switching", "zlb"]
held_out_world = "nk"
train_worlds, test_worlds = get_lowo_world_lists(all_worlds, held_out_world)

# Create datasets
dataset_root = Path("datasets/v1.0")
train_dataset = IRFDataset(
    zarr_root=dataset_root,
    world_ids=train_worlds,  # Only train on 5 worlds
    split="train"
)

test_dataset = IRFDataset(
    zarr_root=dataset_root,
    world_ids=test_worlds,   # Test on held-out world
    split="test_interpolation"
)

# Train model on train_worlds, evaluate on test_worlds
# This tests generalization to unseen simulator families
""")

# Example 5: Full LOWO experiment workflow
print("\n" + "=" * 80)
print("Example 5: Full LOWO Experiment Workflow")
print("=" * 80)

print("""
# Run 6 experiments, each holding out a different world
all_worlds = ["lss", "var", "nk", "rbc", "switching", "zlb"]

results = {}
for held_out_world in all_worlds:
    print(f"\\n=== LOWO Experiment: Held-out world = {held_out_world} ===")

    # Get train/test split
    train_worlds, test_worlds = get_lowo_world_lists(all_worlds, held_out_world)

    # Load datasets
    train_dataset = IRFDataset(zarr_root, train_worlds, split="train")
    test_dataset = IRFDataset(zarr_root, test_worlds, split="test_interpolation")

    # Train model
    model = train_universal_emulator(train_dataset)

    # Evaluate on held-out world
    metrics = evaluate_model(model, test_dataset)
    results[held_out_world] = metrics

    print(f"MSE on {held_out_world}: {metrics['mse']:.4f}")

# Analyze results
avg_mse = sum(r['mse'] for r in results.values()) / len(results)
print(f"\\nAverage LOWO MSE: {avg_mse:.4f}")
print("Per-world LOWO results:", results)
""")

print("\n" + "=" * 80)
print("LOWO Split Implementation Complete!")
print("=" * 80)
print("\nKey Features:")
print("- Deterministic world splits (no randomness)")
print("- Disjoint train/test worlds")
print("- Tests cross-world generalization")
print("- Integrates with existing IRFDataset loader")
print("- Supports all 6 simulator families")
