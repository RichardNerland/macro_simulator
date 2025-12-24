"""
Demonstration of ablation summary generation.

This script shows how to use the ablation_summary module to compare
a full model with various ablations.
"""

import json
import tempfile
from pathlib import Path

from emulator.eval.ablation_summary import generate_ablation_summary


def create_sample_results():
    """Create sample evaluation results for demonstration."""
    # Full model results
    full_results = {
        "aggregate": {
            "nrmse": 0.15,
            "iae": 2.3,
            "sign_at_impact": 0.92,
            "hf_ratio": 0.08,
            "overshoot_ratio": 1.15,
            "sign_flip_count": 2.1,
            "n_samples": 1000,
        },
        "per_world": {
            "lss": {"nrmse": 0.12, "iae": 2.0},
            "var": {"nrmse": 0.18, "iae": 2.6},
        },
    }

    # Ablation results showing impact of removing each component
    ablation_results = {
        "No World ID": {
            "aggregate": {
                "nrmse": 0.18,  # +20% worse
                "iae": 2.8,     # +21.7% worse
                "sign_at_impact": 0.88,
                "hf_ratio": 0.09,
                "overshoot_ratio": 1.18,
                "sign_flip_count": 2.3,
                "n_samples": 1000,
            }
        },
        "No Theta (Parameters)": {
            "aggregate": {
                "nrmse": 0.22,  # +46.7% worse (most important!)
                "iae": 3.1,     # +34.8% worse
                "sign_at_impact": 0.85,
                "hf_ratio": 0.10,
                "overshoot_ratio": 1.22,
                "sign_flip_count": 2.5,
                "n_samples": 1000,
            }
        },
        "No Eps (Shocks)": {
            "aggregate": {
                "nrmse": 0.16,  # +6.7% worse (least important)
                "iae": 2.4,     # +4.3% worse
                "sign_at_impact": 0.91,
                "hf_ratio": 0.08,
                "overshoot_ratio": 1.16,
                "sign_flip_count": 2.2,
                "n_samples": 1000,
            }
        },
        "No History": {
            "aggregate": {
                "nrmse": 0.17,  # +13.3% worse
                "iae": 2.6,     # +13.0% worse
                "sign_at_impact": 0.89,
                "hf_ratio": 0.09,
                "overshoot_ratio": 1.17,
                "sign_flip_count": 2.2,
                "n_samples": 1000,
            }
        },
    }

    return full_results, ablation_results


def main():
    """Run ablation summary demonstration."""
    print("=" * 80)
    print("ABLATION SUMMARY DEMONSTRATION")
    print("=" * 80)

    # Create sample results
    full_results, ablation_results = create_sample_results()

    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Write full model results
        full_path = temp_path / "full_model.json"
        with open(full_path, "w") as f:
            json.dump(full_results, f, indent=2)

        # Write ablation results
        ablation_paths = {}
        for name, results in ablation_results.items():
            # Create filename from ablation name
            filename = f"ablation_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.json"
            ablation_path = temp_path / filename

            with open(ablation_path, "w") as f:
                json.dump(results, f, indent=2)

            ablation_paths[name] = ablation_path

        # Generate summary
        print("\n1. Loading evaluation results...")
        summary = generate_ablation_summary(full_path, ablation_paths)

        # Display summary table
        print("\n2. Generating summary table...")
        summary.print_summary()

        # Save to files
        print("\n3. Saving results...")
        csv_path = temp_path / "ablation_summary.csv"
        md_path = temp_path / "ablation_summary.md"

        summary.save_csv(csv_path)
        summary.save_markdown(md_path)

        # Show markdown output
        print("\n4. Generated Markdown file:")
        print("-" * 80)
        with open(md_path) as f:
            print(f.read())

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)

    # Key takeaways
    print("\nKey Takeaways:")
    print("  1. Theta (parameters) is the most important component (+46.7% NRMSE)")
    print("  2. World ID is moderately important (+20.0% NRMSE)")
    print("  3. History is somewhat important (+13.3% NRMSE)")
    print("  4. Eps (shocks) is least important (+6.7% NRMSE)")
    print("\nThis suggests the model relies heavily on structural parameters,")
    print("followed by world identification and historical context.")


if __name__ == "__main__":
    main()
