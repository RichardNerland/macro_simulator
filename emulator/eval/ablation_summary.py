"""
Ablation study summary generator.

This module provides tools for comparing ablation study results and generating
summary tables showing the impact of removing each input component.

Usage:
    # From Python
    from emulator.eval.ablation_summary import generate_ablation_summary

    summary = generate_ablation_summary(
        full_results_path="results/universal_A.json",
        ablation_results_paths={
            "No World ID": "results/ablation_no_world.json",
            "No Theta": "results/ablation_no_theta.json",
            "No Eps": "results/ablation_no_eps.json",
        }
    )
    summary.save_csv("ablation_summary.csv")
    summary.save_markdown("ablation_summary.md")

    # From CLI
    python -m emulator.eval.ablation_summary \
        --full results/universal_A.json \
        --ablations "No World ID:results/ablation_no_world.json,No Theta:results/ablation_no_theta.json" \
        --output ablation_summary.md

Interpretation:
    - Positive Δ NRMSE % means removing component made performance worse (component helps)
    - Negative Δ NRMSE % means removing component improved performance (component may hurt)
    - Larger absolute Δ % indicates more important component
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


class AblationSummary:
    """Generate ablation study comparison tables.

    Compares a full model with various ablations to measure component importance.
    """

    def __init__(self):
        """Initialize empty ablation summary."""
        self.full_metrics: dict[str, Any] | None = None
        self.ablation_metrics: dict[str, dict[str, Any]] = {}

    def add_full_model(
        self,
        metrics_json_path: Path | str,
    ) -> None:
        """Add full model results.

        Args:
            metrics_json_path: Path to full model evaluation results JSON
        """
        metrics_json_path = Path(metrics_json_path)

        if not metrics_json_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_json_path}")

        with open(metrics_json_path) as f:
            results = json.load(f)

        # Extract aggregate metrics
        aggregate = results.get("aggregate", {})

        self.full_metrics = {
            "nrmse": aggregate.get("nrmse", float("nan")),
            "iae": aggregate.get("iae", float("nan")),
            "sign_at_impact": aggregate.get("sign_at_impact", float("nan")),
            "hf_ratio": aggregate.get("hf_ratio", float("nan")),
            "overshoot_ratio": aggregate.get("overshoot_ratio", float("nan")),
            "sign_flip_count": aggregate.get("sign_flip_count", float("nan")),
            "n_samples": aggregate.get("n_samples", 0),
        }

    def add_ablation(
        self,
        name: str,
        metrics_json_path: Path | str,
    ) -> None:
        """Add ablation model results.

        Args:
            name: Ablation name (e.g., "No World ID", "No Theta")
            metrics_json_path: Path to ablation evaluation results JSON
        """
        metrics_json_path = Path(metrics_json_path)

        if not metrics_json_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_json_path}")

        with open(metrics_json_path) as f:
            results = json.load(f)

        # Extract aggregate metrics
        aggregate = results.get("aggregate", {})

        self.ablation_metrics[name] = {
            "nrmse": aggregate.get("nrmse", float("nan")),
            "iae": aggregate.get("iae", float("nan")),
            "sign_at_impact": aggregate.get("sign_at_impact", float("nan")),
            "hf_ratio": aggregate.get("hf_ratio", float("nan")),
            "overshoot_ratio": aggregate.get("overshoot_ratio", float("nan")),
            "sign_flip_count": aggregate.get("sign_flip_count", float("nan")),
            "n_samples": aggregate.get("n_samples", 0),
        }

    def compute_delta_percent(
        self,
        ablation_value: float,
        full_value: float,
    ) -> float:
        """Compute delta percentage: (ablation - full) / full * 100%.

        Positive means ablation is worse (removing component hurt performance).
        Negative means ablation is better (component may have hurt).

        Args:
            ablation_value: Metric value for ablation model
            full_value: Metric value for full model

        Returns:
            Delta percentage
        """
        if full_value == 0.0 or pd.isna(full_value):
            return float("nan")

        return (ablation_value - full_value) / full_value * 100.0

    def generate_table(
        self,
        sort_by: str = "nrmse",
        ascending: bool = True,
    ) -> pd.DataFrame:
        """Generate ablation comparison table.

        Args:
            sort_by: Metric to sort by (default: "nrmse")
            ascending: Sort order (default: True for lower-is-better metrics)

        Returns:
            DataFrame with columns:
                - Model: Model name
                - NRMSE: Normalized RMSE
                - Δ NRMSE %: Delta vs full model
                - IAE: Integrated absolute error
                - Δ IAE %: Delta vs full model
                - Sign@Impact: Sign-at-impact accuracy
                - HF-ratio: High-frequency ratio
                - Samples: Number of samples
        """
        if self.full_metrics is None:
            raise ValueError("Full model metrics not set. Call add_full_model() first.")

        # Build rows
        rows = []

        # Full model row
        full_row = {
            "Model": "Full Model",
            "NRMSE": self.full_metrics.get("nrmse", float("nan")),
            "Δ NRMSE %": 0.0,  # Baseline
            "IAE": self.full_metrics.get("iae", float("nan")),
            "Δ IAE %": 0.0,  # Baseline
            "Sign@Impact": self.full_metrics.get("sign_at_impact", float("nan")),
            "HF-ratio": self.full_metrics.get("hf_ratio", float("nan")),
            "Samples": self.full_metrics.get("n_samples", 0),
        }
        rows.append(full_row)

        # Ablation rows
        for ablation_name, ablation_metrics in self.ablation_metrics.items():
            row = {
                "Model": ablation_name,
                "NRMSE": ablation_metrics.get("nrmse", float("nan")),
                "Δ NRMSE %": self.compute_delta_percent(
                    ablation_metrics.get("nrmse", float("nan")),
                    self.full_metrics.get("nrmse", float("nan")),
                ),
                "IAE": ablation_metrics.get("iae", float("nan")),
                "Δ IAE %": self.compute_delta_percent(
                    ablation_metrics.get("iae", float("nan")),
                    self.full_metrics.get("iae", float("nan")),
                ),
                "Sign@Impact": ablation_metrics.get("sign_at_impact", float("nan")),
                "HF-ratio": ablation_metrics.get("hf_ratio", float("nan")),
                "Samples": ablation_metrics.get("n_samples", 0),
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort (skip full model row at index 0)
        # Handle case-insensitive column matching
        sort_column = None
        if len(df) > 1:
            # Find matching column (case-insensitive)
            for col in df.columns:
                if col.lower() == sort_by.lower():
                    sort_column = col
                    break

        if sort_column is not None:
            # Sort ablations only
            ablations_df = df.iloc[1:].copy()
            ablations_df = ablations_df.sort_values(by=sort_column, ascending=ascending)
            # Recombine with full model at top
            df = pd.concat([df.iloc[[0]], ablations_df], ignore_index=True)

        return df

    def save_csv(self, output_path: Path | str, **kwargs) -> None:
        """Save ablation summary to CSV file.

        Args:
            output_path: Path to output CSV file
            **kwargs: Additional arguments for generate_table()
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.generate_table(**kwargs)
        df.to_csv(output_path, index=False, float_format="%.4f")

        print(f"Ablation summary saved to: {output_path}")

    def save_markdown(self, output_path: Path | str, **kwargs) -> None:
        """Save ablation summary to Markdown file with explanatory header.

        Args:
            output_path: Path to output Markdown file
            **kwargs: Additional arguments for generate_table()
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.generate_table(**kwargs)

        # Format floats nicely
        df_formatted = df.copy()
        for col in df_formatted.columns:
            if col == "Model" or col == "Samples":
                continue
            elif col in ["Δ NRMSE %", "Δ IAE %"]:
                # Format delta columns with + sign for positive
                df_formatted[col] = df_formatted[col].apply(
                    lambda x: f"+{x:.1f}%" if pd.notna(x) and x > 0
                    else (f"{x:.1f}%" if pd.notna(x) and x != 0 else "baseline")
                )
            elif df_formatted[col].dtype in [float, "float64"]:
                df_formatted[col] = df_formatted[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "—"
                )

        markdown = df_formatted.to_markdown(index=False)

        with open(output_path, "w") as f:
            f.write("# Ablation Study Summary\n\n")
            f.write("## Interpretation\n\n")
            f.write("This table shows the impact of removing each input component from the full model.\n\n")
            f.write("- **Positive Δ %**: Removing the component made performance worse → component helps\n")
            f.write("- **Negative Δ %**: Removing the component improved performance → component may hurt\n")
            f.write("- **Larger absolute Δ %**: More important component\n\n")
            f.write("## Results\n\n")
            f.write(markdown)
            f.write("\n\n")
            f.write("## Metrics\n\n")
            f.write("- **NRMSE**: Normalized Root Mean Square Error (lower is better)\n")
            f.write("- **IAE**: Integrated Absolute Error (lower is better)\n")
            f.write("- **Sign@Impact**: Fraction of correct signs in first 3 horizons (higher is better)\n")
            f.write("- **HF-ratio**: High-frequency energy ratio for oscillation detection (lower is better)\n")
            f.write("\n")

        print(f"Ablation summary (Markdown) saved to: {output_path}")

    def print_summary(self) -> None:
        """Print ablation summary to console."""
        df = self.generate_table()

        print("\n" + "="*100)
        print("ABLATION STUDY SUMMARY")
        print("="*100)
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("="*100)
        print("\nInterpretation:")
        print("  - Positive Δ % = Removing component hurt performance (component helps)")
        print("  - Negative Δ % = Removing component improved performance (component may hurt)")
        print("  - Larger absolute Δ % = More important component")
        print("="*100 + "\n")


def load_ablation_results(
    results_dir: Path | str,
    pattern: str = "ablation_*.json",
) -> dict[str, Path]:
    """Load ablation results from a directory.

    Args:
        results_dir: Directory containing ablation result JSON files
        pattern: Glob pattern for finding ablation files (default: "ablation_*.json")

    Returns:
        Dictionary mapping ablation name to file path
    """
    results_dir = Path(results_dir)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Find all matching JSON files
    json_files = list(results_dir.glob(pattern))

    if not json_files:
        print(f"Warning: No ablation files found matching '{pattern}' in {results_dir}")
        return {}

    # Build dict with cleaned names
    ablations = {}
    for json_file in json_files:
        # Extract ablation name from filename
        # e.g., "ablation_no_world.json" -> "No World"
        name = json_file.stem.replace("ablation_", "").replace("_", " ").title()
        ablations[name] = json_file

    return ablations


def generate_ablation_summary(
    full_results_path: Path | str,
    ablation_results_paths: dict[str, Path | str],
) -> AblationSummary:
    """Generate ablation summary from full model and ablations.

    Args:
        full_results_path: Path to full model evaluation results
        ablation_results_paths: Dict mapping ablation name to results path

    Returns:
        AblationSummary object with loaded results
    """
    summary = AblationSummary()

    # Load full model
    summary.add_full_model(full_results_path)

    # Load ablations
    for name, path in ablation_results_paths.items():
        summary.add_ablation(name, path)

    return summary


def main():
    """CLI entry point for ablation summary generation."""
    parser = argparse.ArgumentParser(
        description="Generate ablation study summary table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # From individual files
    python -m emulator.eval.ablation_summary \\
        --full results/universal_A.json \\
        --ablations "No World:results/ablation_no_world.json,No Theta:results/ablation_no_theta.json" \\
        --output ablation_summary.md

    # Auto-discover ablations in directory
    python -m emulator.eval.ablation_summary \\
        --full results/universal_A.json \\
        --ablations-dir results/ \\
        --output ablation_summary.md
        """
    )

    parser.add_argument(
        "--full",
        type=Path,
        required=True,
        help="Path to full model evaluation results JSON",
    )

    # Mutually exclusive: either specify individual ablations or a directory
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ablations",
        type=str,
        help="Comma-separated list of name:path pairs (e.g., 'No World:results/ablation_no_world.json')",
    )
    group.add_argument(
        "--ablations-dir",
        type=Path,
        help="Directory containing ablation result files (will auto-discover ablation_*.json)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ablation_summary.md"),
        help="Path to output file (.md or .csv, default: ablation_summary.md)",
    )

    parser.add_argument(
        "--sort-by",
        type=str,
        default="nrmse",
        help="Metric to sort ablations by (default: nrmse)",
    )

    args = parser.parse_args()

    # Create summary
    summary = AblationSummary()

    # Load full model
    print(f"Loading full model results: {args.full}")
    summary.add_full_model(args.full)

    # Load ablations
    if args.ablations:
        # Parse individual ablations
        ablation_pairs = args.ablations.split(",")
        for pair in ablation_pairs:
            if ":" not in pair:
                print(f"Warning: Invalid ablation pair '{pair}', expected 'name:path'")
                continue

            name, path = pair.split(":", 1)
            name = name.strip()
            path = Path(path.strip())

            print(f"Loading ablation '{name}': {path}")
            try:
                summary.add_ablation(name, path)
            except Exception as e:
                print(f"Warning: Could not load ablation '{name}': {e}")
                continue

    else:
        # Auto-discover from directory
        print(f"Auto-discovering ablations in: {args.ablations_dir}")
        ablations = load_ablation_results(args.ablations_dir)

        for name, path in ablations.items():
            print(f"Loading ablation '{name}': {path}")
            try:
                summary.add_ablation(name, path)
            except Exception as e:
                print(f"Warning: Could not load ablation '{name}': {e}")
                continue

    # Print to console
    summary.print_summary()

    # Save to file
    if args.output.suffix == ".csv":
        summary.save_csv(args.output, sort_by=args.sort_by)
    else:
        summary.save_markdown(args.output, sort_by=args.sort_by)


if __name__ == "__main__":
    main()
