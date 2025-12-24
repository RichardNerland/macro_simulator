"""
Leaderboard generator for comparing multiple models.

This module provides tools for aggregating and comparing evaluation results
across multiple models. It generates formatted tables suitable for papers,
blog posts, and dashboards.

Usage:
    # Single leaderboard from multiple results
    leaderboard = LeaderboardGenerator()
    leaderboard.add_model("Universal (Regime A)", "results/universal_A.json")
    leaderboard.add_model("Per-world MLP", "results/mlp_baseline.json")
    leaderboard.add_model("Oracle", "results/oracle.json")

    df = leaderboard.generate_table()
    leaderboard.save_csv("leaderboard.csv")
    leaderboard.save_markdown("leaderboard.md")

    # Batch comparison
    compare_models(
        results_dir="experiments/sweep/",
        output_csv="comparison.csv",
        metric="nrmse",
    )

Integration with UniversalEmulator:
    The leaderboard automatically works with evaluation results from:
    - evaluate.py (with UniversalEmulator checkpoint)
    - Baseline models (MLP, Linear, GRU)
    - Oracle/ground-truth results

    All results must be in the standard JSON format produced by evaluate.py:
    {
        "aggregate": {"nrmse": ..., "iae": ..., ...},
        "per_world": {"lss": {...}, "var": {...}, ...},
        "per_split": {...}
    }
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


class LeaderboardGenerator:
    """Generate comparison tables for multiple models.

    Aggregates evaluation results from multiple JSON files and produces
    sorted leaderboards by various metrics.
    """

    def __init__(self):
        """Initialize empty leaderboard."""
        self.models: dict[str, dict[str, Any]] = {}
        self.oracle_metrics: dict[str, float] | None = None

    def add_model(
        self,
        name: str,
        metrics_json_path: Path | str,
        is_oracle: bool = False,
    ) -> None:
        """Add a model's results to the leaderboard.

        Args:
            name: Display name for the model
            metrics_json_path: Path to evaluation results JSON
            is_oracle: Whether this is the oracle baseline (for gap computation)
        """
        metrics_json_path = Path(metrics_json_path)

        if not metrics_json_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_json_path}")

        with open(metrics_json_path) as f:
            results = json.load(f)

        # Extract aggregate metrics
        aggregate = results.get("aggregate", {})

        # Store in models dict
        self.models[name] = {
            "nrmse": aggregate.get("nrmse", float("nan")),
            "iae": aggregate.get("iae", float("nan")),
            "sign_at_impact": aggregate.get("sign_at_impact", float("nan")),
            "hf_ratio": aggregate.get("hf_ratio", float("nan")),
            "overshoot_ratio": aggregate.get("overshoot_ratio", float("nan")),
            "sign_flip_count": aggregate.get("sign_flip_count", float("nan")),
            "n_samples": aggregate.get("n_samples", 0),
            "per_world": results.get("per_world", {}),
            "per_split": results.get("per_split", {}),
        }

        # If oracle, store for gap computation
        if is_oracle:
            self.oracle_metrics = self.models[name]

    def add_model_from_dict(
        self,
        name: str,
        metrics: dict[str, Any],
        is_oracle: bool = False,
    ) -> None:
        """Add a model from a metrics dictionary.

        Args:
            name: Display name for the model
            metrics: Dictionary with aggregate metrics
            is_oracle: Whether this is the oracle baseline
        """
        self.models[name] = metrics

        if is_oracle:
            self.oracle_metrics = metrics

    def compute_gap(self, nrmse: float) -> float:
        """Compute gap vs oracle baseline.

        Gap = (NRMSE_model - NRMSE_oracle) / NRMSE_oracle * 100%

        Args:
            nrmse: Model's NRMSE

        Returns:
            Gap percentage (can be negative if model beats oracle)
        """
        if self.oracle_metrics is None:
            return float("nan")

        oracle_nrmse = self.oracle_metrics.get("nrmse", float("nan"))

        if oracle_nrmse == 0.0:
            return float("nan")

        return (nrmse - oracle_nrmse) / oracle_nrmse * 100.0

    def generate_table(
        self,
        sort_by: str = "nrmse",
        ascending: bool = True,
        include_gap: bool = True,
    ) -> pd.DataFrame:
        """Generate leaderboard table as pandas DataFrame.

        Args:
            sort_by: Metric to sort by (default: "nrmse")
            ascending: Sort order (default: True for metrics where lower is better)
            include_gap: Include gap vs oracle column

        Returns:
            DataFrame with columns:
                - Model: Model name
                - NRMSE: Normalized RMSE
                - IAE: Integrated absolute error
                - Sign@Impact: Sign-at-impact accuracy
                - HF-ratio: High-frequency ratio
                - Overshoot: Overshoot ratio
                - Sign Flips: Sign flip count
                - Gap (%): Gap vs oracle (if include_gap=True)
                - Samples: Number of samples
        """
        if not self.models:
            return pd.DataFrame()

        # Build rows
        rows = []
        for model_name, metrics in self.models.items():
            row = {
                "Model": model_name,
                "NRMSE": metrics.get("nrmse", float("nan")),
                "IAE": metrics.get("iae", float("nan")),
                "Sign@Impact": metrics.get("sign_at_impact", float("nan")),
                "HF-ratio": metrics.get("hf_ratio", float("nan")),
                "Overshoot": metrics.get("overshoot_ratio", float("nan")),
                "Sign Flips": metrics.get("sign_flip_count", float("nan")),
                "Samples": metrics.get("n_samples", 0),
            }

            # Add gap if requested and oracle is available
            if include_gap:
                row["Gap (%)"] = self.compute_gap(metrics.get("nrmse", float("nan")))

            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort
        if sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)

        return df

    def generate_per_world_table(
        self,
        metric: str = "nrmse",
    ) -> pd.DataFrame:
        """Generate per-world breakdown table.

        Args:
            metric: Which metric to show (default: "nrmse")

        Returns:
            DataFrame with rows = models, columns = worlds
        """
        if not self.models:
            return pd.DataFrame()

        # Collect all world IDs
        all_worlds = set()
        for metrics in self.models.values():
            per_world = metrics.get("per_world", {})
            all_worlds.update(per_world.keys())

        all_worlds = sorted(all_worlds)

        # Build rows
        rows = []
        for model_name, metrics in self.models.items():
            row = {"Model": model_name}

            per_world = metrics.get("per_world", {})
            for world_id in all_worlds:
                if world_id in per_world:
                    row[world_id] = per_world[world_id].get(metric, float("nan"))
                else:
                    row[world_id] = float("nan")

            rows.append(row)

        df = pd.DataFrame(rows)

        return df

    def save_csv(self, output_path: Path | str, **kwargs) -> None:
        """Save leaderboard to CSV file.

        Args:
            output_path: Path to output CSV file
            **kwargs: Additional arguments for generate_table()
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.generate_table(**kwargs)
        df.to_csv(output_path, index=False, float_format="%.4f")

        print(f"Leaderboard saved to: {output_path}")

    def save_markdown(self, output_path: Path | str, **kwargs) -> None:
        """Save leaderboard to Markdown file.

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
            if df_formatted[col].dtype in [float, "float64"]:
                df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")

        markdown = df_formatted.to_markdown(index=False)

        with open(output_path, "w") as f:
            f.write("# Model Leaderboard\n\n")
            f.write(markdown)
            f.write("\n")

        print(f"Leaderboard (Markdown) saved to: {output_path}")

    def print_summary(self) -> None:
        """Print leaderboard to console."""
        df = self.generate_table()

        print("\n" + "="*80)
        print("MODEL LEADERBOARD")
        print("="*80)
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("="*80 + "\n")


def compare_models(
    results_dir: Path | str,
    output_csv: Path | str | None = None,
    output_markdown: Path | str | None = None,
    metric: str = "nrmse",
    pattern: str = "*.json",
) -> pd.DataFrame:
    """Compare all models in a directory.

    Loads all JSON files matching pattern in results_dir and generates
    a comparison table.

    Args:
        results_dir: Directory containing evaluation result JSON files
        output_csv: Optional path to save CSV
        output_markdown: Optional path to save Markdown
        metric: Metric to sort by (default: "nrmse")
        pattern: Glob pattern for finding JSON files (default: "*.json")

    Returns:
        Comparison DataFrame
    """
    results_dir = Path(results_dir)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Find all JSON files
    json_files = list(results_dir.glob(pattern))

    if not json_files:
        print(f"Warning: No JSON files found in {results_dir}")
        return pd.DataFrame()

    print(f"Found {len(json_files)} result files")

    # Create leaderboard
    leaderboard = LeaderboardGenerator()

    for json_file in json_files:
        # Use filename (without extension) as model name
        model_name = json_file.stem

        # Check if this is oracle
        is_oracle = "oracle" in model_name.lower()

        try:
            leaderboard.add_model(model_name, json_file, is_oracle=is_oracle)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
            continue

    # Generate table
    df = leaderboard.generate_table(sort_by=metric)

    # Print to console
    leaderboard.print_summary()

    # Save if requested
    if output_csv:
        leaderboard.save_csv(output_csv)

    if output_markdown:
        leaderboard.save_markdown(output_markdown)

    return df


def compute_success_criteria(
    results_json_path: Path | str,
    specialist_results: dict[str, Path | str],
    baseline_results: dict[str, Path | str],
) -> dict[str, Any]:
    """Compute success criteria per spec §7.2.3.

    Checks:
    1. Beat baselines: NRMSE_universal < NRMSE_baseline for all worlds
    2. Mean gap: mean_w(Gap(w)) ≤ 20%
    3. Max gap: max_w(Gap(w)) ≤ 35%
    4. Shape preservation: HF_ratio_universal / HF_ratio_specialist ≤ 1.1

    Args:
        results_json_path: Path to universal model results
        specialist_results: Dict mapping world_id to specialist result JSON path
        baseline_results: Dict mapping baseline_name to result JSON path

    Returns:
        Dictionary with success criteria checks and values
    """
    results_json_path = Path(results_json_path)

    # Load universal model results
    with open(results_json_path) as f:
        universal = json.load(f)

    per_world = universal.get("per_world", {})

    # Load specialist results
    specialist_metrics = {}
    for world_id, path in specialist_results.items():
        with open(Path(path)) as f:
            specialist_metrics[world_id] = json.load(f).get("aggregate", {})

    # Load baseline results
    baseline_metrics = {}
    for baseline_name, path in baseline_results.items():
        with open(Path(path)) as f:
            baseline_metrics[baseline_name] = json.load(f).get("per_world", {})

    # Check 1: Beat baselines
    beat_baselines = {}
    for baseline_name, baseline_per_world in baseline_metrics.items():
        beats_all = True
        for world_id, baseline_world in baseline_per_world.items():
            if world_id not in per_world:
                continue

            universal_nrmse = per_world[world_id].get("nrmse", float("inf"))
            baseline_nrmse = baseline_world.get("nrmse", 0.0)

            if universal_nrmse >= baseline_nrmse:
                beats_all = False
                break

        beat_baselines[baseline_name] = beats_all

    # Check 2 & 3: Gap vs specialists
    gaps = []
    for world_id, world_metrics in per_world.items():
        if world_id not in specialist_metrics:
            continue

        universal_nrmse = world_metrics.get("nrmse", float("inf"))
        specialist_nrmse = specialist_metrics[world_id].get("nrmse", 0.0)

        if specialist_nrmse > 0:
            gap = (universal_nrmse - specialist_nrmse) / specialist_nrmse * 100.0
            gaps.append(gap)

    mean_gap = sum(gaps) / len(gaps) if gaps else float("nan")
    max_gap = max(gaps) if gaps else float("nan")

    # Check 4: Shape preservation
    hf_ratios = []
    for world_id, world_metrics in per_world.items():
        if world_id not in specialist_metrics:
            continue

        universal_hf = world_metrics.get("hf_ratio", 0.0)
        specialist_hf = specialist_metrics[world_id].get("hf_ratio", 1.0)

        if specialist_hf > 0:
            ratio = universal_hf / specialist_hf
            hf_ratios.append(ratio)

    mean_hf_ratio = sum(hf_ratios) / len(hf_ratios) if hf_ratios else float("nan")

    # Assemble results
    criteria = {
        "beat_baselines": beat_baselines,
        "beat_all_baselines": all(beat_baselines.values()),
        "mean_gap": mean_gap,
        "mean_gap_pass": mean_gap <= 20.0,
        "max_gap": max_gap,
        "max_gap_pass": max_gap <= 35.0,
        "mean_hf_ratio": mean_hf_ratio,
        "shape_preservation_pass": mean_hf_ratio <= 1.1,
        "all_criteria_pass": (
            all(beat_baselines.values()) and
            mean_gap <= 20.0 and
            max_gap <= 35.0 and
            mean_hf_ratio <= 1.1
        ),
    }

    return criteria


def main():
    """CLI entry point for leaderboard generation."""
    parser = argparse.ArgumentParser(
        description="Generate leaderboard from evaluation results"
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare all models in a directory",
    )
    compare_parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing evaluation result JSON files",
    )
    compare_parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path to save CSV",
    )
    compare_parser.add_argument(
        "--output-markdown",
        type=Path,
        help="Optional path to save Markdown",
    )
    compare_parser.add_argument(
        "--metric",
        type=str,
        default="nrmse",
        help="Metric to sort by (default: nrmse)",
    )

    # Custom command
    custom_parser = subparsers.add_parser(
        "custom",
        help="Create custom leaderboard from specific files",
    )
    custom_parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated list of name:path pairs (e.g., 'Universal:results/universal.json,MLP:results/mlp.json')",
    )
    custom_parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path to save CSV",
    )
    custom_parser.add_argument(
        "--output-markdown",
        type=Path,
        help="Optional path to save Markdown",
    )
    custom_parser.add_argument(
        "--oracle",
        type=str,
        help="Name of oracle model for gap computation",
    )

    args = parser.parse_args()

    if args.command == "compare":
        compare_models(
            results_dir=args.results_dir,
            output_csv=args.output_csv,
            output_markdown=args.output_markdown,
            metric=args.metric,
        )

    elif args.command == "custom":
        # Parse model list
        leaderboard = LeaderboardGenerator()

        model_pairs = args.models.split(",")
        for pair in model_pairs:
            if ":" not in pair:
                print(f"Warning: Invalid model pair '{pair}', expected 'name:path'")
                continue

            name, path = pair.split(":", 1)
            is_oracle = args.oracle and name.strip() == args.oracle

            try:
                leaderboard.add_model(name.strip(), Path(path.strip()), is_oracle=is_oracle)
            except Exception as e:
                print(f"Warning: Could not load model '{name}': {e}")
                continue

        # Print and save
        leaderboard.print_summary()

        if args.output_csv:
            leaderboard.save_csv(args.output_csv)

        if args.output_markdown:
            leaderboard.save_markdown(args.output_markdown)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
