"""
Final results generation for Sprint 5.

This module provides comprehensive tools for generating all final deliverables:
- Complete metric tables (CSV and Markdown)
- Publication-quality figures
- Results summary document

Usage:
    # Generate all final results
    python -m emulator.eval.final_results \
        --results-dir experiments/runs/ \
        --output-dir results/final/ \
        --universal-results results/universal_A.json \
        --baseline-results "MLP:results/mlp_baseline.json,VAR:results/var_baseline.json" \
        --specialist-results "lss:results/specialist_lss.json,var:results/specialist_var.json"

    # Generate only specific outputs
    python -m emulator.eval.final_results \
        --results-dir experiments/runs/ \
        --output-dir results/final/ \
        --tables-only

    python -m emulator.eval.final_results \
        --results-dir experiments/runs/ \
        --output-dir results/final/ \
        --figures-only

Output structure:
    results/final/
    ├── tables/
    │   ├── main_results.csv
    │   ├── main_results.md
    │   ├── per_world.csv
    │   ├── per_world.md
    │   ├── shape_metrics.csv
    │   ├── shape_metrics.md
    │   ├── regime_comparison.csv
    │   ├── regime_comparison.md
    │   └── success_criteria.json
    ├── figures/
    │   ├── nrmse_bar_chart.png
    │   ├── shape_comparison.png
    │   ├── regime_comparison.png
    │   ├── irf_samples/
    │   │   ├── lss_shock0.png
    │   │   ├── var_shock0.png
    │   │   └── ...
    │   └── ablation_summary.png
    └── RESULTS.md
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from emulator.eval.figures import (
    plot_nrmse_bar_chart,
    plot_regime_comparison,
    plot_shape_comparison,
)
from emulator.eval.leaderboard import LeaderboardGenerator


class FinalResultsGenerator:
    """Generate all final results for publication and reporting.

    This class orchestrates the generation of:
    - Comprehensive metric tables in multiple formats
    - Publication-quality figures
    - Summary documentation
    """

    def __init__(
        self,
        output_dir: Path,
        universal_results: Path | None = None,
        baseline_results: dict[str, Path] | None = None,
        specialist_results: dict[str, Path] | None = None,
        regime_results: dict[str, Path] | None = None,
    ):
        """Initialize results generator.

        Args:
            output_dir: Root output directory
            universal_results: Path to universal model results JSON
            baseline_results: Dict mapping baseline_name -> results path
            specialist_results: Dict mapping world_id -> specialist results path
            regime_results: Dict mapping regime -> results path (e.g., {"A": ..., "B1": ...})
        """
        self.output_dir = Path(output_dir)
        self.tables_dir = self.output_dir / "tables"
        self.figures_dir = self.output_dir / "figures"
        self.irf_samples_dir = self.figures_dir / "irf_samples"

        # Create directories
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.irf_samples_dir.mkdir(parents=True, exist_ok=True)

        # Load results
        self.universal_results = self._load_json(universal_results) if universal_results else None
        self.baseline_results = {
            name: self._load_json(path) for name, path in (baseline_results or {}).items()
        }
        self.specialist_results = {
            world: self._load_json(path) for world, path in (specialist_results or {}).items()
        }
        self.regime_results = {
            regime: self._load_json(path) for regime, path in (regime_results or {}).items()
        }

    @staticmethod
    def _load_json(path: Path | str) -> dict[str, Any]:
        """Load JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")

        with open(path) as f:
            return json.load(f)

    def generate_all(self) -> None:
        """Generate all final results (tables + figures + summary)."""
        print("Generating final results...")
        print(f"Output directory: {self.output_dir}")

        # Generate tables
        print("\n" + "=" * 80)
        print("GENERATING TABLES")
        print("=" * 80)
        self.generate_all_tables()

        # Generate figures
        print("\n" + "=" * 80)
        print("GENERATING FIGURES")
        print("=" * 80)
        self.generate_all_figures()

        # Generate summary document
        print("\n" + "=" * 80)
        print("GENERATING SUMMARY DOCUMENT")
        print("=" * 80)
        self.generate_summary_document()

        print("\n" + "=" * 80)
        print("FINAL RESULTS GENERATION COMPLETE")
        print("=" * 80)
        print(f"\nAll outputs saved to: {self.output_dir}")

    # ========================================================================
    # Table Generation
    # ========================================================================

    def generate_all_tables(self) -> None:
        """Generate all metric tables."""
        print("\nGenerating metric tables...")

        # Table 1: Main results (universal vs baselines)
        self.generate_main_results_table()

        # Table 2: Per-world breakdown
        self.generate_per_world_table()

        # Table 3: Shape metrics
        self.generate_shape_metrics_table()

        # Table 4: Regime comparison (if available)
        if self.regime_results:
            self.generate_regime_comparison_table()

        # Table 5: Success criteria check
        if self.universal_results and self.specialist_results:
            self.generate_success_criteria_table()

        print(f"Tables saved to: {self.tables_dir}")

    def generate_main_results_table(self) -> None:
        """Generate main results table (universal vs baselines)."""
        if not self.universal_results:
            print("Warning: No universal results provided, skipping main results table")
            return

        print("  - Generating main results table...")

        # Use LeaderboardGenerator
        leaderboard = LeaderboardGenerator()

        # Add universal model
        leaderboard.add_model_from_dict("Universal", self.universal_results.get("aggregate", {}))

        # Add baselines
        for baseline_name, baseline_result in self.baseline_results.items():
            leaderboard.add_model_from_dict(
                baseline_name,
                baseline_result.get("aggregate", {}),
            )

        # Generate table
        df = leaderboard.generate_table(sort_by="nrmse", ascending=True, include_gap=False)

        # Save
        csv_path = self.tables_dir / "main_results.csv"
        md_path = self.tables_dir / "main_results.md"

        df.to_csv(csv_path, index=False, float_format="%.4f")
        self._save_dataframe_as_markdown(df, md_path, title="Main Results: Universal vs Baselines")

        print(f"    Saved: {csv_path}")
        print(f"    Saved: {md_path}")

    def generate_per_world_table(self) -> None:
        """Generate per-world NRMSE breakdown table."""
        if not self.universal_results:
            print("Warning: No universal results provided, skipping per-world table")
            return

        print("  - Generating per-world breakdown table...")

        # Prepare data
        rows = []

        # Get all world IDs
        universal_per_world = self.universal_results.get("per_world", {})
        all_worlds = sorted(universal_per_world.keys())

        # Universal row
        universal_row = {"Model": "Universal"}
        for world_id in all_worlds:
            universal_row[world_id.upper()] = universal_per_world.get(world_id, {}).get("nrmse", float("nan"))
        rows.append(universal_row)

        # Baseline rows
        for baseline_name, baseline_result in self.baseline_results.items():
            baseline_per_world = baseline_result.get("per_world", {})
            baseline_row = {"Model": baseline_name}
            for world_id in all_worlds:
                baseline_row[world_id.upper()] = baseline_per_world.get(world_id, {}).get("nrmse", float("nan"))
            rows.append(baseline_row)

        # Specialist rows (if available)
        if self.specialist_results:
            for world_id in all_worlds:
                if world_id in self.specialist_results:
                    specialist_row = {"Model": f"Specialist ({world_id.upper()})"}
                    for w in all_worlds:
                        if w == world_id:
                            specialist_row[w.upper()] = self.specialist_results[world_id].get("aggregate", {}).get("nrmse", float("nan"))
                        else:
                            specialist_row[w.upper()] = float("nan")
                    rows.append(specialist_row)

        df = pd.DataFrame(rows)

        # Save
        csv_path = self.tables_dir / "per_world.csv"
        md_path = self.tables_dir / "per_world.md"

        df.to_csv(csv_path, index=False, float_format="%.4f")
        self._save_dataframe_as_markdown(df, md_path, title="Per-World NRMSE Breakdown")

        print(f"    Saved: {csv_path}")
        print(f"    Saved: {md_path}")

    def generate_shape_metrics_table(self) -> None:
        """Generate shape metrics comparison table."""
        if not self.universal_results:
            print("Warning: No universal results provided, skipping shape metrics table")
            return

        print("  - Generating shape metrics table...")

        # Prepare data
        rows = []

        # Universal model
        universal_agg = self.universal_results.get("aggregate", {})
        rows.append({
            "Model": "Universal",
            "HF-ratio": universal_agg.get("hf_ratio", float("nan")),
            "Overshoot": universal_agg.get("overshoot_ratio", float("nan")),
            "Sign Flips": universal_agg.get("sign_flip_count", float("nan")),
        })

        # Baselines
        for baseline_name, baseline_result in self.baseline_results.items():
            baseline_agg = baseline_result.get("aggregate", {})
            rows.append({
                "Model": baseline_name,
                "HF-ratio": baseline_agg.get("hf_ratio", float("nan")),
                "Overshoot": baseline_agg.get("overshoot_ratio", float("nan")),
                "Sign Flips": baseline_agg.get("sign_flip_count", float("nan")),
            })

        df = pd.DataFrame(rows)

        # Save
        csv_path = self.tables_dir / "shape_metrics.csv"
        md_path = self.tables_dir / "shape_metrics.md"

        df.to_csv(csv_path, index=False, float_format="%.4f")
        self._save_dataframe_as_markdown(df, md_path, title="Shape Metrics Comparison")

        print(f"    Saved: {csv_path}")
        print(f"    Saved: {md_path}")

    def generate_regime_comparison_table(self) -> None:
        """Generate regime comparison table."""
        print("  - Generating regime comparison table...")

        # Prepare data
        rows = []

        # Get all world IDs from regime A (assuming it has all worlds)
        regime_A_results = self.regime_results.get("A")
        if not regime_A_results:
            print("    Warning: No Regime A results, skipping regime comparison")
            return

        per_world_A = regime_A_results.get("per_world", {})
        all_worlds = sorted(per_world_A.keys())

        for regime_name, regime_result in sorted(self.regime_results.items()):
            regime_per_world = regime_result.get("per_world", {})
            row = {"Regime": f"Regime {regime_name}"}

            for world_id in all_worlds:
                row[world_id.upper()] = regime_per_world.get(world_id, {}).get("nrmse", float("nan"))

            rows.append(row)

        df = pd.DataFrame(rows)

        # Save
        csv_path = self.tables_dir / "regime_comparison.csv"
        md_path = self.tables_dir / "regime_comparison.md"

        df.to_csv(csv_path, index=False, float_format="%.4f")
        self._save_dataframe_as_markdown(df, md_path, title="Information Regime Comparison (NRMSE)")

        print(f"    Saved: {csv_path}")
        print(f"    Saved: {md_path}")

    def generate_success_criteria_table(self) -> None:
        """Generate success criteria evaluation."""
        if not self.universal_results or not self.specialist_results:
            print("Warning: Missing universal or specialist results, skipping success criteria")
            return

        print("  - Generating success criteria table...")

        # Compute success criteria
        # For this we need to create temp files since compute_success_criteria expects paths
        # Instead, we'll compute manually
        universal_per_world = self.universal_results.get("per_world", {})

        # Check 1: Beat baselines
        beat_baselines = {}
        for baseline_name, baseline_result in self.baseline_results.items():
            baseline_per_world = baseline_result.get("per_world", {})
            beats_all = True

            for world_id in universal_per_world.keys():
                if world_id not in baseline_per_world:
                    continue

                universal_nrmse = universal_per_world[world_id].get("nrmse", float("inf"))
                baseline_nrmse = baseline_per_world[world_id].get("nrmse", 0.0)

                if universal_nrmse >= baseline_nrmse:
                    beats_all = False
                    break

            beat_baselines[baseline_name] = beats_all

        # Check 2 & 3: Gap vs specialists
        gaps = []
        gap_per_world = {}
        for world_id, world_metrics in universal_per_world.items():
            if world_id not in self.specialist_results:
                continue

            universal_nrmse = world_metrics.get("nrmse", float("inf"))
            specialist_nrmse = self.specialist_results[world_id].get("aggregate", {}).get("nrmse", 0.0)

            if specialist_nrmse > 0:
                gap = (universal_nrmse - specialist_nrmse) / specialist_nrmse * 100.0
                gaps.append(gap)
                gap_per_world[world_id] = gap

        mean_gap = np.mean(gaps) if gaps else float("nan")
        max_gap = np.max(gaps) if gaps else float("nan")

        # Check 4: Shape preservation
        hf_ratios = []
        for world_id, world_metrics in universal_per_world.items():
            if world_id not in self.specialist_results:
                continue

            universal_hf = world_metrics.get("hf_ratio", 0.0)
            specialist_hf = self.specialist_results[world_id].get("aggregate", {}).get("hf_ratio", 1.0)

            if specialist_hf > 0:
                ratio = universal_hf / specialist_hf
                hf_ratios.append(ratio)

        mean_hf_ratio = np.mean(hf_ratios) if hf_ratios else float("nan")

        # Assemble criteria (convert numpy types to Python types for JSON serialization)
        criteria = {
            "beat_baselines": beat_baselines,
            "beat_all_baselines": bool(all(beat_baselines.values()) if beat_baselines else False),
            "mean_gap_percent": float(mean_gap),
            "mean_gap_pass": bool(mean_gap <= 20.0),
            "max_gap_percent": float(max_gap),
            "max_gap_pass": bool(max_gap <= 35.0),
            "mean_hf_ratio": float(mean_hf_ratio),
            "shape_preservation_pass": bool(mean_hf_ratio <= 1.1),
            "all_criteria_pass": bool(
                (all(beat_baselines.values()) and
                mean_gap <= 20.0 and
                max_gap <= 35.0 and
                mean_hf_ratio <= 1.1) if beat_baselines else False
            ),
            "gap_per_world": {k: float(v) for k, v in gap_per_world.items()},
        }

        # Save as JSON
        json_path = self.tables_dir / "success_criteria.json"
        with open(json_path, "w") as f:
            json.dump(criteria, f, indent=2)

        print(f"    Saved: {json_path}")

        # Also create a summary table
        rows = [
            {
                "Criterion": "Beat all baselines",
                "Target": "Yes",
                "Actual": "Yes" if criteria["beat_all_baselines"] else "No",
                "Pass": "✓" if criteria["beat_all_baselines"] else "✗",
            },
            {
                "Criterion": "Mean gap vs specialist",
                "Target": "≤ 20%",
                "Actual": f"{criteria['mean_gap_percent']:.1f}%",
                "Pass": "✓" if criteria["mean_gap_pass"] else "✗",
            },
            {
                "Criterion": "Max gap vs specialist",
                "Target": "≤ 35%",
                "Actual": f"{criteria['max_gap_percent']:.1f}%",
                "Pass": "✓" if criteria["max_gap_pass"] else "✗",
            },
            {
                "Criterion": "Shape preservation (HF-ratio)",
                "Target": "≤ 1.1",
                "Actual": f"{criteria['mean_hf_ratio']:.2f}",
                "Pass": "✓" if criteria["shape_preservation_pass"] else "✗",
            },
        ]

        df = pd.DataFrame(rows)

        # Save
        csv_path = self.tables_dir / "success_criteria.csv"
        md_path = self.tables_dir / "success_criteria.md"

        df.to_csv(csv_path, index=False)
        self._save_dataframe_as_markdown(df, md_path, title="Success Criteria Evaluation")

        print(f"    Saved: {csv_path}")
        print(f"    Saved: {md_path}")

    @staticmethod
    def _save_dataframe_as_markdown(
        df: pd.DataFrame,
        path: Path,
        title: str | None = None,
    ) -> None:
        """Save DataFrame as markdown file."""
        # Format floats nicely
        df_formatted = df.copy()
        for col in df_formatted.columns:
            if df_formatted[col].dtype in [float, "float64", np.float64]:
                df_formatted[col] = df_formatted[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) and not np.isnan(x) else "—"
                )

        markdown = df_formatted.to_markdown(index=False)

        with open(path, "w") as f:
            if title:
                f.write(f"# {title}\n\n")
            f.write(markdown)
            f.write("\n")

    # ========================================================================
    # Figure Generation
    # ========================================================================

    def generate_all_figures(self) -> None:
        """Generate all publication-quality figures."""
        print("\nGenerating figures...")

        # Figure 1: NRMSE bar chart
        if self.universal_results:
            self.generate_nrmse_figure()

        # Figure 2: Shape comparison
        if self.universal_results:
            self.generate_shape_figure()

        # Figure 3: Regime comparison
        if self.regime_results and len(self.regime_results) > 1:
            self.generate_regime_figure()

        # Figure 4: Sample IRF panels (not implemented yet - would need actual data)
        # self.generate_irf_sample_figures()

        print(f"Figures saved to: {self.figures_dir}")

    def generate_nrmse_figure(self) -> None:
        """Generate NRMSE bar chart."""
        print("  - Generating NRMSE bar chart...")

        output_path = self.figures_dir / "nrmse_bar_chart.png"

        try:
            fig = plot_nrmse_bar_chart(
                results=self.universal_results,
                output_path=output_path,
                baseline_results=self.baseline_results,
                dpi=300,
            )
            plt.close(fig)
            print(f"    Saved: {output_path}")
        except Exception as e:
            print(f"    Warning: Could not generate NRMSE bar chart: {e}")

    def generate_shape_figure(self) -> None:
        """Generate shape metrics comparison figure."""
        print("  - Generating shape comparison figure...")

        output_path = self.figures_dir / "shape_comparison.png"

        try:
            fig = plot_shape_comparison(
                results=self.universal_results,
                output_path=output_path,
                baseline_results=self.baseline_results,
                dpi=300,
            )
            plt.close(fig)
            print(f"    Saved: {output_path}")
        except Exception as e:
            print(f"    Warning: Could not generate shape comparison: {e}")

    def generate_regime_figure(self) -> None:
        """Generate regime comparison figure."""
        print("  - Generating regime comparison figure...")

        output_path = self.figures_dir / "regime_comparison.png"

        try:
            # Get regime results in order
            results_A = self.regime_results.get("A")
            results_B1 = self.regime_results.get("B1")
            results_C = self.regime_results.get("C")

            if not results_A:
                print("    Warning: No Regime A results, skipping regime figure")
                return

            fig = plot_regime_comparison(
                results_A=results_A,
                results_B1=results_B1,
                results_C=results_C,
                output_path=output_path,
                dpi=300,
            )
            plt.close(fig)
            print(f"    Saved: {output_path}")
        except Exception as e:
            print(f"    Warning: Could not generate regime comparison: {e}")

    # ========================================================================
    # Summary Document Generation
    # ========================================================================

    def generate_summary_document(self) -> None:
        """Generate RESULTS.md summary document."""
        print("\nGenerating RESULTS.md summary document...")

        output_path = self.output_dir / "RESULTS.md"

        # Build content
        content = self._build_summary_content()

        # Write to file
        with open(output_path, "w") as f:
            f.write(content)

        print(f"Saved: {output_path}")

    def _build_summary_content(self) -> str:
        """Build content for RESULTS.md."""
        lines = []

        # Header
        lines.append("# Universal Macro Emulator: Final Results")
        lines.append("")
        lines.append("**Generated:** Sprint 5 Final Results")
        lines.append("")
        lines.append("This document summarizes the main findings from the Universal Macro Emulator project.")
        lines.append("")

        # Table of contents
        lines.append("## Table of Contents")
        lines.append("")
        lines.append("1. [Overview](#overview)")
        lines.append("2. [Main Results](#main-results)")
        lines.append("3. [Per-World Performance](#per-world-performance)")
        lines.append("4. [Shape Metrics](#shape-metrics)")
        if self.regime_results and len(self.regime_results) > 1:
            lines.append("5. [Information Regime Comparison](#information-regime-comparison)")
        if self.specialist_results:
            lines.append("6. [Success Criteria](#success-criteria)")
        lines.append("7. [Key Figures](#key-figures)")
        lines.append("8. [Conclusion](#conclusion)")
        lines.append("")

        # Overview
        lines.append("## Overview")
        lines.append("")
        lines.append("The Universal Macro Emulator is a neural network that predicts impulse response functions (IRFs)")
        lines.append("across 6 diverse macroeconomic simulator families:")
        lines.append("")
        lines.append("- **LSS**: Linear State Space")
        lines.append("- **VAR**: Vector Autoregression")
        lines.append("- **NK**: New Keynesian DSGE")
        lines.append("- **RBC**: Real Business Cycle")
        lines.append("- **Switching**: Regime-Switching Model")
        lines.append("- **ZLB**: Zero Lower Bound Model")
        lines.append("")
        lines.append("The goal is a single model that generalizes across all simulator families, achieving performance")
        lines.append("comparable to per-world specialists while beating simple baselines.")
        lines.append("")

        # Main Results
        lines.append("## Main Results")
        lines.append("")

        if self.universal_results:
            universal_agg = self.universal_results.get("aggregate", {})
            lines.append("### Universal Emulator Performance")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| NRMSE | {universal_agg.get('nrmse', 0.0):.4f} |")
            lines.append(f"| IAE | {universal_agg.get('iae', 0.0):.4f} |")
            lines.append(f"| Sign@Impact | {universal_agg.get('sign_at_impact', 0.0):.4f} |")
            lines.append(f"| HF-ratio | {universal_agg.get('hf_ratio', 0.0):.4f} |")
            lines.append(f"| Overshoot Ratio | {universal_agg.get('overshoot_ratio', 0.0):.4f} |")
            lines.append(f"| Sign Flips (avg) | {universal_agg.get('sign_flip_count', 0.0):.2f} |")
            lines.append("")

        if self.baseline_results:
            lines.append("### Comparison with Baselines")
            lines.append("")
            lines.append("| Model | NRMSE | IAE | Sign@Impact |")
            lines.append("|-------|-------|-----|-------------|")

            if self.universal_results:
                universal_agg = self.universal_results.get("aggregate", {})
                lines.append(
                    f"| Universal | {universal_agg.get('nrmse', 0.0):.4f} | "
                    f"{universal_agg.get('iae', 0.0):.4f} | "
                    f"{universal_agg.get('sign_at_impact', 0.0):.4f} |"
                )

            for baseline_name, baseline_result in sorted(self.baseline_results.items()):
                baseline_agg = baseline_result.get("aggregate", {})
                lines.append(
                    f"| {baseline_name} | {baseline_agg.get('nrmse', 0.0):.4f} | "
                    f"{baseline_agg.get('iae', 0.0):.4f} | "
                    f"{baseline_agg.get('sign_at_impact', 0.0):.4f} |"
                )
            lines.append("")

        lines.append("See `tables/main_results.md` for complete metrics.")
        lines.append("")

        # Per-World Performance
        lines.append("## Per-World Performance")
        lines.append("")

        if self.universal_results:
            universal_per_world = self.universal_results.get("per_world", {})
            lines.append("### NRMSE by Simulator Family")
            lines.append("")
            lines.append("| World | NRMSE | IAE | HF-ratio |")
            lines.append("|-------|-------|-----|----------|")

            for world_id in sorted(universal_per_world.keys()):
                world_metrics = universal_per_world[world_id]
                lines.append(
                    f"| {world_id.upper()} | {world_metrics.get('nrmse', 0.0):.4f} | "
                    f"{world_metrics.get('iae', 0.0):.4f} | "
                    f"{world_metrics.get('hf_ratio', 0.0):.4f} |"
                )
            lines.append("")

        lines.append("See `tables/per_world.md` for complete per-world breakdown.")
        lines.append("")

        # Shape Metrics
        lines.append("## Shape Metrics")
        lines.append("")
        lines.append("Shape metrics assess whether the emulator preserves IRF dynamics:")
        lines.append("")
        lines.append("- **HF-ratio**: High-frequency spectral content (lower is better, less oscillation)")
        lines.append("- **Overshoot**: Peak-to-impact ratio")
        lines.append("- **Sign Flips**: Number of sign reversals in first differences")
        lines.append("")

        if self.universal_results:
            universal_agg = self.universal_results.get("aggregate", {})
            lines.append("| Metric | Universal |")
            lines.append("|--------|-----------|")
            lines.append(f"| HF-ratio | {universal_agg.get('hf_ratio', 0.0):.4f} |")
            lines.append(f"| Overshoot | {universal_agg.get('overshoot_ratio', 0.0):.4f} |")
            lines.append(f"| Sign Flips | {universal_agg.get('sign_flip_count', 0.0):.2f} |")
            lines.append("")

        lines.append("See `tables/shape_metrics.md` for comparison with baselines.")
        lines.append("")

        # Regime Comparison
        if self.regime_results and len(self.regime_results) > 1:
            lines.append("## Information Regime Comparison")
            lines.append("")
            lines.append("The emulator was evaluated under different information regimes:")
            lines.append("")
            lines.append("- **Regime A**: Full structural assist (world_id, theta, eps_sequence, shock_token)")
            lines.append("- **Regime B1**: Observables + world known (world_id, shock_token, history)")
            lines.append("- **Regime C**: Partial (world_id, theta, shock_token, history, no eps)")
            lines.append("")

            lines.append("| Regime | NRMSE | IAE |")
            lines.append("|--------|-------|-----|")

            for regime_name in sorted(self.regime_results.keys()):
                regime_result = self.regime_results[regime_name]
                regime_agg = regime_result.get("aggregate", {})
                lines.append(
                    f"| Regime {regime_name} | {regime_agg.get('nrmse', 0.0):.4f} | "
                    f"{regime_agg.get('iae', 0.0):.4f} |"
                )
            lines.append("")

            lines.append("See `tables/regime_comparison.md` for per-world regime breakdown.")
            lines.append("")

        # Success Criteria
        if self.specialist_results:
            lines.append("## Success Criteria")
            lines.append("")
            lines.append("Per specification §7.2.3, the universal emulator must satisfy:")
            lines.append("")
            lines.append("1. **Beat baselines**: NRMSE < baseline NRMSE for all worlds")
            lines.append("2. **Mean gap ≤ 20%**: Average gap vs specialist models")
            lines.append("3. **Max gap ≤ 35%**: Maximum gap vs specialist on any world")
            lines.append("4. **Shape preservation**: HF-ratio ≤ 1.1× specialist")
            lines.append("")

            # Load success criteria if it was generated
            criteria_path = self.tables_dir / "success_criteria.json"
            if criteria_path.exists():
                with open(criteria_path) as f:
                    criteria = json.load(f)

                lines.append("| Criterion | Target | Actual | Status |")
                lines.append("|-----------|--------|--------|--------|")
                lines.append(
                    f"| Beat all baselines | Yes | "
                    f"{'Yes' if criteria.get('beat_all_baselines') else 'No'} | "
                    f"{'✓' if criteria.get('beat_all_baselines') else '✗'} |"
                )
                lines.append(
                    f"| Mean gap vs specialist | ≤ 20% | "
                    f"{criteria.get('mean_gap_percent', 0.0):.1f}% | "
                    f"{'✓' if criteria.get('mean_gap_pass') else '✗'} |"
                )
                lines.append(
                    f"| Max gap vs specialist | ≤ 35% | "
                    f"{criteria.get('max_gap_percent', 0.0):.1f}% | "
                    f"{'✓' if criteria.get('max_gap_pass') else '✗'} |"
                )
                lines.append(
                    f"| Shape preservation | ≤ 1.1 | "
                    f"{criteria.get('mean_hf_ratio', 0.0):.2f} | "
                    f"{'✓' if criteria.get('shape_preservation_pass') else '✗'} |"
                )
                lines.append("")

                if criteria.get("all_criteria_pass"):
                    lines.append("**All success criteria met!** ✓")
                else:
                    lines.append("**Some success criteria not met.** See `tables/success_criteria.json` for details.")
                lines.append("")

            lines.append("See `tables/success_criteria.md` for detailed breakdown.")
            lines.append("")

        # Key Figures
        lines.append("## Key Figures")
        lines.append("")
        lines.append("The following figures provide visual summaries of the results:")
        lines.append("")
        lines.append("1. **NRMSE Bar Chart** (`figures/nrmse_bar_chart.png`): Per-world NRMSE comparison")
        lines.append("2. **Shape Comparison** (`figures/shape_comparison.png`): HF-ratio, overshoot, sign-flips")

        if self.regime_results and len(self.regime_results) > 1:
            lines.append("3. **Regime Comparison** (`figures/regime_comparison.png`): NRMSE across information regimes")

        lines.append("")

        # Conclusion
        lines.append("## Conclusion")
        lines.append("")
        lines.append("The Universal Macro Emulator demonstrates the feasibility of a single neural network that can")
        lines.append("generalize across diverse macroeconomic simulator families. Key achievements:")
        lines.append("")

        if self.universal_results:
            universal_agg = self.universal_results.get("aggregate", {})
            lines.append(f"- Aggregate NRMSE: {universal_agg.get('nrmse', 0.0):.4f}")

        if self.baseline_results:
            lines.append(f"- Evaluated against {len(self.baseline_results)} baseline model(s)")

        if self.specialist_results:
            lines.append(f"- Compared with {len(self.specialist_results)} specialist model(s)")

        lines.append("")
        lines.append("For complete results, see:")
        lines.append("")
        lines.append("- `tables/` directory for all metric tables")
        lines.append("- `figures/` directory for all figures")
        lines.append("- Specification document (`spec/spec.md`) for detailed evaluation criteria")
        lines.append("")

        return "\n".join(lines)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate final results for Sprint 5 deliverables"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/final"),
        help="Output directory for all results (default: results/final/)",
    )
    parser.add_argument(
        "--universal-results",
        type=Path,
        help="Path to universal model results JSON",
    )
    parser.add_argument(
        "--baseline-results",
        type=str,
        help="Comma-separated list of name:path pairs for baseline results",
    )
    parser.add_argument(
        "--specialist-results",
        type=str,
        help="Comma-separated list of world_id:path pairs for specialist results",
    )
    parser.add_argument(
        "--regime-results",
        type=str,
        help="Comma-separated list of regime:path pairs (e.g., 'A:results/regime_A.json,B1:results/regime_B1.json')",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Generate only tables (skip figures and summary)",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Generate only figures (skip tables and summary)",
    )

    args = parser.parse_args()

    # Parse baseline results
    baseline_results = {}
    if args.baseline_results:
        for pair in args.baseline_results.split(","):
            if ":" not in pair:
                print(f"Warning: Invalid baseline pair '{pair}', expected 'name:path'")
                continue
            name, path = pair.split(":", 1)
            baseline_results[name.strip()] = Path(path.strip())

    # Parse specialist results
    specialist_results = {}
    if args.specialist_results:
        for pair in args.specialist_results.split(","):
            if ":" not in pair:
                print(f"Warning: Invalid specialist pair '{pair}', expected 'world_id:path'")
                continue
            world_id, path = pair.split(":", 1)
            specialist_results[world_id.strip()] = Path(path.strip())

    # Parse regime results
    regime_results = {}
    if args.regime_results:
        for pair in args.regime_results.split(","):
            if ":" not in pair:
                print(f"Warning: Invalid regime pair '{pair}', expected 'regime:path'")
                continue
            regime, path = pair.split(":", 1)
            regime_results[regime.strip()] = Path(path.strip())

    # Create generator
    generator = FinalResultsGenerator(
        output_dir=args.output_dir,
        universal_results=args.universal_results,
        baseline_results=baseline_results,
        specialist_results=specialist_results,
        regime_results=regime_results,
    )

    # Generate outputs based on flags
    if args.tables_only:
        generator.generate_all_tables()
    elif args.figures_only:
        generator.generate_all_figures()
    else:
        generator.generate_all()


if __name__ == "__main__":
    main()
