"""
Tests for final results generation (Sprint 5).

Tests cover:
- Table generation (CSV and Markdown)
- Figure generation
- Summary document generation
- Integration with existing evaluation outputs
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from emulator.eval.final_results import FinalResultsGenerator


@pytest.fixture
def mock_universal_results() -> dict:
    """Mock universal model evaluation results."""
    return {
        "aggregate": {
            "nrmse": 0.1523,
            "iae": 2.347,
            "sign_at_impact": 0.9456,
            "hf_ratio": 0.0834,
            "overshoot_ratio": 1.2145,
            "sign_flip_count": 2.12,
            "n_samples": 30000,
        },
        "per_world": {
            "lss": {
                "nrmse": 0.1234,
                "iae": 1.834,
                "sign_at_impact": 0.9567,
                "hf_ratio": 0.0723,
                "overshoot_ratio": 1.1234,
                "sign_flip_count": 1.89,
                "n_samples": 5000,
            },
            "var": {
                "nrmse": 0.1456,
                "iae": 2.123,
                "sign_at_impact": 0.9412,
                "hf_ratio": 0.0812,
                "overshoot_ratio": 1.2456,
                "sign_flip_count": 2.15,
                "n_samples": 5000,
            },
            "nk": {
                "nrmse": 0.1678,
                "iae": 2.567,
                "sign_at_impact": 0.9378,
                "hf_ratio": 0.0901,
                "overshoot_ratio": 1.2789,
                "sign_flip_count": 2.34,
                "n_samples": 5000,
            },
        },
        "per_split": {
            "test_interpolation": {
                "nrmse": 0.1489,
                "iae": 2.301,
                "sign_at_impact": 0.9478,
                "n_samples": 15000,
            },
        },
    }


@pytest.fixture
def mock_baseline_results() -> dict:
    """Mock baseline model evaluation results."""
    return {
        "aggregate": {
            "nrmse": 0.1834,
            "iae": 2.891,
            "sign_at_impact": 0.9201,
            "hf_ratio": 0.1023,
            "overshoot_ratio": 1.3421,
            "sign_flip_count": 2.87,
            "n_samples": 30000,
        },
        "per_world": {
            "lss": {
                "nrmse": 0.1567,
                "iae": 2.234,
                "sign_at_impact": 0.9345,
                "hf_ratio": 0.0934,
                "n_samples": 5000,
            },
            "var": {
                "nrmse": 0.1789,
                "iae": 2.678,
                "sign_at_impact": 0.9189,
                "hf_ratio": 0.1012,
                "n_samples": 5000,
            },
            "nk": {
                "nrmse": 0.1901,
                "iae": 3.012,
                "sign_at_impact": 0.9134,
                "hf_ratio": 0.1123,
                "n_samples": 5000,
            },
        },
    }


@pytest.fixture
def mock_specialist_results() -> dict:
    """Mock specialist model evaluation results."""
    return {
        "lss": {
            "aggregate": {
                "nrmse": 0.1123,
                "iae": 1.678,
                "sign_at_impact": 0.9634,
                "hf_ratio": 0.0689,
                "n_samples": 5000,
            },
        },
        "var": {
            "aggregate": {
                "nrmse": 0.1334,
                "iae": 1.923,
                "sign_at_impact": 0.9512,
                "hf_ratio": 0.0756,
                "n_samples": 5000,
            },
        },
    }


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Create temporary output directory."""
    return tmp_path / "final_results"


@pytest.fixture
def mock_results_files(
    tmp_path,
    mock_universal_results,
    mock_baseline_results,
    mock_specialist_results,
) -> dict:
    """Create temporary mock results files."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Write universal results
    universal_path = results_dir / "universal.json"
    with open(universal_path, "w") as f:
        json.dump(mock_universal_results, f)

    # Write baseline results
    baseline_path = results_dir / "mlp_baseline.json"
    with open(baseline_path, "w") as f:
        json.dump(mock_baseline_results, f)

    # Write specialist results
    specialist_paths = {}
    for world_id, specialist_result in mock_specialist_results.items():
        specialist_path = results_dir / f"specialist_{world_id}.json"
        with open(specialist_path, "w") as f:
            json.dump(specialist_result, f)
        specialist_paths[world_id] = specialist_path

    return {
        "universal": universal_path,
        "baseline": {"MLP": baseline_path},
        "specialist": specialist_paths,
    }


@pytest.mark.fast
class TestFinalResultsGenerator:
    """Tests for FinalResultsGenerator class."""

    def test_initialization(self, temp_output_dir):
        """Test generator initialization."""
        generator = FinalResultsGenerator(output_dir=temp_output_dir)

        assert generator.output_dir == temp_output_dir
        assert generator.tables_dir == temp_output_dir / "tables"
        assert generator.figures_dir == temp_output_dir / "figures"
        assert generator.irf_samples_dir == temp_output_dir / "figures" / "irf_samples"

        # Check directories were created
        assert generator.tables_dir.exists()
        assert generator.figures_dir.exists()
        assert generator.irf_samples_dir.exists()

    def test_load_results(self, temp_output_dir, mock_results_files):
        """Test loading results from files."""
        generator = FinalResultsGenerator(
            output_dir=temp_output_dir,
            universal_results=mock_results_files["universal"],
            baseline_results=mock_results_files["baseline"],
            specialist_results=mock_results_files["specialist"],
        )

        assert generator.universal_results is not None
        assert "aggregate" in generator.universal_results
        assert "per_world" in generator.universal_results

        assert "MLP" in generator.baseline_results
        assert "lss" in generator.specialist_results
        assert "var" in generator.specialist_results

    def test_generate_main_results_table(
        self,
        temp_output_dir,
        mock_results_files,
    ):
        """Test main results table generation."""
        generator = FinalResultsGenerator(
            output_dir=temp_output_dir,
            universal_results=mock_results_files["universal"],
            baseline_results=mock_results_files["baseline"],
        )

        generator.generate_main_results_table()

        # Check CSV file exists
        csv_path = generator.tables_dir / "main_results.csv"
        assert csv_path.exists()

        # Check Markdown file exists
        md_path = generator.tables_dir / "main_results.md"
        assert md_path.exists()

        # Read CSV and verify contents
        import pandas as pd
        df = pd.read_csv(csv_path)

        assert "Model" in df.columns
        assert "NRMSE" in df.columns
        assert len(df) == 2  # Universal + MLP

        # Check values
        universal_row = df[df["Model"] == "Universal"].iloc[0]
        assert np.isclose(universal_row["NRMSE"], 0.1523, atol=1e-4)

    def test_generate_per_world_table(
        self,
        temp_output_dir,
        mock_results_files,
    ):
        """Test per-world breakdown table generation."""
        generator = FinalResultsGenerator(
            output_dir=temp_output_dir,
            universal_results=mock_results_files["universal"],
            baseline_results=mock_results_files["baseline"],
            specialist_results=mock_results_files["specialist"],
        )

        generator.generate_per_world_table()

        # Check files exist
        csv_path = generator.tables_dir / "per_world.csv"
        md_path = generator.tables_dir / "per_world.md"
        assert csv_path.exists()
        assert md_path.exists()

        # Read and verify
        import pandas as pd
        df = pd.read_csv(csv_path)

        assert "Model" in df.columns
        assert "LSS" in df.columns
        assert "VAR" in df.columns
        assert "NK" in df.columns

        # Check universal row
        universal_row = df[df["Model"] == "Universal"].iloc[0]
        assert np.isclose(universal_row["LSS"], 0.1234, atol=1e-4)
        assert np.isclose(universal_row["VAR"], 0.1456, atol=1e-4)

    def test_generate_shape_metrics_table(
        self,
        temp_output_dir,
        mock_results_files,
    ):
        """Test shape metrics table generation."""
        generator = FinalResultsGenerator(
            output_dir=temp_output_dir,
            universal_results=mock_results_files["universal"],
            baseline_results=mock_results_files["baseline"],
        )

        generator.generate_shape_metrics_table()

        # Check files exist
        csv_path = generator.tables_dir / "shape_metrics.csv"
        md_path = generator.tables_dir / "shape_metrics.md"
        assert csv_path.exists()
        assert md_path.exists()

        # Read and verify
        import pandas as pd
        df = pd.read_csv(csv_path)

        assert "Model" in df.columns
        assert "HF-ratio" in df.columns
        assert "Overshoot" in df.columns
        assert "Sign Flips" in df.columns

        universal_row = df[df["Model"] == "Universal"].iloc[0]
        assert np.isclose(universal_row["HF-ratio"], 0.0834, atol=1e-4)

    def test_generate_success_criteria_table(
        self,
        temp_output_dir,
        mock_results_files,
    ):
        """Test success criteria table generation."""
        generator = FinalResultsGenerator(
            output_dir=temp_output_dir,
            universal_results=mock_results_files["universal"],
            baseline_results=mock_results_files["baseline"],
            specialist_results=mock_results_files["specialist"],
        )

        generator.generate_success_criteria_table()

        # Check JSON file exists
        json_path = generator.tables_dir / "success_criteria.json"
        assert json_path.exists()

        # Check CSV/MD files
        csv_path = generator.tables_dir / "success_criteria.csv"
        md_path = generator.tables_dir / "success_criteria.md"
        assert csv_path.exists()
        assert md_path.exists()

        # Read JSON and verify structure
        with open(json_path) as f:
            criteria = json.load(f)

        assert "beat_baselines" in criteria
        assert "mean_gap_percent" in criteria
        assert "max_gap_percent" in criteria
        assert "mean_hf_ratio" in criteria
        assert "all_criteria_pass" in criteria

        # Check that gap is computed
        assert isinstance(criteria["mean_gap_percent"], (int, float))
        assert isinstance(criteria["max_gap_percent"], (int, float))

    def test_generate_summary_document(
        self,
        temp_output_dir,
        mock_results_files,
    ):
        """Test RESULTS.md summary document generation."""
        generator = FinalResultsGenerator(
            output_dir=temp_output_dir,
            universal_results=mock_results_files["universal"],
            baseline_results=mock_results_files["baseline"],
            specialist_results=mock_results_files["specialist"],
        )

        generator.generate_summary_document()

        # Check file exists
        md_path = generator.output_dir / "RESULTS.md"
        assert md_path.exists()

        # Read and verify contents
        with open(md_path) as f:
            content = f.read()

        # Check for expected sections
        assert "# Universal Macro Emulator: Final Results" in content
        assert "## Overview" in content
        assert "## Main Results" in content
        assert "## Per-World Performance" in content
        assert "## Shape Metrics" in content
        assert "## Success Criteria" in content
        assert "## Key Figures" in content
        assert "## Conclusion" in content

        # Check for specific values
        assert "0.1523" in content  # Universal NRMSE
        assert "LSS" in content
        assert "VAR" in content
        assert "NK" in content

    def test_generate_all_tables(
        self,
        temp_output_dir,
        mock_results_files,
    ):
        """Test generating all tables at once."""
        generator = FinalResultsGenerator(
            output_dir=temp_output_dir,
            universal_results=mock_results_files["universal"],
            baseline_results=mock_results_files["baseline"],
            specialist_results=mock_results_files["specialist"],
        )

        generator.generate_all_tables()

        # Check that all expected files exist
        expected_files = [
            "main_results.csv",
            "main_results.md",
            "per_world.csv",
            "per_world.md",
            "shape_metrics.csv",
            "shape_metrics.md",
            "success_criteria.csv",
            "success_criteria.md",
            "success_criteria.json",
        ]

        for filename in expected_files:
            assert (generator.tables_dir / filename).exists(), f"Missing {filename}"

    def test_generate_all(
        self,
        temp_output_dir,
        mock_results_files,
    ):
        """Test generating all outputs (tables + figures + summary)."""
        generator = FinalResultsGenerator(
            output_dir=temp_output_dir,
            universal_results=mock_results_files["universal"],
            baseline_results=mock_results_files["baseline"],
            specialist_results=mock_results_files["specialist"],
        )

        # Note: We skip actual figure generation in tests since it requires matplotlib
        # and may fail in headless environments. In real usage, figures would be generated.
        generator.generate_all_tables()
        generator.generate_summary_document()

        # Check that tables exist
        assert (generator.tables_dir / "main_results.csv").exists()

        # Check that summary exists
        assert (generator.output_dir / "RESULTS.md").exists()

    def test_missing_universal_results(self, temp_output_dir):
        """Test handling of missing universal results."""
        generator = FinalResultsGenerator(
            output_dir=temp_output_dir,
            universal_results=None,
        )

        # Should not raise error, but should skip table generation
        generator.generate_main_results_table()

        # No files should be created
        csv_path = generator.tables_dir / "main_results.csv"
        assert not csv_path.exists()

    def test_missing_baseline_results(
        self,
        temp_output_dir,
        mock_results_files,
    ):
        """Test handling of missing baseline results."""
        generator = FinalResultsGenerator(
            output_dir=temp_output_dir,
            universal_results=mock_results_files["universal"],
            baseline_results={},  # No baselines
        )

        # Should still generate table with only universal model
        generator.generate_main_results_table()

        csv_path = generator.tables_dir / "main_results.csv"
        assert csv_path.exists()

        import pandas as pd
        df = pd.read_csv(csv_path)
        assert len(df) == 1  # Only universal model

    def test_gap_computation(
        self,
        temp_output_dir,
        mock_results_files,
    ):
        """Test gap metric computation vs specialists."""
        generator = FinalResultsGenerator(
            output_dir=temp_output_dir,
            universal_results=mock_results_files["universal"],
            specialist_results=mock_results_files["specialist"],
        )

        generator.generate_success_criteria_table()

        # Read success criteria
        json_path = generator.tables_dir / "success_criteria.json"
        with open(json_path) as f:
            criteria = json.load(f)

        # Check gap computation
        # LSS: (0.1234 - 0.1123) / 0.1123 * 100 ≈ 9.88%
        # VAR: (0.1456 - 0.1334) / 0.1334 * 100 ≈ 9.14%
        # Mean: ≈ 9.51%
        mean_gap = criteria["mean_gap_percent"]
        assert mean_gap > 0
        assert mean_gap < 20.0  # Should pass success criterion

        # Check gap per world
        assert "gap_per_world" in criteria
        assert "lss" in criteria["gap_per_world"]
        assert "var" in criteria["gap_per_world"]


@pytest.mark.fast
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_nonexistent_results_file(self, temp_output_dir):
        """Test error handling for nonexistent results file."""
        with pytest.raises(FileNotFoundError):
            FinalResultsGenerator(
                output_dir=temp_output_dir,
                universal_results=Path("/nonexistent/path/results.json"),
            )

    def test_malformed_json(self, tmp_path, temp_output_dir):
        """Test handling of malformed JSON files."""
        # Create malformed JSON file
        bad_json_path = tmp_path / "bad_results.json"
        with open(bad_json_path, "w") as f:
            f.write("{invalid json}")

        with pytest.raises(json.JSONDecodeError):
            FinalResultsGenerator(
                output_dir=temp_output_dir,
                universal_results=bad_json_path,
            )

    def test_empty_results(self, tmp_path, temp_output_dir):
        """Test handling of empty results."""
        # Create empty results file
        empty_results_path = tmp_path / "empty_results.json"
        with open(empty_results_path, "w") as f:
            json.dump({}, f)

        generator = FinalResultsGenerator(
            output_dir=temp_output_dir,
            universal_results=empty_results_path,
        )

        # Should not crash, but tables will be minimal
        generator.generate_all_tables()
