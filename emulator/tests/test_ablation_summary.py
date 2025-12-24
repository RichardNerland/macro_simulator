"""
Tests for ablation summary module.

Tests cover:
1. Loading ablation results from JSON files
2. Computing delta metrics correctly
3. Generating comparison tables
4. Saving to CSV and Markdown
5. CLI interface
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from emulator.eval.ablation_summary import (
    AblationSummary,
    generate_ablation_summary,
    load_ablation_results,
)


@pytest.fixture
def sample_full_results():
    """Sample full model evaluation results."""
    return {
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


@pytest.fixture
def sample_ablation_results():
    """Sample ablation evaluation results."""
    return {
        "No World ID": {
            "aggregate": {
                "nrmse": 0.18,  # Worse by 20%
                "iae": 2.8,     # Worse by ~21.7%
                "sign_at_impact": 0.88,
                "hf_ratio": 0.09,
                "overshoot_ratio": 1.18,
                "sign_flip_count": 2.3,
                "n_samples": 1000,
            }
        },
        "No Theta": {
            "aggregate": {
                "nrmse": 0.22,  # Worse by ~46.7%
                "iae": 3.1,     # Worse by ~34.8%
                "sign_at_impact": 0.85,
                "hf_ratio": 0.10,
                "overshoot_ratio": 1.22,
                "sign_flip_count": 2.5,
                "n_samples": 1000,
            }
        },
        "No Eps": {
            "aggregate": {
                "nrmse": 0.16,  # Worse by ~6.7%
                "iae": 2.4,     # Worse by ~4.3%
                "sign_at_impact": 0.91,
                "hf_ratio": 0.08,
                "overshoot_ratio": 1.16,
                "sign_flip_count": 2.2,
                "n_samples": 1000,
            }
        },
    }


@pytest.mark.fast
def test_ablation_summary_initialization():
    """Test AblationSummary initialization."""
    summary = AblationSummary()

    assert summary.full_metrics is None
    assert summary.ablation_metrics == {}


@pytest.mark.fast
def test_add_full_model_from_file(sample_full_results):
    """Test adding full model from JSON file."""
    summary = AblationSummary()

    # Create temporary JSON file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_full_results, f)
        temp_path = Path(f.name)

    try:
        summary.add_full_model(temp_path)

        assert summary.full_metrics is not None
        assert summary.full_metrics["nrmse"] == 0.15
        assert summary.full_metrics["iae"] == 2.3
        assert summary.full_metrics["sign_at_impact"] == 0.92
        assert summary.full_metrics["n_samples"] == 1000

    finally:
        temp_path.unlink()


@pytest.mark.fast
def test_add_ablation_from_file(sample_ablation_results):
    """Test adding ablation from JSON file."""
    summary = AblationSummary()

    # Create temporary JSON file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_ablation_results["No World ID"], f)
        temp_path = Path(f.name)

    try:
        summary.add_ablation("No World ID", temp_path)

        assert "No World ID" in summary.ablation_metrics
        assert summary.ablation_metrics["No World ID"]["nrmse"] == 0.18
        assert summary.ablation_metrics["No World ID"]["iae"] == 2.8

    finally:
        temp_path.unlink()


@pytest.mark.fast
def test_compute_delta_percent():
    """Test delta percentage computation."""
    summary = AblationSummary()

    # Positive delta (ablation worse)
    delta = summary.compute_delta_percent(0.18, 0.15)
    assert abs(delta - 20.0) < 0.1

    # Negative delta (ablation better)
    delta = summary.compute_delta_percent(0.12, 0.15)
    assert abs(delta - (-20.0)) < 0.1

    # Zero delta (same)
    delta = summary.compute_delta_percent(0.15, 0.15)
    assert abs(delta) < 0.1

    # Division by zero
    delta = summary.compute_delta_percent(0.18, 0.0)
    assert np.isnan(delta)


@pytest.mark.fast
def test_generate_table(sample_full_results, sample_ablation_results):
    """Test generating ablation comparison table."""
    summary = AblationSummary()

    # Create temporary files
    temp_files = []

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_full_results, f)
        full_path = Path(f.name)
        temp_files.append(full_path)

    summary.add_full_model(full_path)

    for name, results in sample_ablation_results.items():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(results, f)
            ablation_path = Path(f.name)
            temp_files.append(ablation_path)

        summary.add_ablation(name, ablation_path)

    try:
        df = summary.generate_table()

        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # Full + 3 ablations
        assert "Model" in df.columns
        assert "NRMSE" in df.columns
        assert "Δ NRMSE %" in df.columns
        assert "IAE" in df.columns
        assert "Δ IAE %" in df.columns

        # Check full model row
        full_row = df[df["Model"] == "Full Model"].iloc[0]
        assert full_row["NRMSE"] == 0.15
        assert full_row["Δ NRMSE %"] == 0.0  # Baseline
        assert full_row["IAE"] == 2.3
        assert full_row["Δ IAE %"] == 0.0

        # Check ablation rows
        no_world_row = df[df["Model"] == "No World ID"].iloc[0]
        assert no_world_row["NRMSE"] == 0.18
        assert abs(no_world_row["Δ NRMSE %"] - 20.0) < 0.1

        no_theta_row = df[df["Model"] == "No Theta"].iloc[0]
        assert no_theta_row["NRMSE"] == 0.22
        assert abs(no_theta_row["Δ NRMSE %"] - 46.67) < 0.1

        no_eps_row = df[df["Model"] == "No Eps"].iloc[0]
        assert no_eps_row["NRMSE"] == 0.16
        assert abs(no_eps_row["Δ NRMSE %"] - 6.67) < 0.1

    finally:
        for temp_file in temp_files:
            temp_file.unlink()


@pytest.mark.fast
def test_generate_table_sorted(sample_full_results, sample_ablation_results):
    """Test table sorting by metric."""
    summary = AblationSummary()

    # Create temporary files
    temp_files = []

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_full_results, f)
        full_path = Path(f.name)
        temp_files.append(full_path)

    summary.add_full_model(full_path)

    for name, results in sample_ablation_results.items():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(results, f)
            ablation_path = Path(f.name)
            temp_files.append(ablation_path)

        summary.add_ablation(name, ablation_path)

    try:
        df = summary.generate_table(sort_by="nrmse", ascending=True)

        # Full model should always be first
        assert df.iloc[0]["Model"] == "Full Model"

        # Ablations should be sorted by NRMSE (ascending)
        ablation_nrmse = df.iloc[1:]["NRMSE"].values
        # Check that each NRMSE is less than or equal to the next
        for i in range(len(ablation_nrmse) - 1):
            assert ablation_nrmse[i] <= ablation_nrmse[i + 1], f"Not sorted: {ablation_nrmse}"

    finally:
        for temp_file in temp_files:
            temp_file.unlink()


@pytest.mark.fast
def test_save_csv(sample_full_results, sample_ablation_results):
    """Test saving to CSV."""
    summary = AblationSummary()

    # Create temporary files for inputs
    temp_files = []

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_full_results, f)
        full_path = Path(f.name)
        temp_files.append(full_path)

    summary.add_full_model(full_path)

    for name, results in sample_ablation_results.items():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(results, f)
            ablation_path = Path(f.name)
            temp_files.append(ablation_path)

        summary.add_ablation(name, ablation_path)

    # Save to CSV
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        output_path = Path(f.name)
        temp_files.append(output_path)

    try:
        summary.save_csv(output_path)

        # Verify file exists and can be read
        assert output_path.exists()

        df = pd.read_csv(output_path)
        assert len(df) == 4
        assert "Model" in df.columns
        assert "NRMSE" in df.columns

    finally:
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()


@pytest.mark.fast
def test_save_markdown(sample_full_results, sample_ablation_results):
    """Test saving to Markdown."""
    summary = AblationSummary()

    # Create temporary files for inputs
    temp_files = []

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_full_results, f)
        full_path = Path(f.name)
        temp_files.append(full_path)

    summary.add_full_model(full_path)

    for name, results in sample_ablation_results.items():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(results, f)
            ablation_path = Path(f.name)
            temp_files.append(ablation_path)

        summary.add_ablation(name, ablation_path)

    # Save to Markdown
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        output_path = Path(f.name)
        temp_files.append(output_path)

    try:
        summary.save_markdown(output_path)

        # Verify file exists and contains expected content
        assert output_path.exists()

        with open(output_path) as f:
            content = f.read()

        assert "# Ablation Study Summary" in content
        assert "## Interpretation" in content
        assert "## Results" in content
        assert "## Metrics" in content
        assert "Full Model" in content
        assert "No World ID" in content
        assert "baseline" in content  # Baseline indicator for full model

    finally:
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()


@pytest.mark.fast
def test_load_ablation_results():
    """Test loading ablation results from directory."""
    # Create temporary directory with ablation files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create sample ablation files
        ablations = {
            "ablation_no_world.json": {"aggregate": {"nrmse": 0.18}},
            "ablation_no_theta.json": {"aggregate": {"nrmse": 0.22}},
            "ablation_no_eps.json": {"aggregate": {"nrmse": 0.16}},
        }

        for filename, data in ablations.items():
            with open(temp_path / filename, "w") as f:
                json.dump(data, f)

        # Load ablations
        results = load_ablation_results(temp_path)

        assert len(results) == 3
        assert "No World" in results
        assert "No Theta" in results
        assert "No Eps" in results


@pytest.mark.fast
def test_generate_ablation_summary(sample_full_results, sample_ablation_results):
    """Test high-level generate_ablation_summary function."""
    # Create temporary files
    temp_files = []

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_full_results, f)
        full_path = Path(f.name)
        temp_files.append(full_path)

    ablation_paths = {}
    for name, results in sample_ablation_results.items():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(results, f)
            ablation_path = Path(f.name)
            temp_files.append(ablation_path)
            ablation_paths[name] = ablation_path

    try:
        summary = generate_ablation_summary(full_path, ablation_paths)

        assert isinstance(summary, AblationSummary)
        assert summary.full_metrics is not None
        assert len(summary.ablation_metrics) == 3

        df = summary.generate_table()
        assert len(df) == 4

    finally:
        for temp_file in temp_files:
            temp_file.unlink()


@pytest.mark.fast
def test_error_handling_missing_file():
    """Test error handling for missing files."""
    summary = AblationSummary()

    # Missing full model file
    with pytest.raises(FileNotFoundError):
        summary.add_full_model("/nonexistent/path.json")

    # Missing ablation file
    with pytest.raises(FileNotFoundError):
        summary.add_ablation("Test", "/nonexistent/path.json")


@pytest.mark.fast
def test_error_handling_no_full_model():
    """Test error when generating table without full model."""
    summary = AblationSummary()

    # Create temp ablation file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"aggregate": {"nrmse": 0.18}}, f)
        temp_path = Path(f.name)

    try:
        summary.add_ablation("Test", temp_path)

        # Should raise error when trying to generate table without full model
        with pytest.raises(ValueError, match="Full model metrics not set"):
            summary.generate_table()

    finally:
        temp_path.unlink()


@pytest.mark.fast
def test_markdown_formatting(sample_full_results, sample_ablation_results):
    """Test Markdown formatting details."""
    summary = AblationSummary()

    # Create temporary files
    temp_files = []

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_full_results, f)
        full_path = Path(f.name)
        temp_files.append(full_path)

    summary.add_full_model(full_path)

    for name, results in sample_ablation_results.items():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(results, f)
            ablation_path = Path(f.name)
            temp_files.append(ablation_path)

        summary.add_ablation(name, ablation_path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        output_path = Path(f.name)
        temp_files.append(output_path)

    try:
        summary.save_markdown(output_path)

        with open(output_path) as f:
            content = f.read()

        # Check for positive delta formatting (with + sign)
        assert "+20" in content or "+6" in content or "+46" in content

        # Check for baseline formatting
        assert "baseline" in content

    finally:
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()


@pytest.mark.fast
def test_negative_delta():
    """Test handling of negative delta (ablation better than full)."""
    summary = AblationSummary()

    # Full model
    full_results = {
        "aggregate": {
            "nrmse": 0.20,
            "iae": 3.0,
            "sign_at_impact": 0.85,
            "hf_ratio": 0.10,
            "n_samples": 1000,
        }
    }

    # Ablation that's better (negative delta)
    ablation_results = {
        "Better Ablation": {
            "aggregate": {
                "nrmse": 0.16,  # 20% better
                "iae": 2.4,     # 20% better
                "sign_at_impact": 0.90,
                "hf_ratio": 0.08,
                "n_samples": 1000,
            }
        }
    }

    temp_files = []

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(full_results, f)
        full_path = Path(f.name)
        temp_files.append(full_path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(ablation_results["Better Ablation"], f)
        ablation_path = Path(f.name)
        temp_files.append(ablation_path)

    try:
        summary.add_full_model(full_path)
        summary.add_ablation("Better Ablation", ablation_path)

        df = summary.generate_table()

        ablation_row = df[df["Model"] == "Better Ablation"].iloc[0]
        assert ablation_row["Δ NRMSE %"] < 0  # Negative delta
        assert abs(ablation_row["Δ NRMSE %"] - (-20.0)) < 0.1

    finally:
        for temp_file in temp_files:
            temp_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
