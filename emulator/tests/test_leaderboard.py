"""
Unit tests for leaderboard generator.

Tests the LeaderboardGenerator class and comparison utilities.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from emulator.eval.leaderboard import (
    LeaderboardGenerator,
    compare_models,
    compute_success_criteria,
)


@pytest.mark.fast
def test_leaderboard_generator_empty():
    """Test that empty leaderboard returns empty DataFrame."""
    leaderboard = LeaderboardGenerator()

    df = leaderboard.generate_table()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


@pytest.mark.fast
def test_leaderboard_add_model_from_dict():
    """Test adding model from metrics dictionary."""
    leaderboard = LeaderboardGenerator()

    metrics = {
        "nrmse": 0.15,
        "iae": 2.3,
        "sign_at_impact": 0.95,
        "hf_ratio": 0.12,
        "overshoot_ratio": 1.5,
        "sign_flip_count": 3.2,
        "n_samples": 100,
    }

    leaderboard.add_model_from_dict("Test Model", metrics)

    df = leaderboard.generate_table()

    assert len(df) == 1
    assert df.loc[0, "Model"] == "Test Model"
    assert df.loc[0, "NRMSE"] == 0.15
    assert df.loc[0, "IAE"] == 2.3
    assert df.loc[0, "Samples"] == 100


@pytest.mark.fast
def test_leaderboard_add_multiple_models():
    """Test adding multiple models."""
    leaderboard = LeaderboardGenerator()

    metrics_1 = {
        "nrmse": 0.15,
        "iae": 2.3,
        "sign_at_impact": 0.95,
        "hf_ratio": 0.12,
        "overshoot_ratio": 1.5,
        "sign_flip_count": 3.2,
        "n_samples": 100,
    }

    metrics_2 = {
        "nrmse": 0.20,
        "iae": 3.5,
        "sign_at_impact": 0.90,
        "hf_ratio": 0.18,
        "overshoot_ratio": 1.8,
        "sign_flip_count": 4.5,
        "n_samples": 150,
    }

    leaderboard.add_model_from_dict("Model A", metrics_1)
    leaderboard.add_model_from_dict("Model B", metrics_2)

    df = leaderboard.generate_table()

    assert len(df) == 2
    assert "Model A" in df["Model"].values
    assert "Model B" in df["Model"].values


@pytest.mark.fast
def test_leaderboard_sorting():
    """Test that leaderboard sorts correctly by metric."""
    leaderboard = LeaderboardGenerator()

    # Add models with different NRMSEs
    for i, nrmse in enumerate([0.25, 0.10, 0.20, 0.15]):
        metrics = {
            "nrmse": nrmse,
            "iae": 2.0,
            "sign_at_impact": 0.9,
            "hf_ratio": 0.1,
            "overshoot_ratio": 1.2,
            "sign_flip_count": 3.0,
            "n_samples": 100,
        }
        leaderboard.add_model_from_dict(f"Model {i}", metrics)

    df = leaderboard.generate_table(sort_by="NRMSE", ascending=True)

    # Should be sorted ascending by NRMSE
    nrmse_values = df["NRMSE"].tolist()
    assert nrmse_values == [0.10, 0.15, 0.20, 0.25]


@pytest.mark.fast
def test_leaderboard_gap_computation():
    """Test gap computation vs oracle."""
    leaderboard = LeaderboardGenerator()

    # Add oracle
    oracle_metrics = {
        "nrmse": 0.10,
        "iae": 2.0,
        "sign_at_impact": 0.95,
        "hf_ratio": 0.10,
        "overshoot_ratio": 1.2,
        "sign_flip_count": 2.5,
        "n_samples": 100,
    }
    leaderboard.add_model_from_dict("Oracle", oracle_metrics, is_oracle=True)

    # Add model with 20% higher NRMSE
    model_metrics = {
        "nrmse": 0.12,
        "iae": 2.5,
        "sign_at_impact": 0.90,
        "hf_ratio": 0.12,
        "overshoot_ratio": 1.5,
        "sign_flip_count": 3.0,
        "n_samples": 100,
    }
    leaderboard.add_model_from_dict("Test Model", model_metrics)

    # Compute gap
    gap = leaderboard.compute_gap(0.12)

    # Gap = (0.12 - 0.10) / 0.10 * 100 = 20%
    assert gap == pytest.approx(20.0)


@pytest.mark.fast
def test_leaderboard_gap_in_table():
    """Test that gap column appears in table."""
    leaderboard = LeaderboardGenerator()

    oracle_metrics = {
        "nrmse": 0.10,
        "iae": 2.0,
        "sign_at_impact": 0.95,
        "hf_ratio": 0.10,
        "overshoot_ratio": 1.2,
        "sign_flip_count": 2.5,
        "n_samples": 100,
    }
    leaderboard.add_model_from_dict("Oracle", oracle_metrics, is_oracle=True)

    model_metrics = {
        "nrmse": 0.12,
        "iae": 2.5,
        "sign_at_impact": 0.90,
        "hf_ratio": 0.12,
        "overshoot_ratio": 1.5,
        "sign_flip_count": 3.0,
        "n_samples": 100,
    }
    leaderboard.add_model_from_dict("Test Model", model_metrics)

    df = leaderboard.generate_table(include_gap=True)

    assert "Gap (%)" in df.columns
    # Oracle should have 0% gap
    oracle_row = df[df["Model"] == "Oracle"].iloc[0]
    assert oracle_row["Gap (%)"] == pytest.approx(0.0)


@pytest.mark.fast
def test_leaderboard_save_csv():
    """Test saving leaderboard to CSV."""
    leaderboard = LeaderboardGenerator()

    metrics = {
        "nrmse": 0.15,
        "iae": 2.3,
        "sign_at_impact": 0.95,
        "hf_ratio": 0.12,
        "overshoot_ratio": 1.5,
        "sign_flip_count": 3.2,
        "n_samples": 100,
    }
    leaderboard.add_model_from_dict("Test Model", metrics)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "leaderboard.csv"

        leaderboard.save_csv(output_path)

        assert output_path.exists()

        # Load and verify
        df = pd.read_csv(output_path)
        assert len(df) == 1
        assert df.loc[0, "Model"] == "Test Model"


@pytest.mark.fast
def test_leaderboard_save_markdown():
    """Test saving leaderboard to Markdown."""
    leaderboard = LeaderboardGenerator()

    metrics = {
        "nrmse": 0.15,
        "iae": 2.3,
        "sign_at_impact": 0.95,
        "hf_ratio": 0.12,
        "overshoot_ratio": 1.5,
        "sign_flip_count": 3.2,
        "n_samples": 100,
    }
    leaderboard.add_model_from_dict("Test Model", metrics)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "leaderboard.md"

        leaderboard.save_markdown(output_path)

        assert output_path.exists()

        # Load and verify contains table
        with open(output_path) as f:
            content = f.read()

        assert "# Model Leaderboard" in content
        assert "Test Model" in content


@pytest.mark.fast
def test_leaderboard_per_world_table():
    """Test per-world breakdown table."""
    leaderboard = LeaderboardGenerator()

    metrics_with_per_world = {
        "nrmse": 0.15,
        "iae": 2.3,
        "sign_at_impact": 0.95,
        "hf_ratio": 0.12,
        "overshoot_ratio": 1.5,
        "sign_flip_count": 3.2,
        "n_samples": 100,
        "per_world": {
            "lss": {"nrmse": 0.12, "iae": 2.0},
            "var": {"nrmse": 0.18, "iae": 2.6},
        },
    }

    leaderboard.add_model_from_dict("Test Model", metrics_with_per_world)

    df = leaderboard.generate_per_world_table(metric="nrmse")

    assert "Model" in df.columns
    assert "lss" in df.columns
    assert "var" in df.columns
    assert df.loc[0, "lss"] == 0.12
    assert df.loc[0, "var"] == 0.18


@pytest.mark.fast
def test_leaderboard_add_model_from_file():
    """Test adding model from JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock results file
        results = {
            "aggregate": {
                "nrmse": 0.15,
                "iae": 2.3,
                "sign_at_impact": 0.95,
                "hf_ratio": 0.12,
                "overshoot_ratio": 1.5,
                "sign_flip_count": 3.2,
                "n_samples": 100,
            }
        }

        results_path = Path(tmpdir) / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f)

        # Add to leaderboard
        leaderboard = LeaderboardGenerator()
        leaderboard.add_model("Test Model", results_path)

        df = leaderboard.generate_table()

        assert len(df) == 1
        assert df.loc[0, "NRMSE"] == 0.15


@pytest.mark.fast
def test_compare_models_empty_directory():
    """Test compare_models with empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        df = compare_models(tmpdir, pattern="*.json")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


@pytest.mark.fast
def test_compare_models_multiple_files():
    """Test compare_models with multiple result files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple result files
        for i in range(3):
            results = {
                "aggregate": {
                    "nrmse": 0.10 + i * 0.05,
                    "iae": 2.0 + i * 0.5,
                    "sign_at_impact": 0.95 - i * 0.02,
                    "hf_ratio": 0.12,
                    "overshoot_ratio": 1.5,
                    "sign_flip_count": 3.0,
                    "n_samples": 100,
                }
            }

            results_path = Path(tmpdir) / f"model_{i}.json"
            with open(results_path, "w") as f:
                json.dump(results, f)

        # Compare
        df = compare_models(tmpdir, pattern="*.json")

        assert len(df) == 3
        assert "model_0" in df["Model"].values
        assert "model_1" in df["Model"].values
        assert "model_2" in df["Model"].values


@pytest.mark.fast
def test_compute_success_criteria():
    """Test success criteria computation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create universal model results
        universal_results = {
            "aggregate": {"nrmse": 0.15},
            "per_world": {
                "lss": {"nrmse": 0.12, "hf_ratio": 0.11},
                "var": {"nrmse": 0.18, "hf_ratio": 0.13},
            },
        }
        universal_path = Path(tmpdir) / "universal.json"
        with open(universal_path, "w") as f:
            json.dump(universal_results, f)

        # Create specialist results
        specialist_lss = {
            "aggregate": {"nrmse": 0.10, "hf_ratio": 0.10},
        }
        specialist_lss_path = Path(tmpdir) / "specialist_lss.json"
        with open(specialist_lss_path, "w") as f:
            json.dump(specialist_lss, f)

        specialist_var = {
            "aggregate": {"nrmse": 0.15, "hf_ratio": 0.12},
        }
        specialist_var_path = Path(tmpdir) / "specialist_var.json"
        with open(specialist_var_path, "w") as f:
            json.dump(specialist_var, f)

        # Create baseline results
        baseline_mlp = {
            "per_world": {
                "lss": {"nrmse": 0.20},
                "var": {"nrmse": 0.25},
            },
        }
        baseline_mlp_path = Path(tmpdir) / "baseline_mlp.json"
        with open(baseline_mlp_path, "w") as f:
            json.dump(baseline_mlp, f)

        # Compute criteria
        criteria = compute_success_criteria(
            universal_path,
            specialist_results={
                "lss": specialist_lss_path,
                "var": specialist_var_path,
            },
            baseline_results={
                "mlp": baseline_mlp_path,
            },
        )

        # Check results
        assert "mean_gap" in criteria
        assert "max_gap" in criteria
        assert "mean_hf_ratio" in criteria
        assert "beat_baselines" in criteria
        assert "all_criteria_pass" in criteria

        # Should beat baseline (0.12 < 0.20 and 0.18 < 0.25)
        assert criteria["beat_baselines"]["mlp"] is True

        # Gap: LSS: (0.12-0.10)/0.10 = 20%, VAR: (0.18-0.15)/0.15 = 20%
        assert criteria["mean_gap"] == pytest.approx(20.0)
        assert criteria["max_gap"] == pytest.approx(20.0)


@pytest.mark.fast
def test_leaderboard_no_gap_without_oracle():
    """Test that gap is NaN when no oracle is provided."""
    leaderboard = LeaderboardGenerator()

    metrics = {
        "nrmse": 0.15,
        "iae": 2.3,
        "sign_at_impact": 0.95,
        "hf_ratio": 0.12,
        "overshoot_ratio": 1.5,
        "sign_flip_count": 3.2,
        "n_samples": 100,
    }
    leaderboard.add_model_from_dict("Test Model", metrics)

    gap = leaderboard.compute_gap(0.15)

    assert np.isnan(gap)
