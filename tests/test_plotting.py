from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mapss.plotting import main, plot_results


def _write_scores(directory: Path) -> None:
    directory.mkdir()
    frame = pd.DataFrame(
        {
            "timestamp_ms": [0.0, 20.0, 40.0],
            "source_1": [0.8, 0.9, float("nan")],
            "source_2": [0.7, 0.75, 0.8],
        }
    )
    frame.to_csv(directory / "ps_scores.csv", index=False)
    frame.to_csv(directory / "pm_scores.csv", index=False)


def test_plot_results_writes_png(tmp_path):
    results = tmp_path / "results"
    _write_scores(results)

    output = plot_results(results, dpi=72)

    assert output == results / "ps_pm_over_time.png"
    assert output.is_file()
    assert output.stat().st_size > 0


def test_plot_cli_accepts_custom_output(tmp_path):
    results = tmp_path / "results"
    _write_scores(results)
    output = tmp_path / "figures" / "scores.png"

    assert main([str(results), "--output", str(output), "--dpi", "72"]) == 0
    assert output.is_file()


def test_plot_rejects_mismatched_sources(tmp_path):
    results = tmp_path / "results"
    _write_scores(results)
    pm = pd.read_csv(results / "pm_scores.csv").rename(columns={"source_2": "other"})
    pm.to_csv(results / "pm_scores.csv", index=False)

    with pytest.raises(ValueError, match="same ordered source columns"):
        plot_results(results)


def test_plot_reports_missing_table(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing PS score table"):
        plot_results(tmp_path)
