from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mapss.plotting import main, plot_results, plot_tables


def _write_scores(directory: Path, *, confidence: bool = False) -> None:
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
    if confidence:
        ci = pd.DataFrame({"timestamp_ms": frame["timestamp_ms"]})
        for source in ("source_1", "source_2"):
            ci[f"{source}_ps_bias"] = [0.01, 0.02, float("nan")]
            ci[f"{source}_ps_prob"] = [0.2, 0.3, float("nan")]
            ci[f"{source}_pm_bias"] = [0.005, 0.006, float("nan")]
            ci[f"{source}_pm_prob"] = [0.1, 0.15, float("nan")]
        ci.to_csv(directory / "confidence.csv", index=False)


def test_plot_results_writes_png(tmp_path):
    results = tmp_path / "results"
    _write_scores(results)

    output = plot_results(results, dpi=72)

    assert output == results / "mapss_over_time.png"
    assert output.is_file()
    assert output.stat().st_size > 0


def test_plot_cli_accepts_custom_output(tmp_path):
    results = tmp_path / "results"
    _write_scores(results)
    output = tmp_path / "figures" / "scores.png"

    assert main([str(results), "--output", str(output), "--dpi", "72"]) == 0
    assert output.is_file()


def test_plot_results_writes_paper_style_figure_with_confidence(tmp_path):
    results = tmp_path / "results"
    _write_scores(results, confidence=True)

    output = plot_results(results, dpi=72, require_confidence=True)

    assert output == results / "mapss_over_time.png"
    assert output.is_file()
    assert output.stat().st_size > 0


def test_plot_requires_confidence_when_requested(tmp_path):
    results = tmp_path / "results"
    _write_scores(results)

    with pytest.raises(ValueError, match="requires confidence data"):
        plot_results(results, require_confidence=True)


def test_plot_rejects_incomplete_confidence_table(tmp_path):
    results = tmp_path / "results"
    _write_scores(results, confidence=True)
    ci = pd.read_csv(results / "confidence.csv").drop(columns="source_2_pm_prob")
    ci.to_csv(results / "confidence.csv", index=False)

    with pytest.raises(ValueError, match="missing required columns.*source_2_pm_prob"):
        plot_results(results, require_confidence=True)


def test_plot_rejects_misaligned_confidence_timestamps(tmp_path):
    results = tmp_path / "results"
    _write_scores(results, confidence=True)
    ci = pd.read_csv(results / "confidence.csv")
    ci.loc[1, "timestamp_ms"] = 21.0
    ci.to_csv(results / "confidence.csv", index=False)

    with pytest.raises(ValueError, match="same frame timestamps"):
        plot_results(results, require_confidence=True)


def test_plot_rejects_duplicate_timestamps(tmp_path):
    results = tmp_path / "results"
    _write_scores(results)
    ps = pd.read_csv(results / "ps_scores.csv")
    ps.loc[1, "timestamp_ms"] = ps.loc[0, "timestamp_ms"]
    ps.to_csv(results / "ps_scores.csv", index=False)

    with pytest.raises(ValueError, match="strictly increasing"):
        plot_results(results)


def test_plot_rejects_source_with_no_finite_scores(tmp_path):
    results = tmp_path / "results"
    _write_scores(results)
    ps = pd.read_csv(results / "ps_scores.csv")
    ps["source_2"] = float("nan")
    ps.to_csv(results / "ps_scores.csv", index=False)

    with pytest.raises(ValueError, match="source_2.*finite values"):
        plot_results(results)


def test_plot_rejects_negative_confidence_bound(tmp_path):
    results = tmp_path / "results"
    _write_scores(results, confidence=True)
    ci = pd.read_csv(results / "confidence.csv")
    ci.loc[0, "source_1_ps_bias"] = -0.1
    ci.to_csv(results / "confidence.csv", index=False)

    with pytest.raises(ValueError, match="negative error bound"):
        plot_results(results, require_confidence=True)


def test_plot_rejects_confidence_bound_above_one(tmp_path):
    results = tmp_path / "results"
    _write_scores(results, confidence=True)
    ci = pd.read_csv(results / "confidence.csv")
    ci.loc[0, "source_1_ps_prob"] = 1.1
    ci.to_csv(results / "confidence.csv", index=False)

    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        plot_results(results, require_confidence=True)


def test_plot_tables_rejects_unsupported_output_extension(tmp_path):
    results = tmp_path / "results"
    _write_scores(results)
    ps = pd.read_csv(results / "ps_scores.csv")
    pm = pd.read_csv(results / "pm_scores.csv")

    with pytest.raises(ValueError, match=".png, .pdf, or .svg"):
        plot_tables(ps, pm, None, tmp_path / "plot.txt")


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
