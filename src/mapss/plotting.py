"""Plot frame-level MAPSS scores and paper-derived confidence quantities."""

from __future__ import annotations

import argparse
from os import PathLike
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


_TIMESTAMP = "timestamp_ms"
_CONFIDENCE_SUFFIXES = ("pm_bias", "ps_bias", "pm_prob", "ps_prob")


def _load_table(path: Path, description: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description} table: {path}")
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read {description} table {path}: {exc}") from exc


def _validate_table(
    table: pd.DataFrame,
    description: str,
    *,
    value_columns: Sequence[str] | None = None,
    unit_interval: bool = False,
) -> pd.DataFrame:
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"{description} must be a pandas DataFrame.")
    if _TIMESTAMP not in table.columns:
        raise ValueError(f"{description} must contain a '{_TIMESTAMP}' column.")
    if table.empty:
        raise ValueError(f"{description} must contain at least one frame.")

    columns = (
        [column for column in table.columns if column != _TIMESTAMP]
        if value_columns is None
        else list(value_columns)
    )
    if not columns:
        raise ValueError(f"{description} does not contain any source columns.")
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(
            f"{description} is missing required columns: {', '.join(missing)}."
        )

    validated = table.copy()
    timestamp = pd.to_numeric(validated[_TIMESTAMP], errors="coerce")
    if timestamp.isna().any() or not np.isfinite(timestamp.to_numpy(dtype=float)).all():
        raise ValueError(f"{description} contains an invalid timestamp.")
    if len(timestamp) > 1 and np.any(np.diff(timestamp.to_numpy(dtype=float)) <= 0):
        raise ValueError(f"{description} timestamps must be strictly increasing.")
    validated[_TIMESTAMP] = timestamp

    for column in columns:
        original = validated[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & numeric.isna()
        if invalid.any():
            raise ValueError(
                f"{description} column {column!r} contains a non-numeric value."
            )
        finite = numeric.dropna().to_numpy(dtype=float)
        if finite.size == 0:
            raise ValueError(
                f"{description} column {column!r} does not contain any finite values."
            )
        if not np.isfinite(finite).all():
            raise ValueError(
                f"{description} column {column!r} contains an infinite value."
            )
        if unit_interval and np.any((finite < -1e-6) | (finite > 1.0 + 1e-6)):
            raise ValueError(
                f"{description} column {column!r} contains a value outside [0, 1]."
            )
        if not unit_interval and np.any(finite < -1e-12):
            raise ValueError(
                f"{description} column {column!r} contains a negative error bound."
            )
        validated[column] = numeric
    return validated


def _validate_plot_inputs(
    ps: pd.DataFrame,
    pm: pd.DataFrame,
    confidence: pd.DataFrame | None,
    *,
    require_confidence: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, list[str]]:
    ps = _validate_table(ps, "PS score table", unit_interval=True)
    pm = _validate_table(pm, "PM score table", unit_interval=True)

    ps_sources = [column for column in ps.columns if column != _TIMESTAMP]
    pm_sources = [column for column in pm.columns if column != _TIMESTAMP]
    if ps_sources != pm_sources:
        raise ValueError("PS and PM tables must contain the same ordered source columns.")
    if len(ps) != len(pm) or not np.allclose(
        ps[_TIMESTAMP].to_numpy(dtype=float),
        pm[_TIMESTAMP].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("PS and PM tables must contain the same frame timestamps.")

    if confidence is None:
        if require_confidence:
            raise ValueError(
                "Paper-style plotting requires confidence data. Run "
                "mapss(..., add_ci=True), then call result.save(..., plot=True)."
            )
        return ps, pm, None, ps_sources

    required = [
        f"{source}_{suffix}"
        for suffix in _CONFIDENCE_SUFFIXES
        for source in ps_sources
    ]
    confidence = _validate_table(
        confidence,
        "confidence table",
        value_columns=required,
        unit_interval=False,
    )
    confidence_values = confidence[required].to_numpy(dtype=float)
    finite_confidence = confidence_values[np.isfinite(confidence_values)]
    if np.any(finite_confidence > 1.0 + 1e-6):
        raise ValueError("Confidence error bounds must lie within [0, 1].")
    if len(confidence) != len(ps) or not np.allclose(
        confidence[_TIMESTAMP].to_numpy(dtype=float),
        ps[_TIMESTAMP].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "Confidence and score tables must contain the same frame timestamps."
        )
    return ps, pm, confidence, ps_sources


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install it with "
            "'python -m pip install mapss-measures[plot]'."
        ) from exc
    return plt


def plot_tables(
    ps: pd.DataFrame,
    pm: pd.DataFrame,
    confidence: pd.DataFrame | None,
    output: str | PathLike[str],
    *,
    dpi: int = 180,
    require_confidence: bool = False,
) -> Path:
    """Plot in-memory MAPSS tables, using the paper's six-panel layout when possible."""
    if not isinstance(dpi, int) or isinstance(dpi, bool) or dpi <= 0:
        raise ValueError("dpi must be a positive integer.")
    if not isinstance(require_confidence, bool):
        raise TypeError("require_confidence must be True or False.")

    ps, pm, confidence, sources = _validate_plot_inputs(
        ps, pm, confidence, require_confidence=require_confidence
    )
    destination = Path(output).expanduser()
    if destination.exists() and destination.is_dir():
        raise ValueError(f"Plot output must be a file path, not a directory: {destination}")
    if destination.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ValueError("Plot output must use a .png, .pdf, or .svg extension.")
    destination.parent.mkdir(parents=True, exist_ok=True)

    plt = _matplotlib()
    panels: list[tuple[str, pd.DataFrame, str | None, bool]] = [
        ("PM", pm, None, True),
        ("PS", ps, None, True),
    ]
    if confidence is not None:
        panels.extend(
            [
                ("PM deterministic error radius", confidence, "pm_bias", False),
                ("PS deterministic error radius", confidence, "ps_bias", False),
                ("PM probabilistic 95% bound", confidence, "pm_prob", False),
                ("PS probabilistic 95% bound", confidence, "ps_prob", False),
            ]
        )

    figure, axes = plt.subplots(
        len(panels),
        1,
        figsize=(11, 2.15 * len(panels) + 0.8),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    time_seconds = ps[_TIMESTAMP] / 1000.0
    line_styles = ("-", "--", "-.", ":")

    handles = []
    labels = []
    for axis, (title, table, suffix, unit_interval) in zip(axes, panels):
        for index, source in enumerate(sources):
            column = source if suffix is None else f"{source}_{suffix}"
            (line,) = axis.plot(
                time_seconds,
                table[column],
                label=source,
                linewidth=1.2,
                linestyle=line_styles[index % len(line_styles)],
            )
            if len(handles) < len(sources):
                handles.append(line)
                labels.append(source)
        axis.set_title(title, fontsize=10)
        axis.set_ylabel(title.split()[0])
        axis.set_ylim((-0.02, 1.02) if unit_interval else (0.0, None))
        axis.grid(alpha=0.22, linewidth=0.6)

    axes[-1].set_xlabel("Time (seconds)")
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=min(4, len(sources)),
        frameon=False,
    )
    figure.savefig(destination, dpi=dpi)
    plt.close(figure)
    return destination


def plot_results(
    results_dir: str | PathLike[str],
    output: str | PathLike[str] | None = None,
    *,
    dpi: int = 180,
    require_confidence: bool = False,
) -> Path:
    """Plot MAPSS tables saved by :meth:`MAPSSResult.save`.

    When ``confidence.csv`` exists, the output follows the paper's six-panel Figure 11
    structure. Older or score-only result directories remain supported with two panels.
    Set ``require_confidence=True`` to reject score-only results.
    """
    directory = Path(results_dir).expanduser()
    if not directory.exists():
        raise FileNotFoundError(f"MAPSS results directory does not exist: {directory}")
    if not directory.is_dir():
        raise ValueError(f"MAPSS results path must be a directory: {directory}")

    ps = _load_table(directory / "ps_scores.csv", "PS score")
    pm = _load_table(directory / "pm_scores.csv", "PM score")
    confidence_path = directory / "confidence.csv"
    confidence = (
        _load_table(confidence_path, "confidence")
        if confidence_path.is_file()
        else None
    )
    destination = (
        directory / "mapss_over_time.png"
        if output is None
        else Path(output).expanduser()
    )
    return plot_tables(
        ps,
        pm,
        confidence,
        destination,
        dpi=dpi,
        require_confidence=require_confidence,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mapss-plot",
        description=(
            "Plot frame-level MAPSS scores and, when available, the paper-derived "
            "confidence quantities for every source."
        ),
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing MAPSS CSV result tables.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: RESULTS_DIR/mapss_over_time.png).",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--require-confidence",
        action="store_true",
        help="Fail instead of creating a score-only plot when confidence.csv is absent.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        destination = plot_results(
            args.results_dir,
            args.output,
            dpi=args.dpi,
            require_confidence=args.require_confidence,
        )
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"MAPSS plotting error: {exc}") from exc
    print(f"Plot saved to: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
