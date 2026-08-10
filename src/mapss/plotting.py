"""Plot frame-level MAPSS scores saved by :class:`MAPSSResult`."""

from __future__ import annotations

import argparse
from os import PathLike
from pathlib import Path

import pandas as pd


def _load_score_table(path: Path, measure: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {measure} score table: {path}")

    table = pd.read_csv(path)
    if "timestamp_ms" not in table.columns:
        raise ValueError(f"{path} must contain a 'timestamp_ms' column.")
    if len(table.columns) < 2:
        raise ValueError(f"{path} does not contain any source score columns.")
    return table


def plot_results(
    results_dir: str | PathLike[str],
    output: str | PathLike[str] | None = None,
    *,
    dpi: int = 180,
) -> Path:
    """Plot saved PS and PM tables for every source.

    Parameters
    ----------
    results_dir:
        Directory created by ``MAPSSResult.save`` or ``mapss --results-dir``.
    output:
        Destination image. Defaults to ``ps_pm_over_time.png`` inside
        ``results_dir``.
    dpi:
        Output resolution in dots per inch.
    """
    if not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer.")

    directory = Path(results_dir).expanduser()
    ps = _load_score_table(directory / "ps_scores.csv", "PS")
    pm = _load_score_table(directory / "pm_scores.csv", "PM")

    ps_sources = [column for column in ps.columns if column != "timestamp_ms"]
    pm_sources = [column for column in pm.columns if column != "timestamp_ms"]
    if ps_sources != pm_sources:
        raise ValueError("PS and PM tables must contain the same ordered source columns.")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install it with "
            "'pip install mapss-measures[plot]'."
        ) from exc

    destination = (
        directory / "ps_pm_over_time.png"
        if output is None
        else Path(output).expanduser()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    for source in ps_sources:
        axes[0].plot(ps["timestamp_ms"] / 1000.0, ps[source], label=source, linewidth=1.2)
        axes[1].plot(pm["timestamp_ms"] / 1000.0, pm[source], label=source, linewidth=1.2)

    for axis, title, ylabel in (
        (axes[0], "Perceptual Separation over time", "PS"),
        (axes[1], "Perceptual Match over time", "PM"),
    ):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")

    axes[1].set_xlabel("Time (seconds)")
    figure.savefig(destination, dpi=dpi)
    plt.close(figure)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mapss-plot",
        description="Plot frame-level MAPSS PS and PM scores for every source.",
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing ps_scores.csv and pm_scores.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: RESULTS_DIR/ps_pm_over_time.png).",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        destination = plot_results(args.results_dir, args.output, dpi=args.dpi)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"MAPSS plotting error: {exc}") from exc
    print(f"Plot saved to: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
