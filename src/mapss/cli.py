"""Command-line interface for MAPSS."""

from __future__ import annotations

import argparse
from pathlib import Path

from ._cli_args import _read_manifest, _validate_and_resolve, _validate_gpus
from .api import mapss
from .engine import compute_mapss_measures


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mapss",
        description="Compute MAPSS Perceptual Separation and Perceptual Match.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest", type=Path, help="Legacy JSON or Python manifest.")
    mode.add_argument(
        "--reference",
        type=Path,
        nargs="+",
        help="Ordered reference source audio files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        nargs="+",
        help="Ordered system output files; required with --reference.",
    )
    parser.add_argument("--source-name", action="append", dest="source_names")
    parser.add_argument("--model", default="wav2vec2")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-gpus", type=int, default=None)
    parser.add_argument("--no-ci", action="store_true")
    parser.add_argument("--length-policy", choices=("error", "trim"), default="error")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    layer, alpha = _validate_and_resolve(args.model, args.layer, args.alpha)
    max_gpus = _validate_gpus(args.max_gpus)

    if args.manifest is not None:
        if args.output is not None:
            raise SystemExit("--output cannot be combined with --manifest.")
        experiment_path = compute_mapss_measures(
            models=[args.model],
            mixtures=_read_manifest(args.manifest),
            layer=layer,
            alpha=alpha,
            add_ci=not args.no_ci,
            seed=args.seed,
            max_gpus=max_gpus,
            verbose=args.verbose,
            results_root=args.results_dir,
        )
        print(f"Results saved to: {experiment_path}")
        return 0

    if not args.output:
        raise SystemExit("--output is required with --reference.")
    result = mapss(
        args.reference,
        args.output,
        source_names=args.source_names,
        model=args.model,
        layer=layer,
        alpha=alpha,
        add_ci=not args.no_ci,
        seed=args.seed,
        max_gpus=max_gpus,
        length_policy=args.length_policy,
        verbose=args.verbose,
    )
    destination = result.save(args.results_dir)
    print(result.summary.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"Results saved to: {destination}")
    return 0
