"""Evaluate ordered music references and estimates with the unreleased MERT backend.

Run from an installed source checkout, for example:
python examples/music_mert.py --reference ref_1.wav ref_2.wav \
    --output est_1.wav est_2.wav --layer 1 --max-gpus 0 --plot
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", nargs="+", type=Path, required=True)
    parser.add_argument("--output", nargs="+", type=Path, required=True)
    parser.add_argument("--source-name", action="append", dest="source_names")
    parser.add_argument("--layer", type=int, choices=range(13), required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-gpus", type=int, default=0,
        help="0 uses CPU; a positive cap uses one visible CUDA device.",
    )
    parser.add_argument("--length-policy", choices=("error", "trim"), default="error")
    parser.add_argument("--results-dir", type=Path, default=Path("mert_results"))
    parser.add_argument("--no-ci", action="store_true", help="Skip confidence calculations.")
    parser.add_argument(
        "--plot", action="store_true",
        help="Save six panels with confidence, or two score panels with --no-ci.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    # Keep --help available without importing the inference stack or loading a model.
    from mapss import mapss

    result = mapss(
        reference=args.reference,
        output=args.output,
        source_names=args.source_names,
        model="mert",
        layer=args.layer,
        alpha=args.alpha,
        add_ci=not args.no_ci,
        seed=args.seed,
        max_gpus=args.max_gpus,
        length_policy=args.length_policy,
        verbose=args.verbose,
    )
    result.save(args.results_dir, plot=args.plot)
    print(result.summary.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"Results saved to: {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
