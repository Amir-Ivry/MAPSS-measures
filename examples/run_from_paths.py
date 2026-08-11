"""Run MAPSS on ordered reference/output audio paths.

Example
-------
python examples/run_from_paths.py \
    --reference reference_1.wav reference_2.wav \
    --output estimate_1.wav estimate_2.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mapss import mapss


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", nargs="+", type=Path, required=True)
    parser.add_argument("--output", nargs="+", type=Path, required=True)
    parser.add_argument("--source-name", action="append", dest="source_names")
    parser.add_argument("--model", default="wav2vec2")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-gpus", type=int, default=None)
    parser.add_argument("--no-ci", action="store_true")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save the paper-style PS/PM and confidence figure.",
    )
    parser.add_argument("--length-policy", choices=("error", "trim"), default="error")
    parser.add_argument("--results-dir", type=Path, default=Path("mapss_results"))
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.plot and args.no_ci:
        raise SystemExit("--plot cannot be combined with --no-ci.")
    result = mapss(
        reference=args.reference,
        output=args.output,
        source_names=args.source_names,
        model=args.model,
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
