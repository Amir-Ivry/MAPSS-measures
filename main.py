"""
Entry point from the CLI into the MAPSS calculation.

IMPORTANT: Output files must be provided in the same order as reference files.
For example, if references are ["ref1.wav", "ref2.wav"],
then outputs must be ["out1.wav", "out2.wav"] in that exact order.
"""

from __future__ import annotations
from pathlib import Path
from engine import compute_mapss_measures
from argshield import _parse_args, _read_manifest, _validate_and_resolve, _validate_gpus

def main():
    args = _parse_args()

    manifest = _read_manifest(Path(args.manifest))
    layer_final, alpha_final = _validate_and_resolve(args.model, args.layer, args.alpha)
    max_gpus_final = _validate_gpus(args.max_gpus)

    results_dir = compute_mapss_measures(
        models=[args.model],
        mixtures=manifest,
        verbose=args.verbose,
        max_gpus=max_gpus_final,
        layer=layer_final,
        alpha=alpha_final,
    )
    print(f"Results saved to: {results_dir}")

if __name__ == "__main__":
    main()
