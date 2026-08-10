"""Run a fast, self-contained MAPSS installation smoke test.

This example uses the raw-waveform development model to avoid downloading a
pretrained checkpoint. Do not use raw-model scores for scientific reporting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from mapss import mapss


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mapss_smoke_results"),
        help="Directory for generated WAV files and MAPSS results.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = args.output_dir
    audio_dir = output_dir / "audio"
    score_dir = output_dir / "scores"
    audio_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 16_000
    time = np.arange(2 * sample_rate, dtype=np.float32) / sample_rate
    envelope = 0.65 + 0.35 * np.sin(2 * np.pi * 1.3 * time) ** 2
    reference_1 = 0.2 * envelope * np.sin(2 * np.pi * 220 * time)
    reference_2 = 0.2 * envelope * np.sin(2 * np.pi * 347 * time + 0.4)
    output_1 = 0.96 * reference_1 + 0.04 * reference_2
    output_2 = 0.96 * reference_2 + 0.04 * reference_1

    references = [audio_dir / "reference_1.wav", audio_dir / "reference_2.wav"]
    outputs = [audio_dir / "output_1.wav", audio_dir / "output_2.wav"]
    for path, waveform in zip(
        references + outputs,
        [reference_1, reference_2, output_1, output_2],
    ):
        sf.write(path, waveform, sample_rate)

    result = mapss(
        reference=references,
        output=outputs,
        source_names=["source_1", "source_2"],
        model="raw",
        layer=0,
        add_ci=False,
        seed=42,
        max_gpus=0,
    )
    result.save(score_dir)

    print("MAPSS installation smoke test passed.")
    print(result.summary.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"Results saved to: {score_dir}")
    print("Note: model='raw' is for installation checks only, not scientific reporting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
