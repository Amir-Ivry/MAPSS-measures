# MAPSS

[![CI](https://github.com/Amir-Ivry/MAPSS-measures/actions/workflows/ci.yml/badge.svg)](https://github.com/Amir-Ivry/MAPSS-measures/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mapss-measures.svg)](https://pypi.org/project/mapss-measures/)
[![Python](https://img.shields.io/pypi/pyversions/mapss-measures.svg)](https://pypi.org/project/mapss-measures/)
[![Paper](https://img.shields.io/badge/ICLR-2026-blue)](https://arxiv.org/abs/2509.09212)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MAPSS (**Manifold-based Assessment of Perceptual Source Separation**) is an ICLR 2026
metric package for source-separation systems. It reports two complementary frame-level
measures in `[0, 1]`, where higher is better:

- **Perceptual Separation (PS):** separation of an estimated source from competing references.
- **Perceptual Match (PM):** perceptual match between an estimated source and its assigned reference.

The public interface accepts `N` ordered reference waveforms and the corresponding `N`
ordered system outputs. It supports WAV paths, NumPy arrays, and PyTorch tensors.
WAV and FLAC paths are tested directly; other formats supported by the installed
`libsndfile` build may also be used.

## Install

MAPSS supports Python 3.10-3.12.

```bash
python -m pip install "mapss-measures==1.1.2"
```

Install the optional plotting command as well:

```bash
python -m pip install "mapss-measures[plot]==1.1.2"
```

The default pretrained backbone is downloaded from Hugging Face on first use. A CUDA GPU
is recommended; CPU execution is supported but slower.

## Python quick start

```python
from mapss import mapss

references = [
    "reference_source_1.wav",
    "reference_source_2.wav",
]
outputs = [
    "estimated_source_1.wav",
    "estimated_source_2.wav",
]

result = mapss(
    reference=references,
    output=outputs,
    source_names=["source_1", "source_2"],
    model="wav2vec2",
    layer=2,
    alpha=1.0,
    seed=42,
)

print(result.summary)
result.save("mapss_results", plot=True)
```

> **Input contract:** `output[i]` must estimate `reference[i]`. MAPSS requires at least
> two sources and does not silently find or change the source assignment.

Inputs are downmixed to mono and resampled to the paper's 16 kHz operating rate. By
default, unequal signal durations are rejected; use `length_policy="trim"` only when
shortest-length alignment is intentional.

## Command line

The same evaluation can be run without writing a Python script:

```bash
python -m mapss \
  --reference reference_1.wav reference_2.wav \
  --output estimate_1.wav estimate_2.wav \
  --source-name source_1 --source-name source_2 \
  --model wav2vec2 --layer 2 --alpha 1.0 --seed 42 \
  --results-dir mapss_results --plot
```

Windows PowerShell uses the same module command with backtick line continuations:

```powershell
python -m mapss `
  --reference "C:\data\reference_1.wav" "C:\data\reference_2.wav" `
  --output "C:\data\estimate_1.wav" "C:\data\estimate_2.wav" `
  --model wav2vec2 --layer 2 `
  --results-dir mapss_results --plot
```

Add more paths to both lists for mixtures with more sources. Keep the two lists the same
length and order.

## Results and plotting

`result.save(...)` and `--results-dir` write:

- `ps_scores.csv`: frame timestamps and PS for every source;
- `pm_scores.csv`: frame timestamps and PM for every source;
- `confidence.csv`: paper-derived error quantities when `add_ci=True`;
- `summary.csv`: convenient per-source means and valid-frame counts.

With `add_ci=True`, `result.save("mapss_results", plot=True)` writes the paper-style,
six-panel time-aligned figure automatically. The panels contain PM, PS, the deterministic
error radius for each measure, and the corresponding probabilistic 95% bounds.

To plot an existing result directory after installing the `plot` extra:

```bash
python -m mapss.plotting mapss_results
```

The figure is saved as `mapss_results/mapss_over_time.png`. If confidence data is absent,
the standalone plotting command remains backward compatible and creates a two-panel PS/PM
figure; `result.save(..., plot=True)` deliberately requires confidence data.

Inactive frames are stored as `NaN` and excluded from convenience means. Do not replace
them with zero. The paper's formal PS utterance aggregation is defined in Appendix B.4 and
is not a plain frame mean.

## Important keyword arguments

| Argument | Default | Meaning |
|---|---:|---|
| `model` | `"wav2vec2"` | Representation backbone; default is wav2vec 2.0 Large. |
| `layer` | paper default | Layer 2 for the default English configuration. |
| `alpha` | `1.0` | Diffusion-map density normalization in `[0, 1]`. |
| `add_ci` | `True` | Compute the paper-derived error components. |
| `seed` | `42` | Seed for MAPSS and the distortion bank. |
| `max_gpus` | all visible | Maximum GPUs; set `0` to force CPU. |
| `length_policy` | `"error"` | Reject unequal lengths; `"trim"` is explicit opt-in. |

Supported representations are `wav2vec2`, `wavlm`, `hubert`, their `_base` variants,
`wav2vec2_xlsr`, and `raw`. The `raw` model is only for fast installation tests, not
scientific reporting.

## Documentation

- [Clean-laptop quick start](https://github.com/Amir-Ivry/MAPSS-measures/blob/main/docs/QUICKSTART.md)
- [Input contract and supported audio](https://github.com/Amir-Ivry/MAPSS-measures/blob/main/docs/INPUTS.md)
- [CPU and CUDA setup](https://github.com/Amir-Ivry/MAPSS-measures/blob/main/docs/HARDWARE.md)
- [Python API reference](https://github.com/Amir-Ivry/MAPSS-measures/blob/main/docs/API.md)
- [Interpreting PS and PM](https://github.com/Amir-Ivry/MAPSS-measures/blob/main/docs/INTERPRETING_RESULTS.md)
- [Troubleshooting](https://github.com/Amir-Ivry/MAPSS-measures/blob/main/docs/TROUBLESHOOTING.md)
- [Reproducible evaluation](https://github.com/Amir-Ivry/MAPSS-measures/blob/main/docs/REPRODUCIBILITY.md)
- [Grand-challenge integration](https://github.com/Amir-Ivry/MAPSS-measures/blob/main/docs/CHALLENGE_GUIDE.md)
- [Examples](https://github.com/Amir-Ivry/MAPSS-measures/tree/main/examples)

## Citation

```bibtex
@inproceedings{ivry2026mapss,
  title     = {MAPSS: Manifold-based Assessment of Perceptual Source Separation},
  author    = {Ivry, Amir and Cornell, Samuele and Watanabe, Shinji},
  booktitle = {International Conference on Learning Representations},
  year      = {2026}
}
```

## License

MAPSS is released under the [MIT License](https://github.com/Amir-Ivry/MAPSS-measures/blob/main/LICENSE).
