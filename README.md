# MAPSS

[![CI](https://github.com/Amir-Ivry/MAPSS-measures/actions/workflows/ci.yml/badge.svg)](https://github.com/Amir-Ivry/MAPSS-measures/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mapss-measures.svg)](https://pypi.org/project/mapss-measures/)
[![Paper](https://img.shields.io/badge/ICLR-2026-blue)](https://arxiv.org/abs/2509.09212)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MAPSS (**Manifold-based Assessment of Perceptual Source Separation**) provides two
perceptually grounded source-separation measures:

- **Perceptual Separation (PS):** how well each output is separated from the other reference sources.
- **Perceptual Match (PM):** how closely each output matches its attributed reference source.

Both measures operate frame by frame in `[0, 1]`; higher is better. MAPSS builds a
perceptual manifold from self-supervised audio representations and controlled distortions.
It can also return the error quantities derived in the [ICLR 2026 paper](https://arxiv.org/abs/2509.09212).

## Install

From PyPI:

```bash
pip install mapss-measures==1.1.0
```

Directly from the current GitHub version:

```bash
pip install "git+https://github.com/Amir-Ivry/MAPSS-measures.git"
```

MAPSS supports Python 3.10-3.12. The default backbone is downloaded from Hugging Face
on first use. A CUDA GPU is recommended; CPU execution is supported but slower.

## Python quick start

```python
from mapss import mapss

result = mapss(
    reference=["reference_speaker_1.wav", "reference_speaker_2.wav"],
    output=["estimate_speaker_1.wav", "estimate_speaker_2.wav"],
)

print(result.summary)
print(result.ps)  # frame-level Perceptual Separation
print(result.pm)  # frame-level Perceptual Match
result.save("mapss_results")
```

Source order is meaningful: `output[i]` must estimate `reference[i]`. MAPSS evaluates
source separation, so at least two reference/output sources are required.

### In-memory waveforms

```python
import soundfile as sf
from mapss import mapss

ref_1, sr = sf.read("reference_1.wav")
ref_2, _ = sf.read("reference_2.wav")
out_1, _ = sf.read("output_1.wav")
out_2, _ = sf.read("output_2.wav")

result = mapss(
    reference=[ref_1, ref_2],
    output=[out_1, out_2],
    sample_rate=sr,
    source_names=["speaker_1", "speaker_2"],
    model="wav2vec2",
    layer=2,
    alpha=1.0,
)
```

Two-dimensional NumPy arrays or PyTorch tensors shaped `(sources, samples)` are also
accepted. Inputs are converted to mono and resampled to the paper's 16 kHz operating rate.

## What the result contains

```python
result.ps          # pandas.DataFrame: timestamp_ms + one column per source
result.pm          # pandas.DataFrame: timestamp_ms + one column per source
result.ci          # confidence/error components, or None when add_ci=False
result.summary     # mean PS/PM and valid-frame counts per source
result.source_names
```

Inactive frames are `NaN` and are excluded from the convenience means. For reported
challenge results, retain the frame tables and state the aggregation you use. The paper's
PM utterance score is a mean over active frames; its PS analysis uses the pooling procedure
defined in Appendix B.4 rather than a plain mean.

## Important keyword arguments

| Argument | Default | Meaning |
|---|---:|---|
| `model` | `"wav2vec2"` | Self-supervised representation; the default is wav2vec 2.0 Large. |
| `layer` | `2` | Paper-selected transformer layer for the default English setup. |
| `alpha` | `1.0` | Diffusion-map density normalization in `[0, 1]`. |
| `add_ci` | `True` | Compute the paper's deterministic/probabilistic error components. |
| `seed` | `42` | Seed used by MAPSS and its distortion bank. |
| `max_gpus` | all available | Maximum GPUs; use `0` to force CPU. |
| `length_policy` | `"error"` | Reject unequal lengths, or use `"trim"` explicitly. |

Supported models are `wav2vec2`, `wavlm`, `hubert`, their `_base` variants,
`wav2vec2_xlsr`, and `raw`. `raw` bypasses self-supervised encoding and is useful for
development smoke tests; it is not the paper's recommended reporting configuration.

## Command line

```bash
mapss \
  --reference reference_1.wav reference_2.wav \
  --output estimate_1.wav estimate_2.wav \
  --model wav2vec2 --layer 2 \
  --results-dir mapss_results
```

The historical manifest workflow remains supported:

```bash
mapss --manifest Manifests/example_English.json --model wav2vec2 --layer 2
```

## Before reporting a challenge result

1. Keep reference and output source order identical.
2. Report the package version, model, layer, `alpha`, seed, and aggregation.
3. Do not replace inactive-frame `NaN` values with zeros.
4. Use the same settings for every submitted system.
5. Cite the MAPSS paper and link this repository.

See the [challenge integration guide](docs/CHALLENGE_GUIDE.md), the complete
[Python API reference](docs/API.md), and [testing instructions](TESTING.md).

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

MIT. See [LICENSE](LICENSE).
