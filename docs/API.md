# Python API

## `mapss(reference, output, **kwargs)`

`from mapss import mapss` is the primary public interface. `evaluate` is an alias.

### Inputs

`reference` and `output` each describe the ordered sources of one mixture. Accepted forms:

- a sequence of audio paths;
- a sequence of one-dimensional NumPy arrays;
- a sequence of one-dimensional PyTorch tensors;
- a two-dimensional NumPy array or tensor shaped `(sources, samples)`.

Audio paths carry their own sample rates. `sample_rate` describes all in-memory arrays.
Stereo files are downmixed to mono. Signals are resampled to 16 kHz before evaluation,
or directly to 24 kHz when `model="mert"` is selected.
WAV and FLAC are tested end to end; see [INPUTS.md](INPUTS.md) for the complete contract.

MAPSS requires at least two sources. The number, order, and duration of references and
outputs must match. By default unequal durations raise an error. Set
`length_policy="trim"` only when deliberate shortest-length alignment is appropriate.
Inputs shorter than 400 ms are rejected because EBU R128 loudness normalization requires
that analysis duration.

### Parameters

```python
mapss(
    reference,
    output,
    *,
    sample_rate=16000,
    source_names=None,
    model="wav2vec2",
    layer=None,
    alpha=1.0,
    add_ci=True,
    seed=42,
    max_gpus=None,
    length_policy="error",
    verbose=False,
)
```

- `source_names`: unique labels used in returned tables.
- `model`: representation backbone. The default reproduces the paper's selected English case study.
- `layer`: transformer layer. `None` chooses the paper-aligned default for the selected model,
  except MERT, which requires an explicit hidden-state index in `0..12`.
- `alpha`: diffusion-map normalization parameter in `[0, 1]`.
- `add_ci`: computes the deterministic and high-probability error components, including
  for MERT. The default is `True`.
- `seed`: controls stochastic distortions and PyTorch/NumPy randomness.
- `max_gpus`: `None` uses all visible GPUs or CPU when none are available; `0` forces
  CPU; a positive value requires CUDA to be visible to PyTorch.
  MERT uses at most one CUDA device regardless of the positive cap.
- `length_policy`: `"error"` rejects unequal lengths; `"trim"` trims to the shortest.
- `verbose`: prints engine progress when true.

### Return value

`MAPSSResult` contains:

- `ps`: frame-level Perceptual Separation table;
- `pm`: frame-level Perceptual Match table;
- `ci`: frame-level error components when requested;
- `summary`: mean PS and PM plus valid-frame counts per source;
- `model`, `layer`, and `sample_rate`: resolved evaluation configuration.

`MAPSSResult.save(directory)` writes `ps_scores.csv`, `pm_scores.csv`,
`confidence.csv` when present, and `summary.csv`.

The MERT backend returns a `MAPSSResult` subclass and additionally saves
`mert_config.json`. Its scores use a 24 kHz evaluation rate, and `add_ci=True` returns
the usual confidence table. See [MERT.md](MERT.md) for the MERT-specific CI construction,
frame alignment, and reproducibility details.

### `MAPSSResult.save(..., plot=True)`

```python
result.save(
    "results/team_system_a",
    plot=True,
    plot_filename="mapss_over_time.png",
    plot_dpi=180,
)
```

`plot=True` adds the paper-style time-aligned diagnostic to the same run directory. It
contains six panels: PM, PS, PM deterministic error radius, PS deterministic error radius,
PM probabilistic 95% bound, and PS probabilistic 95% bound. Every source uses a consistent
color and line style across panels; inactive frames remain blank.

Automatic paper-style plotting requires `add_ci=True` and the optional `plot` dependency.
If either condition is missing, MAPSS raises a direct remediation message instead of
creating an incomplete figure. `plot_filename` must be a relative `.png`, `.pdf`, or `.svg`
path inside the result directory, and `plot_dpi` must be a positive integer.

MERT also supports `result.save(..., plot=True)` with `add_ci=False`, creating a
two-panel PS/PM score figure. With the default `add_ci=True`, it creates the six-panel
figure. The optional `plot` dependency is required in both cases.

### MERT music example

```python
result = mapss(
    reference=references,
    output=estimates,
    model="mert",
    layer=1,
    add_ci=True,
    max_gpus=0,
)
result.save("results/music_mert", plot=True)
```

For in-memory waveforms, pass their actual `sample_rate`. Layer `1` and layer `3` are
the paper's settings for music mixtures with and without drums, respectively; use one
layer consistently across all sources. This backend is available from version 1.2.0 and
requires PyTorch 2.6 or later via the `mert` extra. See [MERT.md](MERT.md) for installation.

### Exceptions and validation

MAPSS fails early with a descriptive exception for:

- fewer than two sources or unequal source counts, including both received counts;
- missing files, empty inputs, all-silent signals, NaN, or infinite samples;
- corrupt/unsupported audio, directory paths, or likely channel-first waveforms;
- duplicate source names;
- unequal durations under the default policy, including every converted duration;
- unsupported model/layer/alpha/GPU options;
- an explicit positive GPU request when CUDA is unavailable;
- any source with no valid PS or PM frame containing at least two active references;
- output scores outside the documented `[0, 1]` range.

## Reproducible evaluation example

```python
from mapss import mapss

result = mapss(
    reference=references,
    output=estimates,
    sample_rate=16_000,
    model="wav2vec2",
    layer=2,
    alpha=1.0,
    add_ci=True,
    seed=42,
    max_gpus=1,
)
result.save("results/team_system_a", plot=True)
```

Record all arguments and the installed `mapss-measures` version with any leaderboard result.

## Plotting saved results

Install the optional plotting dependency and plot all PS/PM source columns:

```bash
python -m pip install "mapss-measures[plot]==1.2.0"
python -m mapss.plotting results/team_system_a
```

The equivalent console command is `mapss-plot results/team_system_a`. When
`confidence.csv` exists, the plotting utility produces the same six-panel figure and writes
`mapss_over_time.png` by default. A legacy score-only directory produces a two-panel PS/PM
figure unless `--require-confidence` is supplied.
