# MERT for music evaluation (unreleased)

The source checkout adds `model="mert"` using
[`m-a-p/MERT-v1-95M`](https://huggingface.co/m-a-p/MERT-v1-95M). This feature is not in
the published `mapss-measures==1.1.2` package. Install this checkout from its repository
root, using a Python 3.10–3.12 environment with PyTorch 2.6 or later:

```bash
python -m pip install -e ".[plot]"
```

The PyTorch 2.6 minimum applies to MERT's upstream `.bin` checkpoint; requirements for
the other backbones are unchanged. Keep a compatible PyTorch/CUDA installation for your
machine. The first run downloads the checkpoint from Hugging Face.

## Python example

```python
from mapss import mapss

result = mapss(
    reference=["drums_ref.wav", "bass_ref.wav", "other_ref.wav"],
    output=["drums_est.wav", "bass_est.wav", "other_est.wav"],
    source_names=["drums", "bass", "other"],
    model="mert",
    layer=1,
    alpha=1.0,
    add_ci=True,
    seed=42,
    max_gpus=0,
)
result.save("mert_results", plot=True)
print(result.summary)
```

Provide at least two aligned, ordered source pairs from the same mixture. The example
names are placeholders; an estimate must correspond to the reference at the same index.
Use `max_gpus=0` for CPU or `max_gpus=1` for one visible CUDA GPU.

For NumPy arrays or tensors, also set `sample_rate` to the actual input sample rate,
for example `sample_rate=44_100`. This argument describes the input; the MERT evaluation
rate is always 24 kHz. File inputs carry their own sample rates.

## Command line

```bash
python -m mapss \
  --reference drums_ref.wav bass_ref.wav other_ref.wav \
  --output drums_est.wav bass_est.wav other_est.wav \
  --source-name drums --source-name bass --source-name other \
  --model mert --layer 1 \
  --alpha 1.0 --seed 42 --max-gpus 0 \
  --results-dir mert_results --plot
```

[`examples/music_mert.py`](../examples/music_mert.py) provides the same explicit music
configuration. Its `--help` command does not load a model.

## Required configuration

- **Choose a layer explicitly:** supported hidden-state indices are `0` through `12`.
  Index `0` is the representation before the transformer blocks; `1` and `3` are the
  outputs of the first and third blocks. No layer is selected automatically for MERT.
- **Confidence calculations are enabled by default:** `add_ci=True` returns the
  deterministic and probabilistic error components for PS and PM. Use `add_ci=False`
  or CLI `--no-ci` when only point scores are needed.
- **Use one layer for the whole evaluation:** the paper uses layer `1` for music mixtures
  with drums and layer `3` for mixtures without drums. These are mixture-level settings;
  they do not assign different layers to different stems in one graph. A new music
  dataset still requires its own validation.

The explicit-layer check applies to `mapss()` and both CLI modes. If calling the legacy
`mapss.engine.compute_mapss_measures()` helper directly, always pass `layer=1` or your
chosen layer: its existing `layer=2` default is preserved for compatibility. Run MERT
separately from other backbones in manifest evaluations.

## Sampling and frame alignment

Inputs are downmixed to mono and resampled directly from their input rate to **24 kHz**.
References, estimates, and distortion generation all use that rate. EBU R128 loudness
normalization targets −23 LUFS with peak protection, as in the existing pipeline.

The historical music script and the checkpoint use different temporal grids:

| Grid | Window / receptive field | Hop | Use |
|---|---:|---:|---|
| Music activity and reported scores | 480 samples / 20 ms | 324 samples / 13.5 ms | Activity decisions and output timestamps |
| Native MERT convolution features | 400 samples / 16.67 ms | 320 samples / 13.33 ms | Feature extraction before the transformer |

The implementation matches activity-frame centers to native feature-frame centers
before applying the common activity mask. This replaces the historical script's
interpolation and sequential assignment of compacted frames, which could associate
scores with the wrong times. At the signal boundary, a mapping beyond the available
native grid uses the nearest available feature frame; metadata records boundary clipping
and the actual center error. The `L=324` music figure in the paper corresponds to the
historical activity-grid hop; it is not MERT's native receptive field or stride.
Transformer features also depend on context beyond the convolution receptive field.

Waveforms are encoded with their full supplied context. There is no automatic chunking
of long tracks. Splitting a track changes its model context and must be recorded as an
evaluation choice. Memory use grows with duration, source count, and the distortion bank.
Full distortion-bank evaluation can take several minutes on CPU even for short excerpts.

## Diffusion maps and scores

Each time frame with at least two active references is evaluated jointly across its
active sources. For each metric pass, the graph includes their references, distorted
references, and estimates. Estimates are excluded from the reference-cluster statistics.
PM uses the attributed source's statistics within that joint embedding; it does not fit
a separate diffusion map for each source.

PS and PM retain the package's separate distortion banks and separate graph passes.
Both use the same MERT checkpoint and selected layer for every waveform in a pass.
The existing diffusion-map and score functions are reused without global changes.

## Confidence intervals

With `add_ci=True`, MERT uses the retained and complementary coordinates from each
metric's joint diffusion map to calculate the paper-derived error components. The
confidence table has `timestamp_ms` and four columns per source:
`{source}_ps_bias`, `{source}_ps_prob`, `{source}_pm_bias`, and `{source}_pm_prob`.
The `bias` columns describe deterministic truncation error; the `prob` columns describe
finite-sample probabilistic error. Inactive entries remain `NaN`, aligned with the scores.

For a point score `s`, combine its two components as
`[max(0, s - bias - prob), min(1, s + bias + prob)]`. The `prob` component alone does
not include truncation error.

These calculations use MERT-only helpers. PM's covariance blocks use the same
reference-centered moments as its point score, and the helpers respect the selected CPU
or CUDA device. They leave the existing backbones' CI helpers unchanged. If there are no
discarded coordinates, the truncation component is zero; finite-sample uncertainty can
still be nonzero. Error radii are bounded by one, the width of the score range.

The nominal setting is `delta=0.05`, using paper-derived practical plug-in CI components
that follow the original music implementation's conventions. PM retains the 50% local
parameter-perturbation cap. PS retains its effective-count factor of 0.7, 5% eigenvalue
floor, and Euclidean combination of error terms. Their interpretation depends on the
paper's stated assumptions; numerical checks do not establish empirical 95% coverage
on a new dataset. These frame-level quantities also are not an utterance-level confidence
interval for the convenience mean in `result.summary`.

## Results and execution

The result is a `MAPSSResult` subclass with the usual `ps`, `pm`, `ci`, and `summary`
tables. With `add_ci=False`, `ci` is `None`. Saving writes `ps_scores.csv`,
`pm_scores.csv`, `summary.csv`, `mert_config.json`, and `confidence.csv` when CIs are
enabled. Manifest evaluations use the established `ci_mert.csv` layout. The metadata
records the MERT evaluation configuration and should be kept with the scores.

For MERT, `result.save(..., plot=True)` and CLI `--plot` create the **six-panel figure**
when CIs are enabled. With `add_ci=False` / `--no-ci`, they create a **two-panel PS/PM
score figure**. The plotting behavior of other backbones remains unchanged. Inactive
scores remain `NaN` and are excluded from the convenience frame means.

MERT model inference uses float32 on CPU or one CUDA device. `max_gpus=None` selects one visible CUDA
device when available, otherwise CPU; `0` forces CPU. A positive cap requires CUDA, but
MERT still uses only one device even if the cap is greater than one.

The checkpoint and custom model code are pinned to revision
`12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`. Loading uses the checkpoint's custom code
(`trust_remote_code=True`), eager attention, and its upstream `pytorch_model.bin` weights.
PyTorch 2.6 or later is required for this loading path. No `nnAudio` or `accelerate`
installation is needed for this configuration.

## Relation to the paper

See the [ICLR 2026 paper](https://arxiv.org/abs/2509.09212), especially Sections 4 and 6
and Appendix A. This is a music-backbone integration with a corrected temporal mapping,
not a claim that the paper's result tables have been reproduced.

Existing core conventions are retained: the kernel uses twice the median squared
distance as its denominator, and coordinates use the symmetric normalized kernel's
eigenvectors directly. These differ from the literal forms of Equations 2 and 5–6.
The existing PM bank also omits delay/tail variants longer than its 20 ms processing
block. This change does not revise those shared numerical or distortion policies.

Record the evaluator commit, environment, input hashes, layer, seed, preprocessing,
full-context duration, and aggregation policy alongside `mert_config.json`. A successful
example validates that run; it does not establish perceptual validity on a new dataset
or exact numerical parity with the historical experimental script.

## Score and feature validation

Local Linux CPU validation on 4 November 2026 used PyTorch 2.6.0 and Transformers
4.52.4. The score-only integration passed 428 tests, including the existing end-to-end suite and compatibility
checks for all eight existing backbones and their 140 explicit layer selections.
The wheel and source package built successfully; wheel imports and package metadata
checks also passed. Shared model, audio, distortion, metric, and plotting files and
dependency bounds remained unchanged.

A fresh `mapss(..., model="mert", layer=1, add_ci=False)` run evaluated the complete
10-second SASSEC `wdrums_inst_sim` excerpt with `Algo10` estimates from the
[SEBASS evaluation data](https://www.audiolabs-erlangen.de/content/resources/00_2019-WASPAA-SEBASS/SASSEC.zip).
It encoded 390 unique waveforms and produced 1,087 finite scores per metric across
453 eligible activity frames. Against the earlier standalone diagnostic with corrected
timestamp alignment, the largest absolute differences were 0.00000948 for PS and
0.00006865 for PM. The first two waveforms used different batching in the two runs;
the separate check reusing identical cached features matched within 1.2e-16.

Real layer-3 extraction was also checked on the full no-drums reference/estimate pair:
both produced finite `749 × 768` embeddings, and prefix extraction exactly matched
the full model's requested hidden state. This was an extraction check, not a full
layer-3 score evaluation. These score and extraction checks did not test the CI extension.
CUDA, full MUSDB18 evaluation, empirical CI coverage, and the paper's published correlation
tables have not been validated by these checks.

### CI extension validation

After adding confidence support, all **451 tests** passed, including independent NumPy
checks of PM moments and Gamma bounds, a Schur-complement check against direct full-space
inversion, zero-complement cases, device selection, and CI serialization/plotting. Package
builds, metadata checks, and import of the CI module from the built wheel also passed.

The same complete music excerpt was evaluated with CIs enabled and disabled using its
390 hash-verified, previously computed real MERT embeddings. Both runs used identical
CPU thread settings: **PS and PM were bitwise identical between the two modes**.
The CI run produced 4,348 finite error components, aligned with the 1,087 finite scores
per metric; all components were nonnegative and within `[0, 1]`, with no degenerate-Gamma
fallbacks. Saving produced the six-panel figure. Neural inference was reused for this CI
check; the fresh inference and layer-3 checks above remain separate validation steps.
