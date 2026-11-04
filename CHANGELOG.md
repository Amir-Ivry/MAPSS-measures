# Changelog

## 1.2.0 - 2026-11-04

- Add an isolated MERT music backend with direct 24 kHz input conversion, an explicit
  layer, pinned checkpoint revision, and corrected feature/activity frame alignment.
- Support MERT confidence calculations with the existing `add_ci=True` default and
  six-panel plotting; retain optional score-only evaluation and two-panel plotting.
  Keep MERT CI processing separate, with PM reference-centered moments and explicit
  device handling. Save evaluation metadata alongside the results.
- Run MERT on CPU or one CUDA device, with PyTorch 2.6 or later for MERT only;
  the optional `mert` installation extra supplies this requirement.
- Add music examples and document the distinction from exact paper-result
  reproduction. Preserve the existing backbones and shared metric/distortion behavior.

## 1.1.2 - 2026-08-11

- Add `MAPSSResult.save(..., plot=True)` and CLI `--plot` for the paper-style six-panel
  PM, PS, deterministic-radius, and probabilistic-95%-bound figure.
- Validate the complete raw engine with confidence and plotting for N=2, N=3, and N=4.
- Add end-to-end coverage for mixed WAV/FLAC inputs, sample-rate conversion, stereo
  downmixing, and explicit shortest-duration trimming.
- Add early, source-specific guardrails for count and duration mismatches, corrupt audio,
  invalid waveform layouts, naming conflicts, option types, CUDA requests, and malformed
  plotting tables.
- Add dedicated input-contract and CPU/CUDA documentation, including first-run checkpoint
  and Hugging Face cache-warning guidance.

## 1.1.1 - 2026-08-11

- Add a clean-laptop quick start for Windows PowerShell, Linux, and macOS.
- Add tested Python and PowerShell examples for ordered reference/output paths.
- Add the optional `mapss-plot` command for plotting frame-level PS and PM scores.
- Add interpretation, troubleshooting, and reproducibility documentation.
- Replace machine-specific SASSEC manifest paths with portable templates.
- Constrain NumPy/Numba compatibility to prevent invalid clean-environment resolution.
- Validate examples, documentation links, plotting, and built-wheel installation in CI.

## 1.1.0 - 2026-08-09

- Add the installable `mapss-measures` distribution and `mapss` import package.
- Add the two-argument `mapss(reference, output, **kwargs)` Python API.
- Accept paths, NumPy arrays, PyTorch tensors, and batched source arrays.
- Return structured frame-level PS/PM scores and utterance-level summaries.
- Add strict validation for source order, counts, duration, silence, and finite samples.
- Align the default English configuration with the paper: wav2vec 2.0 Large, layer 2.
- Fix CPU inference, source identifiers containing hyphens, and multi-model result retention.
- Add unit/integration tests, CI, build checks, and tag-based PyPI release automation.

## 1.0.1 — 2025-09-12
- Fix single-GPU embedding to return CPU tensors (prevents GPU OOM and .numpy() errors).
- Add GPU→CPU fallback in diffusion maps when CUDA OOM occurs.
- Add robust import and fallback for `scipy.optimize.linear_sum_assignment`.
- Clamp audio after loudness normalization and silence benign pyloudnorm warnings.
- Introduce experiment logger writing to `experiment.log`.
- Add GitHub repo auxiliary files and CI.
