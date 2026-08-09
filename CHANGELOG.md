# Changelog

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
