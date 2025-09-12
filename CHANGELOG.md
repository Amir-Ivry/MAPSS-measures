# Changelog

## 1.0.1 — 2025-09-12
- Fix single-GPU embedding to return CPU tensors (prevents GPU OOM and .numpy() errors).
- Add GPU→CPU fallback in diffusion maps when CUDA OOM occurs.
- Add robust import and fallback for `scipy.optimize.linear_sum_assignment`.
- Clamp audio after loudness normalization and silence benign pyloudnorm warnings.
- Introduce experiment logger writing to `experiment.log`.
- Add GitHub repo auxiliary files and CI.
