# PM/PS Metrics (Diffusion‑Map Based)

Clean, documented, and reproducible implementation of **Perceptual Match (PM)**
and **Perceptual Separation (PS)** metrics computed over **diffusion map**
embeddings of SSL (self‑supervised) speech representations.

This repo reorganizes and hardens the original research code into a
production‑grade package with a simple CLI, CPU/GPU fallbacks, and extensive
docstrings.

> If you are migrating from the internal research scripts, the functionality of
> `run_experiment` and the grid driver formerly in `main_exp_reproduce.py`
> are preserved via the `pspm-run` CLI and `scripts/run_grid.py`.

## Quickstart

```bash
# 1) create a fresh environment (recommended)
python -m venv .venv && source .venv/bin/activate

# 2) install (editable)
pip install -e ".[dev]"

# 3) run a smoke test (CPU only)
pspm-run --help
```

### Running an experiment

Edit `configs/example.yaml` and then:

```bash
pspm-run --config configs/example.yaml
```

Or supply flags directly:
```bash
pspm-run   --data-root "/path/to/Signals"   --results-root "./results"   --models wav2vec2   --distortions all   --alpha-dm 1.0   --pm-method gamma   --dm-solver full   --layer 13   --add-ci   --mixtures-file configs/mixtures.example.json
```

### Key features
- Clear package layout (`src/pspm`), type hints and NumPy‑style docstrings
- Robust **CPU / single‑GPU / multi‑GPU** execution with graceful fallbacks
- Unified, readable naming (no Greek letters in code)
- Logging instead of prints + reproducible seeding
- Simple YAML/CLI configuration and a maintained `run_experiment` pipeline
- CI‑ready formatting (`black`, `ruff`) and test scaffold

## License

MIT — see `LICENSE`. Replace the author details in `pyproject.toml`.
