# How to run the grid tests

## Quick start (from project root)

1. Create/activate your Python environment and install dependencies.
2. Run the grid tests using the provided demo manifest and tiny audio assets:

```bash
python -m tests.run_grid_tests --manifest tests/manifest_example.py
```

This runs **all models** (including `raw`) at their **default layers** (see `argshield.MODEL_DEFAULT_LAYER`)
and tries `--max-gpus 0 1 2` with `alpha=1.0`. It validates that:
- the pipeline executes without crashing,
- the generated CSVs are present,
- and the PS/PM per-frame outputs are **non-empty** and **not constant**.

### Faster, subset runs

If you first want a quicker sanity check:

```bash
python -m tests.run_grid_tests --subset raw wavlm_base --gpus 0
```

## Running the main program manually

Example:

```bash
python -m main --manifest tests/manifest_example.py --model wavlm --layer 24 --alpha 1.0 --max-gpus 1 --verbose
```

## Notes

- The demo manifest and tiny WAV files live under `tests/`. They are short, low amplitude, and
  are sufficient to exercise masking, diffusion maps, and PS/PM metric computations.
- The tests don't change any pipeline functionality. They only _run_ the code and check results.
- The CLI "argument shielding" (validation/parsing) now lives in `argshield.py`. See
  `argshield._validate_gpus` for the explicit check that `--max-gpus` is an integer ≥ 0.
