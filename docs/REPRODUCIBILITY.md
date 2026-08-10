# Reproducible MAPSS evaluation

MAPSS scores depend on the package version, representation configuration, ordered source
assignment, input signals, and aggregation policy. Record these before comparing systems.

## Recommended paper-aligned configuration

```python
result = mapss(
    reference=reference_paths,
    output=output_paths,
    source_names=source_names,
    model="wav2vec2",
    layer=2,
    alpha=1.0,
    add_ci=True,
    seed=42,
    max_gpus=1,
    length_policy="error",
)
```

`max_gpus` controls hardware use rather than the scientific definition, but it should still
be recorded with the environment.

## Evaluation record

Preserve:

- `mapss-measures` version;
- Python, PyTorch, Transformers, NumPy, and SciPy versions;
- model, layer, `alpha`, `add_ci`, seed, and length policy;
- source identifiers and ordered reference/output assignments;
- cryptographic hashes of input WAV files;
- frame-level PS, PM, and error tables;
- the exact aggregation code and weighting policy.

Useful environment commands:

```bash
python -c "import mapss; print(mapss.__version__)"
python -m pip freeze > environment.txt
```

Windows PowerShell can record WAV hashes with:

```powershell
Get-FileHash -Algorithm SHA256 "C:\data\reference_1.wav"
```

## Aggregation

Freeze aggregation before inspecting challenge submissions. State whether cross-item
aggregation weights mixtures, speakers, frames, or duration equally. Retain inactive-frame
`NaN` values and exclude them according to the declared policy rather than converting them
to zeros.

The convenience `summary.csv` uses finite-frame means. For paper-level reporting, follow
the PM aggregation and the Appendix B.4 PS pooling procedure defined in the ICLR 2026 paper.

## Release pinning

Install an exact version in evaluation infrastructure:

```text
mapss-measures==1.1.1
```

Do not evaluate some systems from a moving GitHub `main` branch and others from PyPI.
Archive the evaluator commit or container image used for the final results.
