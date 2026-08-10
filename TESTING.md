# Testing MAPSS

Create a Python 3.10-3.12 environment and install the editable package with test tools:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Run fast unit tests and static checks:

```bash
ruff check .
pytest -m "not integration"
```

Run the CPU-only, end-to-end raw-waveform smoke test:

```bash
pytest -m integration
python examples/smoke_test.py --output-dir smoke_results
```

Build and validate both distribution formats:

```bash
python -m build
python -m twine check dist/*
```

Verify the built wheel, rather than the editable source tree, in a clean environment:

```bash
python -m venv wheel-check
wheel-check/bin/python -m pip install "dist/mapss_measures-1.1.1-py3-none-any.whl[plot]"
wheel-check/bin/python -c "import mapss; print(mapss.__version__)"
wheel-check/bin/python -m mapss --help
wheel-check/bin/python -m mapss.plotting --help
```

On Windows, replace `wheel-check/bin/python` with `wheel-check\Scripts\python.exe`.

The real pretrained-model path downloads large public checkpoints. Before a tagged release,
also run one wav2vec 2.0 layer-2 example on a CUDA machine and preserve the package version,
command, input hashes, and output CSVs as a release artifact.
