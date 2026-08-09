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
```

Build and validate both distribution formats:

```bash
python -m build
python -m twine check dist/*
```

The real pretrained-model path downloads large public checkpoints. Before a tagged release,
also run one wav2vec 2.0 layer-2 example on a CUDA machine and preserve the package version,
command, input hashes, and output CSVs as a release artifact.
