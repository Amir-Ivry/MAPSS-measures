# Clean-laptop quick start

This guide starts with an empty directory and ends with PS/PM CSV files and a plot. It
uses the released package from PyPI; cloning the repository is not required.

## Requirements

- Python 3.10, 3.11, or 3.12 (64-bit recommended)
- Internet access during installation and the first pretrained-model run
- `N >= 2` reference audio files and `N` corresponding output audio files

WAV and FLAC are tested directly. Stereo files and mixed sample rates are accepted: MAPSS
downmixes each file to mono and resamples it to 16 kHz before checking duration alignment.

The ordering is part of the evaluation: output `i` must estimate reference `i`.

## Windows PowerShell

### 1. Create a clean environment

```powershell
New-Item -ItemType Directory -Force "$HOME\mapss_run" | Out-Null
Set-Location "$HOME\mapss_run"

py -3.11 -m venv .venv
$python = ".\.venv\Scripts\python.exe"

& $python -m pip install --upgrade pip
& $python -m pip install "mapss-measures[plot]==1.1.2"
& $python -m pip check
& $python -c 'import mapss; print("MAPSS version:", mapss.__version__); print(mapss.__file__)'
```

`pip check` should print `No broken requirements found.` Activation is optional because
the commands below call the environment's Python executable directly.

If your prompt already begins with `(.venv)`, you may use `python` instead of `& $python`
in every command below.

### 2. Enter the ordered paths

Replace the paths below and add or remove entries as needed:

```powershell
$references = @(
    "C:\data\reference_source_1.wav"
    "C:\data\reference_source_2.wav"
)

$outputs = @(
    "C:\data\estimated_source_1.wav"
    "C:\data\estimated_source_2.wav"
)
```

Validate the input before the expensive model run:

```powershell
if ($references.Count -lt 2) {
    throw "MAPSS requires at least two sources."
}
if ($references.Count -ne $outputs.Count) {
    throw "Found $($references.Count) references and $($outputs.Count) outputs."
}
foreach ($path in ($references + $outputs)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Audio file does not exist: $path"
    }
}
Write-Host "Validated $($references.Count) aligned sources."
```

### 3. Run the paper-recommended configuration

```powershell
$mapssArguments = @(
    "-m", "mapss", "--reference"
) + $references + @(
    "--output"
) + $outputs + @(
    "--model", "wav2vec2",
    "--layer", "2",
    "--alpha", "1.0",
    "--seed", "42",
    "--max-gpus", "0",
    "--length-policy", "error",
    "--results-dir", "mapss_results",
    "--plot",
    "--verbose"
)

& $python @mapssArguments
if ($LASTEXITCODE -ne 0) {
    throw "MAPSS failed with exit code $LASTEXITCODE."
}
```

`--max-gpus 0` forces CPU. The first `wav2vec2` run downloads approximately 1.3 GB of
model weights. After the download reaches 100%, CPU inference can remain quiet for several
minutes; `--verbose` shows MAPSS progress. Hugging Face symlink and `hf_xet` messages are
cache/performance warnings, not failures.

For one working CUDA GPU, first follow [HARDWARE.md](HARDWARE.md), verify that PyTorch
reports `CUDA available: True`, and change the value to `--max-gpus 1`.

### 4. Inspect and plot the results

```powershell
Import-Csv ".\mapss_results\summary.csv" | Format-Table
Get-ChildItem ".\mapss_results"
```

The result directory contains `ps_scores.csv`, `pm_scores.csv`, `summary.csv`,
`confidence.csv`, and `mapss_over_time.png`. The figure follows the paper's six-panel
time-aligned layout for PM, PS, deterministic error radii, and probabilistic 95% bounds.

## Linux and macOS

```bash
mkdir -p "$HOME/mapss_run"
cd "$HOME/mapss_run"
python3.11 -m venv .venv
python=.venv/bin/python

"$python" -m pip install --upgrade pip
"$python" -m pip install "mapss-measures[plot]==1.1.2"
"$python" -m pip check
"$python" -c 'import mapss; print(mapss.__version__)'

"$python" -m mapss \
  --reference /data/reference_1.wav /data/reference_2.wav \
  --output /data/output_1.wav /data/output_2.wav \
  --model wav2vec2 --layer 2 --alpha 1.0 --seed 42 \
  --results-dir mapss_results --plot --verbose
```

Use `--max-gpus 0` to force CPU. Add more paths to both lists for additional sources.

## Fast installation-only check

The repository includes [`examples/smoke_test.py`](../examples/smoke_test.py), which
generates its own audio and uses `model="raw"` to avoid a checkpoint download:

```bash
python examples/smoke_test.py
```

This verifies installation and the end-to-end engine. Raw-model scores are not intended
for papers, reports, or challenge leaderboards.

## Python API equivalent

The same automatic figure is available from Python:

```python
result = mapss(reference=references, output=outputs, add_ci=True, max_gpus=0)
result.save("mapss_results", plot=True)
```

`plot=True` and `--plot` intentionally require confidence computation. If you set
`add_ci=False` or use `--no-ci`, save the tables without plotting or use the standalone
two-panel score plot afterward.
