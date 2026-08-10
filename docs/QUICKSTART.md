# Clean-laptop quick start

This guide starts with an empty directory and ends with PS/PM CSV files and a plot. It
uses the released package from PyPI; cloning the repository is not required.

## Requirements

- Python 3.10, 3.11, or 3.12 (64-bit recommended)
- Internet access during installation and the first pretrained-model run
- `N >= 2` reference WAV files and `N` corresponding output WAV files

The ordering is part of the evaluation: output `i` must estimate reference `i`.

## Windows PowerShell

### 1. Create a clean environment

```powershell
New-Item -ItemType Directory -Force "$HOME\mapss_run" | Out-Null
Set-Location "$HOME\mapss_run"

py -3.11 -m venv .venv
$python = ".\.venv\Scripts\python.exe"

& $python -m pip install --upgrade pip
& $python -m pip install "mapss-measures[plot]==1.1.1"
& $python -m pip check
& $python -c 'import mapss; print("MAPSS version:", mapss.__version__); print(mapss.__file__)'
```

`pip check` should print `No broken requirements found.` Activation is optional because
the commands below call the environment's Python executable directly.

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
    "--verbose"
)

& $python @mapssArguments
if ($LASTEXITCODE -ne 0) {
    throw "MAPSS failed with exit code $LASTEXITCODE."
}
```

`--max-gpus 0` forces CPU. Change it to `--max-gpus 1` for one working CUDA GPU. The
first `wav2vec2` run downloads model weights and can therefore take longer.

### 4. Inspect and plot the results

```powershell
Import-Csv ".\mapss_results\summary.csv" | Format-Table
& $python -m mapss.plotting ".\mapss_results"
Get-ChildItem ".\mapss_results"
```

The result directory contains `ps_scores.csv`, `pm_scores.csv`, `summary.csv`, optional
`confidence.csv`, and `ps_pm_over_time.png`.

## Linux and macOS

```bash
mkdir -p "$HOME/mapss_run"
cd "$HOME/mapss_run"
python3.11 -m venv .venv
python=.venv/bin/python

"$python" -m pip install --upgrade pip
"$python" -m pip install "mapss-measures[plot]==1.1.1"
"$python" -m pip check
"$python" -c 'import mapss; print(mapss.__version__)'

"$python" -m mapss \
  --reference /data/reference_1.wav /data/reference_2.wav \
  --output /data/output_1.wav /data/output_2.wav \
  --model wav2vec2 --layer 2 --alpha 1.0 --seed 42 \
  --results-dir mapss_results --verbose

"$python" -m mapss.plotting mapss_results
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
