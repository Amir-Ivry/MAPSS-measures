# Troubleshooting

## `ModuleNotFoundError: No module named 'mapss'`

The package was installed into a different Python environment. Use the same interpreter for
installation and execution:

```powershell
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install "mapss-measures==1.1.1"
& $python -c 'import mapss; print(mapss.__version__); print(mapss.__file__)'
```

## A dependency such as NumPy is missing

Do not install MAPSS with `--no-deps`. Repair the environment with:

```powershell
& $python -m pip install --no-cache-dir --force-reinstall "mapss-measures==1.1.1"
& $python -m pip check
```

## `run_mapss.py` does not exist

No local script is required. Run the installed command directly:

```powershell
& $python -m mapss --reference ref1.wav ref2.wav --output out1.wav out2.wav
```

The [`examples/run_from_paths.py`](../examples/run_from_paths.py) script is provided only as
an editable Python example.

## A WAV file cannot be found

Use absolute paths or verify the current directory:

```powershell
Get-Location
Test-Path -LiteralPath "C:\full\path\to\audio.wav"
```

Paths containing spaces are supported when quoted.

## Reference and output counts differ

MAPSS expects exactly `N` references and `N` outputs. Output `i` must estimate reference
`i`; MAPSS does not reorder sources automatically.

## Durations differ

The default `length_policy="error"` rejects unequal lengths. Correct the upstream alignment
when the difference is unintended. Use `length_policy="trim"` or `--length-policy trim`
only when trimming every signal to the shortest duration matches the evaluation protocol.

## Silent or very short inputs

MAPSS rejects empty, non-finite, entirely silent, and sub-400 ms inputs. Confirm that the
files contain valid signal data and cover the intended evaluation interval.

## The first run appears slow

The default wav2vec 2.0 Large checkpoint is downloaded on first use. CPU evaluation is also
substantially slower than CUDA execution. Use `max_gpus=0` only when CPU execution is
intended; use `max_gpus=1` for one visible CUDA GPU.

## Plotting says matplotlib is missing

Install the optional plotting dependency in the same environment:

```powershell
& $python -m pip install "mapss-measures[plot]==1.1.1"
```

Then run:

```powershell
& $python -m mapss.plotting mapss_results
```

## PowerShell blocks virtual-environment activation

Activation is optional. Call the environment executable directly:

```powershell
$python = ".\.venv\Scripts\python.exe"
& $python -m pip --version
```

## Requesting help

Open a GitHub issue and include the MAPSS version, Python version, operating system,
CPU/GPU, full command, and complete traceback. Do not upload private challenge audio.
