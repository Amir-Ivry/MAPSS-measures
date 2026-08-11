# CPU and CUDA setup

CUDA is optional. MAPSS runs on CPU or on one or more NVIDIA GPUs visible to PyTorch.

## Check the current environment

Run this inside the same virtual environment in which MAPSS is installed:

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Interpretation:

- `CUDA available: False`: use `max_gpus=0` or `--max-gpus 0` for CPU.
- `CUDA available: True`: use `max_gpus=1` or `--max-gpus 1` for one GPU.
- `max_gpus=None`: MAPSS uses the visible GPUs automatically and otherwise uses CPU.

An explicit positive `max_gpus` value now fails immediately when PyTorch cannot access
CUDA. This prevents a requested GPU run from silently continuing on CPU.

## Install a CUDA-enabled PyTorch build

MAPSS depends on PyTorch, but CUDA compatibility depends on the machine, NVIDIA driver,
operating system, Python version, and selected PyTorch build. Use the official
[PyTorch installation selector](https://docs.pytorch.org/get-started/locally/) and select
your operating system, `Pip`, `Python`, and a CUDA compute platform. Run the generated
command inside the MAPSS virtual environment, then repeat the availability check above.

Do not assume that having a system CUDA toolkit means the active Python environment can
use CUDA. `torch.cuda.is_available()` is the decisive MAPSS check.

## CPU expectations

The paper-selected default uses wav2vec 2.0 Large. Its first run downloads approximately
1.3 GB of weights, and CPU inference can take many minutes depending on audio duration and
hardware. Use `verbose=True` or `--verbose` for progress. The model is cached after the
first successful download.

Hugging Face warnings about Windows symlink support or missing `hf_xet` affect cache space
or download performance; they do not indicate a failed model download. A progress line
ending in `model.safetensors: 100%` means the checkpoint download completed and MAPSS is
moving into model loading/inference.
