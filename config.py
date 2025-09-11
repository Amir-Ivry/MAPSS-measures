"""Configuration settings."""

import os

import torch

# Audio settings
SR = 16_000
RESULTS_ROOT = "results"
BATCH_SIZE = 4
ENERGY_WIN_MS = 20
ENERGY_HOP_MS = 20
SILENCE_RATIO = 0.1
EPS = 1e-4
COV_TOL = 1e-6

# Experiment parameters
DEFAULT_LAYER = 2
DEFAULT_ADD_CI = True
DEFAULT_DELTA_CI = 0.05

# CUDA performance hints
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,expandable_segments:True"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
