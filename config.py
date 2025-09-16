"""
Basic configuration and default values used in the MAPSS computations.
"""
import os
import torch
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"^expandable_segments not supported on this platform"
)

SR = 16_000
RESULTS_ROOT = "results"
BATCH_SIZE = 2
ENERGY_WIN_MS = 20
ENERGY_HOP_MS = 20
SILENCE_RATIO = 0.1
EPS = 1e-4
COV_TOL = 1e-6

DEFAULT_LAYER = 2
DEFAULT_ADD_CI = True
DEFAULT_DELTA_CI = 0.05
DEFAULT_ALPHA = 1.0

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True,garbage_collection_threshold:0.6"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)