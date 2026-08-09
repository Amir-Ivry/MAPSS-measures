"""Basic configuration and defaults used by MAPSS."""

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

# Defaults reproduce the paper's selected shallow-layer configuration for English.
MODEL_DEFAULT_LAYER = {
    "raw": 0,
    "wavlm": 3,
    "wav2vec2": 2,
    "hubert": 3,
    "wavlm_base": 2,
    "wav2vec2_base": 2,
    "hubert_base": 2,
    "wav2vec2_xlsr": 2,
}

MODEL_MAX_LAYER = {
    "raw": 0,
    "wavlm": 24,
    "wav2vec2": 24,
    "hubert": 24,
    "wavlm_base": 12,
    "wav2vec2_base": 12,
    "hubert_base": 12,
    "wav2vec2_xlsr": 24,
}
