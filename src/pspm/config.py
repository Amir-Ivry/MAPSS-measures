from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """
    Configuration for a PM/PS experiment.

    Attributes
    ----------
    results_root:
        Output directory for CSV results and logs.
    data_root:
        Root directory that contains algorithm output folders with WAV signals.
    mixtures_file:
        JSON file containing a "mixtures" key – a list of mixtures,
        each mixture is a list of speakers, and each speaker is an object
        with keys {"id": str, "ref": str}.
    models:
        SSL model names to embed with. Supported: ["wavlm", "wav2vec2", "hubert", "wav2vec2_xlsr"].
    distortion_keys:
        Either "all" or a subset from {"notch","comb","tremolo","noise","harmonic",
        "reverb","noisegate","pitch_shift","lowpass","highpass","echo","clipping","vibrato"}.
    alpha_dm:
        Diffusion map alpha (symmetric normalization uses alpha=0.0; keep 0.0 or 1.0).
    dm_solver:
        "lobpcg" or "full" eigensolver for symmetric kernel.
    pm_method:
        "gamma" or "rank".
    use_mlm:
        If True, set model.train() for masked-LM style stochasticity.
    mlm_oversample:
        If True, doubles samples for masked-LM.
    advanced_distortions:
        If True, use content-aware framewise distortions; else simple whole-signal versions.
    layer:
        SSL model hidden-state layer to use (implementation forwards this index to transformers models).
    misalign_max_ms:
        Apply a deterministic max shift (ms) to "out" signals for robustness testing (0 disables).
    add_ci:
        If True, compute CI components for PM and PS (bias and probabilistic radii).
    seed:
        Random seed for numpy/torch/python.
    batch_size:
        Batch size during embedding. Reduce to avoid OOM.
    energy_win_ms, energy_hop_ms, silence_ratio:
        Parameters for voiced-frame mask generation.
    delta_ci:
        Confidence parameter delta for CI computations.
    cov_tolerance:
        Ridge added to singular covariance matrices.

    Notes
    -----
    This is a direct, typed mapping of the original research script defaults.
    """

    # Paths
    results_root: str = "results"
    data_root: str = "."
    mixtures_file: str = ""

    # Core algorithm
    models: List[str] = field(default_factory=lambda: ["wav2vec2"])
    distortion_keys: str = "all"
    alpha_dm: float = 1.0
    dm_solver: str = "full"
    pm_method: str = "gamma"
    use_mlm: bool = False
    mlm_oversample: bool = False
    advanced_distortions: bool = False
    layer: int = 13
    misalign_max_ms: int = 0
    add_ci: bool = False

    # Reproducibility & performance
    seed: int = 42
    batch_size: int = 4

    # Signal processing
    energy_win_ms: int = 20
    energy_hop_ms: int = 20
    silence_ratio: float = 0.1

    # CI specifics
    delta_ci: float = 0.05
    cov_tolerance: float = 1e-6
