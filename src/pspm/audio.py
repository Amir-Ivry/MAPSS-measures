from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pyloudnorm as pyln
import torch
import torch.nn.functional as F


def loudness_normalize(wav: np.ndarray, sr: int = 16_000, target_lufs: float = -23.0) -> np.ndarray:
    """Normalize an audio signal to target LUFS with peak safety.

    Parameters
    ----------
    wav : np.ndarray
        Input mono signal in range [-1, 1].
    sr : int
        Sampling rate (Hz).
    target_lufs : float
        Target LUFS.

    Returns
    -------
    np.ndarray
        Loudness-normalized waveform.
    """
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wav)
    normalized_wav = pyln.normalize.loudness(wav, loudness, target_lufs)
    max_peak = np.max(np.abs(normalized_wav))
    if max_peak > 1.0:
        normalized_wav = normalized_wav / max_peak
    return normalized_wav


def frame_rms_torch(sig: torch.Tensor, win: int, hop: int) -> torch.Tensor:
    """Frame-wise RMS on a 1D tensor.

    Returns floor((len(sig) - win) / hop) + 1 frames.

    Parameters
    ----------
    sig : torch.Tensor
        1D signal tensor.
    win : int
        Window length in samples.
    hop : int
        Hop length in samples.

    Returns
    -------
    torch.Tensor
        RMS per frame (shape: [n_frames]).
    """
    frames = sig.unfold(0, win, hop)  # (n_frames, win)
    rms = torch.sqrt((frames ** 2).mean(dim=1) + 1e-12)
    return rms


def make_union_voiced_mask(refs: list[np.ndarray], win: int, hop: int, silence_ratio: float) -> torch.Tensor:
    """Compute a union voiced mask across references based on frame RMS.

    Parameters
    ----------
    refs : list of np.ndarray
        Reference waveforms.
    win : int
        Window length (samples).
    hop : int
        Hop length (samples).
    silence_ratio : float
        Threshold multiplier on global RMS to label silence.

    Returns
    -------
    torch.Tensor
        Boolean vector (shape: [max_n_frames]) where True marks voiced frames.
    """
    rms_vecs = []
    for r in refs:
        t = torch.as_tensor(r, dtype=torch.float32)
        rms_vecs.append(frame_rms_torch(t, win, hop))

    lengths = [v.numel() for v in rms_vecs]
    L_max = max(lengths)
    silent_union = torch.zeros(L_max, dtype=torch.bool)

    for rms, r in zip(rms_vecs, refs):
        t = torch.as_tensor(r, dtype=torch.float32)
        thr = silence_ratio * torch.sqrt((t ** 2).mean())
        sil = rms <= thr
        silent_union[: sil.numel()] |= sil

    return ~silent_union  # True = voiced
