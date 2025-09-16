import numpy as np
import pyloudnorm as pyln
import torch

from config import SILENCE_RATIO, SR
import warnings

warnings.filterwarnings("ignore", message="Possible clipped samples in output.")


def loudness_normalize(wav, sr=SR, target_lufs=-23.0):
    """
    Apply loudness normalization on an audio signal.
    :param wav: waveform signal to normalize.
    :param sr: sampling rate.
    :param target_lufs: LUFS points to normalize to.
    :return: normalized signal.
    """
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wav)
    normalized_wav = pyln.normalize.loudness(wav, loudness, target_lufs)
    peak = np.max(np.abs(normalized_wav))
    if peak > 1.0:
        normalized_wav = normalized_wav / max(peak, 1e-12)
    return np.clip(normalized_wav, -1.0, 1.0)


def frame_rms_torch(sig, win, hop):
    """
    Calculates the RMS of a signal with a moving window.
    :param sig: signal for calculation.
    :param win: analysis window size.
    :param hop: analysis window hop size.
    :return: RMS of signal.
    """
    dev = sig.device
    frames = sig.unfold(0, win, hop)
    if frames.size(0) and (frames.size(0) - 1) * hop == sig.numel() - win:
        frames = frames[:-1]
    rms = torch.sqrt((frames ** 2).mean(1) + 1e-12)
    return rms.to(dev)


def compute_speaker_activity_masks(refs_tensors, win, hop):
    """
    Computes individual voice activity for each speaker and determines which frames
    have at least 2 active speakers.
    :param refs_tensors: references that compose the mixture.
    :param win: analysis window size.
    :param hop: analysis window hop size.
    :return: (multi_speaker_mask, individual_speaker_masks)
        - multi_speaker_mask: boolean mask of frames where at least 2 speakers are active
        - individual_speaker_masks: list of boolean masks, one per speaker
    """
    device = refs_tensors[0].device
    individual_masks = []
    lengths = []

    for ref in refs_tensors:
        rms = frame_rms_torch(ref, win, hop)
        threshold = SILENCE_RATIO * torch.sqrt((ref ** 2).mean())
        voiced = rms > threshold
        individual_masks.append(voiced)
        lengths.append(voiced.numel())

    L_max = max(lengths)
    padded_masks = []
    for mask, L in zip(individual_masks, lengths):
        if L < L_max:
            padded = torch.cat([mask, torch.zeros(L_max - L, dtype=torch.bool, device=device)])
        else:
            padded = mask
        padded_masks.append(padded)

    stacked = torch.stack(padded_masks, dim=0)
    active_count = stacked.sum(dim=0)
    multi_speaker_mask = active_count >= 2
    return multi_speaker_mask, padded_masks