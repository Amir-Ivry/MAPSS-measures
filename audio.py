import numpy as np
import torch
import librosa
import pyloudnorm as pyln
from config import SR, SILENCE_RATIO, ENERGY_WIN_MS, ENERGY_HOP_MS
from utils import safe_corr_np, hungarian

def loudness_normalize(wav, sr=SR, target_lufs=-23.0):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wav)
    normalized_wav = pyln.normalize.loudness(wav, loudness, target_lufs)
    peak = np.max(np.abs(normalized_wav))
    if peak > 1.0:
        normalized_wav = normalized_wav / peak
    return normalized_wav

def frame_rms_torch(sig, win, hop):
    dev = sig.device
    frames = sig.unfold(0, win, hop)
    if frames.size(0) and (frames.size(0) - 1) * hop == sig.numel() - win:
        frames = frames[:-1]
    rms = torch.sqrt((frames ** 2).mean(1) + 1e-12)
    return rms.to(dev)

def make_union_voiced_mask(refs_tensors, win, hop):
    device = refs_tensors[0].device
    rms_vecs = [frame_rms_torch(r, win, hop) for r in refs_tensors]
    lengths = [v.numel() for v in rms_vecs]
    L_max = max(lengths)
    silent_union = torch.zeros(L_max, dtype=torch.bool, device=device)
    for rms, L in zip(rms_vecs, lengths):
        ref_idx = lengths.index(L)
        thr = SILENCE_RATIO * torch.sqrt((refs_tensors[ref_idx] ** 2).mean())
        sil = (rms <= thr)
        silent_union[:L] |= sil
    return ~silent_union

def assign_outputs_to_refs_by_corr(ref_paths, out_paths):
    if not out_paths:
        return [None] * len(ref_paths)
    refs = [loudness_normalize(librosa.load(str(p), sr=SR)[0]) for p in ref_paths]
    outs = [loudness_normalize(librosa.load(str(p), sr=SR)[0]) for p in out_paths]
    n, m = len(refs), len(outs)
    K = max(n, m)
    C = np.ones((K, K), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            r = safe_corr_np(refs[i], outs[j])
            C[i, j] = 1.0 - (r + 1.0) * 0.5  # lower = better
    ri, cj = hungarian(C)
    mapping = [None] * n
    for i, j in zip(ri, cj):
        if i < n and j < m:
            mapping[i] = int(j)
    return mapping