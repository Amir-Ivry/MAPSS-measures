"""Utility functions preserving all original functionality."""
import gc
import warnings
import threading
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment as _lsa

warnings.filterwarnings("ignore", message="Some weights of Wav2Vec2Model")


def get_gpu_count(max_gpus=None):
    """Get number of GPUs to use."""
    ngpu = torch.cuda.device_count()
    if max_gpus is not None:
        ngpu = min(ngpu, max_gpus)
    return ngpu


def clear_gpu_memory():
    """Clear GPU memory exactly as in original."""
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    gc.collect()


def get_gpu_memory_info(verbose=False):
    """Get GPU memory information."""
    if not verbose:
        return
    for i in range(torch.cuda.device_count()):
        mem_allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
        mem_reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
        free_memory = total_memory - mem_reserved
        print(f"GPU {i}: {mem_allocated:.2f}GB allocated, {free_memory:.2f}GB free")


def write_wav_16bit(path, x, sr=16000):
    """Write WAV file exactly as original."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
        sf.write(str(path), x.astype(np.float32), sr)
    except Exception:
        from scipy.io.wavfile import write
        write(str(path), sr, (np.clip(x, -1, 1) * 32767).astype(np.int16))


def safe_corr_np(a, b):
    """Safe correlation computation from original."""
    L = min(len(a), len(b))
    if L <= 1:
        return 0.0
    a = a[:L].astype(np.float64)
    b = b[:L].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    da = a.std()
    db = b.std()
    if da <= 1e-12 or db <= 1e-12:
        return 0.0
    r = float((a * b).mean() / (da * db))
    return max(-1.0, min(1.0, r))


def hungarian(cost):
    """Hungarian algorithm from original."""
    try:
        return _lsa(cost)
    except Exception:
        used = set()
        rows, cols = [], []
        for i in range(cost.shape[0]):
            j = int(np.argmin([cost[i, k] if k not in used else 1e12
                               for k in range(cost.shape[1])]))
            used.add(j)
            rows.append(i)
            cols.append(j)
        return np.asarray(rows), np.asarray(cols)


class GPUWorkDistributor:
    """Distribute work across GPUs exactly as original."""

    def __init__(self, max_gpus=None):
        ngpu = get_gpu_count(max_gpus)
        self.gpu_locks = [threading.Lock() for _ in range(max(1, min(ngpu, 2)))]
        self.gpu_load = [0 for _ in range(max(1, min(ngpu, 2)))]
        self.ngpu = ngpu

    def get_least_loaded_gpu(self):
        return int(np.argmin(self.gpu_load))

    def execute_on_gpu(self, func, *args, **kwargs):
        if self.ngpu == 0:
            kwargs.pop('device', None)
            return func(*args, **kwargs)
        gid = self.get_least_loaded_gpu()
        with self.gpu_locks[gid]:
            self.gpu_load[gid] += 1
            try:
                with torch.cuda.device(gid):
                    kwargs['device'] = f'cuda:{gid}'
                    result = func(*args, **kwargs)
                return result
            finally:
                self.gpu_load[gid] -= 1


@dataclass
class Mixture:
    """Mixture data structure."""
    mixture_id: str
    refs: list[Path]
    systems: dict[str, list[Path]]
    speaker_ids: list[str]


def canonicalize_mixtures(mixtures, systems=None):
    """Convert mixtures to canonical format exactly as original."""
    canon = []
    if not mixtures:
        return canon

    def as_paths(seq):
        return [p if isinstance(p, Path) else Path(str(p)) for p in seq]

    def speaker_id_from_ref(ref_path, idx, mixture_id):
        stem = (ref_path.stem or "").strip()
        if not stem:
            stem = f"spk{idx:02d}"
        return f"{mixture_id}__{stem}"

    # New style manifest
    if isinstance(mixtures[0], dict):
        for m in mixtures:
            mid = str(m.get("mixture_id") or m.get("id") or "").strip()
            if not mid:
                raise ValueError("Each mixture must include 'mixture_id'.")
            refs = as_paths(m.get("references", []))
            if not refs:
                raise ValueError(f"Mixture {mid}: 'references' must be non-empty.")
            sysmap = {}
            if isinstance(m.get("systems"), dict):
                for algo, outs in m["systems"].items():
                    sysmap[str(algo)] = as_paths(outs)
            spk_ids = [speaker_id_from_ref(r, i, mid) for i, r in enumerate(refs)]
            canon.append(Mixture(mid, refs, sysmap, spk_ids))
        return canon

    # Legacy style
    if isinstance(mixtures[0], list):
        for i, group in enumerate(mixtures):
            mid = f"mix_{i:03d}"
            refs, spk_ids = [], []
            for d in group:
                if not isinstance(d, dict) or "ref" not in d or "id" not in d:
                    raise ValueError("Legacy mixtures expect dicts with 'id' and 'ref'.")
                rp = Path(d["ref"])
                refs.append(rp)
                spk_ids.append(f"{mid}__{str(d['id']).strip()}")
            sysmap = {}
            if systems:
                for algo, per_mix in systems.items():
                    if mid in per_mix:
                        sysmap[algo] = as_paths(per_mix[mid])
            canon.append(Mixture(mid, refs, sysmap, spk_ids))
        return canon

    raise ValueError("Unsupported 'mixtures' format.")


def random_misalign(sig, sr, max_ms, mode='single', rng=None):
    """Random misalignment exactly as original."""
    import random
    if rng is None:
        rng = random
    max_samples = int(sr * max_ms / 1000)
    if max_samples == 0:
        return sig
    shift = rng.randint(-max_samples, max_samples) if mode == 'range' else int(max_samples)
    if shift == 0:
        return sig
    if isinstance(sig, torch.Tensor):
        z = torch.zeros(abs(shift), dtype=sig.dtype, device=sig.device)
        return torch.cat([z, sig[:-shift]]) if shift > 0 else torch.cat([sig[-shift:], z])
    else:
        z = np.zeros(abs(shift), dtype=sig.dtype)
        return np.concatenate([z, sig[:-shift]]) if shift > 0 else np.concatenate([sig[-shift:], z])


def safe_cov_torch(X):
    """Safe covariance computation from original."""
    Xc = X - X.mean(dim=0, keepdim=True)
    cov = Xc.T @ Xc / (Xc.shape[0] - 1)
    if torch.linalg.matrix_rank(cov) < cov.shape[0]:
        cov += torch.eye(cov.shape[0], device=cov.device) * 1e-6
    return cov


def mahalanobis_torch(x, mu, inv):
    """Mahalanobis distance from original."""
    diff = x - mu
    return torch.sqrt(diff @ inv @ diff.T + 1e-6)