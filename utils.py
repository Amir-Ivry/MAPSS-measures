import gc
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
warnings.filterwarnings("ignore", message="Some weights of Wav2Vec2Model")


def get_gpu_count(max_gpus=None):
    """
    Get the number of available GPUs.
    :param max_gpus: maximal number of GPUs to utilize.
    """
    ngpu = torch.cuda.device_count()
    if max_gpus is not None:
        ngpu = min(ngpu, max_gpus)
    return ngpu


def clear_gpu_memory():
    """
    Enhanced GPU memory clearing
    """
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()


def get_gpu_memory_info(verbose=False):
    """
    Get GPU memory info.
    :param verbose: if True, get info.
    """
    if not verbose:
        return
    for i in range(torch.cuda.device_count()):
        try:
            free_b, total_b = torch.cuda.mem_get_info(i)
            free_gb = free_b / 1024**3
            total_gb = total_b / 1024**3
        except Exception:
            total_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
            free_gb = total_gb - (torch.cuda.memory_reserved(i) / 1024**3)
        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
        print(f"GPU {i}: {mem_allocated:.2f}GB allocated, {free_gb:.2f}GB free / {total_gb:.2f}GB total")


class GPUWorkDistributor:
    """
    Distribute GPU memory into multiple GPUs.
    """
    def __init__(self, max_gpus=None):
        ngpu = get_gpu_count(max_gpus)
        self.gpu_locks = [threading.Lock() for _ in range(max(1, min(ngpu, 2)))]
        self.gpu_load = [0 for _ in range(max(1, min(ngpu, 2)))]
        self.ngpu = ngpu

    def get_least_loaded_gpu(self):
        return int(np.argmin(self.gpu_load))

    def execute_on_gpu(self, func, *args, **kwargs):
        if self.ngpu == 0:
            kwargs.pop("device", None)
            return func(*args, **kwargs)
        gid = self.get_least_loaded_gpu()
        with self.gpu_locks[gid]:
            self.gpu_load[gid] += 1
            try:
                with torch.cuda.device(gid):
                    kwargs["device"] = f"cuda:{gid}"
                    result = func(*args, **kwargs)
                    torch.cuda.empty_cache()
                return result
            finally:
                self.gpu_load[gid] -= 1


@dataclass
class Mixture:

    mixture_id: str
    refs: list[Path]
    systems: dict[str, list[Path]]
    speaker_ids: list[str]


def canonicalize_mixtures(mixtures, systems=None):
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

    if isinstance(mixtures[0], list):
        for i, group in enumerate(mixtures):
            mid = f"mix_{i:03d}"
            refs, spk_ids = [], []
            for d in group:
                if not isinstance(d, dict) or "ref" not in d or "id" not in d:
                    raise ValueError(
                        "Legacy mixtures expect dicts with 'id' and 'ref'."
                    )
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

def safe_cov_torch(X):
    """
    Compute the covariance matrix of X.
    :param X: array to compute covariance matrix of.
    :return: regularized covariance matrix.
    """
    Xc = X - X.mean(dim=0, keepdim=True)
    cov = Xc.T @ Xc / (Xc.shape[0] - 1)
    if torch.linalg.matrix_rank(cov) < cov.shape[0]:
        cov += torch.eye(cov.shape[0], device=cov.device) * 1e-6
    return cov


def mahalanobis_torch(x, mu, inv):
    """
    Compute the mahalanobis distance with x centered around mu with inverse covariance matrix inv.
    :param x: point to calculates distance from.
    :param mu: x is centered around mu.
    :param inv: the inverse covariance matrix.
    :return: Mahalanobis distance.
    """
    diff = x - mu
    diff_T = diff.transpose(-1, -2) if diff.ndim >= 2 else diff
    return torch.sqrt(diff @ inv @ diff_T + 1e-6)