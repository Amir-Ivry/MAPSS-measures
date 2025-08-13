from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np
import torch

from .models import get_device
from .logging_utils import setup_logger

logger = setup_logger(__name__)


def safe_cov_torch(X: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
    """Sample covariance with a minimal ridge to ensure invertibility."""
    Xc = X - X.mean(dim=0, keepdim=True)
    cov = Xc.T @ Xc / max(Xc.shape[0] - 1, 1)
    # Ridge if ill-conditioned
    if torch.linalg.matrix_rank(cov) < cov.shape[0]:
        cov = cov + torch.eye(cov.shape[0], device=cov.device) * ridge
    return cov


def mahalanobis_torch(x: torch.Tensor, mu: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
    """Robust Mahalanobis distance with numerical floor to avoid sqrt(neg)."""
    diff = x - mu
    return torch.sqrt((diff @ inv @ diff.T).clamp_min(1e-12))


def diffusion_map_torch(
    X_np: np.ndarray,
    labels_by_mix: list[str],
    *,
    cutoff: float = 0.99,
    tol: float = 1e-3,
    diffusion_time: int = 1,
    alpha: float = 0.0,
    eig_solver: str = "lobpcg",
    k: int | None = None,
    device: str | torch.device | None = None,
    return_eigs: bool = False,
    return_complement: bool = False,
    return_cval: bool = False,
) -> Tuple[np.ndarray, ...]:
    """Compute symmetric-normalized diffusion maps with CPU/GPU fallback.

    Returns Psi (principal coordinates), optionally complement and eigvals.
    """
    dev = torch.device(device) if device is not None else get_device()
    X = torch.as_tensor(X_np, dtype=torch.float32, device=dev)
    N = X.shape[0]

    # Distance matrix (chunked if needed)
    if N > 1000:
        chunk = min(500, N)
        D2 = torch.zeros(N, N, device=dev)
        for i in range(0, N, chunk):
            end_i = min(i + chunk, N)
            for j in range(0, N, chunk):
                end_j = min(j + chunk, N)
                D2[i:end_i, j:end_j] = torch.cdist(X[i:end_i], X[j:end_j]).pow_(2)
    else:
        D2 = torch.cdist(X, X).pow_(2)

    i, j = torch.triu_indices(N, N, offset=1, device=dev)
    eps = torch.median(D2[i, j])
    K = torch.exp(-D2 / (2 * eps))

    d = K.sum(dim=1)
    if alpha != 0.0:
        d_alpha_inv = d.pow(-alpha)
        K = d_alpha_inv[:, None] * K * d_alpha_inv[None, :]
        d = K.sum(dim=1)

    D_half_inv = torch.diag(torch.rsqrt(d.clamp_min(1e-12)))
    K_sym = D_half_inv @ K @ D_half_inv

    if eig_solver == "lobpcg":
        m = k if k is not None else min(N - 1, 50)
        init = torch.randn(N, m, device=dev)
        vals, vecs = torch.lobpcg(K_sym, k=m, X=init, niter=200, tol=tol, largest=True)
    elif eig_solver == "full":
        vals, vecs = torch.linalg.eigh(K_sym)
        vals, vecs = vals.flip(0), vecs.flip(1)
        if k is not None:
            vecs = vecs[:, : k + 1]
            vals = vals[: k + 1]
    else:
        raise ValueError(f"Unknown eig_solver '{eig_solver}'")

    psi = vecs[:, 1:]
    lam = vals[1:]

    cum = torch.cumsum(lam, dim=0)
    L = int((cum / cum[-1] < cutoff).sum().item()) + 1

    lam_pow = lam.pow(diffusion_time)
    psi_all = psi * lam_pow

    Psi = psi_all[:, :L]
    Psi_rest = psi_all[:, L:]

    if return_cval:
        indices_with_out = [i for i, name in enumerate(labels_by_mix) if "out" in name]
        valid_idx = torch.tensor([i for i in range(N) if i not in indices_with_out], device=dev)
        pi_min = d[valid_idx].min() / d[valid_idx].sum()
        c_val = lam_pow[0] * pi_min.rsqrt() / math.log(2.0)

    pieces = [Psi.cpu().numpy()]
    if return_complement:
        pieces.append(Psi_rest.cpu().numpy())
    if return_eigs:
        pieces.append(lam.cpu().numpy())
    if return_cval:
        pieces.append(float(c_val))
    return tuple(pieces)
