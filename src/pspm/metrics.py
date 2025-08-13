from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import gammaincc
from scipy.stats import gamma as gamma_dist

from .diffusion import safe_cov_torch, mahalanobis_torch
from .models import get_device


def compute_ps(coords: np.ndarray, labels: list[str]) -> Dict[str, float]:
    """Compute PS score per speaker from diffusion coordinates.

    PS = B / (A+B) in the *truncated* space where
    A is distance to own cluster, B is distance to nearest other.
    """
    dev = get_device()
    coords_t = torch.tensor(coords, device=dev)
    spks = sorted({l.split("-")[0] for l in labels})
    out: Dict[str, float] = {}
    for s in spks:
        idxs = [i for i, l in enumerate(labels) if l.startswith(s)]
        out_i = labels.index(f"{s}-out")
        ref_is = [i for i in idxs if i != out_i]

        mu = coords_t[ref_is].mean(0)
        cov = safe_cov_torch(coords_t[ref_is])
        inv = torch.linalg.inv(cov)
        A = mahalanobis_torch(coords_t[out_i], mu, inv)

        Bs = []
        for o in spks:
            if o == s:
                continue
            o_idxs = [i for i, l in enumerate(labels) if l.startswith(o) and not l.endswith("-out")]
            mu_o = coords_t[o_idxs].mean(0)
            inv_o = torch.linalg.inv(safe_cov_torch(coords_t[o_idxs]))
            Bs.append(mahalanobis_torch(coords_t[out_i], mu_o, inv_o))
        B_min = torch.min(torch.stack(Bs)) if Bs else torch.tensor(0.0, device=dev)
        out[s] = (1 - A / (A + B_min + 1e-6)).item()
    return out


def _pm_tail_gamma(d_out_sq: float, sq_dists: torch.Tensor) -> float:
    mu = sq_dists.mean().item()
    var = sq_dists.var(unbiased=True).item()
    if var == 0.0:
        return 1.0
    k = (mu ** 2) / var
    theta = var / mu
    return float(1.0 - gamma_dist.cdf(d_out_sq, a=k, scale=theta))


def _pm_tail_rank(d_out_sq: float, sq_dists: torch.Tensor) -> float:
    rank = int((sq_dists < d_out_sq).sum().item())
    n = sq_dists.numel()
    return 1.0 - (rank + 0.5) / (n + 1.0)


def compute_pm(coords: np.ndarray, labels: list[str], method: str = "gamma") -> Dict[str, float]:
    """Compute PM score per speaker from diffusion coordinates."""
    dev = get_device()
    coords_t = torch.tensor(coords, device=dev)
    spks = sorted({l.split("-")[0] for l in labels})
    out: Dict[str, float] = {}
    for s in spks:
        idxs = [i for i, l in enumerate(labels) if l.startswith(s)]
        ref_i = labels.index(f"{s}-ref")
        out_i = labels.index(f"{s}-out")
        d_idx = [i for i in idxs if i not in {ref_i, out_i}]
        if len(d_idx) < 2:
            out[s] = 0.0
            continue
        ref_v = coords_t[ref_i]
        dist = coords_t[d_idx] - ref_v
        N, D = dist.shape

        cov = dist.T @ dist / (max(N - 1, 1))
        if torch.linalg.matrix_rank(cov) < D:
            cov += torch.eye(D, device=dev) * 1e-6
        inv = torch.linalg.inv(cov)

        sq_dists = torch.stack([mahalanobis_torch(coords_t[i], ref_v, inv) ** 2 for i in d_idx])
        d_out_sq = float(mahalanobis_torch(coords_t[out_i], ref_v, inv) ** 2)

        pm_score = _pm_tail_rank(d_out_sq, sq_dists) if method == "rank" else _pm_tail_gamma(d_out_sq, sq_dists)
        out[s] = float(np.clip(pm_score, 0.0, 1.0))
    return out


# ========================== CI COMPONENTS ================================
def pm_ci_components_full(
    coords_d: np.ndarray,
    coords_rest: np.ndarray,
    eigvals: np.ndarray,
    labels: list[str],
    *,
    delta: float = 0.05,
) -> tuple[dict[str, float], dict[str, float]]:
    """Half-width components for PM confidence intervals.

    Returns
    -------
    bias_ci, prob_ci : dict[str, float], dict[str, float]
        Truncation bias and finite-sample probabilistic radii.
    """
    dev = get_device()
    D = coords_d.shape[1]
    m = coords_rest.shape[1]
    if m == 0:
        z = {s: 0.0 for s in {l.split("-")[0] for l in labels}}
        return z.copy(), z.copy()

    lam_tail = eigvals[D:]
    X_d = torch.tensor(coords_d, device=dev)
    X_c = torch.tensor(coords_rest, device=dev)
    spk_ids = sorted({l.split("-")[0] for l in labels})

    bias_ci: dict[str, float] = {}
    prob_ci: dict[str, float] = {}

    def _safe_div(a: float, b: float) -> float:
        return a / max(b, 1e-12)

    def _grad_Q_norm(k, theta, a, h=1e-6):
        """Finite-diff gradient norm of Q = gammaincc(k, a/theta)."""
        def F(k_, th_, a_):
            return gammaincc(k_, _safe_div(a_, th_))

        hk = max(h, k * 1e-5)
        hth = max(h, theta * 1e-5)
        ha = max(h, a * 1e-5)
        dFk = (F(k + hk, theta, a) - F(k - hk, theta, a)) / (2 * hk)
        dFth = (F(k, theta + hth, a) - F(k, theta - hth, a)) / (2 * hth)
        dFa = (F(k, theta, a + ha) - F(k, theta, a - ha)) / (2 * ha)
        return math.sqrt(dFk * dFk + dFth * dFth + dFa * dFa)

    for s in spk_ids:
        idxs = [i for i, l in enumerate(labels) if l.startswith(s)]
        ref_i = labels.index(f"{s}-ref")
        out_i = labels.index(f"{s}-out")
        dist_is = [i for i in idxs if i not in {ref_i, out_i}]
        n_p = len(dist_is)
        if n_p < 2:
            bias_ci[s] = 0.0
            prob_ci[s] = 0.0
            continue

        ref_d = X_d[ref_i]
        ref_c = X_c[ref_i]

        D_mat = X_d[dist_is] - ref_d
        C_mat = X_c[dist_is] - ref_c

        Sigma_d = safe_cov_torch(D_mat)
        Sigma_c = safe_cov_torch(C_mat)
        C_dc = D_mat.T @ C_mat / max(n_p - 1, 1)

        inv_Sd = torch.linalg.inv(Sigma_d)

        S_i = Sigma_c - C_dc.T @ inv_Sd @ C_dc + torch.eye(Sigma_c.shape[0], device=dev) * 1e-9
        S_inv = torch.linalg.inv(S_i)

        diff_out_d = X_d[out_i] - ref_d
        diff_out_c = X_c[out_i] - ref_c
        r_out = diff_out_c - C_dc.T @ inv_Sd @ diff_out_d
        delta_a = float(r_out @ S_inv @ r_out)

        r_list = []
        for p in dist_is:
            d_p = X_d[p] - ref_d
            c_p = X_c[p] - ref_c
            r_p = c_p - C_dc.T @ inv_Sd @ d_p
            r_list.append(r_p)
        R_p = torch.stack(r_list, dim=0)
        delta_g_i = torch.sum(R_p @ S_inv * R_p, dim=1)
        delta_g_mu_max = float(delta_g_i.max())

        mah_sq = torch.stack([(X_d[i] - ref_d) @ inv_Sd @ (X_d[i] - ref_d) for i in dist_is])
        mu_g = float(mah_sq.mean())
        var_g = float(mah_sq.var(unbiased=True) + 1e-12)

        full_sq = mah_sq + delta_g_i
        mu_full = float(full_sq.mean())
        var_full = float(full_sq.var(unbiased=True) + 1e-12)

        factor = delta_g_mu_max * n_p / max(n_p - 1, 1)
        delta_k = factor * (mu_full + mu_g) / max(var_g, 1e-12)
        delta_theta = factor * (var_full + var_g) / max(mu_g ** 2, 1e-12)

        k_d = mu_g ** 2 / max(var_g, 1e-12)
        theta_d = var_g / max(mu_g, 1e-12)
        a_d = float(diff_out_d @ inv_Sd @ diff_out_d)

        pm_center = gammaincc(k_d, _safe_div(a_d, theta_d))

        corner_vals = []
        for s_k in (-1, 1):
            for s_th in (-1, 1):
                for s_a in (-1, 1):
                    k_c = max(k_d + s_k * delta_k, 1e-6)
                    th_c = max(theta_d + s_th * delta_theta, 1e-6)
                    a_c = max(a_d + s_a * delta_a, 1e-8)
                    corner_vals.append(gammaincc(k_c, _safe_div(a_c, th_c)))
        corner_diff = max(abs(v - pm_center) for v in corner_vals)
        bias_ci[s] = corner_diff

        # Probabilistic radius via concentration
        R_sq = float(mah_sq.max()) + 1e-12
        log_term = math.log(6.0 / delta)
        eps_mu = math.sqrt(2 * var_g * log_term / n_p) + 3 * R_sq * log_term / n_p
        eps_sigma = math.sqrt(2 * R_sq ** 2 * log_term / n_p) + 3 * R_sq ** 2 * log_term / n_p
        delta_a_prob = R_sq * math.sqrt(2 * log_term / n_p)

        g1_x = 2.0 * mu_g / max(var_g, 1e-12)
        g1_y = -2.0 * mu_g ** 2 / (max(var_g, 1e-12) ** 1.5)
        g2_x = -var_g / max(mu_g ** 2, 1e-12)
        g2_y = 2.0 * math.sqrt(var_g) / max(mu_g, 1e-12)

        delta_k_prob = abs(g1_x) * eps_mu + abs(g1_y) * eps_sigma
        delta_th_prob = abs(g2_x) * eps_mu + abs(g2_y) * eps_sigma

        rel_clip = 0.5
        delta_k_prob = min(delta_k_prob, rel_clip * k_d)
        delta_th_prob = min(delta_th_prob, rel_clip * theta_d)
        delta_a_prob = min(delta_a_prob, rel_clip * a_d + 1e-12)

        pm_center = gammaincc(k_d, _safe_div(a_d, theta_d))
        pm_corners = []
        for s_k in (-1, 1):
            for s_th in (-1, 1):
                for s_a in (-1, 1):
                    k_c = k_d + s_k * delta_k_prob
                    th_c = theta_d + s_th * delta_th_prob
                    a_c = max(a_d + s_a * delta_a_prob, 1e-8)
                    pm_corners.append(gammaincc(k_c, _safe_div(a_c, th_c)))
        prob_ci[s] = max(abs(pm - pm_center) for pm in pm_corners)

    return bias_ci, prob_ci


def ps_ci_components_full(
    coords_d: np.ndarray, coords_rest: np.ndarray, eigvals: np.ndarray, labels: list[str], *, delta: float = 0.05
) -> tuple[dict[str, float], dict[str, float]]:
    """Half-width components for PS confidence intervals (bias + probabilistic)."""
    dev = get_device()
    D = coords_d.shape[1]
    m = coords_rest.shape[1]
    if m == 0:
        z = {s: 0.0 for s in {l.split("-")[0] for l in labels}}
        return z.copy(), z.copy()

    lam_tail = eigvals[D:]
    X_d = torch.tensor(coords_d, device=dev)
    X_c = torch.tensor(coords_rest, device=dev)

    spk_ids = sorted({l.split("-")[0] for l in labels})
    bias: dict[str, float] = {}
    prob: dict[str, float] = {}

    for s in spk_ids:
        idxs = [i for i, l in enumerate(labels) if l.startswith(s)]
        out_i = labels.index(f"{s}-out")
        ref_is = [i for i in idxs if i != out_i]

        mu_d = X_d[ref_is].mean(0)
        mu_c = X_c[ref_is].mean(0)
        Sigma_d = safe_cov_torch(X_d[ref_is])
        Sigma_c = safe_cov_torch(X_c[ref_is])
        C_dc = (X_d[ref_is] - mu_d).T @ (X_c[ref_is] - mu_c) / max(len(ref_is) - 1, 1)

        inv_Sd = torch.linalg.inv(Sigma_d)

        diff_d = X_d[out_i] - mu_d
        diff_c = X_c[out_i] - mu_c
        A_d = float(mahalanobis_torch(X_d[out_i], mu_d, inv_Sd))

        r_i = diff_c - C_dc.T @ inv_Sd @ diff_d
        S_i = Sigma_c - C_dc.T @ inv_Sd @ C_dc + torch.eye(Sigma_c.shape[0], device=dev) * 1e-9
        term_i = math.sqrt(float(r_i @ torch.linalg.solve(S_i, r_i)))

        # nearest foreign cluster j
        spk_ids_ = [o for o in spk_ids if o != s]
        B_d, term_j = float("inf"), 0.0
        Sig_o = None
        for o in spk_ids_:
            o_idxs = [i for i, l in enumerate(labels) if l.startswith(o) and not l.endswith("-out")]
            muo_d = X_d[o_idxs].mean(0)
            muo_c = X_c[o_idxs].mean(0)
            Sig_o = safe_cov_torch(X_d[o_idxs])
            inv_So = torch.linalg.inv(Sig_o)
            this_B = float(mahalanobis_torch(X_d[out_i], muo_d, inv_So))
            diff_do = X_d[out_i] - muo_d
            diff_co = X_c[out_i] - muo_c
            if this_B < B_d:
                B_d = this_B
                C_oc = ((X_d[o_idxs] - muo_d).T @ (X_c[o_idxs] - muo_c)) / max(len(o_idxs) - 1, 1)
                r_j = diff_co - C_oc.T @ inv_So @ diff_do
                S_j = safe_cov_torch(X_c[o_idxs]) - C_oc.T @ inv_So @ C_oc + torch.eye(Sigma_c.shape[0], device=dev) * 1e-9
                term_j = math.sqrt(float(r_j @ torch.linalg.solve(S_j, r_j)))

        denom = (A_d + B_d)
        bias[s] = (B_d * term_i + A_d * term_j) / (denom ** 2)

        # Probabilistic part (sub-Gaussian bounds for Mahalanobis deviations)
        lam_min = torch.linalg.eigvalsh(Sigma_d).min().clamp_min(1e-9).item()
        lam_max = torch.linalg.eigvalsh(Sigma_d).max().item()
        trace = torch.trace(Sigma_d).item()

        lam_min_o = torch.linalg.eigvalsh(Sig_o).min().clamp_min(1e-9).item() if Sig_o is not None else lam_min
        lam_max_o = torch.linalg.eigvalsh(Sig_o).max().item() if Sig_o is not None else lam_max
        trace_o = torch.trace(Sig_o).item() if Sig_o is not None else trace

        n_eff = max(int(0.7 * len(ref_is)), 3)
        RIDGE = 0.05
        lam_min_eff = max(lam_min, RIDGE * lam_max)
        lam_min_o_eff = max(lam_min_o, RIDGE * lam_max_o)

        def _mean_dev(lam_max, delta, n_eff):
            return math.sqrt(2 * lam_max * math.log(2 / delta) / n_eff)

        def _rel_cov_dev(lam_max, trace, delta, n_eff, C=1.0):
            r = trace / lam_max
            abs_dev = C * lam_max * (math.sqrt(r / n_eff) + (r + math.log(2 / delta)) / n_eff)
            return abs_dev / lam_max

        def _maha_eps_m(a_hat, lam_min, lam_max, mean_dev, rel_cov_dev):
            term1 = 2 * math.sqrt(max(a_hat, 1e-12)) * mean_dev * math.sqrt(lam_max / lam_min)
            term2 = a_hat * rel_cov_dev
            return term1 + term2

        eps_i_sg = _maha_eps_m(
            A_d, lam_min_eff, lam_max, _mean_dev(lam_max, delta / 2, n_eff), _rel_cov_dev(lam_max, trace, delta / 2, n_eff)
        )
        eps_j_sg = _maha_eps_m(
            B_d, lam_min_o_eff, lam_max_o, _mean_dev(lam_max_o, delta / 2, n_eff), _rel_cov_dev(lam_max_o, trace_o, delta / 2, n_eff)
        )

        grad_l2 = math.hypot(A_d, B_d) / (A_d + B_d) ** 2
        ps_radius = grad_l2 * math.hypot(eps_i_sg, eps_j_sg)
        prob[s] = min(1.0, ps_radius)

    return bias, prob
