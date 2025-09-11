"""Metrics computation preserving ALL original functionality including CI."""

import math

import numpy as np
import torch
from scipy.special import gammaincc
from scipy.stats import gamma

from config import COV_TOL, DEFAULT_DELTA_CI
from utils import get_gpu_count, mahalanobis_torch, safe_cov_torch


def pm_tail_gamma(d_out_sq, sq_dists):
    """PM tail gamma exactly as original."""
    mu = sq_dists.mean().item()
    var = sq_dists.var(unbiased=True).item()
    if var == 0.0:
        return 1.0
    k = (mu**2) / var
    theta = var / mu
    return float(1.0 - gamma.cdf(d_out_sq, a=k, scale=theta))


def pm_tail_rank(d_out_sq, sq_dists):
    """PM tail rank exactly as original."""
    rank = int((sq_dists < d_out_sq).sum().item())
    n = sq_dists.numel()
    return 1.0 - (rank + 0.5) / (n + 1.0)


def diffusion_map_torch(
    X_np,
    labels_by_mix,
    *,
    cutoff=0.99,
    tol=1e-3,
    diffusion_time=1,
    alpha=0.0,
    eig_solver="lobpcg",
    k=None,
    device=None,
    return_eigs=False,
    return_complement=False,
    return_cval=False,
):
    """Diffusion map computation exactly as original."""
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    X = torch.as_tensor(X_np, dtype=torch.float32, device=device)
    N = X.shape[0]

    if device != "cpu" and torch.cuda.is_available():
        stream = torch.cuda.Stream(device=device)
        ctx_dev = torch.cuda.device(device)
        ctx_stream = torch.cuda.stream(stream)
    else:
        from contextlib import nullcontext

        stream = None
        ctx_dev = nullcontext()
        ctx_stream = nullcontext()

    with ctx_dev:
        with ctx_stream:
            if N > 1000:
                chunk = min(500, N)
                D2 = torch.zeros(N, N, device=device)
                for i in range(0, N, chunk):
                    ei = min(i + chunk, N)
                    for j in range(0, N, chunk):
                        ej = min(j + chunk, N)
                        D2[i:ei, j:ej] = torch.cdist(X[i:ei], X[j:ej]).pow_(2)
            else:
                D2 = torch.cdist(X, X).pow_(2)

            i, j = torch.triu_indices(
                N, N, offset=1, device=None if device == "cpu" else device
            )
            eps = torch.median(D2[i, j])
            K = torch.exp(-D2 / (2 * eps))
            d = K.sum(dim=1)

            if alpha != 0.0:
                d_alpha_inv = d.pow(-alpha)
                K *= d_alpha_inv[:, None] * d_alpha_inv[None, :]
                d = K.sum(dim=1)

            D_half_inv = torch.diag(torch.rsqrt(d))
            K_sym = D_half_inv @ K @ D_half_inv

            if eig_solver == "lobpcg":
                m = k if k is not None else min(N - 1, 50)
                init = torch.randn(N, m, device=device)
                vals, vecs = torch.lobpcg(
                    K_sym, k=m, X=init, niter=200, tol=tol, largest=True
                )
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
                indices_with_out = [
                    ii for ii, name in enumerate(labels_by_mix) if "out" in name
                ]
                valid_idx = torch.tensor(
                    [ii for ii in range(N) if ii not in indices_with_out], device=device
                )
                pi_min = d[valid_idx].min() / d[valid_idx].sum()
                c_val = lam_pow[0] * pi_min.rsqrt() / math.log(2.0)

            if stream is not None:
                stream.synchronize()

    if return_complement and return_eigs and return_cval:
        return (
            Psi.cpu().numpy(),
            Psi_rest.cpu().numpy(),
            lam.cpu().numpy(),
            float(c_val),
        )
    if return_complement and return_eigs:
        return Psi.cpu().numpy(), Psi_rest.cpu().numpy(), lam.cpu().numpy()
    if return_complement:
        return Psi.cpu().numpy(), Psi_rest.cpu().numpy()
    if return_eigs:
        return Psi.cpu().numpy(), lam.cpu().numpy()
    return Psi.cpu().numpy()


def compute_ps(coords, labels, max_gpus=None):
    """Compute PS exactly as original."""
    ngpu = get_gpu_count(max_gpus)

    if ngpu == 0:
        coords_t = torch.tensor(coords)
        spks_here = sorted({l.split("-")[0] for l in labels})
        out = {}
        for s in spks_here:
            idxs = [i for i, l in enumerate(labels) if l.startswith(s)]
            out_i = labels.index(f"{s}-out")
            ref_is = [i for i in idxs if i != out_i]
            mu = coords_t[ref_is].mean(0)
            cov = safe_cov_torch(coords_t[ref_is])
            inv = torch.linalg.inv(cov)
            A = mahalanobis_torch(coords_t[out_i], mu, inv)
            B_list = []
            for o in spks_here:
                if o == s:
                    continue
                o_idxs = [
                    i
                    for i, l in enumerate(labels)
                    if l.startswith(o) and not l.endswith("-out")
                ]
                mu_o = coords_t[o_idxs].mean(0)
                inv_o = torch.linalg.inv(safe_cov_torch(coords_t[o_idxs]))
                B_list.append(mahalanobis_torch(coords_t[out_i], mu_o, inv_o))
            B_min = torch.min(torch.stack(B_list)) if B_list else torch.tensor(0.0)
            out[s] = (1 - A / (A + B_min + 1e-6)).item()
        return out

    # GPU version
    device = min(ngpu - 1, 1)  # Use second GPU if available
    device_str = f"cuda:{device}"
    coords_t = torch.tensor(coords, device=device_str)
    spks_here = sorted({l.split("-")[0] for l in labels})
    out = {}

    stream = torch.cuda.Stream(device=device_str)
    with torch.cuda.device(device):
        with torch.cuda.stream(stream):
            for s in spks_here:
                idxs = [i for i, l in enumerate(labels) if l.startswith(s)]
                out_i = labels.index(f"{s}-out")
                ref_is = [i for i in idxs if i != out_i]
                mu = coords_t[ref_is].mean(0)
                cov = safe_cov_torch(coords_t[ref_is])
                inv = torch.linalg.inv(cov)
                A = mahalanobis_torch(coords_t[out_i], mu, inv)
                B_list = []
                for o in spks_here:
                    if o == s:
                        continue
                    o_idxs = [
                        i
                        for i, l in enumerate(labels)
                        if l.startswith(o) and not l.endswith("-out")
                    ]
                    mu_o = coords_t[o_idxs].mean(0)
                    inv_o = torch.linalg.inv(safe_cov_torch(coords_t[o_idxs]))
                    B_list.append(mahalanobis_torch(coords_t[out_i], mu_o, inv_o))
                B_min = (
                    torch.min(torch.stack(B_list))
                    if B_list
                    else torch.tensor(0.0, device=device_str)
                )
                out[s] = (1 - A / (A + B_min + 1e-6)).item()
            stream.synchronize()
    return out


def compute_pm(coords, labels, pm_method, max_gpus=None):
    """Compute PM exactly as original."""
    ngpu = get_gpu_count(max_gpus)

    if ngpu == 0:
        coords_t = torch.tensor(coords)
        spks_here = sorted({l.split("-")[0] for l in labels})
        out = {}
        for s in spks_here:
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
            cov = dist.T @ dist / (N - 1)
            if torch.linalg.matrix_rank(cov) < D:
                cov += torch.eye(D) * COV_TOL
            inv = torch.linalg.inv(cov)
            sq_dists = torch.stack(
                [mahalanobis_torch(coords_t[i], ref_v, inv) ** 2 for i in d_idx]
            )
            d_out_sq = float(mahalanobis_torch(coords_t[out_i], ref_v, inv) ** 2)
            pm_score = (
                pm_tail_rank(d_out_sq, sq_dists)
                if pm_method == "rank"
                else pm_tail_gamma(d_out_sq, sq_dists)
            )
            out[s] = float(np.clip(pm_score, 0.0, 1.0))
        return out

    # GPU version
    device = min(ngpu - 1, 1)
    device_str = f"cuda:{device}"
    coords_t = torch.tensor(coords, device=device_str)
    spks_here = sorted({l.split("-")[0] for l in labels})
    out = {}

    stream = torch.cuda.Stream(device=device_str)
    with torch.cuda.device(device):
        with torch.cuda.stream(stream):
            for s in spks_here:
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
                cov = dist.T @ dist / (N - 1)
                if torch.linalg.matrix_rank(cov) < D:
                    cov += torch.eye(D, device=device_str) * COV_TOL
                inv = torch.linalg.inv(cov)
                sq_dists = torch.stack(
                    [mahalanobis_torch(coords_t[i], ref_v, inv) ** 2 for i in d_idx]
                )
                d_out_sq = float(mahalanobis_torch(coords_t[out_i], ref_v, inv) ** 2)
                pm_score = (
                    pm_tail_rank(d_out_sq, sq_dists)
                    if pm_method == "rank"
                    else pm_tail_gamma(d_out_sq, sq_dists)
                )
                out[s] = float(np.clip(pm_score, 0.0, 1.0))
            stream.synchronize()
    return out


def pm_ci_components_full(
    coords_d, coords_rest, eigvals, labels, *, delta=0.05, K=1.0, C1=1.0, C2=1.0
):
    """PM CI components exactly as original - complete implementation."""
    _EPS = 1e-12

    def _safe_x(a, theta):
        return a / max(theta, _EPS)

    D = coords_d.shape[1]
    m = coords_rest.shape[1]
    if m == 0:
        z = {s: 0.0 for s in {l.split("-")[0] for l in labels}}
        return z.copy(), z.copy()

    X_d = torch.tensor(
        coords_d, device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    X_c = torch.tensor(
        coords_rest, device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    spk_ids = sorted({l.split("-")[0] for l in labels})
    bias_ci = {}
    prob_ci = {}

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
        C_dc = D_mat.T @ C_mat / (n_p - 1)
        inv_Sigma_d = torch.linalg.inv(Sigma_d)

        S_i = (
            Sigma_c
            - C_dc.T @ inv_Sigma_d @ C_dc
            + torch.eye(X_c.shape[1], device=X_c.device) * 1e-9
        )
        S_inv = torch.linalg.inv(S_i)

        diff_out_d = X_d[out_i] - ref_d
        diff_out_c = X_c[out_i] - ref_c
        r_out = diff_out_c - C_dc.T @ inv_Sigma_d @ diff_out_d
        delta_Gi_a = float(r_out @ S_inv @ r_out)

        r_list = []
        for p in dist_is:
            d_p = X_d[p] - ref_d
            c_p = X_c[p] - ref_c
            r_p = c_p - C_dc.T @ inv_Sigma_d @ d_p
            r_list.append(r_p)
        R_p = torch.stack(r_list, dim=0)
        delta_Gi_p = torch.sum(R_p @ S_inv * R_p, dim=1)
        delta_Gi_mu_max = float(delta_Gi_p.max())

        mah_sq = torch.stack(
            [(X_d[i] - ref_d) @ inv_Sigma_d @ (X_d[i] - ref_d) for i in dist_is]
        )
        mu_g = float(mah_sq.mean())
        sigma2_g = float(mah_sq.var(unbiased=True) + 1e-12)
        sigma_g = math.sqrt(sigma2_g)

        full_sq = mah_sq + delta_Gi_p
        mu_full = float(full_sq.mean())
        sigma2_full = float(full_sq.var(unbiased=True) + 1e-12)

        if sigma2_g == 0.0:
            delta_Gi_k = delta_Gi_theta = 0.0
        else:
            factor = delta_Gi_mu_max * n_p / (n_p - 1)
            delta_Gi_k = 1.0 * factor * (mu_full + mu_g) / sigma2_g
            delta_Gi_theta = 1.0 * factor * (sigma2_full + sigma2_g) / (mu_g**2 + 1e-9)

        k_d = (mu_g**2) / max(sigma2_g, 1e-12)
        theta_d = sigma2_g / max(mu_g, 1e-12)
        a_d = float(diff_out_d @ inv_Sigma_d @ diff_out_d)

        pm_center = gammaincc(k_d, _safe_x(a_d, theta_d))

        corner_vals = []
        for s_k in (-1, 1):
            for s_theta in (-1, 1):
                for s_a in (-1, 1):
                    k_c = max(k_d + s_k * delta_Gi_k, 1e-6)
                    theta_c = max(theta_d + s_theta * delta_Gi_theta, 1e-6)
                    a_c = max(a_d + s_a * delta_Gi_a, 1e-8)
                    corner_vals.append(gammaincc(k_c, _safe_x(a_c, theta_c)))

        bias_ci[s] = max(abs(v - pm_center) for v in corner_vals)

        # Probabilistic half-width
        R_sq = float(mah_sq.max()) + 1e-12
        log_term = math.log(6.0 / delta)
        eps_mu = math.sqrt(2 * sigma2_g * log_term / n_p) + 3 * R_sq * log_term / n_p
        eps_sigma = (
            math.sqrt(2 * R_sq**2 * log_term / n_p) + 3 * R_sq**2 * log_term / n_p
        )

        g1_x = 2.0 * mu_g / (sigma2_g + 1e-9)
        g1_y = -2.0 * mu_g**2 / (sigma_g**3 + 1e-9)
        g2_x = -sigma2_g / (mu_g**2 + 1e-9)
        g2_y = 2.0 * sigma_g / (mu_g + 1e-9)

        delta_k = min(abs(g1_x) * eps_mu + abs(g1_y) * eps_sigma, 0.5 * k_d)
        delta_theta = min(abs(g2_x) * eps_mu + abs(g2_y) * eps_sigma, 0.5 * theta_d)
        delta_a = min(R_sq * math.sqrt(2 * log_term / n_p), 0.5 * a_d + 1e-12)

        pm_corners = []
        for s_k in (-1, 1):
            for s_theta in (-1, 1):
                for s_a in (-1, 1):
                    k_c = k_d + s_k * delta_k
                    theta_c = theta_d + s_theta * delta_theta
                    a_c = max(a_d + s_a * delta_a, 1e-8)
                    pm_corners.append(gammaincc(k_c, _safe_x(a_c, theta_c)))

        prob_ci[s] = max(abs(pm - pm_center) for pm in pm_corners)

    return bias_ci, prob_ci


def ps_ci_components_full(coords_d, coords_rest, eigvals, labels, *, delta=0.05):
    """PS CI components exactly as original - complete implementation."""

    def _mean_dev(lam_max, delta, n_eff):
        return math.sqrt(2 * lam_max * math.log(2 / delta) / n_eff)

    def _rel_cov_dev(lam_max, trace, delta, n_eff, C=1.0):
        r = trace / lam_max
        abs_dev = (
            C * lam_max * (math.sqrt(r / n_eff) + (r + math.log(2 / delta)) / n_eff)
        )
        return abs_dev / lam_max

    def _maha_eps_m(a_hat, lam_min, lam_max, mean_dev, rel_cov_dev):
        term1 = 2 * math.sqrt(a_hat) * mean_dev * math.sqrt(lam_max / lam_min)
        term2 = a_hat * rel_cov_dev
        return term1 + term2

    D = coords_d.shape[1]
    m = coords_rest.shape[1]
    if m == 0:
        z = {s: 0.0 for s in set(l.split("-")[0] for l in labels)}
        return z.copy(), z.copy()

    X_d = torch.tensor(
        coords_d, device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    X_c = torch.tensor(
        coords_rest, device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    spk_ids = sorted({l.split("-")[0] for l in labels})
    bias = {}
    prob = {}

    for s in spk_ids:
        idxs = [i for i, l in enumerate(labels) if l.startswith(s)]
        out_i = labels.index(f"{s}-out")
        ref_is = [i for i in idxs if i != out_i]

        mu_d = X_d[ref_is].mean(0)
        mu_c = X_c[ref_is].mean(0)
        Sigma_d = safe_cov_torch(X_d[ref_is])
        Sigma_c = safe_cov_torch(X_c[ref_is])
        C_dc = (X_d[ref_is] - mu_d).T @ (X_c[ref_is] - mu_c) / (len(ref_is) - 1)
        inv_Sd = torch.linalg.inv(Sigma_d)

        lam_min = torch.linalg.eigvalsh(Sigma_d).min().clamp_min(1e-9).item()
        lam_max = torch.linalg.eigvalsh(Sigma_d).max()
        trace = torch.trace(Sigma_d).item()

        diff_d = X_d[out_i] - mu_d
        diff_c = X_c[out_i] - mu_c
        A_d = float(mahalanobis_torch(X_d[out_i], mu_d, inv_Sd))

        r_i = diff_c - C_dc.T @ inv_Sd @ diff_d
        S_i = (
            Sigma_c
            - C_dc.T @ inv_Sd @ C_dc
            + torch.eye(X_c.shape[1], device=X_c.device) * 1e-9
        )
        term_i = math.sqrt(float(r_i @ torch.linalg.solve(S_i, r_i)))

        B_d, term_j = float("inf"), 0.0
        Sig_o = None
        for o in spk_ids:
            if o == s:
                continue
            o_idxs = [
                i
                for i, l in enumerate(labels)
                if l.startswith(o) and not l.endswith("-out")
            ]
            muo_d = X_d[o_idxs].mean(0)
            muo_c = X_c[o_idxs].mean(0)
            Sig_o_tmp = safe_cov_torch(X_d[o_idxs])
            inv_So = torch.linalg.inv(Sig_o_tmp)
            this_B = float(mahalanobis_torch(X_d[out_i], muo_d, inv_So))

            if this_B < B_d:
                B_d = this_B
                Sig_o = Sig_o_tmp
                diff_do = X_d[out_i] - muo_d
                diff_co = X_c[out_i] - muo_c
                C_oc = (
                    (X_d[o_idxs] - muo_d).T @ (X_c[o_idxs] - muo_c) / (len(o_idxs) - 1)
                )
                r_j = diff_co - C_oc.T @ inv_So @ diff_do
                S_j = (
                    safe_cov_torch(X_c[o_idxs])
                    - C_oc.T @ inv_So @ C_oc
                    + torch.eye(X_c.shape[1], device=X_c.device) * 1e-9
                )
                term_j = math.sqrt(float(r_j @ torch.linalg.solve(S_j, r_j)))

        denom = A_d + B_d
        bias[s] = (B_d * term_i + A_d * term_j) / (denom**2)

        if Sig_o is not None:
            lam_min_o = torch.linalg.eigvalsh(Sig_o).min().clamp_min(1e-9).item()
            lam_max_o = torch.linalg.eigvalsh(Sig_o).max().item()
            trace_o = torch.trace(Sig_o).item()

            n_eff = max(int(0.7 * len(ref_is)), 3)
            RIDGE = 0.05
            lam_min_eff = max(lam_min, RIDGE * lam_max.item())
            lam_min_o_eff = max(lam_min_o, RIDGE * lam_max_o)

            eps_i_sg = _maha_eps_m(
                A_d,
                lam_min_eff,
                lam_max.item(),
                _mean_dev(lam_max.item(), delta / 2, n_eff),
                _rel_cov_dev(lam_max.item(), trace, delta / 2, n_eff),
            )
            eps_j_sg = _maha_eps_m(
                B_d,
                lam_min_o_eff,
                lam_max_o,
                _mean_dev(lam_max_o, delta / 2, n_eff),
                _rel_cov_dev(lam_max_o, trace_o, delta / 2, n_eff),
            )

            grad_l2 = math.hypot(A_d, B_d) / (A_d + B_d) ** 2
            ps_radius = grad_l2 * math.hypot(eps_i_sg, eps_j_sg)
            prob[s] = min(1.0, ps_radius)
        else:
            prob[s] = 0.0

    return bias, prob
