"""MERT-only error radii following the original music experiment's conventions.

The deterministic component concerns diffusion-coordinate truncation; the
probability component concerns finite distortion-cluster statistics at nominal
``1-delta`` confidence. These are paper-derived plug-in quantities, not an
empirically calibrated coverage claim for a new music dataset. PS retains the
experimental 0.7 effective-sample factor, 5% spectral floor, and Euclidean
propagation formula; PM retains the 50% local-box clipping and unit constants.

Unlike the historical PM CI helper, PM moment blocks are centered at the
reference, exactly as the point-score definition and paper Eq. 19 require.
All CI linear algebra is float64 on the explicitly requested device. No shared
point-score, confidence helper, or global device setting is changed.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.special import gammaincc
import torch

from .config import COV_TOL


_DISTANCE_EPS = 1e-6  # Existing mahalanobis_torch: sqrt(clamp(q, 0) + 1e-6).
_POSITIVE = np.finfo(np.float64).tiny


def _inputs(coords_d, coords_rest, labels, delta, device):
    if not 0 < float(delta) < 1:
        raise ValueError("CI delta must be strictly between 0 and 1.")
    retained = torch.as_tensor(coords_d, dtype=torch.float64, device=device)
    complement = torch.as_tensor(coords_rest, dtype=torch.float64, device=device)
    if (retained.ndim != 2 or complement.ndim != 2 or retained.shape[1] < 1
            or retained.shape[0] != complement.shape[0]
            or len(labels) != retained.shape[0]):
        raise ValueError("CI coordinates and labels have incompatible shapes.")
    if not bool(torch.isfinite(retained).all() and torch.isfinite(complement).all()):
        raise ValueError("CI coordinates must be finite.")
    if len(set(labels)) != len(labels):
        raise ValueError("CI labels must be unique.")
    sources = sorted({label.rsplit("-", 1)[0] for label in labels})
    groups = {}
    for source in sources:
        if f"{source}-ref" not in labels or f"{source}-out" not in labels:
            raise ValueError(f"CI source {source} requires a reference and output.")
        ref, out = labels.index(f"{source}-ref"), labels.index(f"{source}-out")
        distortions = [i for i, label in enumerate(labels)
                       if label.rsplit("-", 1)[0] == source and i not in {ref, out}]
        if len(distortions) < 2:
            raise ValueError(f"CI source {source} requires at least two distortions.")
        groups[source] = (ref, out, distortions)
    return retained, complement, groups


def _record(report, key, amount=1):
    if report is not None:
        report[key] = report.get(key, 0) + amount


def _center(source, fitted, scores, report):
    if not math.isfinite(fitted):
        raise RuntimeError(f"MERT CI fitted score for {source} is non-finite.")
    value = float(fitted if scores is None else scores[source])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"CI point-score center for {source} must be in [0, 1].")
    difference = abs(value - fitted)
    if report is not None:
        report["max_score_center_difference"] = max(
            report.get("max_score_center_difference", 0.0), difference
        )
    _record(report, "source_evaluations")
    return value, difference


def _radius(value, kind, report):
    value = float(value)
    if not math.isfinite(value) or value < -1e-10:
        raise RuntimeError(f"MERT {kind} CI calculation produced an invalid radius: {value}.")
    if value > 1:
        _record(report, f"{kind}_cap_count")
    return min(1.0, max(0.0, value))


def _moment_blocks(retained, complement):
    """Gram blocks for differences already centered by the metric definition."""
    denominator = retained.shape[0] - 1
    covariance = retained.T @ retained / denominator
    covariance = covariance + COV_TOL * torch.eye(
        retained.shape[1], dtype=retained.dtype, device=retained.device
    )
    tail_covariance = complement.T @ complement / denominator
    tail_covariance = tail_covariance + COV_TOL * torch.eye(
        complement.shape[1], dtype=complement.dtype, device=complement.device
    )
    cross = retained.T @ complement / denominator
    return covariance, cross, tail_covariance


def _schur_residuals(covariance, cross, tail_covariance, query_d, query_c,
                     diagnostics=None):
    """Return full-minus-retained squared-distance corrections for query rows.

    All three blocks include the same regularization as the corresponding full
    Gram matrix. This allows independent validation against direct block inversion.
    """
    if query_c.shape[1] == 0:
        return torch.zeros(query_d.shape[0], dtype=query_d.dtype, device=query_d.device)
    regression = torch.linalg.solve(covariance, cross)
    schur = tail_covariance - cross.T @ regression
    schur = (schur + schur.T) / 2
    residual = query_c - query_d @ regression
    correction = torch.sum(residual * torch.linalg.solve(schur, residual.T).T, dim=1)
    if not bool(torch.isfinite(correction).all()):
        raise RuntimeError("MERT CI Schur correction is non-finite.")
    tolerance = 1e-10 * max(1.0, float(correction.abs().max()))
    if float(correction.min()) < -tolerance:
        raise RuntimeError("MERT CI Schur correction is negative beyond numerical precision.")
    _record(diagnostics, "roundoff_negative_residual_count", int((correction < 0).sum()))
    return correction.clamp_min(0)


def _squared_distances(differences, inverse):
    quadratic = torch.sum((differences @ inverse) * differences, dim=1)
    # Preserve the existing score's sqrt-then-square convention, including epsilon.
    return torch.sqrt(quadratic.clamp_min(0) + _DISTANCE_EPS).square()


def _gamma_corner_radius(shape, scale, distance, widths, center, diagnostics):
    # Positivity guards must not move a valid small central Gamma parameter.
    # Unlike the old 1e-6/1e-8 floors, these admit every positive shape/scale
    # and the valid zero-distance endpoint Q(k, 0) = 1.
    values = []
    for signs in itertools.product((-1, 1), repeat=3):
        k = max(shape + signs[0] * widths[0], _POSITIVE)
        theta = max(scale + signs[1] * widths[1], _POSITIVE)
        a = max(distance + signs[2] * widths[2], 0.0)
        value = float(gammaincc(k, a / theta))
        if not math.isfinite(value):
            # The score is bounded even if numerical Gamma evaluation degenerates.
            _record(diagnostics, "nonfinite_gamma_corner_count")
            return max(center, 1 - center)
        values.append(abs(value - center))
    return max(values)


def pm_ci_components(coords_d, coords_rest, eigvals, labels, *, delta=0.05,
                     device="cpu", scores=None, diagnostics=None):
    """Return PM truncation and finite-cluster radii around supplied point scores.

    ``eigvals`` is accepted for the established CI interface; the already scaled
    retained/complement coordinates contain the needed spectral information.
    Degenerate Gamma moments produce a conservative radius covering [0, 1],
    rather than an unsupported zero-width confidence assertion.
    """
    retained, complement, groups = _inputs(coords_d, coords_rest, labels, delta, device)
    bias, probability = {}, {}
    if diagnostics is not None:
        diagnostics.update(covariance_center="reference", device=str(device), dtype="float64",
                           confidence="nominal paper-derived plug-in components; coverage not empirically calibrated")
    for source, (ref, out, distortions) in groups.items():
        differences = retained[distortions] - retained[ref]
        tail_differences = complement[distortions] - complement[ref]
        query_d = retained[out:out + 1] - retained[ref]
        query_c = complement[out:out + 1] - complement[ref]
        covariance, cross, tail_covariance = _moment_blocks(differences, tail_differences)
        inverse = torch.linalg.inv(covariance)
        squared = _squared_distances(differences, inverse)
        output_squared = float(_squared_distances(query_d, inverse)[0])
        mean, variance = float(squared.mean()), float(squared.var(unbiased=True))
        count = len(distortions)
        if not math.isfinite(mean) or not math.isfinite(variance) or mean <= 0 or variance <= 0:
            fallback_center = 1.0 if output_squared <= 1e-12 else 0.0
            center, _ = _center(source, fallback_center, scores, diagnostics)
            bias[source] = 0.0 if complement.shape[1] == 0 else max(center, 1 - center)
            probability[source] = max(center, 1 - center)
            _record(diagnostics, "degenerate_gamma_count")
            continue
        shape, scale = mean**2 / variance, variance / mean
        fitted = (float(gammaincc(shape, output_squared / scale))
                  if math.isfinite(shape) and math.isfinite(scale) and scale > 0
                  else math.nan)
        if not math.isfinite(fitted):
            if scores is None:
                covering_radius = 1.0
            else:
                center = float(scores[source])
                if not math.isfinite(center) or not 0 <= center <= 1:
                    raise ValueError(f"CI point-score center for {source} must be in [0, 1].")
                covering_radius = max(center, 1 - center)
            bias[source] = 0.0 if complement.shape[1] == 0 else covering_radius
            probability[source] = covering_radius
            _record(diagnostics, "source_evaluations")
            _record(diagnostics, "nonfinite_gamma_fit_count")
            continue
        center, _ = _center(source, fitted, scores, diagnostics)

        if complement.shape[1] == 0:
            bias[source] = 0.0
        else:
            residual_squared = _schur_residuals(
                covariance, cross, tail_covariance,
                torch.cat([differences, query_d]), torch.cat([tail_differences, query_c]),
                diagnostics,
            )
            full_squared = squared + residual_squared[:-1]
            full_mean = float(full_squared.mean())
            full_variance = float(full_squared.var(unbiased=True))
            factor = float(residual_squared[:-1].max()) * count / (count - 1)
            dk = factor * (full_mean + mean) / variance
            dt = factor * (full_variance + variance) / (mean**2 + 1e-9)
            da = float(residual_squared[-1])
            bias[source] = _radius(
                _gamma_corner_radius(shape, scale, output_squared, (dk, dt, da),
                                     center, diagnostics), "bias", diagnostics
            )

        maximum = float(squared.max()) + 1e-12
        log_term = math.log(6.0 / delta)
        sigma = math.sqrt(variance)
        eps_mu = math.sqrt(2 * variance * log_term / count) + 3 * maximum * log_term / count
        eps_sigma = (math.sqrt(2 * maximum**2 * log_term / count)
                     + 3 * maximum**2 * log_term / count)
        dk = min(abs(2 * mean / (variance + 1e-9)) * eps_mu
                 + abs(-2 * mean**2 / (sigma**3 + 1e-9)) * eps_sigma, 0.5 * shape)
        dt = min(abs(-variance / (mean**2 + 1e-9)) * eps_mu
                 + abs(2 * sigma / (mean + 1e-9)) * eps_sigma, 0.5 * scale)
        da = min(maximum * math.sqrt(2 * log_term / count), 0.5 * output_squared + 1e-12)
        probability[source] = _radius(
            _gamma_corner_radius(shape, scale, output_squared, (dk, dt, da),
                                 center, diagnostics), "probability", diagnostics
        )
    return bias, probability


def _ps_sampling_deviation(distance, covariance, count, delta):
    eigenvalues = torch.linalg.eigvalsh(covariance)
    minimum = max(float(eigenvalues[0]), 1e-9)
    maximum = float(eigenvalues[-1])
    trace = float(torch.trace(covariance))
    effective_count = max(int(0.7 * count), 3)
    floor = max(minimum, 0.05 * maximum)
    mean_deviation = math.sqrt(2 * maximum * math.log(2 / delta) / effective_count)
    rank = trace / maximum
    relative_covariance_deviation = (
        math.sqrt(rank / effective_count)
        + (rank + math.log(2 / delta)) / effective_count
    )
    return (2 * math.sqrt(distance) * mean_deviation * math.sqrt(maximum / floor)
            + distance * relative_covariance_deviation)


def ps_ci_components(coords_d, coords_rest, eigvals, labels, *, delta=0.05,
                     device="cpu", scores=None, diagnostics=None):
    """Return PS truncation and finite-cluster radii using the music conventions."""
    retained, complement, groups = _inputs(coords_d, coords_rest, labels, delta, device)
    if len(groups) < 2:
        raise ValueError("PS confidence calculations require at least two source clusters.")
    if diagnostics is not None:
        diagnostics.update(covariance_center="cluster_mean", device=str(device), dtype="float64",
                           effective_sample_fraction=0.7, minimum_eigenvalue_fraction=0.05,
                           confidence="nominal paper-derived plug-in components; coverage not empirically calibrated")
    clusters = {}
    for source, (ref, _, distortions) in groups.items():
        indices = [ref] + distortions
        mean_d, mean_c = retained[indices].mean(0), complement[indices].mean(0)
        covariance, cross, tail_covariance = _moment_blocks(
            retained[indices] - mean_d, complement[indices] - mean_c
        )
        clusters[source] = (mean_d, mean_c, covariance, cross, tail_covariance,
                            torch.linalg.inv(covariance), len(indices))

    def distance_to_cluster(output, source):
        mean_d, _, _, _, _, inverse, _ = clusters[source]
        return math.sqrt(float(_squared_distances(retained[output:output + 1] - mean_d, inverse)[0]))

    def tail_distance_to_cluster(output, source):
        mean_d, mean_c, covariance, cross, tail_covariance, _, _ = clusters[source]
        value = _schur_residuals(
            covariance, cross, tail_covariance,
            retained[output:output + 1] - mean_d,
            complement[output:output + 1] - mean_c, diagnostics,
        )
        return math.sqrt(float(value[0]))

    bias, probability = {}, {}
    for source, (_, output, _) in groups.items():
        own_distance = distance_to_cluster(output, source)
        other = min((name for name in groups if name != source),
                    key=lambda name: distance_to_cluster(output, name))
        foreign_distance = distance_to_cluster(output, other)
        denominator = own_distance + foreign_distance
        fitted = 1 - own_distance / denominator
        _, center_difference = _center(source, fitted, scores, diagnostics)
        if complement.shape[1] == 0:
            bias[source] = 0.0
        else:
            bias[source] = _radius(
                (foreign_distance * tail_distance_to_cluster(output, source)
                 + own_distance * tail_distance_to_cluster(output, other)) / denominator**2,
                "bias", diagnostics,
            )
        own_deviation = _ps_sampling_deviation(
            own_distance, clusters[source][2], clusters[source][-1], delta / 2
        )
        foreign_deviation = _ps_sampling_deviation(
            foreign_distance, clusters[other][2], clusters[other][-1], delta / 2
        )
        gradient_norm = math.hypot(own_distance, foreign_distance) / denominator**2
        probability[source] = _radius(
            gradient_norm * math.hypot(own_deviation, foreign_deviation) + center_difference,
            "probability", diagnostics,
        )
    return bias, probability
