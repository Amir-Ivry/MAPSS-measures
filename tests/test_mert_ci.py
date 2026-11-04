"""Independent numerical checks for MERT's isolated confidence calculations."""

from __future__ import annotations

import importlib
import itertools
import math

import numpy as np
import pytest
from scipy.special import gammaincc
import torch


@pytest.fixture
def ci_module():
    return importlib.import_module("mapss._mert_ci")


def pm_cloud():
    # Distortions are deliberately shifted away from the reference: mean-centering
    # them a second time gives a different PM fit and invalid mixed Schur blocks.
    distortions = np.array([
        [1.0, .5, .3], [2.0, .1, .9], [1.4, 1.2, -.1], [.7, 1.6, .6],
        [1.7, .9, 1.2], [2.2, 1.8, .1], [.3, .4, .8], [1.1, 1.9, -.4],
    ], dtype=np.float64)
    full = np.vstack([np.zeros(3), [1.7, .5, .4], distortions])
    labels = ["src000-ref", "src000-out"] + [f"src000-d{i}" for i in range(len(distortions))]
    return full[:, :2], full[:, 2:], np.array([.8, .5, .2]), labels


def independent_pm_statistics(retained, complement):
    differences = retained[2:] - retained[0]
    query = retained[1] - retained[0]
    sample_count, dimension = differences.shape
    covariance = differences.T @ differences / (sample_count - 1) + 1e-6 * np.eye(dimension)
    inverse = np.linalg.inv(covariance)
    radii = np.einsum("ij,jk,ik->i", differences, inverse, differences) + 1e-6
    query_radius = float(query @ inverse @ query + 1e-6)
    mean, variance = float(radii.mean()), float(radii.var(ddof=1))
    shape, scale = mean**2 / variance, variance / mean
    point_score = float(gammaincc(shape, query_radius / scale))
    all_differences = np.column_stack([differences, complement[2:] - complement[0]])
    all_query = np.concatenate([query, complement[1] - complement[0]])
    full_covariance = (all_differences.T @ all_differences / (sample_count - 1)
                       + 1e-6 * np.eye(all_differences.shape[1]))
    full_inverse = np.linalg.inv(full_covariance)
    full_radii = np.einsum("ij,jk,ik->i", all_differences, full_inverse, all_differences) + 1e-6
    full_query_radius = float(all_query @ full_inverse @ all_query + 1e-6)
    return dict(mean=mean, variance=variance, shape=shape, scale=scale,
                point_score=point_score, radii=radii, query_radius=query_radius,
                full_radii=full_radii, full_query_radius=full_query_radius,
                n=sample_count, covariance=covariance)


def independent_pm_probability_radius(stats, center, delta=.05):
    # Equation-level NumPy/SciPy oracle, independent of torch/block calculations.
    mean, variance, n = stats["mean"], stats["variance"], stats["n"]
    sigma = math.sqrt(variance)
    maximum = float(stats["radii"].max()) + 1e-12
    log_term = math.log(6 / delta)
    eps_mean = math.sqrt(2 * variance * log_term / n) + 3 * maximum * log_term / n
    eps_sigma = math.sqrt(2 * maximum**2 * log_term / n) + 3 * maximum**2 * log_term / n
    dk = min(abs(2 * mean / (variance + 1e-9)) * eps_mean
             + abs(-2 * mean**2 / (sigma**3 + 1e-9)) * eps_sigma,
             .5 * stats["shape"])
    dt = min(abs(-variance / (mean**2 + 1e-9)) * eps_mean
             + abs(2 * sigma / (mean + 1e-9)) * eps_sigma,
             .5 * stats["scale"])
    da = min(maximum * math.sqrt(2 * log_term / n), .5 * stats["query_radius"] + 1e-12)
    corners = [
        gammaincc(stats["shape"] + k * dk,
                  max(stats["query_radius"] + a * da, 1e-8) / (stats["scale"] + t * dt))
        for k, t, a in itertools.product((-1, 1), repeat=3)
    ]
    return min(1., max(abs(value - center) for value in corners))


def test_schur_correction_equals_direct_full_minus_retained_inverse(ci_module):
    retained, complement, _, _ = pm_cloud()
    differences = retained[2:] - retained[0]
    tail_differences = complement[2:] - complement[0]
    full_differences = np.column_stack([differences, tail_differences])
    full_covariance = (full_differences.T @ full_differences / (len(differences) - 1)
                       + 1e-6 * np.eye(full_differences.shape[1]))
    queries = np.column_stack([retained[1:] - retained[0], complement[1:] - complement[0]])
    full_quadratic = np.einsum("ij,jk,ik->i", queries, np.linalg.inv(full_covariance), queries)
    kept_quadratic = np.einsum("ij,jk,ik->i", queries[:, :2],
                               np.linalg.inv(full_covariance[:2, :2]), queries[:, :2])
    # Inputs are built independently of the production block-construction helper.
    correction = ci_module._schur_residuals(
        torch.from_numpy(full_covariance[:2, :2]),
        torch.from_numpy(full_covariance[:2, 2:]),
        torch.from_numpy(full_covariance[2:, 2:]),
        torch.from_numpy(queries[:, :2]), torch.from_numpy(queries[:, 2:]),
    )
    np.testing.assert_allclose(correction.numpy(), full_quadratic - kept_quadratic,
                               atol=1e-12, rtol=1e-12)
    assert bool((correction >= 0).all())


def test_pm_uses_reference_centered_moments_and_unchanged_point_score(ci_module):
    retained, complement, eigenvalues, labels = pm_cloud()
    stats = independent_pm_statistics(retained, complement)
    centered = retained[2:] - retained[2:].mean(axis=0)
    wrong_covariance = centered.T @ centered / (len(centered) - 1) + 1e-6 * np.eye(2)
    assert np.linalg.norm(stats["covariance"] - wrong_covariance) > 1
    diagnostics = {}
    bias, probability = ci_module.pm_ci_components(
        retained, complement, eigenvalues, labels, device="cpu",
        scores={"src000": stats["point_score"]}, diagnostics=diagnostics,
    )
    assert diagnostics["max_score_center_difference"] < 1e-8
    assert probability["src000"] == pytest.approx(
        independent_pm_probability_radius(stats, stats["point_score"]), abs=1e-8
    )
    assert 0 <= bias["src000"] <= 1
    assert stats["full_query_radius"] >= stats["query_radius"]
    assert np.all(stats["full_radii"] >= stats["radii"])
    full_mean = float(stats["full_radii"].mean())
    full_variance = float(stats["full_radii"].var(ddof=1))
    full_score = gammaincc(full_mean**2 / full_variance,
                          stats["full_query_radius"] / (full_variance / full_mean))
    assert abs(full_score - stats["point_score"]) <= bias["src000"] + 1e-8


def test_pm_radii_are_centered_on_supplied_point_score(ci_module):
    retained, complement, eigenvalues, labels = pm_cloud()
    stats = independent_pm_statistics(retained, complement)
    center = .2  # Distinct from the covariance-fit center to expose silent recentering.
    assert abs(center - stats["point_score"]) > .05
    diagnostics = {}
    _, probability = ci_module.pm_ci_components(
        retained, complement, eigenvalues, labels, scores={"src000": center},
        device="cpu", diagnostics=diagnostics,
    )
    assert probability["src000"] == pytest.approx(
        independent_pm_probability_radius(stats, center), abs=1e-8
    )
    assert diagnostics["max_score_center_difference"] == pytest.approx(
        abs(center - stats["point_score"]), abs=1e-8
    )


def two_source_cloud():
    generator = np.random.default_rng(13)
    rows, labels = [], []
    for source, center in [("src000", np.array([0., 0., 0.])),
                           ("src001", np.array([3., 1., 2.]))]:
        rows.extend([center, center + [.3, .1, .2]])
        rows.extend(center + generator.normal(size=(12, 3)) * .4)
        labels.extend([f"{source}-ref", f"{source}-out"]
                      + [f"{source}-d{i}" for i in range(12)])
    full = np.asarray(rows)
    return full[:, :2], full[:, 2:], np.array([.9, .6, .2]), labels


@pytest.mark.parametrize("metric", ["ps", "pm"])
def test_zero_complement_has_no_truncation_but_retains_sampling_uncertainty(metric, ci_module):
    retained, _, eigenvalues, labels = two_source_cloud()
    no_tail = np.empty((len(retained), 0))
    bias, probability = getattr(ci_module, f"{metric}_ci_components")(
        retained, no_tail, eigenvalues[:2], labels, device="cpu"
    )
    assert set(bias) == {"src000", "src001"}
    assert all(value == 0 for value in bias.values())
    assert all(0 < value <= 1 for value in probability.values())


@pytest.mark.parametrize("metric", ["ps", "pm"])
def test_ci_cpu_selection_ignores_advertised_cuda(metric, ci_module, monkeypatch):
    retained, complement, eigenvalues, labels = two_source_cloud()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    bias, probability = getattr(ci_module, f"{metric}_ci_components")(
        retained, complement, eigenvalues, labels, device="cpu"
    )
    assert all(np.isfinite(value) and 0 <= value <= 1
               for value in [*bias.values(), *probability.values()])


@pytest.mark.parametrize("metric", ["ps", "pm"])
def test_ci_is_invariant_to_common_translation(metric, ci_module):
    retained, complement, eigenvalues, labels = two_source_cloud()
    function = getattr(ci_module, f"{metric}_ci_components")
    ordinary = function(retained, complement, eigenvalues, labels, device="cpu")
    translated = function(retained + [10., -4.], complement + 7., eigenvalues,
                          labels, device="cpu")
    for original, moved in zip(ordinary, translated):
        assert original.keys() == moved.keys()
        for source in original:
            assert original[source] == pytest.approx(moved[source], abs=1e-9)
