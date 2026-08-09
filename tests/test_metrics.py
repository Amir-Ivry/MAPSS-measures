import numpy as np
import pytest
import torch

from mapss.metrics import compute_pm, compute_ps, diffusion_map_torch, pm_tail_gamma


def test_diffusion_map_handles_duplicate_points():
    points = np.ones((6, 4), dtype=np.float32)
    labels = [f"source-a-d{index}" for index in range(6)]
    coordinates = diffusion_map_torch(points, labels, eig_solver="full", device="cpu")
    assert np.isfinite(coordinates).all()


def test_metrics_support_hyphens_in_source_ids():
    generator = np.random.default_rng(3)
    coordinates = generator.normal(size=(10, 3)).astype(np.float32)
    labels = [
        "source-a-ref",
        "source-a-out",
        "source-a-d0",
        "source-a-d1",
        "source-a-d2",
        "source-b-ref",
        "source-b-out",
        "source-b-d0",
        "source-b-d1",
        "source-b-d2",
    ]
    ps = compute_ps(coordinates, labels, max_gpus=0)
    pm = compute_pm(coordinates, labels, "gamma", max_gpus=0)
    assert set(ps) == {"source-a", "source-b"}
    assert set(pm) == {"source-a", "source-b"}
    assert all(0 <= score <= 1 for score in (*ps.values(), *pm.values()))


@pytest.mark.parametrize("distance, expected", [(0.0, 1.0), (1.0, 0.0)])
def test_degenerate_gamma_tail_is_finite(distance, expected):
    values = torch.zeros(4)
    assert pm_tail_gamma(distance, values) == expected
