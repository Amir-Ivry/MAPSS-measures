"""Conservative CI fallback when Gamma numerics cannot supply a finite fit."""

import numpy as np
import pytest

from mapss import _mert_ci


@pytest.mark.parametrize("tail_width", [0, 1])
@pytest.mark.parametrize("supplied_score", [None, 0.3])
def test_nonfinite_gamma_fit_covers_full_score_range(monkeypatch, tail_width, supplied_score):
    retained = np.array([[0.0], [0.8], [0.1], [0.3], [0.7], [1.2]])
    tail = np.arange(len(retained), dtype=float)[:, None] * 0.1
    complement = tail[:, :tail_width]
    labels = ["music-ref", "music-out", "music-d0", "music-d1", "music-d2", "music-d3"]
    scores = None if supplied_score is None else {"music": supplied_score}
    diagnostics = {}
    monkeypatch.setattr(_mert_ci, "gammaincc", lambda *args: np.nan)

    bias, probability = _mert_ci.pm_ci_components(
        retained, complement, np.ones(1 + tail_width), labels,
        scores=scores, diagnostics=diagnostics, device="cpu",
    )

    covering_radius = 1.0 if supplied_score is None else 0.7
    assert probability["music"] == pytest.approx(covering_radius)
    assert bias["music"] == pytest.approx(0.0 if tail_width == 0 else covering_radius)
    assert diagnostics["nonfinite_gamma_fit_count"] == 1
    assert diagnostics["source_evaluations"] == 1
    # An unavailable fitted center must not masquerade as measured agreement.
    assert "max_score_center_difference" not in diagnostics
