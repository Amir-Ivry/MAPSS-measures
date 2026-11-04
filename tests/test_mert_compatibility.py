"""Regression contracts for every pre-MERT representation and valid layer.

The engine is replaced only at its expensive execution boundary. Audio conversion,
option resolution, source naming, result loading and public API dispatch remain real.
"""
from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from mapss import mapss
from mapss._cli_args import _validate_and_resolve
from mapss.api import _resolve_options
from mapss.config import MODEL_DEFAULT_LAYER, MODEL_MAX_LAYER


# Public v1.1.2 configuration, independent of the candidate's registry.
LEGACY_MODELS = {
    "raw": (0, 0),
    "wavlm": (3, 24),
    "wav2vec2": (2, 24),
    "hubert": (3, 24),
    "wavlm_base": (2, 12),
    "wav2vec2_base": (2, 12),
    "hubert_base": (2, 12),
    "wav2vec2_xlsr": (2, 24),
}
LEGACY_LAYERS = [
    (name, layer)
    for name, (_, maximum) in LEGACY_MODELS.items()
    for layer in range(maximum + 1)
]


@pytest.mark.parametrize("name", LEGACY_MODELS)
def test_every_legacy_model_retains_its_default_and_layer_range(name):
    default, maximum = LEGACY_MODELS[name]
    assert MODEL_DEFAULT_LAYER[name] == default
    assert MODEL_MAX_LAYER[name] == maximum
    assert _resolve_options(name, None, 1.0, 0) == (name, default)
    assert _validate_and_resolve(name, None, None) == (default, 1.0)


@pytest.mark.parametrize("name,layer", LEGACY_LAYERS)
def test_all_140_legacy_layer_selections_resolve_identically(name, layer):
    assert _resolve_options(name, layer, 0.3, 0) == (name, layer)
    assert _validate_and_resolve(name, layer, 0.3) == (layer, 0.3)


@pytest.mark.parametrize(
    "name,layer", LEGACY_LAYERS + [pytest.param(None, None, id="default_wav2vec2_layer2")]
)
def test_every_legacy_layer_uses_original_engine_and_16khz_audio(
    name, layer, monkeypatch
):
    """A MERT import, rate change or argument change fails for any old layer."""
    default_call = name is None
    name, layer = ("wav2vec2", 2) if default_call else (name, layer)
    captured = {}
    engine = types.ModuleType("mapss.engine")

    def compute(**kwargs):
        captured.update(kwargs)
        mixture = kwargs["mixtures"][0]
        assert mixture["mixture_id"] == "api"
        assert tuple(mixture["systems"]) == ("system",)
        for path in mixture["references"] + mixture["systems"]["system"]:
            wave, rate = sf.read(path)
            assert rate == 16_000
            assert wave.shape == (8_000,)
        root = Path(kwargs["results_root"]) / "experiment_api" / "api"
        root.mkdir(parents=True)
        columns = {
            "timestamp_ms": [0.0, 20.0],
            "system_api__reference_00": [0.8, np.nan],
            "system_api__reference_01": [0.4, 0.6],
        }
        for metric in ("ps", "pm"):
            pd.DataFrame(columns).to_csv(root / f"{metric}_scores_{name}.csv", index=False)
        if kwargs["add_ci"]:
            pd.DataFrame(
                {
                    "timestamp_ms": [0.0, 20.0],
                    "system_api__reference_00_ps_bias": [0.01, np.nan],
                    "system_api__reference_01_pm_prob": [0.2, 0.3],
                }
            ).to_csv(root / f"ci_{name}.csv", index=False)
        return root.parent

    engine.compute_mapss_measures = compute
    monkeypatch.setitem(sys.modules, "mapss.engine", engine)
    original_import = builtins.__import__

    def forbid_mert_import(module_name, *args, **kwargs):
        if module_name.rsplit(".", 1)[-1] == "_mert":
            raise AssertionError("An existing model attempted to import MERT")
        return original_import(module_name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_mert_import)
    time = np.arange(8_000, dtype=np.float32) / 16_000
    refs = [np.sin(2 * np.pi * frequency * time) for frequency in (220, 330)]
    ci = bool(layer % 2)
    alpha = (0.0, 0.3, 1.0)[layer % 3]
    max_gpus = None if layer % 2 else 0
    result = mapss(
        refs,
        [wave * 0.9 for wave in refs],
        **({} if default_call else {"model": name, "layer": layer}),
        alpha=alpha,
        add_ci=ci,
        seed=137,
        source_names=["drum", "drum-extra"],
        max_gpus=max_gpus,
        length_policy="error",
        verbose=False,
    )
    for key, value in {
        "models": [name],
        "experiment_id": "api",
        "layer": layer,
        "add_ci": ci,
        "alpha": alpha,
        "seed": 137,
        "on_missing": "error",
        "verbose": False,
        "max_gpus": max_gpus,
    }.items():
        assert captured[key] == value
    assert result.model == name
    assert result.layer == layer
    assert result.sample_rate == 16_000
    assert result.source_names == ("drum", "drum-extra")
    assert result.ps.timestamp_ms.tolist() == [0.0, 20.0]
    assert result.ps["drum"].iloc[0] == 0.8
    assert np.isnan(result.ps["drum"].iloc[1])
    assert result.summary.loc["drum", "ps_frames"] == 1
    if ci:
        assert "drum_ps_bias" in result.ci
        assert "drum-extra_pm_prob" in result.ci
    else:
        assert result.ci is None


@pytest.mark.parametrize("name", LEGACY_MODELS)
def test_legacy_out_of_range_layers_still_rejected(name):
    maximum = LEGACY_MODELS[name][1]
    for layer in (-1, maximum + 1):
        with pytest.raises(ValueError, match="layer"):
            _resolve_options(name, layer, 1.0, 0)

