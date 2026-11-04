"""MERT manifest dispatch and existing CSV layout, without model downloads."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mapss.engine import compute_mapss_measures


@pytest.fixture
def music_manifest(tmp_path, monkeypatch):
    calls = []
    references = [tmp_path / "drums.wav", tmp_path / "bass.wav"]
    outputs = [tmp_path / "drums_est.wav", tmp_path / "bass_est.wav"]
    for path in references + outputs:
        path.touch()

    def evaluate(refs, outs, **kwargs):
        calls.append((refs, outs, kwargs))
        names = kwargs["source_names"]
        table = pd.DataFrame({"timestamp_ms": [0.0, 13.5],
                              names[0]: [0.8, np.nan], names[1]: [0.6, np.nan]})
        ci = None
        if kwargs["add_ci"]:
            ci = pd.DataFrame({"timestamp_ms": table["timestamp_ms"]})
            for source in names:
                for suffix in ("ps_bias", "ps_prob", "pm_bias", "pm_prob"):
                    ci[f"{source}_{suffix}"] = [0.1, np.nan]
        return SimpleNamespace(ps=table, pm=table.copy(), ci=ci,
                               metadata={"layer": kwargs["layer"]})

    monkeypatch.setattr(importlib.import_module("mapss.api"), "mapss", evaluate)
    mixtures = [{"mixture_id": "music", "references": references,
                 "systems": {"separator": outputs, "second": outputs}}]
    return mixtures, calls


def run(mixtures, tmp_path, **kwargs):
    options = dict(models=["mert"], mixtures=mixtures, layer=1, add_ci=False,
                   results_root=tmp_path / "results", experiment_id="test", max_gpus=0)
    options.update(kwargs)
    return Path(compute_mapss_measures(**options))


@pytest.mark.parametrize("add_ci", [False, True])
def test_mert_manifest_dispatch_and_csv_layout(music_manifest, tmp_path, add_ci):
    mixtures, calls = music_manifest
    destination = run(mixtures, tmp_path, add_ci=add_ci)
    assert len(calls) == 2
    for refs, outputs, options in calls:
        assert refs == mixtures[0]["references"]
        assert outputs == mixtures[0]["systems"]["separator"]
        assert options["model"] == "mert"
        assert options["layer"] == 1
        assert options["add_ci"] is add_ci
        assert options["max_gpus"] == 0
        assert options["source_names"] == ["music__drums", "music__bass"]
    saved = pd.read_csv(destination / "music/ps_scores_mert.csv")
    assert saved.columns.tolist() == ["timestamp_ms", "second_music__drums", "second_music__bass",
                                      "separator_music__drums", "separator_music__bass"]
    assert saved.iloc[1, 1:].isna().all()
    assert (destination / "music/pm_scores_mert.csv").exists()
    metadata = json.loads((destination / "music/mert_config.json").read_text())
    assert metadata == {"second": {"layer": 1}, "separator": {"layer": 1}}
    assert json.loads((destination / "params.json").read_text())["add_ci"] is add_ci
    ci_path = destination / "music/ci_mert.csv"
    if add_ci:
        ci = pd.read_csv(ci_path)
        expected = ["timestamp_ms"] + [
            f"{source}_{suffix}" for source in saved.columns[1:]
            for suffix in ("ps_bias", "ps_prob", "pm_bias", "pm_prob")
        ]
        assert ci.columns.tolist() == expected
        assert np.array_equal(ci["timestamp_ms"], saved["timestamp_ms"])
        assert ci.iloc[1, 1:].isna().all()
        assert (ci.iloc[0, 1:] == 0.1).all()
    else:
        assert not ci_path.exists()


def test_mert_manifest_selects_requested_system(music_manifest, tmp_path):
    mixtures, calls = music_manifest
    run(mixtures, tmp_path, algos=["separator"])
    assert len(calls) == 1


def test_mert_manifest_missing_policy(music_manifest, tmp_path):
    mixtures, calls = music_manifest
    mixtures[0]["systems"]["missing"] = [tmp_path / "missing.wav"] * 2
    with pytest.raises(ValueError, match="Missing audio"):
        run(mixtures, tmp_path, on_missing="error")
    assert not calls
    with pytest.warns(UserWarning, match="Skipping this system"):
        run(mixtures, tmp_path, on_missing="skip", verbose=True)
    assert len(calls) == 2


def test_mert_manifest_refuses_mixed_backbones(music_manifest, tmp_path):
    mixtures, calls = music_manifest
    with pytest.raises(ValueError, match="separately"):
        run(mixtures, tmp_path, models=["mert", "raw"])
    assert not calls


def test_mert_manifest_score_only_rerun_removes_stale_confidence(music_manifest, tmp_path):
    mixtures, _ = music_manifest
    destination = run(mixtures, tmp_path, add_ci=True)
    ci_path = destination / "music/ci_mert.csv"
    assert ci_path.exists()
    run(mixtures, tmp_path, add_ci=False)
    assert not ci_path.exists()


@pytest.mark.parametrize("table_name", ["pm", "ci"])
def test_mert_manifest_rejects_mismatched_frame_timestamps(
    music_manifest, tmp_path, monkeypatch, table_name
):
    mixtures, _ = music_manifest
    api = importlib.import_module("mapss.api")
    evaluate = api.mapss

    def misaligned(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        getattr(result, table_name).loc[0, "timestamp_ms"] = 1.0
        return result

    monkeypatch.setattr(api, "mapss", misaligned)
    with pytest.raises(RuntimeError, match="different frame timestamps"):
        run(mixtures, tmp_path, add_ci=True)


@pytest.mark.parametrize("missing", ["table", "column"])
def test_mert_manifest_rejects_incomplete_confidence(
    music_manifest, tmp_path, monkeypatch, missing
):
    mixtures, _ = music_manifest
    api = importlib.import_module("mapss.api")
    evaluate = api.mapss

    def incomplete(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        if missing == "table":
            result.ci = None
        else:
            result.ci = result.ci.drop(columns=[result.ci.columns[1]])
        return result

    monkeypatch.setattr(api, "mapss", incomplete)
    with pytest.raises(RuntimeError, match="confidence data"):
        run(mixtures, tmp_path, add_ci=True)


@pytest.mark.parametrize("mixture_id", ["../escape", "nested/music", ".", ".."])
def test_mert_manifest_rejects_unsafe_output_directory(music_manifest, tmp_path, mixture_id):
    mixtures, calls = music_manifest
    mixtures[0]["mixture_id"] = mixture_id
    with pytest.raises(ValueError, match="directory name"):
        run(mixtures, tmp_path)
    assert not calls


def test_mert_manifest_empty_selection_is_an_error(music_manifest, tmp_path):
    mixtures, calls = music_manifest
    with pytest.raises(ValueError, match="No valid MERT"):
        run(mixtures, tmp_path, algos=["absent"])
    assert not calls
