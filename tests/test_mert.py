"""MERT integration contracts without downloading a neural checkpoint.

Only model loading/feature extraction and distortion-bank size are replaced.
Input conversion, loudness, activity, temporal alignment, diffusion maps, scores,
source naming and result serialization are exercised together.
"""
from __future__ import annotations

import importlib
import json
import random
import sys
import types

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from mapss import mapss


def signals(rate=24_000, samples=None, count=2):
    count_samples = samples if samples is not None else rate // 2
    time = np.arange(count_samples, dtype=np.float32) / rate
    references = [
        np.sin(2 * np.pi * frequency * time).astype(np.float32)
        for frequency in (220, 330, 440)[:count]
    ]
    return references, [np.roll(wave, 4) for wave in references]


@pytest.fixture
def mert_module():
    return importlib.import_module("mapss._mert")


@pytest.fixture
def small_mert(monkeypatch, mert_module):
    """Deterministic native-frame features plus complete small test banks."""
    observed = {"loads": [], "waves": [], "bank_rates": [], "clouds": [], "layers": []}

    def load_model(device):
        observed["loads"].append(str(device))
        # Loading an actual checkpoint may consume all three RNGs.
        random.random()
        np.random.random()
        torch.rand(1)
        return object(), object()

    def extract(waveforms, model, extractor, layer, device, **kwargs):
        observed["layers"].append(layer)
        result = []
        for wave in waveforms:
            wave = np.asarray(wave)
            observed["waves"].append(wave.copy())
            frames = np.lib.stride_tricks.sliding_window_view(wave, 400)[::320]
            # The first coordinate independently exposes the native frame selected
            # downstream. The other coordinates distinguish source/distortion audio.
            result.append(
                np.column_stack(
                    [
                        np.arange(len(frames)),
                        frames[:, 0],
                        frames[:, 73],
                        frames[:, 211],
                        frames.mean(axis=1),
                        np.sqrt(np.mean(frames**2, axis=1)),
                    ]
                ).astype(np.float32)
            )
        return result

    def bank(reference, kind, sr):
        observed["bank_rates"].append((kind, sr))
        time = np.arange(len(reference), dtype=np.float32) / sr
        # Use random draws to check local seeding/restoration in the real evaluator.
        offset = 0.01 * (random.random() + np.random.random() + torch.rand(1).item())
        variants = [
            np.roll(reference, 9),
            np.clip(reference, -0.15, 0.15),
            reference + (0.04 + offset) * np.sin(2 * np.pi * 137 * time),
        ]
        if kind == "PM":
            variants.append(reference[::-1].copy())
        return variants

    def ps_bank(reference, keys, sr):
        assert keys == "all"
        return bank(reference, "PS", sr)

    def pm_bank(reference, keys, sr):
        assert keys == "all"
        return bank(reference, "PM", sr)

    original_diffusion = mert_module.diffusion_map_torch

    def diffusion(cloud, labels, **kwargs):
        observed["clouds"].append((np.asarray(cloud).copy(), tuple(labels)))
        return original_diffusion(cloud, labels, **kwargs)

    monkeypatch.setattr(mert_module, "_load_mert_model", load_model)
    monkeypatch.setattr(mert_module, "_extract_features", extract)
    monkeypatch.setattr(mert_module, "apply_ps_distortions", ps_bank)
    monkeypatch.setattr(mert_module, "apply_pm_distortions", pm_bank)
    monkeypatch.setattr(mert_module, "diffusion_map_torch", diffusion)
    return observed


def evaluate(references, outputs, **options):
    settings = {
        "model": "mert",
        "layer": 1,
        "sample_rate": 24_000,
        "add_ci": False,
        "max_gpus": 0,
    }
    settings.update(options)
    return mapss(references, outputs, **settings)


@pytest.mark.parametrize("layer", [None, -1, 13, True, np.bool_(False), 1.5])
def test_invalid_or_missing_mert_layer_fails_before_model_load(layer, monkeypatch, mert_module):
    def forbidden_load(*args, **kwargs):
        pytest.fail("Invalid options must fail before loading MERT")

    monkeypatch.setattr(mert_module, "_load_mert_model", forbidden_load)
    references, outputs = signals()
    with pytest.raises((TypeError, ValueError), match="layer"):
        evaluate(references, outputs, layer=layer)


def test_mert_default_ci_keeps_point_scores_and_uses_exact_score_mask(small_mert):
    references, _ = signals(samples=24_000, count=3)
    references[1][14_400:] = 0
    references[2][:9_600] = 0
    outputs = [np.roll(wave, 4) for wave in references]
    names = ["a", "a-b", "a-b-out"]
    plain = evaluate(references, outputs, source_names=names)
    # Deliberately omit add_ci: the standard API default now also applies to MERT.
    with_ci = mapss(references, outputs, source_names=names, sample_rate=24_000,
                    model="mert", layer=1, max_gpus=0)
    pd.testing.assert_frame_equal(with_ci.ps, plain.ps, check_exact=True)
    pd.testing.assert_frame_equal(with_ci.pm, plain.pm, check_exact=True)
    assert with_ci.ci is not None
    assert with_ci.metadata["confidence_intervals"]["computed"] is True
    assert with_ci.ci.columns.tolist() == [
        "timestamp_ms", *[f"{name}_{kind}" for name in names
                          for kind in ("ps_bias", "ps_prob", "pm_bias", "pm_prob")]
    ]
    np.testing.assert_array_equal(with_ci.ci.timestamp_ms, with_ci.ps.timestamp_ms)
    for name in names:
        for metric, table in [("ps", with_ci.ps), ("pm", with_ci.pm)]:
            for kind in ("bias", "prob"):
                values = with_ci.ci[f"{name}_{metric}_{kind}"]
                np.testing.assert_array_equal(values.notna(), table[name].notna())
                assert values.dropna().between(0, 1).all()
    assert with_ci.ci.filter(like="_prob").max().max() > 0


def test_ci_failure_restores_random_state(small_mert, mert_module, monkeypatch):
    references, outputs = signals()
    before = random_states()

    def fail_confidence(*args, **kwargs):
        random.random()
        np.random.random()
        torch.rand(1)
        raise RuntimeError("test confidence failure")

    monkeypatch.setattr(mert_module, "ps_ci_components", fail_confidence)
    with pytest.raises(RuntimeError, match="test confidence failure"):
        evaluate(references, outputs, add_ci=True)
    assert_same_random_states(before, random_states())


def test_ci_public_api_honors_forced_cpu_on_a_cuda_host(small_mert, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    references, outputs = signals()
    result = evaluate(references, outputs, add_ci=True, max_gpus=0)
    assert result.ci is not None
    assert small_mert["loads"] == ["cpu"]
    assert result.metadata["device"] == "cpu"


@pytest.mark.parametrize("layer", range(13))
def test_every_mert_layer_reaches_extractor_and_cli(layer, small_mert):
    from mapss._cli_args import _validate_and_resolve

    references, outputs = signals()
    result = evaluate(references, outputs, layer=layer)
    assert result.layer == layer
    assert small_mert["layers"] == [layer]
    assert _validate_and_resolve("mert", layer, None) == (layer, 1.0)


@pytest.mark.parametrize("layer", [None, -1, 13])
def test_cli_requires_explicit_valid_mert_layer(layer):
    from mapss._cli_args import _validate_and_resolve

    with pytest.raises(SystemExit, match="layer"):
        _validate_and_resolve("mert", layer, None)


@pytest.mark.parametrize("rate", [16_000, 24_000, 44_100, 48_000])
@pytest.mark.parametrize("input_kind", ["array", "file"])
def test_file_and_array_rates_reach_mert_at_24khz(rate, input_kind, small_mert, tmp_path):
    references, outputs = signals(rate)
    supplied_rate = rate
    if input_kind == "file":
        paths = []
        for index, wave in enumerate(references + outputs):
            path = tmp_path / f"source_{index}.wav"
            sf.write(path, np.column_stack([wave, 0.7 * wave]), rate, subtype="FLOAT")
            paths.append(path)
        references, outputs = paths[:2], paths[2:]
        # For files, metadata takes precedence over the in-memory-array argument.
        supplied_rate = 8_000
    result = evaluate(references, outputs, sample_rate=supplied_rate)
    assert result.sample_rate == 24_000
    assert result.model == "mert"
    assert result.layer == 1
    assert result.ci is None
    assert small_mert["loads"] == ["cpu"]
    assert small_mert["waves"]
    assert all(wave.ndim == 1 and len(wave) == 12_000 for wave in small_mert["waves"])
    assert all(np.isfinite(wave).all() for wave in small_mert["waves"])
    assert all(sr == 24_000 for _, sr in small_mert["bank_rates"])
    frequencies = np.fft.rfftfreq(12_000, 1 / 24_000)
    first_reference = small_mert["waves"][0]
    dominant = frequencies[np.abs(np.fft.rfft(first_reference)).argmax()]
    assert dominant == pytest.approx(220, abs=2)
    assert result.ps.timestamp_ms.iloc[1] - result.ps.timestamp_ms.iloc[0] == 13.5


def test_equal_duration_mixed_rate_files_do_not_need_trimming(small_mert, tmp_path):
    paths = []
    for index, rate in enumerate((16_000, 24_000, 44_100, 48_000)):
        references, _ = signals(rate)
        path = tmp_path / f"mixed_{index}.wav"
        sf.write(path, references[index % 2], rate, subtype="FLOAT")
        paths.append(path)
    result = evaluate(paths[:2], paths[2:], length_policy="error")
    assert all(len(wave) == 12_000 for wave in small_mert["waves"])
    assert result.metadata["input_samples_at_24000_before_trim"] == [12_000] * 4
    assert result.metadata["length_policy"] == "error"


@pytest.mark.parametrize("problem", ["silence", "nonfinite", "too_short", "unequal"])
def test_invalid_audio_fails_before_model_loading(problem, monkeypatch, mert_module):
    monkeypatch.setattr(
        mert_module, "_load_mert_model", lambda *args, **kwargs: pytest.fail("Loaded invalid audio")
    )
    references, outputs = signals()
    if problem == "silence":
        references[0][:] = 0
    elif problem == "nonfinite":
        outputs[1][5] = np.nan
    elif problem == "too_short":
        references = [wave[:9_599] for wave in references]
        outputs = [wave[:9_599] for wave in outputs]
    else:
        outputs[1] = outputs[1][:-1]
    with pytest.raises(ValueError):
        evaluate(references, outputs)


def test_explicit_trim_policy_reaches_shortest_common_duration(small_mert):
    references, outputs = signals()
    outputs[1] = outputs[1][:11_760]
    result = evaluate(references, outputs, length_policy="trim")
    assert all(len(wave) == 11_760 for wave in small_mert["waves"])
    assert result.summary.ps_frames.min() > 0


def test_source_names_do_not_change_scores_or_confuse_prefixes(small_mert):
    references, outputs = signals(count=3)
    ordinary = evaluate(references, outputs, source_names=["one", "two", "three"])
    prefixed = evaluate(references, outputs, source_names=["a", "a-b", "a-b-out"])
    assert prefixed.source_names == ("a", "a-b", "a-b-out")
    np.testing.assert_allclose(ordinary.ps.to_numpy(), prefixed.ps.to_numpy(), equal_nan=True)
    np.testing.assert_allclose(ordinary.pm.to_numpy(), prefixed.pm.to_numpy(), equal_nan=True)


def test_joint_active_source_graphs_use_correct_native_times_and_separate_banks(small_mert):
    references, outputs = signals(samples=24_000, count=3)
    references[1][14_400:] = 0
    references[2][:9_600] = 0
    outputs = [np.roll(wave, 4) for wave in references]
    result = evaluate(references, outputs, source_names=["a", "a-b", "a-b-out"])
    from mapss.audio import compute_speaker_activity_masks
    from mapss._mert import loudness_normalize

    normalized = [torch.from_numpy(loudness_normalize(wave, sr=24_000)) for wave in references]
    multi, individual = compute_speaker_activity_masks(normalized, 480, 324)
    eligible = np.flatnonzero(multi.numpy())
    masks = np.stack([item.numpy() for item in individual])
    expected_native = np.rint((eligible * 324 + 40) / 320).astype(int)
    expected_active_counts = masks[:, eligible].sum(axis=0)
    assert set(expected_active_counts) == {2, 3}
    assert len(small_mert["clouds"]) == 2 * len(eligible)
    for metric_index, rows_per_source in enumerate((5, 6)):
        clouds = small_mert["clouds"][metric_index * len(eligible):(metric_index + 1) * len(eligible)]
        for (cloud, labels), native, active in zip(clouds, expected_native, expected_active_counts):
            assert len(labels) == active * rows_per_source
            assert len(labels) == len(set(labels))
            assert len({label.rsplit("-", 1)[0] for label in labels}) == active
            np.testing.assert_array_equal(cloud[:, 0], native)
        table = result.ps if metric_index == 0 else result.pm
        for source_index, name in enumerate(result.source_names):
            np.testing.assert_array_equal(
                table[name].notna().to_numpy(), multi.numpy() & masks[source_index]
            )
    np.testing.assert_allclose(result.ps.timestamp_ms, np.arange(len(multi)) * 13.5)


def test_irregular_tail_uses_last_available_feature_and_keeps_activity_frame(small_mert):
    references, outputs = signals(samples=10_525)
    result = evaluate(references, outputs)
    # 32 native features; the final activity center is nearest unavailable index32,
    # so nearest-available alignment selects31 and retains that final score.
    assert len(result.ps) == 32
    assert result.ps.timestamp_ms.iloc[-1] == pytest.approx(31 * 13.5)
    assert result.ps.iloc[-1, 1:].notna().all()
    assert result.pm.iloc[-1, 1:].notna().all()
    assert max(int(cloud[:, 0].max()) for cloud, _ in small_mert["clouds"]) == 31
    np.testing.assert_array_equal(small_mert["clouds"][31][0][:, 0], 31)
    np.testing.assert_array_equal(small_mert["clouds"][-1][0][:, 0], 31)
    timing = result.metadata["timing"]
    assert timing["boundary_clipped_waveform_frame_pairs"] == len(small_mert["waves"])
    assert timing["max_center_error_seconds"] == pytest.approx(164 / 24_000)
    assert timing["native_frame_counts"] == [32]


def random_states():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def restore_random_states(states):
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.random.set_rng_state(states[2])


def assert_same_random_states(left, right):
    assert left[0] == right[0]
    assert left[1][0] == right[1][0]
    np.testing.assert_array_equal(left[1][1], right[1][1])
    assert left[1][2:] == right[1][2:]
    assert torch.equal(left[2], right[2])


@pytest.mark.parametrize("fail", [False, True])
def test_mert_restores_callers_random_states_on_success_and_failure(
    fail, small_mert, mert_module, monkeypatch
):
    saved = random_states()
    try:
        random.seed(9)
        np.random.seed(10)
        torch.manual_seed(11)
        before = random_states()
        references, outputs = signals()
        if fail:
            def broken_extractor(*args, **kwargs):
                random.random()
                np.random.random()
                torch.rand(1)
                raise RuntimeError("test feature extraction failure")

            monkeypatch.setattr(mert_module, "_extract_features", broken_extractor)
            with pytest.raises(RuntimeError, match="test feature extraction failure"):
                evaluate(references, outputs, seed=72)
        else:
            evaluate(references, outputs, seed=72)
        assert_same_random_states(before, random_states())
    finally:
        restore_random_states(saved)


def test_mert_result_saves_score_only_plot_and_metadata(small_mert, tmp_path):
    references, outputs = signals()
    result = evaluate(references, outputs)
    destination = result.save(tmp_path / "music", plot=True, plot_dpi=72)
    assert (destination / "mapss_over_time.png").stat().st_size > 1_000
    assert (destination / "mert_config.json").is_file()
    metadata = json.loads((destination / "mert_config.json").read_text())
    assert metadata["sample_rate"] == 24_000
    assert metadata["layer"] == 1
    assert metadata["confidence_intervals"]["computed"] is False
    assert not (destination / "confidence.csv").exists()
    saved = pd.read_csv(destination / "ps_scores.csv")
    pd.testing.assert_frame_equal(saved, result.ps, check_exact=False)


def test_mert_ci_saves_six_panels_and_score_only_overwrite_removes_stale_ci(
    small_mert, monkeypatch, tmp_path
):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    references, outputs = signals()
    with_ci = evaluate(references, outputs, add_ci=True)
    axes_counts = []
    original_savefig = Figure.savefig

    def record_savefig(figure, *args, **kwargs):
        axes_counts.append(len(figure.axes))
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", record_savefig)
    destination = with_ci.save(tmp_path / "music", plot=True, plot_dpi=72)
    assert axes_counts == [6]
    assert (destination / "mapss_over_time.png").stat().st_size > 1_000
    saved = pd.read_csv(destination / "confidence.csv")
    pd.testing.assert_frame_equal(saved, with_ci.ci, check_exact=False)
    metadata = json.loads((destination / "mert_config.json").read_text())
    assert metadata["confidence_intervals"]["computed"] is True

    plain = evaluate(references, outputs, add_ci=False)
    plain.save(destination, plot=True, plot_dpi=72)
    assert axes_counts == [6, 2]
    assert not (destination / "confidence.csv").exists()
    metadata = json.loads((destination / "mert_config.json").read_text())
    assert metadata["confidence_intervals"]["computed"] is False
    assert not plt.get_fignums()


class CountedBlock(torch.nn.Module):
    def __init__(self, increment):
        super().__init__()
        self.increment = increment
        self.calls = 0

    def forward(self, hidden):
        self.calls += 1
        return hidden + self.increment


class TinyMert(torch.nn.Module):
    """Transformer-shaped test double with independently predictable states."""

    def __init__(self, mismatch_layer=None, fail_on_call=None):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.layers = torch.nn.ModuleList([CountedBlock(i + 1) for i in range(12)])
        self.mismatch_layer = mismatch_layer
        self.fail_on_call = fail_on_call
        self.calls = 0

    def forward(self, values, *, output_hidden_states):
        self.calls += 1
        assert output_hidden_states is True
        assert values.dtype == torch.float32
        assert not torch.is_grad_enabled()
        if self.calls == self.fail_on_call:
            raise RuntimeError("test encoder failure")
        frames = values.unfold(1, 400, 320)
        hidden = torch.stack([frames[:, :, 0], frames[:, :, -1]], dim=-1)
        states = [hidden]
        for block in self.encoder.layers:
            hidden = block(hidden)
            states.append(hidden)
        if self.mismatch_layer is not None:
            states[self.mismatch_layer] = states[self.mismatch_layer] + 0.25
        return types.SimpleNamespace(hidden_states=tuple(states))


class TinyExtractor:
    def __init__(self):
        self.batches = []

    def __call__(self, waves, *, sampling_rate, return_tensors, padding):
        assert sampling_rate == 24_000
        assert return_tensors == "pt"
        assert padding is True
        assert len(set(map(len, waves))) == 1, "Unequal audio must not share a padded batch"
        self.batches.append([len(wave) for wave in waves])
        return types.SimpleNamespace(input_values=torch.tensor(np.stack(waves), dtype=torch.float64))


def feature_inputs():
    return [
        np.full(length, index + 1, dtype=np.float32)
        for index, length in enumerate((1040, 1360, 1040, 1680, 1360))
    ]


def assert_feature_values(features, waves, layer, offset=0):
    increment = layer * (layer + 1) / 2 + offset
    assert len(features) == len(waves)
    for index, (feature, wave) in enumerate(zip(features, waves)):
        assert feature.shape == ((len(wave) - 400) // 320 + 1, 2)
        assert feature.dtype == np.float32
        np.testing.assert_array_equal(feature, index + 1 + increment)


@pytest.mark.parametrize("layer", [0, 3])
def test_feature_prefix_is_verified_then_skips_unused_blocks_and_preserves_order(layer, mert_module):
    model, extractor, diagnostics = TinyMert(), TinyExtractor(), {}
    waves = feature_inputs()
    features = mert_module._extract_features(
        waves, model, extractor, layer, "cpu", diagnostics=diagnostics
    )
    assert_feature_values(features, waves, layer)
    assert extractor.batches == [[1040, 1040], [1360, 1360], [1680]]
    assert model.calls == 4  # One full pass, one validation and two prefix passes.
    assert model.encoder.layers[layer].calls == 1
    assert diagnostics["prefix_validation"] == "exact"
    assert diagnostics["prefix_torch_equal"] is True
    assert diagnostics["prefix_max_abs_difference"] == 0
    assert all(not block._forward_pre_hooks for block in model.encoder.layers)


def test_feature_prefix_mismatch_uses_full_hidden_states_for_every_batch(mert_module):
    model, extractor, diagnostics = TinyMert(mismatch_layer=3), TinyExtractor(), {}
    waves = feature_inputs()
    features = mert_module._extract_features(
        waves, model, extractor, 3, "cpu", diagnostics=diagnostics
    )
    assert_feature_values(features, waves, 3, offset=0.25)
    assert model.encoder.layers[-1].calls == 3
    assert diagnostics["prefix_validation"] == "unequal_full_forward"
    assert diagnostics["prefix_torch_equal"] is False
    assert diagnostics["prefix_max_abs_difference"] == 0.25
    assert all(not block._forward_pre_hooks for block in model.encoder.layers)


def test_final_layer_uses_full_forward_without_a_following_block(mert_module):
    model, extractor, diagnostics = TinyMert(), TinyExtractor(), {}
    waves = feature_inputs()
    features = mert_module._extract_features(
        waves, model, extractor, 12, "cpu", diagnostics=diagnostics
    )
    assert_feature_values(features, waves, 12)
    assert model.calls == 3
    assert diagnostics["prefix_validation"] == "unavailable_full_forward"
    assert all(not block._forward_pre_hooks for block in model.encoder.layers)


@pytest.mark.parametrize("failure_call", [2, 3])
def test_prefix_hooks_are_removed_when_encoder_raises(failure_call, mert_module):
    model, extractor, diagnostics = TinyMert(fail_on_call=failure_call), TinyExtractor(), {}
    waves = feature_inputs()
    if failure_call == 2:
        features = mert_module._extract_features(
            waves, model, extractor, 3, "cpu", diagnostics=diagnostics
        )
        assert_feature_values(features, waves, 3)
        assert diagnostics["prefix_validation"] == "unavailable_full_forward"
    else:
        with pytest.raises(RuntimeError, match="test encoder failure"):
            mert_module._extract_features(waves, model, extractor, 3, "cpu")
    assert all(not block._forward_pre_hooks for block in model.encoder.layers)


def test_loader_pins_checkpoint_and_custom_code_and_freezes_float32_model(monkeypatch, mert_module):
    calls = {}
    model, extractor = torch.nn.Linear(2, 2), object()

    def load_model(name, **kwargs):
        calls["model"] = (name, kwargs)
        return model

    def load_extractor(name, **kwargs):
        calls["extractor"] = (name, kwargs)
        return extractor

    fake_transformers = types.SimpleNamespace(
        AutoModel=types.SimpleNamespace(from_pretrained=load_model),
        Wav2Vec2FeatureExtractor=types.SimpleNamespace(from_pretrained=load_extractor),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(torch, "__version__", "2.6.0+cpu")
    actual_model, actual_extractor = mert_module._load_mert_model("cpu")
    assert actual_model is model and actual_extractor is extractor
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert calls["model"] == (
        "m-a-p/MERT-v1-95M",
        {
            "revision": "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5",
            "trust_remote_code": True,
            "use_safetensors": False,
            "torch_dtype": torch.float32,
            "attn_implementation": "eager",
            "output_hidden_states": True,
        },
    )
    assert calls["extractor"] == (
        "m-a-p/MERT-v1-95M", {"revision": "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"}
    )


def test_loader_refuses_old_torch_before_checkpoint_loading(monkeypatch, mert_module):
    forbidden = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: pytest.fail("Old PyTorch reached checkpoint loading")
    )
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoModel=forbidden, Wav2Vec2FeatureExtractor=forbidden
    ))
    monkeypatch.setattr(torch, "__version__", "2.5.1+cpu")
    with pytest.raises(RuntimeError, match="PyTorch >=2.6"):
        mert_module._load_mert_model("cpu")


@pytest.mark.parametrize("with_ci", [False, True])
def test_cli_mert_can_save_score_or_confidence_plot(with_ci, monkeypatch, tmp_path):
    from mapss import cli

    captured = {}

    def save(directory, *, plot):
        captured["save"] = (directory, plot)
        return directory

    def evaluate_cli(references, outputs, **options):
        captured["evaluate"] = (references, outputs, options)
        return types.SimpleNamespace(summary=pd.DataFrame({"ps_mean": [0.8]}), save=save)

    monkeypatch.setattr(cli, "mapss", evaluate_cli)
    arguments = [
        "--reference", "a.wav", "b.wav", "--output", "x.wav", "y.wav",
        "--model", "mert", "--layer", "3", "--plot",
        "--max-gpus", "0", "--results-dir", str(tmp_path),
    ]
    if not with_ci:
        arguments.append("--no-ci")
    assert cli.main(arguments) == 0
    assert captured["evaluate"][2]["model"] == "mert"
    assert captured["evaluate"][2]["layer"] == 3
    assert captured["evaluate"][2]["add_ci"] is with_ci
    assert captured["evaluate"][2]["max_gpus"] == 0
    assert captured["save"] == (tmp_path, True)


@pytest.mark.parametrize("options", [[], ["--layer", "-1"], ["--layer", "13", "--no-ci"]])
def test_cli_rejects_incomplete_mert_options_before_evaluation(options, monkeypatch):
    from mapss import cli

    monkeypatch.setattr(cli, "mapss", lambda *args, **kwargs: pytest.fail("Invalid CLI reached inference"))
    with pytest.raises(SystemExit, match="layer"):
        cli.main([
            "--reference", "a.wav", "b.wav", "--output", "x.wav", "y.wav",
            "--model", "mert", *options,
        ])

