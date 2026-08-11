from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from mapss import mapss
from mapss.api import _as_mono_array


def _signals(count: int = 2, length: int = 8_000) -> tuple[list[np.ndarray], list[np.ndarray]]:
    time = np.arange(length, dtype=np.float32) / 16_000
    references = [
        0.15 * np.sin(2 * np.pi * (220 + 71 * index) * time + 0.2 * index)
        for index in range(count)
    ]
    outputs = [
        0.96 * reference + 0.01 * references[(index + 1) % count]
        for index, reference in enumerate(references)
    ]
    return references, outputs


@pytest.mark.parametrize("reference_count,output_count", [(3, 2), (2, 3)])
def test_count_mismatch_reports_both_counts(reference_count, output_count):
    references, _ = _signals(reference_count)
    _, outputs = _signals(output_count)

    with pytest.raises(
        ValueError,
        match=(
            rf"received {reference_count} reference source\(s\) and "
            rf"{output_count} output source\(s\)"
        ),
    ):
        mapss(references, outputs, model="raw", add_ci=False)


def test_duration_error_identifies_each_input_length():
    references, outputs = _signals()
    outputs[1] = outputs[1][:-100]

    with pytest.raises(
        ValueError,
        match=r"reference\[0\]=.*output\[1\]=.*length_policy='trim'",
    ):
        mapss(references, outputs, model="raw", add_ci=False)


@pytest.mark.parametrize(
    "source_names,expected",
    [
        ("speaker", "one name per source"),
        (["speaker"], "must contain 2 names"),
        (["speaker", " speaker "], "must be unique"),
        (["speaker", "   "], "whitespace-only"),
        (["speaker", "timestamp_ms"], "reserved name"),
    ],
)
def test_source_name_guardrails(source_names, expected):
    references, outputs = _signals()

    with pytest.raises((TypeError, ValueError), match=expected):
        mapss(
            references,
            outputs,
            source_names=source_names,
            model="raw",
            add_ci=False,
        )


@pytest.mark.parametrize(
    "keyword,value,error",
    [
        ("sample_rate", True, "positive integer"),
        ("add_ci", "yes", "True or False"),
        ("verbose", 1, "True or False"),
        ("seed", 4.2, "must be an integer"),
        ("layer", 0.5, "must be an integer"),
        ("alpha", float("nan"), "alpha must be in"),
        ("max_gpus", True, "non-negative integer"),
    ],
)
def test_option_types_fail_before_expensive_processing(keyword, value, error):
    references, outputs = _signals()
    options = {"model": "raw", "add_ci": False, keyword: value}

    with pytest.raises((TypeError, ValueError), match=error):
        mapss(references, outputs, **options)


def test_explicit_gpu_request_fails_when_cuda_is_unavailable(monkeypatch):
    references, outputs = _signals()
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with pytest.raises(RuntimeError, match=r"max_gpus=1.*cannot access a CUDA GPU"):
        mapss(
            references,
            outputs,
            model="raw",
            add_ci=False,
            max_gpus=1,
        )


def test_corrupt_audio_reports_path_and_decoding_problem(tmp_path):
    corrupt = tmp_path / "not_audio.mp3"
    corrupt.write_text("This is not an audio file.", encoding="utf-8")

    with pytest.raises(ValueError, match=r"reference\[0\].*could not be decoded.*not_audio.mp3"):
        _as_mono_array(corrupt, 16_000, "reference[0]")


def test_directory_path_reports_that_an_audio_file_is_required(tmp_path):
    with pytest.raises(ValueError, match="audio file, not a directory"):
        _as_mono_array(tmp_path, 16_000, "output[1]")


def test_non_numeric_samples_report_the_source_label():
    with pytest.raises(TypeError, match=r"output\[0\].*numeric audio samples"):
        _as_mono_array(["not", "audio"], 16_000, "output[0]")


def test_channel_first_array_is_rejected_with_transpose_guidance():
    waveform = np.ones((2, 8_000), dtype=np.float32)

    with pytest.raises(ValueError, match=r"channels, samples.*transpose"):
        _as_mono_array(waveform, 16_000, "reference[0]")


def test_wav_flac_stereo_and_resampling_are_supported(tmp_path):
    time_8k = np.arange(4_000, dtype=np.float32) / 8_000
    mono = 0.15 * np.sin(2 * np.pi * 220 * time_8k)
    stereo = np.column_stack((mono, 0.8 * mono))
    wav_path = tmp_path / "source.wav"
    flac_path = tmp_path / "source.flac"
    sf.write(wav_path, stereo, 8_000)
    sf.write(flac_path, mono, 8_000)

    wav = _as_mono_array(wav_path, 16_000, "reference[0]")
    flac = _as_mono_array(flac_path, 16_000, "output[0]")

    assert wav.ndim == flac.ndim == 1
    assert len(wav) == len(flac) == 8_000
    assert np.isfinite(wav).all()
    assert np.isfinite(flac).all()


def test_three_dimensional_source_batch_is_rejected():
    references = np.ones((2, 2, 8_000), dtype=np.float32)
    _, outputs = _signals()

    with pytest.raises(ValueError, match="reference must be one- or two-dimensional"):
        mapss(references, outputs, model="raw", add_ci=False)


def test_audio_shorter_than_loudness_window_is_rejected():
    references, outputs = _signals(length=6_399)

    with pytest.raises(ValueError, match="at least 0.400 seconds"):
        mapss(references, outputs, model="raw", add_ci=False)
