from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from mapss import mapss


def _mixture(count: int, duration: float = 0.5, sample_rate: int = 16_000):
    time = np.arange(int(duration * sample_rate), dtype=np.float32) / sample_rate
    references = [
        0.15
        * (0.7 + 0.3 * np.sin(2 * np.pi * (1.1 + 0.1 * index) * time) ** 2)
        * np.sin(2 * np.pi * (220 + 83 * index) * time + 0.2 * index)
        for index in range(count)
    ]
    outputs = [
        0.96 * reference + 0.01 * references[(index + 1) % count]
        for index, reference in enumerate(references)
    ]
    return references, outputs


@pytest.mark.integration
@pytest.mark.parametrize("source_count", [2, 3, 4])
def test_raw_end_to_end_for_two_three_and_four_sources(source_count, tmp_path):
    references, outputs = _mixture(source_count)
    names = [f"source_{index + 1}" for index in range(source_count)]

    result = mapss(
        references,
        outputs,
        source_names=names,
        model="raw",
        add_ci=True,
        max_gpus=0,
    )
    result.save(tmp_path / f"n{source_count}", plot=True, plot_dpi=72)

    assert result.source_names == tuple(names)
    assert result.ps.drop(columns="timestamp_ms").notna().any().all()
    assert result.pm.drop(columns="timestamp_ms").notna().any().all()
    assert result.ci is not None
    assert result.summary[["ps", "pm"]].apply(
        lambda column: column.between(0, 1)
    ).all().all()
    assert (tmp_path / f"n{source_count}" / "mapss_over_time.png").is_file()


@pytest.mark.integration
def test_mixed_wav_flac_sample_rates_and_stereo_run_end_to_end(tmp_path):
    references_16k, outputs_16k = _mixture(2, duration=0.5, sample_rate=16_000)
    references_8k, outputs_8k = _mixture(2, duration=0.5, sample_rate=8_000)
    reference_paths = [tmp_path / "reference_1.wav", tmp_path / "reference_2.flac"]
    output_paths = [tmp_path / "output_1.flac", tmp_path / "output_2.wav"]

    sf.write(reference_paths[0], np.column_stack((references_16k[0], references_16k[0])), 16_000)
    sf.write(reference_paths[1], references_8k[1], 8_000)
    sf.write(output_paths[0], outputs_16k[0], 16_000)
    sf.write(output_paths[1], np.column_stack((outputs_8k[1], outputs_8k[1])), 8_000)

    result = mapss(
        reference_paths,
        output_paths,
        model="raw",
        add_ci=False,
        max_gpus=0,
    )

    assert result.ps.drop(columns="timestamp_ms").notna().any().all()
    assert result.pm.drop(columns="timestamp_ms").notna().any().all()


@pytest.mark.integration
def test_trim_policy_runs_at_the_shortest_aligned_duration():
    references, outputs = _mixture(2, duration=0.55)
    references[1] = references[1][:-400]
    outputs[0] = outputs[0][:-800]

    result = mapss(
        references,
        outputs,
        model="raw",
        add_ci=False,
        max_gpus=0,
        length_policy="trim",
    )

    assert result.ps.drop(columns="timestamp_ms").notna().any().all()
    assert result.pm.drop(columns="timestamp_ms").notna().any().all()
