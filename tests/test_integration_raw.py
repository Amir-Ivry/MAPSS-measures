import numpy as np
import pytest

from mapss import mapss


@pytest.mark.integration
def test_raw_end_to_end_smoke():
    sample_rate = 16_000
    time = np.arange(int(0.5 * sample_rate), dtype=np.float32) / sample_rate
    references = [
        0.15 * np.sin(2 * np.pi * 220 * time),
        0.15 * np.sin(2 * np.pi * 370 * time),
    ]
    outputs = [
        references[0] + 0.01 * references[1],
        references[1] + 0.01 * references[0],
    ]
    result = mapss(
        references,
        outputs,
        sample_rate=sample_rate,
        model="raw",
        add_ci=False,
        max_gpus=0,
    )
    assert result.ps.drop(columns="timestamp_ms").notna().any().all()
    assert result.pm.drop(columns="timestamp_ms").notna().any().all()
    assert result.summary[["ps", "pm"]].apply(lambda column: column.between(0, 1)).all().all()
