from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from mapss import MAPSSResult, mapss


def _signals(length=8000):
    time = np.arange(length, dtype=np.float32) / 16_000
    refs = [np.sin(2 * np.pi * 220 * time), np.sin(2 * np.pi * 330 * time)]
    outs = [refs[0] * 0.95, refs[1] * 0.90]
    return refs, outs


class TestMAPSSResult(unittest.TestCase):
    def test_summary_ignores_inactive_frames(self):
        ps = pd.DataFrame(
            {"timestamp_ms": [0, 20, 40], "alice": [0.8, np.nan, 0.6]}
        )
        pm = pd.DataFrame(
            {"timestamp_ms": [0, 20, 40], "alice": [0.5, np.nan, 0.9]}
        )
        result = MAPSSResult(ps, pm, None, "raw", 0, 16_000)
        self.assertAlmostEqual(result.summary.loc["alice", "ps"], 0.7)
        self.assertAlmostEqual(result.summary.loc["alice", "pm"], 0.7)
        self.assertEqual(result.summary.loc["alice", "ps_frames"], 2)

    def test_save_writes_expected_tables(self):
        ps = pd.DataFrame({"timestamp_ms": [0], "source_0": [0.8]})
        pm = pd.DataFrame({"timestamp_ms": [0], "source_0": [0.7]})
        result = MAPSSResult(ps, pm, None, "raw", 0, 16_000)
        with tempfile.TemporaryDirectory() as directory:
            result.save(directory)
            self.assertTrue((Path(directory) / "ps_scores.csv").is_file())
            self.assertTrue((Path(directory) / "pm_scores.csv").is_file())
            self.assertTrue((Path(directory) / "summary.csv").is_file())

    def test_save_can_write_paper_style_plot(self):
        ps = pd.DataFrame(
            {"timestamp_ms": [0, 20], "source_1": [0.8, 0.7], "source_2": [0.9, 0.6]}
        )
        pm = pd.DataFrame(
            {"timestamp_ms": [0, 20], "source_1": [0.7, 0.6], "source_2": [0.8, 0.5]}
        )
        ci = pd.DataFrame({"timestamp_ms": [0, 20]})
        for source in ("source_1", "source_2"):
            ci[f"{source}_ps_bias"] = [0.01, 0.02]
            ci[f"{source}_ps_prob"] = [0.2, 0.3]
            ci[f"{source}_pm_bias"] = [0.005, 0.006]
            ci[f"{source}_pm_prob"] = [0.1, 0.15]
        result = MAPSSResult(ps, pm, ci, "raw", 0, 16_000)

        with tempfile.TemporaryDirectory() as directory:
            result.save(directory, plot=True, plot_dpi=72)
            self.assertTrue((Path(directory) / "mapss_over_time.png").is_file())

    def test_save_plot_requires_confidence_data(self):
        ps = pd.DataFrame({"timestamp_ms": [0], "source_1": [0.8]})
        pm = pd.DataFrame({"timestamp_ms": [0], "source_1": [0.7]})
        result = MAPSSResult(ps, pm, None, "raw", 0, 16_000)

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "requires confidence data"):
                result.save(directory, plot=True)

    def test_save_rejects_unsafe_or_unsupported_plot_paths(self):
        ps = pd.DataFrame({"timestamp_ms": [0], "source_1": [0.8]})
        pm = pd.DataFrame({"timestamp_ms": [0], "source_1": [0.7]})
        ci = pd.DataFrame(
            {
                "timestamp_ms": [0],
                "source_1_ps_bias": [0.01],
                "source_1_ps_prob": [0.2],
                "source_1_pm_bias": [0.005],
                "source_1_pm_prob": [0.1],
            }
        )
        result = MAPSSResult(ps, pm, ci, "raw", 0, 16_000)

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "relative path"):
                result.save(directory, plot=True, plot_filename="../outside.png")
            with self.assertRaisesRegex(ValueError, ".png, .pdf, or .svg"):
                result.save(directory, plot=True, plot_filename="figure.txt")


class TestInputValidation(unittest.TestCase):
    def test_requires_at_least_two_sources(self):
        refs, outs = _signals()
        with self.assertRaisesRegex(ValueError, "at least two"):
            mapss(refs[:1], outs[:1], model="raw", add_ci=False)

    def test_requires_matching_source_counts(self):
        refs, outs = _signals()
        with self.assertRaisesRegex(ValueError, "same number"):
            mapss(refs, outs[:1], model="raw", add_ci=False)

    def test_rejects_nonfinite_audio(self):
        refs, outs = _signals()
        refs[0][5] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            mapss(refs, outs, model="raw", add_ci=False)

    def test_rejects_silent_audio(self):
        refs, outs = _signals()
        refs[0] = np.zeros_like(refs[0])
        with self.assertRaisesRegex(ValueError, "entirely silent"):
            mapss(refs, outs, model="raw", add_ci=False)

    def test_rejects_unequal_lengths_by_default(self):
        refs, outs = _signals()
        outs[1] = outs[1][:-1]
        with self.assertRaisesRegex(ValueError, "equal duration"):
            mapss(refs, outs, model="raw", add_ci=False)

    def test_rejects_invalid_configuration(self):
        refs, outs = _signals()
        with self.assertRaisesRegex(ValueError, "alpha"):
            mapss(refs, outs, model="raw", alpha=2.0, add_ci=False)
        with self.assertRaisesRegex(ValueError, "layer"):
            mapss(refs, outs, model="raw", layer=1, add_ci=False)


class TestPublicAPI(unittest.TestCase):
    def test_two_argument_api_returns_named_dataframes(self):
        refs, outs = _signals()
        fake_soundfile = types.ModuleType("soundfile")
        fake_soundfile.write = lambda path, *_args, **_kwargs: Path(path).touch()

        captured = {}
        fake_engine = types.ModuleType("mapss.engine")

        def fake_compute(**kwargs):
            captured.update(kwargs)
            root = Path(kwargs["results_root"]) / "experiment_api" / "api"
            root.mkdir(parents=True)
            columns = {
                "timestamp_ms": [0.0, 20.0],
                "system_api__reference_00": [0.8, 0.7],
                "system_api__reference_01": [0.6, 0.5],
            }
            pd.DataFrame(columns).to_csv(root / "ps_scores_raw.csv", index=False)
            pd.DataFrame(columns).to_csv(root / "pm_scores_raw.csv", index=False)
            return root.parent

        fake_engine.compute_mapss_measures = fake_compute
        with patch.dict(
            sys.modules, {"soundfile": fake_soundfile, "mapss.engine": fake_engine}
        ):
            result = mapss(
                refs,
                outs,
                source_names=["speaker-a", "speaker-b"],
                model="raw",
                add_ci=False,
                max_gpus=0,
            )

        self.assertEqual(result.source_names, ("speaker-a", "speaker-b"))
        self.assertEqual(result.layer, 0)
        self.assertEqual(captured["on_missing"], "error")
        self.assertEqual(captured["max_gpus"], 0)

    def test_rejects_a_source_with_no_valid_frames(self):
        refs, outs = _signals()
        fake_soundfile = types.ModuleType("soundfile")
        fake_soundfile.write = lambda path, *_args, **_kwargs: Path(path).touch()
        fake_engine = types.ModuleType("mapss.engine")

        def fake_compute(**kwargs):
            root = Path(kwargs["results_root"]) / "experiment_api" / "api"
            root.mkdir(parents=True)
            values = {
                "timestamp_ms": [0.0, 20.0],
                "system_api__reference_00": [0.8, 0.7],
                "system_api__reference_01": [np.nan, np.nan],
            }
            pd.DataFrame(values).to_csv(root / "ps_scores_raw.csv", index=False)
            pd.DataFrame(values).to_csv(root / "pm_scores_raw.csv", index=False)
            return root.parent

        fake_engine.compute_mapss_measures = fake_compute
        with patch.dict(
            sys.modules, {"soundfile": fake_soundfile, "mapss.engine": fake_engine}
        ):
            with self.assertRaisesRegex(
                RuntimeError, r"PS\[speaker-b\].*PM\[speaker-b\]"
            ):
                mapss(
                    refs,
                    outs,
                    source_names=["speaker-a", "speaker-b"],
                    model="raw",
                    add_ci=False,
                    max_gpus=0,
                )


if __name__ == "__main__":
    unittest.main()
