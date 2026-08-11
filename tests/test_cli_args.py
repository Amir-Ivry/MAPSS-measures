import unittest

from mapss._cli_args import _validate_and_resolve, _validate_gpus
from mapss.cli import main


class TestCLIArguments(unittest.TestCase):
    def test_paper_aligned_default_layer(self):
        layer, alpha = _validate_and_resolve("wav2vec2", None, None)
        self.assertEqual(layer, 2)
        self.assertEqual(alpha, 1.0)

    def test_gpu_limit_validation(self):
        self.assertIsNone(_validate_gpus(None))
        self.assertEqual(_validate_gpus(0), 0)
        with self.assertRaises(SystemExit):
            _validate_gpus(-1)

    def test_plot_rejects_no_ci_before_loading_audio(self):
        with self.assertRaisesRegex(SystemExit, "cannot be combined with --no-ci"):
            main(
                [
                    "--reference",
                    "reference_1.wav",
                    "reference_2.wav",
                    "--output",
                    "output_1.wav",
                    "output_2.wav",
                    "--model",
                    "raw",
                    "--plot",
                    "--no-ci",
                ]
            )


if __name__ == "__main__":
    unittest.main()
