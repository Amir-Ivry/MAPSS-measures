import unittest

from mapss._cli_args import _validate_and_resolve, _validate_gpus


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


if __name__ == "__main__":
    unittest.main()
