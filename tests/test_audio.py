import unittest

import numpy as np

from mapss.audio import loudness_normalize


class TestLoudnessNormalization(unittest.TestCase):
    def test_very_quiet_gated_signal_uses_finite_fallback(self):
        signal = np.zeros(8_000, dtype=np.float32)
        signal[0] = 1e-6

        normalized = loudness_normalize(signal)

        self.assertTrue(np.isfinite(normalized).all())
        self.assertGreater(float(np.max(np.abs(normalized))), 0.0)
        self.assertLessEqual(float(np.max(np.abs(normalized))), 1.0)


if __name__ == "__main__":
    unittest.main()
