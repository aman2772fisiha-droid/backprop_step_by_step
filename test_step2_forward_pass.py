import unittest
import numpy as np
from step2_forward_pass import forward_hidden_layer, forward_output_layer, forward_pass


class TestStep2ForwardPass(unittest.TestCase):
    def test_forward_hidden_layer(self):
        x = np.array([0.0, 1.0])
        W1 = np.array([[0.5, 0.2], [-0.5, 0.8]])
        b1 = np.array([0.0, 0.0])
        z_h, a_h = forward_hidden_layer(x, W1, b1)
        np.testing.assert_allclose(z_h, np.array([-0.5, 0.8]), atol=1e-6)
        np.testing.assert_allclose(a_h, np.array([0.37754067, 0.68997448]), atol=1e-6)

    def test_forward_output_layer(self):
        a_h = np.array([0.3775, 0.6900])
        W2 = np.array([0.6, -0.4])
        b2 = 0.0
        z3, y_hat = forward_output_layer(a_h, W2, b2)
        self.assertAlmostEqual(z3, -0.0495, places=4)
        self.assertAlmostEqual(y_hat, 0.487627, places=4)

    def test_full_forward_pass_matches_hand_trace(self):
        params = {
            "x": np.array([0.0, 1.0]),
            "y": 1.0,
            "W1": np.array([[0.5, 0.2], [-0.5, 0.8]]),
            "b1": np.array([0.0, 0.0]),
            "W2": np.array([0.6, -0.4]),
            "b2": 0.0,
        }
        cache = forward_pass(params)
        self.assertAlmostEqual(cache["z1"], -0.5, places=4)
        self.assertAlmostEqual(cache["a1"], 0.3775, places=4)
        self.assertAlmostEqual(cache["z2"], 0.8, places=4)
        self.assertAlmostEqual(cache["a2"], 0.6900, places=4)
        self.assertAlmostEqual(cache["z3"], -0.0495, places=3)
        self.assertAlmostEqual(cache["y_hat"], 0.4876, places=4)
        self.assertAlmostEqual(cache["loss"], 0.1313, places=4)


if __name__ == "__main__":
    unittest.main()
