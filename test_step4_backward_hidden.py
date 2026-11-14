import unittest
import numpy as np
from step4_backward_hidden import compute_hidden_deltas, backward_hidden_layer


class TestStep4BackwardHidden(unittest.TestCase):
    def test_compute_hidden_deltas(self):
        delta_3 = -0.1280
        W2 = np.array([0.6, -0.4])
        a_hidden = np.array([0.3775, 0.6900])
        deltas = compute_hidden_deltas(delta_3, W2, a_hidden)
        self.assertAlmostEqual(deltas[0], -0.0180, places=4)
        self.assertAlmostEqual(deltas[1], 0.0110, places=4)

    def test_backward_hidden_layer(self):
        x = np.array([0.0, 1.0])
        delta_3 = -0.1280
        W2 = np.array([0.6, -0.4])
        a_hidden = np.array([0.3775, 0.6900])

        grads = backward_hidden_layer(x, delta_3, W2, a_hidden)
        self.assertAlmostEqual(grads["dw11"], 0.0000, places=4)
        self.assertAlmostEqual(grads["dw21"], -0.0180, places=4)
        self.assertAlmostEqual(grads["db1_val"], -0.0180, places=4)
        self.assertAlmostEqual(grads["dw12"], 0.0000, places=4)
        self.assertAlmostEqual(grads["dw22"], 0.0110, places=4)
        self.assertAlmostEqual(grads["db2_val"], 0.0110, places=4)


if __name__ == "__main__":
    unittest.main()
