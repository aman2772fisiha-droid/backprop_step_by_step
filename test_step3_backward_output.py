import unittest
import numpy as np
from step3_backward_output import compute_output_delta, backward_output_layer


class TestStep3BackwardOutput(unittest.TestCase):
    def test_compute_output_delta(self):
        # Using rounded forward value y_hat = 0.4876
        delta_3 = compute_output_delta(y_hat=0.4876, y_true=1.0)
        self.assertAlmostEqual(delta_3, -0.1280, places=4)

    def test_backward_output_layer(self):
        a_hidden = np.array([0.3775, 0.6900])
        grads = backward_output_layer(y_hat=0.4876, y_true=1.0, a_hidden=a_hidden)
        self.assertAlmostEqual(grads["delta_3"], -0.1280, places=4)
        self.assertAlmostEqual(grads["dv1"], -0.0483, places=4)
        self.assertAlmostEqual(grads["dv2"], -0.0883, places=4)
        self.assertAlmostEqual(grads["db3"], -0.1280, places=4)


if __name__ == "__main__":
    unittest.main()
