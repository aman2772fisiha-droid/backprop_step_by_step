import unittest
import numpy as np
from step5_gradient_descent_step import apply_gradient_descent


class TestStep5GradientDescentStep(unittest.TestCase):
    def test_apply_gradient_descent_directions(self):
        params = {
            "x": np.array([0.0, 1.0]),
            "y": 1.0,
            "W1": np.array([[0.5, 0.2], [-0.5, 0.8]]),
            "b1": np.array([0.0, 0.0]),
            "W2": np.array([0.6, -0.4]),
            "b2": 0.0,
        }
        out_grads = {"dW2": np.array([-0.0483, -0.0883]), "db3": -0.1280}
        hid_grads = {
            "dW1": np.array([[0.0, 0.0], [-0.0180, 0.0110]]),
            "db1": np.array([-0.0180, 0.0110]),
        }

        updated = apply_gradient_descent(params, out_grads, hid_grads, lr=0.5)

        # Verify parameter update signs match summary table directions
        self.assertGreater(updated["W2"][0], params["W2"][0])      # v1 increases
        self.assertGreater(updated["W2"][1], params["W2"][1])      # v2 increases
        self.assertGreater(updated["b2"], params["b2"])            # b3 increases
        self.assertEqual(updated["W1"][0, 0], params["W1"][0, 0])  # w11 unchanged
        self.assertGreater(updated["W1"][1, 0], params["W1"][1, 0])# w21 increases
        self.assertLess(updated["W1"][1, 1], params["W1"][1, 1])   # w22 decreases


if __name__ == "__main__":
    unittest.main()
