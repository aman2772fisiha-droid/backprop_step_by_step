import unittest
import numpy as np
from step1_primitives import (
    sigmoid,
    sigmoid_derivative_from_activation,
    mse_loss,
    mse_loss_derivative,
    get_initial_network_params,
)


class TestStep1Primitives(unittest.TestCase):
    def test_sigmoid_values(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5, places=6)
        self.assertAlmostEqual(sigmoid(-0.5), 0.37754067, places=6)
        self.assertAlmostEqual(sigmoid(0.8), 0.68997448, places=6)

    def test_sigmoid_derivative(self):
        a = sigmoid(-0.5)
        self.assertAlmostEqual(sigmoid_derivative_from_activation(a), 0.23500371, places=6)

    def test_mse_loss_and_derivative(self):
        y_pred = 0.4876
        y_true = 1.0
        self.assertAlmostEqual(mse_loss(y_pred, y_true), 0.13127688, places=6)
        self.assertAlmostEqual(mse_loss_derivative(y_pred, y_true), -0.5124, places=6)

    def test_initial_params_shapes_and_values(self):
        params = get_initial_network_params()
        np.testing.assert_array_equal(params["x"], np.array([0.0, 1.0]))
        self.assertEqual(params["y"], 1.0)
        self.assertEqual(params["W1"].shape, (2, 2))
        self.assertEqual(params["b1"].shape, (2,))
        self.assertEqual(params["W2"].shape, (2,))


if __name__ == "__main__":
    unittest.main()
