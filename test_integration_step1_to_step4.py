import unittest
import numpy as np
from step1_primitives import get_initial_network_params
from step2_forward_pass import forward_pass
from step3_backward_output import backward_output_layer
from step4_backward_hidden import backward_hidden_layer


class TestIntegrationStep1ToStep4(unittest.TestCase):
    def test_full_forward_and_backward_pass_with_numerical_gradient_check(self):
        """
        Integration test:
        1. Runs full Step 1 -> Step 2 -> Step 3 -> Step 4 pipeline.
        2. Verifies against hand-traced values.
        3. Performs finite-difference numerical gradient check to prove mathematical exactness.
        """
        params = get_initial_network_params()
        cache = forward_pass(params)
        out_grads = backward_output_layer(cache["y_hat"], params["y"], cache["a_hidden"])
        hid_grads = backward_hidden_layer(params["x"], out_grads["delta_3"], params["W2"], cache["a_hidden"])

        # Check hand-traced values
        self.assertAlmostEqual(hid_grads["dw11"], 0.0, places=4)
        self.assertAlmostEqual(hid_grads["dw21"], -0.0180, places=3)
        self.assertAlmostEqual(hid_grads["dw12"], 0.0, places=4)
        self.assertAlmostEqual(hid_grads["dw22"], 0.0110, places=3)

        # Finite-difference numerical gradient check on W1 and W2
        eps = 1e-6
        for i in range(2):
            for j in range(2):
                params_plus = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in params.items()}
                params_minus = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in params.items()}
                params_plus["W1"][i, j] += eps
                params_minus["W1"][i, j] -= eps
                num_grad = (forward_pass(params_plus)["loss"] - forward_pass(params_minus)["loss"]) / (2 * eps)
                self.assertAlmostEqual(hid_grads["dW1"][i, j], num_grad, places=6)

        for i in range(2):
            params_plus = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in params.items()}
            params_minus = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in params.items()}
            params_plus["W2"][i] += eps
            params_minus["W2"][i] -= eps
            num_grad = (forward_pass(params_plus)["loss"] - forward_pass(params_minus)["loss"]) / (2 * eps)
            self.assertAlmostEqual(out_grads["dW2"][i], num_grad, places=6)


if __name__ == "__main__":
    unittest.main()
