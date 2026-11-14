import unittest
from step1_primitives import get_initial_network_params
from step2_forward_pass import forward_pass
from step3_backward_output import backward_output_layer


class TestIntegrationStep1ToStep3(unittest.TestCase):
    def test_forward_to_output_backward(self):
        """Integration test: Step 1 params -> Step 2 forward -> Step 3 output gradients."""
        params = get_initial_network_params()
        cache = forward_pass(params)
        out_grads = backward_output_layer(cache["y_hat"], params["y"], cache["a_hidden"])

        self.assertAlmostEqual(out_grads["delta_3"], -0.1280, places=3)
        self.assertAlmostEqual(out_grads["dv1"], -0.0483, places=3)
        self.assertAlmostEqual(out_grads["dv2"], -0.0883, places=3)
        self.assertAlmostEqual(out_grads["db3"], -0.1280, places=3)


if __name__ == "__main__":
    unittest.main()
