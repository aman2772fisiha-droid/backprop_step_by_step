import unittest
from step1_primitives import get_initial_network_params
from step2_forward_pass import forward_pass


class TestIntegrationStep1ToStep2(unittest.TestCase):
    def test_initial_params_through_forward_pass(self):
        """Integration test: feed Step 1 initial parameters into Step 2 forward pass."""
        params = get_initial_network_params()
        cache = forward_pass(params)

        # Verify end-to-end consistency with hand-traced values
        self.assertAlmostEqual(cache["a1"], 0.37754, places=4)
        self.assertAlmostEqual(cache["a2"], 0.68997, places=4)
        self.assertAlmostEqual(cache["y_hat"], 0.48763, places=4)
        self.assertAlmostEqual(cache["loss"], 0.13126, places=4)


if __name__ == "__main__":
    unittest.main()
