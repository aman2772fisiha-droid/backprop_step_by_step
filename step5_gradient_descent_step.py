import numpy as np
from step2_forward_pass import forward_pass
from step3_backward_output import backward_output_layer
from step4_backward_hidden import backward_hidden_layer


def apply_gradient_descent(params: dict, out_grads: dict, hid_grads: dict, lr: float = 0.5) -> dict:
    """Return updated parameter dictionary after one gradient descent step."""
    updated = {
        "x": params["x"].copy(),
        "y": params["y"],
        "W1": params["W1"] - lr * hid_grads["dW1"],
        "b1": params["b1"] - lr * hid_grads["db1"],
        "W2": params["W2"] - lr * out_grads["dW2"],
        "b2": float(params["b2"] - lr * out_grads["db3"]),
    }
    return updated


def train_single_step(params: dict, lr: float = 0.5) -> tuple[dict, dict]:
    """Execute one complete forward pass, backward pass, and parameter update."""
    cache = forward_pass(params)
    out_grads = backward_output_layer(cache["y_hat"], params["y"], cache["a_hidden"])
    hid_grads = backward_hidden_layer(params["x"], out_grads["delta_3"], params["W2"], cache["a_hidden"])
    updated_params = apply_gradient_descent(params, out_grads, hid_grads, lr=lr)

    summary = {
        "forward": cache,
        "output_grads": out_grads,
        "hidden_grads": hid_grads,
    }
    return updated_params, summary
