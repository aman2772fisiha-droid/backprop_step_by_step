import numpy as np
from step1_primitives import sigmoid, mse_loss


def forward_hidden_layer(x: np.ndarray, W1: np.ndarray, b1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute pre-activations z_hidden = [z1, z2] and activations a_hidden = [a1, a2]."""
    z_hidden = x @ W1 + b1
    a_hidden = sigmoid(z_hidden)
    return z_hidden, a_hidden


def forward_output_layer(a_hidden: np.ndarray, W2: np.ndarray, b2: float) -> tuple[float, float]:
    """Compute output pre-activation z3 and final prediction y_hat."""
    z3 = float(a_hidden @ W2 + b2)
    y_hat = float(sigmoid(z3))
    return z3, y_hat


def forward_pass(params: dict) -> dict:
    """Execute full forward pass and return intermediate cache and loss."""
    z_hidden, a_hidden = forward_hidden_layer(params["x"], params["W1"], params["b1"])
    z3, y_hat = forward_output_layer(a_hidden, params["W2"], params["b2"])
    loss = mse_loss(y_hat, params["y"])

    return {
        "z1": float(z_hidden[0]),
        "z2": float(z_hidden[1]),
        "a1": float(a_hidden[0]),
        "a2": float(a_hidden[1]),
        "a_hidden": a_hidden,
        "z3": z3,
        "y_hat": y_hat,
        "loss": loss,
    }
