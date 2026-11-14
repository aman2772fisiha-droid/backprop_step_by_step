import numpy as np
from step1_primitives import sigmoid_derivative_from_activation


def compute_hidden_deltas(delta_3: float, W2: np.ndarray, a_hidden: np.ndarray) -> np.ndarray:
    """Compute hidden layer error signals [delta_1, delta_2]."""
    local_deriv = sigmoid_derivative_from_activation(a_hidden)
    return (delta_3 * W2) * local_deriv


def backward_hidden_layer(
    x: np.ndarray,
    delta_3: float,
    W2: np.ndarray,
    a_hidden: np.ndarray,
) -> dict:
    """Compute gradients for input->hidden weights W1 and hidden biases b1 = [b1, b2]."""
    deltas = compute_hidden_deltas(delta_3, W2, a_hidden)
    # Outer product: x shape (2, 1) @ deltas shape (1, 2) -> dW1 shape (2, 2)
    dW1 = np.outer(x, deltas)
    db1 = deltas

    return {
        "delta_1": float(deltas[0]),
        "delta_2": float(deltas[1]),
        "dW1": dW1,
        "db1": db1,
        "dw11": float(dW1[0, 0]),
        "dw21": float(dW1[1, 0]),
        "db1_val": float(db1[0]),
        "dw12": float(dW1[0, 1]),
        "dw22": float(dW1[1, 1]),
        "db2_val": float(db1[1]),
    }
