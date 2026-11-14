import numpy as np
from step1_primitives import mse_loss_derivative, sigmoid_derivative_from_activation


def compute_output_delta(y_hat: float, y_true: float) -> float:
    """Compute output layer error signal delta_3 = dL/dz3."""
    dL_dyhat = mse_loss_derivative(y_hat, y_true)
    dyhat_dz3 = sigmoid_derivative_from_activation(y_hat)
    return float(dL_dyhat * dyhat_dz3)


def backward_output_layer(y_hat: float, y_true: float, a_hidden: np.ndarray) -> dict:
    """Compute gradients for output weights W2 = [v1, v2] and output bias b2 (b3)."""
    delta_3 = compute_output_delta(y_hat, y_true)
    dW2 = delta_3 * a_hidden  # [dL/dv1, dL/dv2]
    db2 = delta_3             # dL/db3

    return {
        "delta_3": delta_3,
        "dW2": dW2,
        "dv1": float(dW2[0]),
        "dv2": float(dW2[1]),
        "db3": float(db2),
    }
