import numpy as np


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Compute the sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative_from_activation(a: np.ndarray | float) -> np.ndarray | float:
    """Compute derivative of sigmoid given its already-computed activation a = sigmoid(z)."""
    return a * (1.0 - a)


def mse_loss(y_pred: np.ndarray | float, y_true: np.ndarray | float) -> float:
    """Compute 1/2 Mean Squared Error for a single sample."""
    return float(0.5 * np.mean((y_pred - y_true) ** 2))


def mse_loss_derivative(y_pred: np.ndarray | float, y_true: np.ndarray | float) -> np.ndarray | float:
    """Compute dL / dy_pred for L = 0.5 * (y_pred - y_true)^2."""
    return y_pred - y_true


def get_initial_network_params() -> dict:
    """Return the exact initial weights, biases, and sample from the hand-traced walkthrough."""
    return {
        "x": np.array([0.0, 1.0]),
        "y": 1.0,
        # Input -> Hidden weights: shape (2, 2) where W1[i, j] is weight from x_{i+1} to h_{j+1}
        "W1": np.array([
            [0.5, 0.2],   # w11 = 0.5, w12 = 0.2
            [-0.5, 0.8],  # w21 = -0.5, w22 = 0.8
        ]),
        "b1": np.array([0.0, 0.0]),  # [b1, b2]
        # Hidden -> Output weights: shape (2,) where W2[i] is v_{i+1}
        "W2": np.array([0.6, -0.4]),  # [v1, v2]
        "b2": 0.0,  # b3
    }
