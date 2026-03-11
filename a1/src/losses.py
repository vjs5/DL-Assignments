"""
Loss functions and their derivatives for MLP.
"""

import numpy as np


# -----------------------------
# Mean Squared Error (MSE)
# -----------------------------

def mse(y_true, y_pred):
    """
    Mean squared error loss
    """
    return np.mean((y_true - y_pred) ** 2)


def mse_derivative(y_true, y_pred):
    """
    Derivative of MSE w.r.t predictions
    """
    return 2 * (y_pred - y_true) / y_true.shape[0]


# -----------------------------
# Softmax
# -----------------------------

def softmax(x):
    """
    Numerically stable softmax
    """
    x_shift = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# -----------------------------
# Cross Entropy
# -----------------------------

def cross_entropy(y_true, y_pred):
    """
    Cross entropy loss for classification.
    Assumes y_true is one-hot encoded.
    """
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)

    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def cross_entropy_derivative(y_true, y_pred):
    """
    Gradient of cross-entropy with softmax output.
    """
    return (y_pred - y_true) / y_true.shape[0]


# -----------------------------
# Mean Absolute Error (MAE)
# -----------------------------

def mae(y_true, y_pred):
    """
    Mean absolute error loss
    """
    return np.mean(np.abs(y_true - y_pred))


def mae_derivative(y_true, y_pred):
    """
    Derivative of MAE w.r.t predictions
    """
    return np.sign(y_pred - y_true) / y_true.shape[0]


# -----------------------------
# Huber Loss
# -----------------------------

def huber(y_true, y_pred, delta=1.0):
    """
    Huber loss combines MSE and MAE behavior
    """
    error = y_true - y_pred
    abs_error = np.abs(error)

    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic

    return np.mean(0.5 * quadratic**2 + delta * linear)


def huber_derivative(y_true, y_pred, delta=1.0):
    """
    Derivative of Huber loss
    """
    error = y_pred - y_true

    grad = np.where(
        np.abs(error) <= delta,
        error,
        delta * np.sign(error)
    )

    return grad / y_true.shape[0]


# -----------------------------
# Loss lookup dictionaries
# -----------------------------

LOSSES = {
    "mse": mse,
    "mae": mae,
    "cross_entropy": cross_entropy,
    "huber": huber
}

LOSS_DERIVATIVES = {
    "mse": mse_derivative,
    "mae": mae_derivative,
    "cross_entropy": cross_entropy_derivative,
    "huber": huber_derivative
}