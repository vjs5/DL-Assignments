"""
Activation functions and their derivatives for MLP.

All functions operate on NumPy arrays.
"""

import numpy as np


# -----------------------------
# Sigmoid
# -----------------------------

def sigmoid(x):
    """Sigmoid activation"""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Derivative of sigmoid"""
    s = sigmoid(x)
    return s * (1 - s)


# -----------------------------
# Tanh
# -----------------------------

def tanh(x):
    """Tanh activation"""
    return np.tanh(x)


def tanh_derivative(x):
    """Derivative of tanh"""
    return 1 - np.tanh(x) ** 2


# -----------------------------
# ReLU
# -----------------------------

def relu(x):
    """ReLU activation"""
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)


# -----------------------------
# Leaky ReLU
# -----------------------------

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation"""
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.01):
    """Derivative of Leaky ReLU"""
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx

# -----------------------------
# Linear (for regression)
# -----------------------------

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)


# -----------------------------
# Activation lookup dictionaries
# -----------------------------

ACTIVATIONS = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
    "leaky_relu": leaky_relu,
    "linear": linear
}

DERIVATIVES = {
    "sigmoid": sigmoid_derivative,
    "tanh": tanh_derivative,
    "relu": relu_derivative,
    "leaky_relu": leaky_relu_derivative,
    "linear": linear_derivative
}