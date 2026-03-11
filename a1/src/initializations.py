"""
Weight initialization schemes for neural networks.
"""

import numpy as np


# -----------------------------
# Random Initialization
# -----------------------------

def random_init(shape):
    """
    Initialize weights with small random values.
    """
    return np.random.randn(*shape) * 0.01


# -----------------------------
# Xavier / Glorot Initialization
# -----------------------------

def xavier_init(shape):
    """
    Xavier (Glorot) initialization.
    Good for sigmoid / tanh activations.
    """
    fan_in, fan_out = shape
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)


# -----------------------------
# He Initialization
# -----------------------------

def he_init(shape):
    """
    He initialization.
    Good for ReLU activations.
    """
    fan_in = shape[0]
    std = np.sqrt(2 / fan_in)
    return np.random.randn(*shape) * std


# -----------------------------
# Initialization lookup
# -----------------------------

INITIALIZERS = {
    "random": random_init,
    "xavier": xavier_init,
    "he": he_init
}