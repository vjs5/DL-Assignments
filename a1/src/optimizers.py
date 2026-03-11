"""
Optimization algorithms for training neural networks.
"""

import numpy as np


# ---------------------------------------------------
# SGD
# ---------------------------------------------------

class SGD:

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for k in params:
            params[k] -= self.lr * grads[k]


# ---------------------------------------------------
# Momentum
# ---------------------------------------------------

class Momentum:

    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = {}

    def update(self, params, grads):

        for k in params:

            if k not in self.v:
                self.v[k] = np.zeros_like(params[k])

            self.v[k] = self.beta * self.v[k] + (1 - self.beta) * grads[k]

            params[k] -= self.lr * self.v[k]


# ---------------------------------------------------
# Nesterov Accelerated GD
# ---------------------------------------------------

class Nesterov:

    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = {}

    def update(self, params, grads):

        for k in params:

            if k not in self.v:
                self.v[k] = np.zeros_like(params[k])

            v_prev = self.v[k].copy()

            self.v[k] = self.beta * self.v[k] + self.lr * grads[k]

            params[k] -= self.beta * v_prev + (1 + self.beta) * self.v[k]


# ---------------------------------------------------
# AdaGrad
# ---------------------------------------------------

class AdaGrad:

    def __init__(self, lr=0.01, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.cache = {}

    def update(self, params, grads):

        for k in params:

            if k not in self.cache:
                self.cache[k] = np.zeros_like(params[k])

            self.cache[k] += grads[k] ** 2

            params[k] -= self.lr * grads[k] / (np.sqrt(self.cache[k]) + self.eps)


# ---------------------------------------------------
# RMSProp
# ---------------------------------------------------

class RMSProp:

    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.cache = {}

    def update(self, params, grads):

        for k in params:

            if k not in self.cache:
                self.cache[k] = np.zeros_like(params[k])

            self.cache[k] = self.beta * self.cache[k] + (1 - self.beta) * grads[k] ** 2

            params[k] -= self.lr * grads[k] / (np.sqrt(self.cache[k]) + self.eps)


# ---------------------------------------------------
# Adam
# ---------------------------------------------------

class Adam:

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):

        self.t += 1

        for k in params:

            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])

            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k] ** 2)

            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)

            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------
# Muon Optimizer
# ---------------------------------------------------

class Muon:

    def __init__(self, lr=0.01, eps=1e-8):
        self.lr = lr
        self.eps = eps

    def update(self, params, grads):

        for k in params:

            g = grads[k]

            norm = np.linalg.norm(g)

            params[k] -= self.lr * g / (norm + self.eps)


# ---------------------------------------------------
# Optimizer factory
# ---------------------------------------------------

def get_optimizer(name, lr):

    name = name.lower()

    if name == "sgd":
        return SGD(lr)

    elif name == "momentum":
        return Momentum(lr)

    elif name == "nesterov":
        return Nesterov(lr)

    elif name == "adagrad":
        return AdaGrad(lr)

    elif name == "rmsprop":
        return RMSProp(lr)

    elif name == "adam":
        return Adam(lr)
    
    elif name == "muon":
        return Muon(lr)

    else:
        raise ValueError(f"Unknown optimizer: {name}")