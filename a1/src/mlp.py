"""
Flexible Multi-Layer Perceptron implementation.

Supports:
- Multiple hidden layers
- Different activation functions
- Multiple optimizers
- Mini-batch training
- L1 / L2 regularization
"""

import numpy as np

from src.activations import ACTIVATIONS, DERIVATIVES
from src.losses import LOSSES, LOSS_DERIVATIVES
from src.initializations import INITIALIZERS
from src.optimizers import get_optimizer


class MLP:

    def __init__(
        self,
        layer_sizes,
        activations,
        loss="mse",
        learning_rate=0.01,
        optimizer="sgd",
        batch_size=32,
        weight_init="xavier",
        regularization=None,
        lambda_reg=0.0
    ):

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.loss_name = loss
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.regularization = regularization
        self.lambda_reg = lambda_reg

        self.L = len(layer_sizes) - 1

        # loss functions
        self.loss = LOSSES[loss]
        self.loss_derivative = LOSS_DERIVATIVES[loss]

        # optimizer
        self.optimizer = get_optimizer(optimizer, learning_rate)

        # parameters
        self.params = {}

        # caches
        self.z = {}
        self.a = {}

        # weight initialization
        initializer = INITIALIZERS[weight_init]

        for l in range(1, self.L + 1):

            input_size = layer_sizes[l - 1]
            output_size = layer_sizes[l]

            self.params[f"W{l}"] = initializer((output_size, input_size))
            self.params[f"b{l}"] = np.zeros((output_size, 1))

    # ---------------------------------------------------
    # Forward Pass
    # ---------------------------------------------------

    def forward(self, X):

        a = X.T
        self.a[0] = a

        for l in range(1, self.L + 1):

            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]

            z = W @ a + b
            self.z[l] = z

            activation_fn = ACTIVATIONS[self.activations[l - 1]]
            a = activation_fn(z)

            self.a[l] = a

        return a

    # ---------------------------------------------------
    # Backward Pass
    # ---------------------------------------------------

    def backward(self, y):

        grads = {}

        m = y.shape[0]
        y = y.T

        # output layer delta
        aL = self.a[self.L]
        delta = self.loss_derivative(y, aL) * DERIVATIVES[self.activations[self.L - 1]](self.z[self.L])

        for l in reversed(range(1, self.L + 1)):

            a_prev = self.a[l - 1]

            dW = delta @ a_prev.T
            db = np.sum(delta, axis=1, keepdims=True)

            # regularization
            if self.regularization == "l2":
                dW += self.lambda_reg * self.params[f"W{l}"]

            if self.regularization == "l1":
                dW += self.lambda_reg * np.sign(self.params[f"W{l}"])

            grads[f"W{l}"] = dW / m
            grads[f"b{l}"] = db / m

            if l > 1:

                W = self.params[f"W{l}"]
                delta = (W.T @ delta) * DERIVATIVES[self.activations[l - 2]](self.z[l - 1])

        return grads

    # ---------------------------------------------------
    # Training
    # ---------------------------------------------------

    def fit(self, X, y, epochs=100, verbose=True):

        n_samples = X.shape[0]

        history = []

        for epoch in range(epochs):

            indices = np.random.permutation(n_samples)

            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, n_samples, self.batch_size):

                end = start + self.batch_size

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # forward
                y_pred = self.forward(X_batch)

                # backward
                grads = self.backward(y_batch)

                # update
                self.optimizer.update(self.params, grads)

            # compute epoch loss
            y_pred_full = self.forward(X)

            loss_value = self.loss(y.T, y_pred_full)

            history.append(loss_value)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch} Loss {loss_value:.4f}")

        return history

    # ---------------------------------------------------
    # Prediction
    # ---------------------------------------------------

    def predict(self, X):

        y_pred = self.forward(X)

        return y_pred.T