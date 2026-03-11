"""
Flexible Multi-Layer Perceptron implementation.

Supports:
- Multiple hidden layers
- Different activation functions
- Multiple optimizers
- Mini-batch training
- L1 / L2 regularization
- Validation loss tracking + early stopping
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

        aL = self.a[self.L]

        if self.loss_name == "cross_entropy":
            delta = self.loss_derivative(y, aL)   # shape: (output_size, m)
        else:
            # General case: chain rule through the output activation.
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

            if self.loss_name == "cross_entropy":
                grads[f"W{l}"] = dW
                grads[f"b{l}"] = db
            else:
                grads[f"W{l}"] = dW / m
                grads[f"b{l}"] = db / m

            if l > 1:
                W = self.params[f"W{l}"]
                delta = (W.T @ delta) * DERIVATIVES[self.activations[l - 2]](self.z[l - 1])

        return grads

    # ---------------------------------------------------
    # Training  (FIX 3: added X_val / y_val + early stopping)
    # ---------------------------------------------------

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=100, patience=10, verbose=True):

        n_samples = X_train.shape[0]

        train_losses = []
        val_losses   = []

        best_val_loss   = np.inf
        best_params     = None
        epochs_no_improve = 0

        for epoch in range(epochs):

            # --- mini-batch loop ---
            indices   = np.random.permutation(n_samples)
            X_shuf    = X_train[indices]
            y_shuf    = y_train[indices]

            for start in range(0, n_samples, self.batch_size):
                end     = start + self.batch_size
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]

                self.forward(X_batch)
                grads = self.backward(y_batch)
                self.optimizer.update(self.params, grads)

            # --- epoch metrics ---
            y_pred_train = self.forward(X_train)
            train_loss   = self.loss(y_train.T, y_pred_train)
            train_losses.append(train_loss)

            if X_val is not None and y_val is not None:
                y_pred_val = self.forward(X_val)
                val_loss   = self.loss(y_val.T, y_pred_val)
                val_losses.append(val_loss)

                # early stopping
                if val_loss < best_val_loss:
                    best_val_loss      = val_loss
                    best_params        = {k: v.copy() for k, v in self.params.items()}
                    epochs_no_improve  = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch} "
                                  f"(best val loss: {best_val_loss:.4f})")
                        # restore best weights
                        self.params = best_params
                        break

            if verbose and epoch % 10 == 0:
                msg = f"Epoch {epoch:4d}  train loss: {train_loss:.4f}"
                if val_losses:
                    msg += f"  val loss: {val_losses[-1]:.4f}"
                print(msg)

        history = {"train_loss": train_losses}
        if val_losses:
            history["val_loss"] = val_losses

        return history

    # ---------------------------------------------------
    # Prediction
    # ---------------------------------------------------

    def predict(self, X):
        return self.forward(X).T