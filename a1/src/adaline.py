import numpy as np


class Adaline:

    def __init__(self, learning_rate=1.0, max_iterations=1000):
        """
        Initialize the ADALINE model.

        learning_rate : step size for gradient descent
        max_iterations : number of training epochs
        """

        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Train ADALINE using gradient descent.

        X : (n_samples, n_features)
        y : (n_samples,)
        """

        n_samples, n_features = X.shape

        # initialize parameters
        self.w = np.zeros(n_features)
        self.b = 0.0

        history = []

        for _ in range(self.max_iterations):

            # linear output
            y_pred = X @ self.w + self.b

            # compute error
            error = y - y_pred

            # MSE
            mse = np.mean(error**2)

            if not np.isfinite(mse):
                print("MSE is not finite. Stopping training.")
                break

            history.append(mse)

            # gradients
            dw = (-2 / n_samples) * (X.T @ error)
            db = (-2 / n_samples) * np.sum(error)

            # update parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

        return history

    def predict(self, X):
        """Return predicted outputs"""
        return X @ self.w + self.b

    def score(self, X, y):
        """Return mean squared error"""
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)