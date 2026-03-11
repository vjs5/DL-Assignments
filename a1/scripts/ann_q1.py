import numpy as np


class SimpleANN:

    def __init__(self, input_size=2, hidden_size=2, output_size=1, lr=0.1):

        self.lr = lr

        # weight initialization
        self.W1 = np.random.randn(hidden_size, input_size)
        self.b1 = np.zeros((hidden_size,1))

        self.W2 = np.random.randn(output_size, hidden_size)
        self.b2 = np.zeros((output_size,1))


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)


    def forward(self, x):

        self.z1 = self.W1 @ x + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2


    def backward(self, x, y):

        y_hat = self.a2

        dz2 = (y_hat - y) * self.sigmoid_derivative(self.z2)

        dW2 = dz2 @ self.a1.T
        db2 = dz2

        dz1 = (self.W2.T @ dz2) * self.sigmoid_derivative(self.z1)

        dW1 = dz1 @ x.T
        db1 = dz1

        # update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1