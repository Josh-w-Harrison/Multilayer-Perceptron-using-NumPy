import numpy as np
import matplotlib.pyplot as plt

class NeuralNet:
import numpy as np

class NeuralNet:
    def __init__(self, input_dim, hidden_dim, seed=0):
        np.random.seed(seed)
        self.W1 = 0.01 * np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = 0.01 * np.random.randn(hidden_dim, 1)
        self.b2 = np.zeros((1, 1))
        self.cache = {}

    def _sigmoid(self, x):
        pos_mask = (x >= 0)
        neg_mask = ~pos_mask
        z = np.empty_like(x)
        z[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
        exp_x = np.exp(x[neg_mask])
        z[neg_mask] = exp_x / (1 + exp_x)
        return z

    def feed_forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self._sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        y_hat = self._sigmoid(z2)
        self.cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "y_hat": y_hat}
        return y_hat

    def loss(self, y_hat, y_true):
        eps = 1e-12
        y_hat = np.clip(y_hat, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))

    def backprop(self, y_true):
        X   = self.cache["X"]
        a1  = self.cache["a1"]
        y_hat = self.cache["y_hat"]

        N = X.shape[0]
        dz2 = (y_hat - y_true) / N
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (a1 * (1 - a1))

        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def step(self, grads, lr=0.1):
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]

    def predict(self, X, threshold=0.5):
        probs = self.feed_forward(X)
        return (probs >= threshold).astype(int)



