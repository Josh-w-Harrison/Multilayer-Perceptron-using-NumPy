import numpy as np
import matplotlib.pyplot as plt
class NeuralNet:
    def __init__(self, input_dim, hidden_dim, seed=0):
        np.random.seed(seed)
        self.W1 = 0.01 * np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = 0.01 * np.random.randn(hidden_dim, 1)
        self.b2 = np.zeros((1, 1))
        self.cache = {}

    # ----- activations -----
    def _sigmoid(self, x):
        # numerically stable sigmoid
        pos = x >= 0
        neg = ~pos
        z = np.empty_like(x)
        z[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        expx = np.exp(x[neg])
        z[neg] = expx / (1.0 + expx)
        return z

    def softmax(z):
        z_stable = z - np.max(z, axis=1, keepdims=True)  # subtract max for stability
        expz = np.exp(z_stable)
        return expz / np.sum(expz, axis=1, keepdims=True)

    def relu(x):
        return np.maximum(0, x)

    def relu_grad(x):
        return (x > 0).astype(float)

    # ----- forward -----
    def feed_forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self._sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        y_hat = self._sigmoid(z2)
        self.cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "y_hat": y_hat}
        return y_hat

    # ----- loss -----
    def loss(self, y_hat, y_true):
        eps = 1e-12
        y_hat = np.clip(y_hat, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))

    # ----- backprop -----
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

    # ----- sgd step -----
    def step(self, grads, lr=0.1):
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]

    # ----- predict / score -----
    def predict(self, X, threshold=0.5):
        probs = self.feed_forward(X)
        return (probs >= threshold).astype(int)

    def score(self, X, y_true, threshold=0.5):
        y_pred = self.predict(X, threshold=threshold)
        return (y_pred == y_true).mean()

    # ----- training loop -----
    def fit(self, X, y, epochs=1000, lr=0.1, batch_size=None, verbose_every=100, shuffle=True):
        """
        X: (N, D)
        y: (N, 1) with {0,1}
        If batch_size is None or >= N, uses full-batch gradient descent.
        """
        N = X.shape[0]
        for epoch in range(1, epochs + 1):
            if batch_size is None or batch_size >= N:
                # Full-batch
                y_hat = self.feed_forward(X)
                grads = self.backprop(y)
                self.step(grads, lr=lr)
                loss_val = self.loss(y_hat, y)
            else:
                # Mini-batch SGD
                if shuffle:
                    idx = np.random.permutation(N)
                    X, y = X[idx], y[idx]
                loss_accum = 0.0
                for start in range(0, N, batch_size):
                    end = start + batch_size
                    Xb, yb = X[start:end], y[start:end]
                    y_hat_b = self.feed_forward(Xb)
                    grads = self.backprop(yb)
                    self.step(grads, lr=lr)
                    loss_accum += self.loss(y_hat_b, yb) * len(Xb)
                loss_val = loss_accum / N

            if verbose_every and (epoch == 1 or epoch % verbose_every == 0 or epoch == epochs):
                acc = self.score(X, y)
                print(f"epoch {epoch:4d} | loss={loss_val:.4f} | acc={acc:.3f}")

# Toy dataset (linearly separable)
np.random.seed(1)
N = 400
X = np.random.randn(N, 2)
y = (X[:, 0] + 0.5*X[:, 1] > 0).astype(int).reshape(-1, 1)

net = NeuralNet(input_dim=2, hidden_dim=4, seed=0)
net.fit(X, y, epochs=1000, lr=0.5, batch_size=None, verbose_every=200)  # full-batch
print("Final accuracy:", net.score(X, y))



