import numpy as np

class NeuralNet:
    def __init__(self, input_dim, hidden_dim, seed=0):
        np.random.seed(seed)
        # He init for ReLU layers
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, 1))
        self.cache = {}

    # ----- activations -----
    @staticmethod
    def _relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_grad(x):
        # derivative wrt pre-activation (z1)
        return (x > 0.0).astype(x.dtype)

    @staticmethod
    def _sigmoid(x):
        # numerically stable sigmoid
        pos = x >= 0
        neg = ~pos
        z = np.empty_like(x)
        z[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        expx = np.exp(x[neg])
        z[neg] = expx / (1.0 + expx)
        return z

    # ----- forward -----
    def feed_forward(self, X):
        z1 = X @ self.W1 + self.b1           # (N, H)
        a1 = self._relu(z1)                  # ReLU hidden
        z2 = a1 @ self.W2 + self.b2          # (N, 1)
        y_hat = self._sigmoid(z2)            # sigmoid output
        self.cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "y_hat": y_hat}
        return y_hat

    # ----- loss -----
    @staticmethod
    def loss(y_hat, y_true):
        eps = 1e-12
        y_hat = np.clip(y_hat, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))

    # ----- backprop -----
    def backprop(self, y_true):
        X   = self.cache["X"]
        z1  = self.cache["z1"]
        a1  = self.cache["a1"]
        y_hat = self.cache["y_hat"]

        N = X.shape[0]

        # Output layer (sigmoid + BCE)
        dz2 = (y_hat - y_true) / N                   # (N, 1)
        dW2 = a1.T @ dz2                             # (H, 1)
        db2 = dz2.sum(axis=0, keepdims=True)         # (1, 1)

        # Hidden layer (through ReLU)
        da1 = dz2 @ self.W2.T                        # (N, H)
        dz1 = da1 * self._relu_grad(z1)              # (N, H)

        dW1 = X.T @ dz1                              # (D, H)
        db1 = dz1.sum(axis=0, keepdims=True)         # (1, H)

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
        N = X.shape[0]
        for epoch in range(1, epochs + 1):
            if batch_size is None or batch_size >= N:
                y_hat = self.feed_forward(X)
                grads = self.backprop(y)
                self.step(grads, lr=lr)
                loss_val = self.loss(y_hat, y)
            else:
                if shuffle:
                    idx = np.random.permutation(N)
                    X, y = X[idx], y[idx]
                loss_accum = 0.0
                for s in range(0, N, batch_size):
                    e = s + batch_size
                    Xb, yb = X[s:e], y[s:e]
                    y_hat_b = self.feed_forward(Xb)
                    grads = self.backprop(yb)
                    self.step(grads, lr=lr)
                    loss_accum += self.loss(y_hat_b, yb) * len(Xb)
                loss_val = loss_accum / N

            if verbose_every and (epoch == 1 or epoch % verbose_every == 0 or epoch == epochs):
                acc = self.score(X, y)
                print(f"epoch {epoch:4d} | loss={loss_val:.4f} | acc={acc:.3f}")


                
class NeuralNetMC:
    def __init__(self, input_dim, hidden_dim, num_classes, seed=0):
        np.random.seed(seed)
        # He init for ReLU
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        # Xavier-ish for logits layer
        self.W2 = np.random.randn(hidden_dim, num_classes) * np.sqrt(1.0 / hidden_dim)
        self.b2 = np.zeros((1, num_classes))
        self.cache = {}

    # ----- activations -----
    @staticmethod
    def _relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_grad(x):
        return (x > 0.0).astype(x.dtype)

    @staticmethod
    def _softmax(logits):
        z = logits - logits.max(axis=1, keepdims=True)  # stability
        expz = np.exp(z)
        return expz / expz.sum(axis=1, keepdims=True)

    # ----- forward -----
    def feed_forward(self, X):
        z1 = X @ self.W1 + self.b1               # (N, H)
        a1 = self._relu(z1)                      # (N, H)
        z2 = a1 @ self.W2 + self.b2              # (N, C)
        probs = self._softmax(z2)                # (N, C)
        self.cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "probs": probs}
        return probs

    # ----- loss (categorical cross-entropy) -----
    @staticmethod
    def _to_one_hot(y, C):
        if y.ndim == 2 and y.shape[1] == C:  # already one-hot
            return y
        Y = np.zeros((y.size, C))
        Y[np.arange(y.size), y.reshape(-1)] = 1.0
        return Y

    def loss(self, probs, y, C):
        Y = self._to_one_hot(y, C) if probs.shape[1] == C else y
        eps = 1e-12
        probs = np.clip(probs, eps, 1 - eps)
        return -np.mean(np.sum(Y * np.log(probs), axis=1))

    # ----- backprop -----
    def backprop(self, y, C):
        X   = self.cache["X"]
        z1  = self.cache["z1"]
        a1  = self.cache["a1"]
        probs = self.cache["probs"]

        N = X.shape[0]
        Y = self._to_one_hot(y, C) if probs.shape[1] == C else y

        # dL/dz2 = (probs - Y) / N  (softmax + CE)
        dZ2 = (probs - Y) / N                      # (N, C)
        dW2 = a1.T @ dZ2                           # (H, C)
        db2 = dZ2.sum(axis=0, keepdims=True)       # (1, C)

        dA1 = dZ2 @ self.W2.T                      # (N, H)
        dZ1 = dA1 * self._relu_grad(z1)            # (N, H)

        dW1 = X.T @ dZ1                            # (D, H)
        db1 = dZ1.sum(axis=0, keepdims=True)       # (1, H)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    # ----- sgd step -----
    def step(self, grads, lr=0.1, weight_decay=0.0):
        # Optional L2 on weights (not biases)
        if weight_decay:
            grads["W1"] += weight_decay * self.W1
            grads["W2"] += weight_decay * self.W2
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]

    # ----- predict / score -----
    def predict(self, X):
        probs = self.feed_forward(X)
        return np.argmax(probs, axis=1).reshape(-1, 1)   # class indices

    def score(self, X, y_true):
        y_pred = self.predict(X)
        y_true_idx = y_true if y_true.ndim == 1 or y_true.shape[1] == 1 \
                     else np.argmax(y_true, axis=1).reshape(-1, 1)
        return (y_pred == y_true_idx).mean()

    # ----- training loop -----
    def fit(self, X, y, epochs=1000, lr=0.1, batch_size=None, verbose_every=100,
            shuffle=True, weight_decay=0.0):
        N, C = X.shape[0], self.W2.shape[1]
        # Ensure y has shape (N,1) if integer labels
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        for epoch in range(1, epochs + 1):
            if batch_size is None or batch_size >= N:
                probs = self.feed_forward(X)
                grads = self.backprop(y, C)
                self.step(grads, lr=lr, weight_decay=weight_decay)
                loss_val = self.loss(probs, y, C)
            else:
                if shuffle:
                    idx = np.random.permutation(N)
                    X, y = X[idx], y[idx]
                loss_accum = 0.0
                for s in range(0, N, batch_size):
                    e = s + batch_size
                    Xb, yb = X[s:e], y[s:e]
                    probs_b = self.feed_forward(Xb)
                    grads = self.backprop(yb, C)
                    self.step(grads, lr=lr, weight_decay=weight_decay)
                    loss_accum += self.loss(probs_b, yb, C) * len(Xb)
                loss_val = loss_accum / N

            if verbose_every and (epoch == 1 or epoch % verbose_every == 0 or epoch == epochs):
                acc = self.score(X, y)
                print(f"epoch {epoch:4d} | loss={loss_val:.4f} | acc={acc:.3f}")