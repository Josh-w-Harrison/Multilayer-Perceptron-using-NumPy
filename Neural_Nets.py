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

class MLP:
    def __init__(self, layer_sizes, seed=0):
        """
        layer_sizes: list like [input_dim, h1, h2, ..., num_classes]
        """
        assert len(layer_sizes) >= 2
        self.sizes = layer_sizes
        self.L = len(layer_sizes) - 1  # number of weight layers
        rng = np.random.RandomState(seed)

        # Params: W[l] is (sizes[l], sizes[l+1]); b[l] is (1, sizes[l+1])
        self.W = []
        self.b = []
        for l in range(self.L):
            fan_in, fan_out = layer_sizes[l], layer_sizes[l+1]
            if l < self.L - 1:
                # He init for ReLU hidden layers
                Wl = rng.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            else:
                # last layer: logits; modest scale
                Wl = rng.randn(fan_in, fan_out) * np.sqrt(1.0 / fan_in)
            bl = np.zeros((1, fan_out))
            self.W.append(Wl)
            self.b.append(bl)

        self.cache = {}

    @staticmethod
    def _relu(x): return np.maximum(0.0, x)
    @staticmethod
    def _relu_grad(x): return (x > 0.0).astype(x.dtype)

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def _to_one_hot(y, C):
        if y.ndim == 2 and y.shape[1] == C:
            return y
        Y = np.zeros((y.size, C))
        Y[np.arange(y.size), y.reshape(-1)] = 1.0
        return Y

    # ----- forward -----
    def forward(self, X):
        A = [X]      # activations
        Z = []       # pre-activations
        for l in range(self.L):
            z = A[-1] @ self.W[l] + self.b[l]
            Z.append(z)
            if l < self.L - 1:
                a = self._relu(z)
            else:
                a = self._softmax(z)
            A.append(a)

        self.cache = {"A": A, "Z": Z}
        return A[-1]  # probs

    # ----- loss (cross-entropy) -----
    def loss(self, probs, y):
        C = probs.shape[1]
        Y = self._to_one_hot(y, C)
        eps = 1e-12
        probs = np.clip(probs, eps, 1 - eps)
        return -np.mean(np.sum(Y * np.log(probs), axis=1))

    # ----- backprop -----
    def backprop(self, y):
        A, Z = self.cache["A"], self.cache["Z"]
        N = A[0].shape[0]
        C = A[-1].shape[1]
        Y = self._to_one_hot(y, C)

        dW = [None] * self.L
        db = [None] * self.L

        # output layer: softmax + CE
        dZ = (A[-1] - Y) / N                 # (N, C)
        dW[self.L-1] = A[-2].T @ dZ
        db[self.L-1] = dZ.sum(axis=0, keepdims=True)

        # hidden layers (ReLU)
        for l in range(self.L-2, -1, -1):
            dA = dZ @ self.W[l+1].T
            dZ = dA * self._relu_grad(Z[l])
            dW[l] = A[l].T @ dZ
            db[l] = dZ.sum(axis=0, keepdims=True)

        return dW, db

    # ----- SGD step -----
    def step(self, dW, db, lr=0.1, weight_decay=0.0):
        for l in range(self.L):
            if weight_decay:
                dW[l] = dW[l] + weight_decay * self.W[l]
            self.W[l] -= lr * dW[l]
            self.b[l] -= lr * db[l]

    # ----- predict / score -----
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1).reshape(-1, 1)

    def score(self, X, y):
        yp = self.predict(X)
        y_idx = y if (y.ndim == 1 or y.shape[1] == 1) else np.argmax(y, axis=1).reshape(-1, 1)
        return (yp == y_idx).mean()

    # ----- training loop -----
    def fit(self, X, y, epochs=500, lr=0.1, batch_size=None, shuffle=True,
            verbose_every=100, weight_decay=0.0):
        N = X.shape[0]
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        for epoch in range(1, epochs+1):
            if batch_size is None or batch_size >= N:
                probs = self.forward(X)
                dW, db = self.backprop(y)
                self.step(dW, db, lr=lr, weight_decay=weight_decay)
                loss_val = self.loss(probs, y)
            else:
                if shuffle:
                    idx = np.random.permutation(N)
                    X, y = X[idx], y[idx]
                loss_sum = 0.0
                seen = 0
                for s in range(0, N, batch_size):
                    e = s + batch_size
                    Xb, yb = X[s:e], y[s:e]
                    probs_b = self.forward(Xb)
                    dW, db = self.backprop(yb)
                    self.step(dW, db, lr=lr, weight_decay=weight_decay)
                    loss_sum += self.loss(probs_b, yb) * len(Xb)
                    seen += len(Xb)
                loss_val = loss_sum / max(1, seen)

            if verbose_every and (epoch == 1 or epoch % verbose_every == 0 or epoch == epochs):
                acc = self.score(X, y)
                print(f"epoch {epoch:4d} | loss={loss_val:.4f} | acc={acc:.3f}")

    # ----- save / load -----
    def save(self, path):
        np.savez_compressed(
            path,
            sizes=np.array(self.sizes, dtype=np.int64),
            **{f"W{l}": self.W[l] for l in range(self.L)},
            **{f"b{l}": self.b[l] for l in range(self.L)}
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        sizes = data["sizes"].tolist()
        model = cls(sizes, seed=0)  # will be overwritten
        for l in range(model.L):
            model.W[l] = data[f"W{l}"]
            model.b[l] = data[f"b{l}"]
        return model
