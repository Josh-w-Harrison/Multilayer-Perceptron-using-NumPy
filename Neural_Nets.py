import numpy as np

__all__ = ["MLP", "Standardiser", "train_val_split", "confusion_matrix"]

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
        assert isinstance(layer_sizes, (list, tuple)) and len(layer_sizes) >= 2
        self.sizes = list(layer_sizes)
        self.L = len(self.sizes) - 1
        rng = np.random.RandomState(seed)

        self.W = []
        self.b = []
        for l in range(self.L):
            fan_in, fan_out = self.sizes[l], self.sizes[l+1]
            if l < self.L - 1:
                Wl = rng.randn(fan_in, fan_out) * math.sqrt(2.0 / fan_in)
            else:
                Wl = rng.randn(fan_in, fan_out) * math.sqrt(1.0 / fan_in)
            bl = np.zeros((1, fan_out))
            self.W.append(Wl)
            self.b.append(bl)

        self.cache = {}

    @staticmethod
    def _relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_grad(x):
        return (x > 0.0).astype(x.dtype)

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

    def forward(self, X):
        A = [X]
        Z = []
        for l in range(self.L):
            z = A[-1] @ self.W[l] + self.b[l]
            Z.append(z)
            if l < self.L - 1:
                a = self._relu(z)
            else:
                a = self._softmax(z)
            A.append(a)
        self.cache = {"A": A, "Z": Z}
        return A[-1]

    def loss(self, probs, y):
        C = probs.shape[1]
        Y = self._to_one_hot(y, C)
        eps = 1e-12
        probs = np.clip(probs, eps, 1 - eps)
        return -np.mean(np.sum(Y * np.log(probs), axis=1))

    def backprop(self, y):
        A, Z = self.cache["A"], self.cache["Z"]
        N = A[0].shape[0]
        C = A[-1].shape[1]
        Y = self._to_one_hot(y, C)

        dW = [None] * self.L
        db = [None] * self.L

        dZ = (A[-1] - Y) / N
        dW[self.L-1] = A[-2].T @ dZ
        db[self.L-1] = dZ.sum(axis=0, keepdims=True)

        for l in range(self.L-2, -1, -1):
            dA = dZ @ self.W[l+1].T
            dZ = dA * self._relu_grad(Z[l])
            dW[l] = A[l].T @ dZ
            db[l] = dZ.sum(axis=0, keepdims=True)

        return dW, db

    def step(self, dW, db, lr=0.1, weight_decay=0.0):
        for l in range(self.L):
            if weight_decay:
                dW[l] = dW[l] + weight_decay * self.W[l]
            self.W[l] -= lr * dW[l]
            self.b[l] -= lr * db[l]

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1).reshape(-1, 1)

    def score(self, X, y):
        y_pred = self.predict(X)
        y_idx = y if (y.ndim == 1 or y.shape[1] == 1) else np.argmax(y, axis=1).reshape(-1, 1)
        return (y_pred == y_idx).mean()

    def fit(self, X, y, epochs=500, lr=0.1, batch_size=None, shuffle=True,
            verbose_every=100, weight_decay=0.0, record_history=False):
        N = X.shape[0]
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        history = {"loss": [], "acc": []} if record_history else None

        for epoch in range(1, epochs + 1):
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
                n_seen = 0
                for s in range(0, N, batch_size):
                    e = s + batch_size
                    Xb, yb = X[s:e], y[s:e]
                    probs_b = self.forward(Xb)
                    dW, db = self.backprop(yb)
                    self.step(dW, db, lr=lr, weight_decay=weight_decay)
                    loss_sum += self.loss(probs_b, yb) * len(Xb)
                    n_seen += len(Xb)
                loss_val = loss_sum / max(1, n_seen)

            if record_history:
                acc = self.score(X, y)
                history["loss"].append(loss_val)
                history["acc"].append(acc)

            if verbose_every and (epoch == 1 or epoch % verbose_every == 0 or epoch == epochs):
                acc = self.score(X, y)
                print(f"epoch {epoch:4d} | loss={loss_val:.4f} | acc={acc:.3f}")

        return history

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
        model = cls(sizes, seed=0)
        for l in range(model.L):
            model.W[l] = data[f"W{l}"]
            model.b[l] = data[f"b{l}"]
        return model
    
class Standardiser:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.std_ = X.std(axis=0, keepdims=True) + 1e-12
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def save(self, path):
        np.savez_compressed(path, mean=self.mean_, std=self.std_)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        obj = cls()
        obj.mean_ = data["mean"]
        obj.std_  = data["std"]
        return obj


def train_val_split(X, y, val_ratio=0.25, seed=0, stratify=True):
    rng = np.random.RandomState(seed)
    N = X.shape[0]
    if stratify:
        classes = np.unique(y.reshape(-1))
        tr_idx, va_idx = [], []
        for c in classes:
            idx = np.where(y.reshape(-1) == c)[0]
            rng.shuffle(idx)
            n_val = int(round(len(idx) * val_ratio))
            va_idx.extend(idx[:n_val].tolist())
            tr_idx.extend(idx[n_val:].tolist())
        tr_idx, va_idx = np.array(tr_idx), np.array(va_idx)
    else:
        idx = np.arange(N)
        rng.shuffle(idx)
        n_val = int(round(N * val_ratio))
        va_idx = idx[:n_val]
        tr_idx = idx[n_val:]
    return X[tr_idx], X[va_idx], y[tr_idx], y[va_idx]


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true.reshape(-1), y_pred.reshape(-1)):
        cm[t, p] += 1
    return cm


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a from-scratch MLP (NumPy).")
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated sizes, e.g. '4,16,3'.")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.25)
    parser.add_argument("--model_path", type=str, default="mlp_model.npz")
    parser.add_argument("--scaler_path", type=str, default="scaler.npz")
    args = parser.parse_args()

    try:
        from sklearn.datasets import load_iris
        iris = load_iris()
        X = iris.data.astype(np.float64)
        y = iris.target.reshape(-1, 1)
        print(f"Loaded Iris: X={X.shape}, classes={np.unique(y).size}")
    except Exception:
        print("scikit-learn not available; using synthetic 3-class blobs.")
        rng = np.random.RandomState(args.seed)
        N, D, C = 450, 2, 3
        centers = np.array([[ 2.0,  0.0],
                            [-2.0,  0.5],
                            [ 0.0, -2.0]])
        X_list, y_list = [], []
        for ci in range(C):
            Xc = rng.randn(N//C, D) + centers[ci]
            yc = np.full((N//C, 1), ci, dtype=int)
            X_list.append(Xc); y_list.append(yc)
        X = np.vstack(X_list)
        y = np.vstack(y_list)
        print(f"Synthetic blobs: X={X.shape}, classes={np.unique(y).size}")

    X_tr, X_va, y_tr, y_va = train_val_split(X, y, val_ratio=args.val_ratio, seed=args.seed, stratify=True)
    scaler = Standardiser().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)

    if args.layers:
        sizes = [int(s) for s in args.layers.split(",")]
    else:
        D = X_tr_s.shape[1]
        C = int(np.unique(y_tr).size)
        sizes = [D, 16, C]

    print("Architecture:", sizes)
    mlp = MLP(sizes, seed=args.seed)

    mlp.fit(
        X_tr_s, y_tr,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        shuffle=True, verbose_every=max(1, args.epochs // 4),
        weight_decay=args.weight_decay, record_history=False
    )

    print("Train acc:", mlp.score(X_tr_s, y_tr))
    print("Val   acc:", mlp.score(X_va_s, y_va))

    mlp.save(args.model_path)
    scaler.save(args.scaler_path)
    print(f"Saved model to {args.model_path} and {args.scaler_path}")
