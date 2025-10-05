from Neural_Nets import NeuralNet, NeuralNetMC, MLP
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Dataset: Iris (3 classes)
    iris = load_iris()
    X = iris.data.astype(np.float64)   # shape (150,4)
    y = iris.target.reshape(-1, 1)     # ints in {0,1,2}

    # Standardise features (important!)
    X_mean, X_std = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True) + 1e-12
    X = (X - X_mean) / X_std

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

    # Build a small net: 4 -> 16 -> 3
    net = MLP([X.shape[1], 16, 3], seed=0)
    net.fit(X_tr, y_tr, epochs=600, lr=0.1, batch_size=32, verbose_every=150, weight_decay=1e-4)
    print("Train acc:", net.score(X_tr, y_tr))
    print("Test  acc:", net.score(X_te, y_te))

    # Save and load
    net.save("iris_mlp.npz")
    loaded = MLP.load("iris_mlp.npz")
    print("Loaded test acc:", loaded.score(X_te, y_te))


