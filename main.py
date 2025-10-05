from Neural_Nets import NeuralNet, NeuralNetMC
import numpy as np

if __name__ == "__main__":
    # Make a simple 3-class toy dataset (blobs)
    np.random.seed(7)
    N = 450
    D = 2
    C = 3
    X = np.vstack([
        np.random.randn(N//3, D) + np.array([ 2.0,  0.0]),
        np.random.randn(N//3, D) + np.array([-2.0,  0.5]),
        np.random.randn(N//3, D) + np.array([ 0.0, -2.0]),
    ])
    y = np.concatenate([
        np.zeros(N//3, dtype=int),
        np.ones (N//3, dtype=int),
        np.full (N//3, 2, dtype=int),
    ]).reshape(-1, 1)

    net = NeuralNetMC(input_dim=D, hidden_dim=16, num_classes=C, seed=0)
    net.fit(X, y, epochs=800, lr=0.1, batch_size=64, verbose_every=200, weight_decay=1e-4)
    print("Final accuracy:", net.score(X, y))

