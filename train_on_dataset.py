
#!/usr/bin/env python3
# Train the from-scratch MLP on real datasets (Iris/Wine/Digits) or simple synthetics.
import argparse
import numpy as np

# Local module (same folder) — MLP/Standardiser utilities
from Neural_Nets import MLP, Standardiser, train_val_split, confusion_matrix

def load_dataset(name, seed=0):
    name = name.lower()
    if name in ("iris", "wine", "digits"):
        try:
            if name == "iris":
                from sklearn.datasets import load_iris
                data = load_iris()
                X = data.data.astype(np.float64)
                y = data.target.reshape(-1, 1)
            elif name == "wine":
                from sklearn.datasets import load_wine
                X, y = load_wine(return_X_y=True)
                X = X.astype(np.float64)
                y = y.reshape(-1, 1)
            else:  # digits (8x8 images flattened -> 64 features)
                from sklearn.datasets import load_digits
                X, y = load_digits(return_X_y=True)
                X = X.astype(np.float64)
                # scale pixel range approx 0..16 to 0..1 (helps a bit even with Standardiser)
                X /= 16.0
                y = y.reshape(-1, 1)
            return X, y
        except Exception as e:
            raise RuntimeError(f"scikit-learn is required for dataset '{name}'. Install with 'pip install scikit-learn'.") from e
    elif name in ("moons", "blobs"):
        # Built-in synthetic datasets without sklearn (basic versions)
        rng = np.random.RandomState(seed)
        if name == "moons":
            # simple two half-moons (2 classes)
            N = 1000
            # quick-and-dirty moons generator
            t = rng.rand(N//2) * np.pi
            x1 = np.c_[np.cos(t), np.sin(t)] + 0.05*rng.randn(N//2, 2)
            x2 = np.c_[1 - np.cos(t), 1 - np.sin(t) - 0.5] + 0.05*rng.randn(N//2, 2)
            X = np.vstack([x1, x2])
            y = np.vstack([np.zeros((N//2,1), int), np.ones((N//2,1), int)])
        else:
            # 3-class blobs
            N, D, C = 600, 2, 3
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
        return X, y
    else:
        raise ValueError("Unknown dataset. Choose from: iris, wine, digits, moons, blobs.")

def main():
    p = argparse.ArgumentParser(description="Train a from-scratch NumPy MLP on real datasets.")
    p.add_argument("--dataset", type=str, default="iris",
                   help="iris | wine | digits | moons | blobs")
    p.add_argument("--layers", type=str, default=None,
                   help="Comma-separated sizes like '4,16,3'. If omitted, inferred from data.")
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model_path", type=str, default="mlp_model.npz")
    p.add_argument("--scaler_path", type=str, default="scaler.npz")
    p.add_argument("--show_cm", action="store_true", help="Print validation confusion matrix")
    args = p.parse_args()

    # Load data
    X, y = load_dataset(args.dataset, seed=args.seed)
    C = int(np.unique(y).size)
    print(f"Loaded '{args.dataset}': X={X.shape}, classes={C}")

    # Split & scale
    X_tr, X_va, y_tr, y_va = train_val_split(X, y, val_ratio=args.val_ratio, seed=args.seed, stratify=True)
    scaler = Standardiser().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)

    # Architecture
    if args.layers:
        sizes = [int(s) for s in args.layers.split(",")]
    else:
        D = X_tr_s.shape[1]
        # Good defaults per dataset if layers omitted
        default_map = {
            "iris":   [D, 16, C],
            "wine":   [D, 32, C],
            "digits": [D, 64, C],  # 64->64->10 works well
            "moons":  [D, 32, C],
            "blobs":  [D, 16, C],
        }
        sizes = default_map.get(args.dataset.lower(), [D, 32, C])

    print("Architecture:", sizes)
    mlp = MLP(sizes, seed=args.seed)

    # Train
    mlp.fit(
        X_tr_s, y_tr,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        shuffle=True, verbose_every=max(1, args.epochs // 4),
        weight_decay=args.weight_decay, record_history=False
    )

    # Metrics
    tr_acc = mlp.score(X_tr_s, y_tr)
    va_acc = mlp.score(X_va_s, y_va)
    print(f"Train acc: {tr_acc:.3f}")
    print(f"Val   acc: {va_acc:.3f}")

    if args.show_cm:
        y_pred = mlp.predict(X_va_s)
        cm = confusion_matrix(y_va, y_pred, C)
        print("Confusion matrix (rows=true, cols=pred):\n", cm)

    # Save
    mlp.save(args.model_path)
    scaler.save(args.scaler_path)
    print(f"Saved model to {args.model_path} and scaler to {args.scaler_path}")

if __name__ == "__main__":
    main()
