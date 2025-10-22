#!/usr/bin/env python3
# Train the from-scratch MLP on real datasets (Iris/Wine/Digits/MNIST) or CSV/synthetics.
import argparse
import numpy as np

# Local module (same folder) — MLP/Standardiser utilities
from Neural_Nets import MLP, Standardiser, train_val_split, confusion_matrix

# -----------------------------
# Dataset loaders
# -----------------------------
def load_dataset(name, seed=0):
    """
    Built-in datasets that may rely on scikit-learn (iris, wine, digits).
    Synthetics (moons, blobs) require no external deps.
    """
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

def load_mnist_via_sklearn():
    """
    MNIST via scikit-learn's OpenML fetcher.
    Returns X in [0,1], shape (70000, 784); y in {0..9} shape (70000,1).
    """
    try:
        from sklearn.datasets import fetch_openml
    except Exception as e:
        raise RuntimeError("MNIST requires scikit-learn: pip install scikit-learn") from e
    ds = fetch_openml("mnist_784", version=1, as_frame=False)
    X = ds.data.astype(np.float64) / 255.0
    y = ds.target.astype(int).reshape(-1, 1)
    return X, y

def load_csv(path, target, delimiter=",", has_header=True):
    """
    Generic CSV loader.
    Args:
      path: CSV file
      target: column name (if has_header) or 0-based index
      delimiter: field delimiter
      has_header: whether first row is header
    Returns:
      X (float64, N x D), y (int, N x 1), info dict {'classes': [...], 'target_name': str}
    """
    import csv
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV appears to be empty.")

    if has_header:
        header = rows[0]
        data_rows = rows[1:]
        if isinstance(target, str):
            if target not in header:
                raise ValueError(f"Target column '{target}' not found in header: {header}")
            t_idx = header.index(target)
            target_name = target
        else:
            t_idx = int(target)
            if t_idx < 0 or t_idx >= len(header):
                raise ValueError(f"Target index {t_idx} out of range for header with {len(header)} columns.")
            target_name = header[t_idx]
    else:
        header = None
        data_rows = rows
        t_idx = int(target)
        target_name = f"col_{t_idx}"

    # Split into X and y
    X_raw, y_raw = [], []
    for r in data_rows:
        if not r:
            continue
        y_raw.append(r[t_idx])
        X_raw.append([c for i, c in enumerate(r) if i != t_idx])

    # Convert X to float64
    try:
        X = np.array(X_raw, dtype=np.float64)
    except ValueError as e:
        raise ValueError("Failed to convert some feature columns to float. "
                         "Ensure your CSV features are numeric or preprocessed.") from e

    # Encode labels (could be strings) to 0..C-1
    y_raw = np.array(y_raw)
    classes, y_idx = np.unique(y_raw, return_inverse=True)
    y = y_idx.reshape(-1, 1).astype(int)

    info = {"classes": classes.tolist(), "target_name": target_name}
    return X, y, info

# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="Train a from-scratch NumPy MLP on real datasets or CSV.")
    ds_help = "iris | wine | digits | mnist | moons | blobs (or use --csv)"
    p.add_argument("--dataset", type=str, default=None, help=ds_help)
    p.add_argument("--csv", type=str, default=None, help="Path to a CSV file to train on.")
    p.add_argument("--target", type=str, default=None,
                   help="Target column name (if CSV has header) or 0-based index (use with --no_header).")
    p.add_argument("--delimiter", type=str, default=",", help="CSV delimiter (default ',').")
    p.add_argument("--no_header", action="store_true", help="Set if CSV has no header row.")
    p.add_argument("--layers", type=str, default=None,
                   help="Comma-separated sizes like '4,16,3'. If omitted, inferred from data.")
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default=None,
                   help="sgd | momentum | rmsprop | adam")
    p.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD with momentum")
    p.add_argument("--rho", type=float, default=0.9, help="RMSprop rho")
    p.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    p.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")
    p.add_argument("--eps", type=float, default=1e-8, help="Optimizer epsilon")
    p.add_argument("--val_ratio", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model_path", type=str, default="mlp_model.npz")
    p.add_argument("--scaler_path", type=str, default="scaler.npz")
    p.add_argument("--record_history", action="store_true", help="Store training/validation curves")
    p.add_argument("--plot_curves", action="store_true", help="Save loss/accuracy plots as PNG")
    p.add_argument("--plot_prefix", type=str, default="training", help="Filename prefix for plots")
    p.add_argument("--show_cm", action="store_true", help="Print validation confusion matrix")
    p.add_argument("--early_stop", action="store_true", help="Enable early stopping on validation set")
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs without improvement)")
    p.add_argument("--min_delta", type=float, default=0.0, help="Minimum improvement to reset patience")
    p.add_argument("--monitor", type=str, default="val_loss", help="Metric to monitor: val_loss | val_acc | train_loss | acc")
    p.add_argument("--restore_best", action="store_true", help="Restore best weights at the end")
    args = p.parse_args()

    # Load data with priority: CSV > dataset
    if args.csv:
        if args.target is None:
            raise SystemExit("--csv requires --target (name or 0-based index)")
        tgt = args.target if not args.no_header else int(args.target)
        X, y, info = load_csv(args.csv, target=tgt, delimiter=args.delimiter, has_header=(not args.no_header))
        C = int(np.unique(y).size)
        print(f"Loaded CSV '{args.csv}': X={X.shape}, classes={C}, target={info['target_name']}")
        dataset_key = "csv"
    elif args.dataset:
        name = args.dataset.lower()
        if name == "mnist":
            X, y = load_mnist_via_sklearn()
        else:
            X, y = load_dataset(name, seed=args.seed)  # iris|wine|digits|moons|blobs
        C = int(np.unique(y).size)
        print(f"Loaded '{args.dataset}': X={X.shape}, classes={C}")
        dataset_key = name
    else:
        raise SystemExit("Specify --dataset or --csv. See --help for options.")

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
        default_map = {
            "iris":   [D, 16, C],
            "wine":   [D, 32, C],
            "digits": [D, 64, C],
            "mnist":  [D, 128, 64, C],
            "moons":  [D, 32, C],
            "blobs":  [D, 16, C],
            "csv":    [D, 32, C],
        }
        sizes = default_map.get(dataset_key, [D, 32, C])

    print("Architecture:", sizes)
    mlp = MLP(sizes, seed=args.seed)

    # Train
    history = mlp.fit(
        X_tr_s, y_tr,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        shuffle=True, verbose_every=max(1, args.epochs // 4),
        weight_decay=args.weight_decay, record_history=args.record_history,
        optimizer=args.optimizer,
        X_val=X_va_s, y_val=y_va, monitor=args.monitor,
        early_stop=args.early_stop, patience=args.patience, min_delta=args.min_delta,
        restore_best=args.restore_best,
        momentum=args.momentum, rho=args.rho, beta1=args.beta1, beta2=args.beta2, eps=args.eps
    )

    # Metrics
    tr_acc = mlp.score(X_tr_s, y_tr)
    va_acc = mlp.score(X_va_s, y_va)
    print(f"Train acc: {tr_acc:.3f}")
    print(f"Val   acc: {va_acc:.3f}")

    if args.show_cm:
        y_pred = mlp.predict(X_va_s)
        cm = confusion_matrix(y_va, y_pred, C)
        print("Confusion matrix (rows=true, cols=pred): n", cm)

    # Save
    mlp.save(args.model_path)
    scaler.save(args.scaler_path)
    print(f"Saved model to {args.model_path} and scaler to {args.scaler_path}")
    # Save plots if requested
    if args.record_history and args.plot_curves and history is not None:
        try:
            from Neural_Nets import plot_history
            plot_history(history, prefix=args.plot_prefix, show=False, save=True)
            print(f"Saved plots: {args.plot_prefix}_loss.png, {args.plot_prefix}_acc.png")
        except Exception as e:
            print("Plotting failed:", e)


if __name__ == "__main__":
    main()
