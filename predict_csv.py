#!/usr/bin/env python3
# Predict classes (and optional probabilities) for a CSV using a saved from-scratch MLP.
import argparse
import csv
import json
import numpy as np
from Neural_Nets import MLP, Standardiser

def read_csv_features(path, delimiter=",", has_header=True, exclude=None):
    exclude = set([] if exclude is None else [str(x) for x in exclude])
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        rows = list(reader)
    if not rows:
        raise SystemExit("CSV appears to be empty.")
    if has_header:
        header = rows[0]
        data_rows = rows[1:]
        keep_idx = [i for i, col in enumerate(header) if (col not in exclude and str(i) not in exclude)]
        feat_header = [header[i] for i in keep_idx]
    else:
        header = None
        data_rows = rows
        n_cols = len(rows[0])
        keep_idx = [i for i in range(n_cols) if (str(i) not in exclude)]
        feat_header = None
    X_list = []
    for r in data_rows:
        if not r:
            continue
        try:
            X_list.append([float(r[i]) for i in keep_idx])
        except ValueError as e:
            bad_cols = [i for i in keep_idx if (i >= len(r) or not r[i].replace('.','',1).replace('-','',1).isdigit())]
            raise SystemExit(
                "Non-numeric value encountered in features.\n"
                f"Row sample: {r}\n"
                f"Consider using --exclude to drop non-numeric columns; indices to try: {bad_cols}"
            ) from e
    X = np.array(X_list, dtype=np.float64)
    return X, feat_header

def maybe_load_class_names(path, num_classes):
    if not path:
        return None
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            names = json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
    if len(names) != num_classes:
        raise SystemExit(f"--class_map contains {len(names)} names but model has {num_classes} classes.")
    return names

def write_output(path, preds, probs=None, header=None, class_names=None):
    num_classes = probs.shape[1] if probs is not None else None
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        pred_col = "pred_class"
        if class_names:
            pred_col = "pred_label"
        cols = [pred_col]
        if probs is not None:
            if class_names:
                prob_cols = [f"prob_{name}" for name in class_names]
            else:
                prob_cols = [f"prob_{i}" for i in range(num_classes)]
            cols.extend(prob_cols)
        w.writerow(cols)
        for i in range(preds.shape[0]):
            row = []
            if class_names:
                row.append(class_names[int(preds[i,0])])
            else:
                row.append(int(preds[i,0]))
            if probs is not None:
                row.extend(list(map(lambda x: f"{x:.6f}", probs[i].tolist())))
            w.writerow(row)

def main():
    ap = argparse.ArgumentParser(description="Predict a CSV using a saved from-scratch NumPy MLP.")
    ap.add_argument("--input", required=True, help="Input CSV with feature columns (no labels required).")
    ap.add_argument("--output", default="predictions.csv", help="Output CSV path (default: predictions.csv).")
    ap.add_argument("--model", default="mlp_model.npz", help="Path to saved model (.npz).")
    ap.add_argument("--scaler", default="scaler.npz", help="Path to saved scaler (.npz).")
    ap.add_argument("--delimiter", default=",", help="CSV delimiter (default ',').")
    ap.add_argument("--no_header", action="store_true", help="Set if CSV has no header row.")
    ap.add_argument("--exclude", type=str, default=None,
                    help="Comma-separated list of columns to drop (names if header, else 0-based indices).")
    ap.add_argument("--probs", action="store_true", help="Also output class probabilities.")
    ap.add_argument("--class_map", type=str, default=None,
                    help="Optional file with class names in index order (JSON list or newline-separated).")
    args = ap.parse_args()

    exclude = args.exclude.split(",") if args.exclude else None

    model = MLP.load(args.model)
    scaler = Standardiser.load(args.scaler)

    X, _ = read_csv_features(args.input, delimiter=args.delimiter, has_header=(not args.no_header), exclude=exclude)
    Xs = scaler.transform(X)

    preds = model.predict(Xs)
    probs = None
    if args.probs:
        probs = model.forward(Xs)

    class_names = maybe_load_class_names(args.class_map, model.sizes[-1]) if args.class_map else None
    write_output(args.output, preds, probs=probs, class_names=class_names)
    print(f"Wrote predictions to {args.output}")

if __name__ == "__main__":
    main()
