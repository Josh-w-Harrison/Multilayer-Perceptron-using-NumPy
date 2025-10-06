
# From‑Scratch Multiclass MLP (NumPy)

A lightweight, framework‑free **multilayer perceptron (MLP)** for **multiclass classification**, built only with **NumPy**.

## What's here
- `Neural_Nets.py` — reusable module with:
  - `MLP` (ReLU hidden layers, Softmax output, Cross‑Entropy loss, SGD)
  - `Standardiser` (fit/transform/save/load)
  - `train_val_split`, `confusion_matrix`
- `train_on_dataset.py` — CLI trainer for **Iris/Wine/Digits/MNIST**, **synthetics** (moons/blobs), or **any CSV**.

## Install
```bash
pip install numpy matplotlib scikit-learn  # sklearn used for datasets like Iris/Wine/Digits/MNIST
```

## Quick start (Iris)
```bash
python train_on_dataset.py --dataset iris --layers 4,16,3 --epochs 600 --lr 0.1 --batch_size 32 --show_cm
```

## Train on MNIST
Uses OpenML via scikit‑learn, flattens 28×28 to 784 features and scales to [0,1].
```bash
python train_on_dataset.py --dataset mnist --epochs 15 --lr 0.05 --batch_size 128 --show_cm
# or with a deeper net
python train_on_dataset.py --dataset mnist --layers 784,256,128,10 --epochs 15 --lr 0.05 --batch_size 128 --show_cm
```

## Train on Digits (8×8 images → 64 features)
```bash
python train_on_dataset.py --dataset digits --layers 64,64,10 --epochs 800 --lr 0.1 --batch_size 64 --show_cm
```

## Train on any CSV
The loader accepts a header or not, and maps string labels to 0..C‑1.

With a header (target by name):
```bash
python train_on_dataset.py --csv mydata.csv --target label --epochs 500 --lr 0.1 --batch_size 64 --show_cm
```

No header (target by index, e.g., last column):
```bash
python train_on_dataset.py --csv mydata_noheader.csv --no_header --target 10 --epochs 500 --lr 0.1
```

Options:
- `--delimiter` to change the CSV delimiter (default `,`).
- `--val_ratio` (default `0.25`) uses a stratified split.

## Architecture defaults
If `--layers` is omitted, the trainer picks sensible defaults:
- iris → `[D, 16, C]`
- wine → `[D, 32, C]`
- digits → `[D, 64, C]`
- mnist → `[D, 128, 64, C]`
- moons → `[D, 32, C]`
- blobs → `[D, 16, C]`
- csv → `[D, 32, C]`

You can always override, e.g., `--layers 32,32,10`.

## Saving & loading
Training saves:
- `mlp_model.npz` — network weights
- `scaler.npz` — feature standardiser

At inference time, load both and apply the scaler before calling `MLP.predict`.

## Notes
- Always **fit** the `Standardiser` on the **training** split only; use it to transform validation/test.
- For images (MNIST/Digits), inputs are numeric and scaled; standardising still helps.
- This MLP is educational; CNNs outperform it on image tasks.

## License
MIT
