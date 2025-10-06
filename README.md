
# From‑Scratch Multiclass MLP (NumPy)

A lightweight, framework‑free **multilayer perceptron (MLP)** for **multiclass classification**, built only with **NumPy**.

## Features
- Configurable depth/width: pass `layer_sizes` like `[D, 32, 32, C]`
- **ReLU** hidden layers, **Softmax + Cross‑Entropy** output
- Mini‑batch **SGD** with optional **L2 weight decay**
- Simple **Standardiser** (persistable) for feature scaling
- **Save/Load** model and scaler via `.npz`

## Files
- `mlp_from_scratch.py` — the module (classes: `MLP`, `Standardiser`; utils: `train_val_split`, `confusion_matrix`)
- `from_scratch_mlp.ipynb` — companion notebook (plots & walkthrough)

## Install
```
pip install numpy matplotlib scikit-learn  # sklearn optional (Iris demo)
```

## Quick Start (Python)
```python
import numpy as np
from mlp_from_scratch import MLP, Standardiser, train_val_split

# Data (3 classes)
N, D, C = 300, 4, 3
X = np.random.randn(N, D)
y = np.random.randint(0, C, size=(N, 1))

# Split & scale
X_tr, X_va, y_tr, y_va = train_val_split(X, y, val_ratio=0.25, seed=0, stratify=True)
scaler = Standardiser().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_va_s = scaler.transform(X_va)

# Model
net = MLP([D, 16, C], seed=0)
net.fit(X_tr_s, y_tr, epochs=500, lr=0.1, batch_size=32, weight_decay=1e-4)

print("Val acc:", net.score(X_va_s, y_va))

# Save
net.save("mlp_model.npz")
scaler.save("scaler.npz")
```

## CLI Demo
```
python mlp_from_scratch.py --layers 4,16,3 --epochs 600 --lr 0.1 --batch_size 32 --weight_decay 1e-4
```
Saves `mlp_model.npz` and `scaler.npz` and prints train/val accuracy.

### CLI Options
- `--layers` e.g. `4,32,16,3` (default inferred from data)
- `--epochs` training epochs (default 600)
- `--lr` learning rate (default 0.1)
- `--batch_size` mini‑batch size (default 32; use large value for full‑batch)
- `--weight_decay` L2 coefficient (default 1e-4)
- `--val_ratio` validation split ratio (default 0.25)
- `--seed` RNG seed (default 0)
- `--model_path`, `--scaler_path` output file paths

## Tips
- Always standardise features for tabular data and reuse the same scaler for inference.
- Increase width/depth if underfitting, e.g. `[D, 64, 64, C]`.
- If training is unstable, try a smaller `--lr` or larger `--batch_size`.
- For image datasets (e.g., MNIST), flatten images first; CNNs will outperform MLPs on images.

## License
MIT
