# NumPy Neural Network Framework

A lightweight, from-scratch neural network framework built entirely in NumPy, designed for learning and experimentation.  
It includes modular components for:
- Fully-connected multi-layer perceptrons (MLPs)
- Multiple optimizers (SGD, Momentum, RMSprop, Adam)
- Early stopping and training-curve visualisation
- Command-line support for real datasets (Iris, Wine, Digits, MNIST) or custom CSVs

---

## Features
- Pure NumPy implementation — no PyTorch or TensorFlow required  
- Optimizers: SGD, SGD with Momentum, RMSprop, Adam  
- Early stopping with patience and best-model restoration  
- Training and validation history tracking (loss / accuracy)  
- Plotting utilities to visualise learning curves  
- Command-line interface to train on real datasets or CSVs  
- Lightweight utilities: standardisation, confusion matrices, model save/load  

---

## Project Structure
```
Neural_Nets.py        # Core implementation (NeuralNet, NeuralNetMC, MLP, Optimizers, Plotting)
train_on_dataset.py   # CLI training script (datasets, early stopping, plots)
main.py               # Minimal example (Iris dataset)
```

---

## Installation
Install dependencies:
```bash
pip install numpy matplotlib scikit-learn
```

---

## Quick Start

### 1. Run a simple demo (Iris)
```bash
python main.py
```
This trains a small 2-layer MLP on the Iris dataset, prints accuracy, and saves the model (`iris_mlp.npz`).

---

### 2. Train via the CLI on built-in datasets
Supported datasets: `iris`, `wine`, `digits`, `mnist`, `moons`, `blobs`

Example: MNIST with Adam optimizer and early stopping
```bash
python train_on_dataset.py --dataset mnist --optimizer adam --lr 0.001   --epochs 40 --batch_size 128   --early_stop --patience 5 --min_delta 0.0001 --monitor val_loss --restore_best   --record_history --plot_curves --plot_prefix mnist_run
```

Output:
```
Train acc: 0.992
Val   acc: 0.967
Saved model to mlp_model.npz and scaler to scaler.npz
Saved plots: mnist_run_loss.png, mnist_run_acc.png
```

---

### 3. Train on your own CSV
```bash
python train_on_dataset.py --csv data.csv --target label_column_name   --optimizer adam --lr 0.001 --epochs 100 --record_history --plot_curves
```

If your CSV has no header:
```bash
python train_on_dataset.py --csv data.csv --no_header --target 4
```
(where column index `4` is the label)

---

## CLI Reference

| Argument | Description |
|-----------|--------------|
| `--dataset` | iris, wine, digits, mnist, moons, blobs |
| `--csv` | Path to CSV file instead of built-in dataset |
| `--target` | Target column name or index (for CSV) |
| `--layers` | Custom architecture, e.g. `784,128,64,10` |
| `--optimizer` | sgd, momentum, rmsprop, adam |
| `--lr` | Learning rate |
| `--batch_size` | Mini-batch size |
| `--epochs` | Number of epochs |
| `--weight_decay` | L2 regularisation coefficient |
| `--early_stop` | Enable early stopping |
| `--patience` | Epochs to wait without improvement |
| `--min_delta` | Required improvement to reset patience |
| `--monitor` | Metric to monitor (val_loss, val_acc, etc.) |
| `--restore_best` | Restore best weights after stopping |
| `--record_history` | Store training/validation curves |
| `--plot_curves` | Save loss / accuracy plots to PNG |
| `--plot_prefix` | Prefix for saved plot files |
| `--show_cm` | Print confusion matrix on validation set |

---

## Plotting Example

After training with `--record_history --plot_curves`, two figures are saved:
- `training_loss.png` – training and validation loss  
- `training_acc.png` – training and validation accuracy  

Programmatic example:
```python
from Neural_Nets import plot_history
plot_history(history, prefix="experiment1", show=True)
```

---

## Model Persistence
Save and load trained models:
```python
mlp.save("model.npz")
loaded = MLP.load("model.npz")
print("Restored accuracy:", loaded.score(X_test, y_test))
```

---

## Example: Custom usage in Python
```python
from Neural_Nets import MLP, Standardiser, plot_history
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
y = y.reshape(-1, 1)
scaler = Standardiser().fit(X)
X_s = scaler.transform(X)
X_tr, X_va, y_tr, y_va = train_test_split(X_s, y, test_size=0.25, random_state=0)

mlp = MLP([64, 64, 10], seed=0)
history = mlp.fit(X_tr, y_tr, epochs=40, batch_size=64, lr=1e-3,
                  optimizer="adam", X_val=X_va, y_val=y_va,
                  early_stop=True, record_history=True)
plot_history(history, prefix="digits_demo", save=True)
```

---

## POssible Future Extensions
- Learning-rate schedulers (step or cosine decay)
- Checkpoint saving and auto-resume
- Decision-boundary visualiser for 2D datasets
