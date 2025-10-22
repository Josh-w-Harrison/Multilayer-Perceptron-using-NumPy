
# NumPy MLP (Updated)

This package contains your updated from-scratch MLP framework with:
- Optimizers: **SGD**, **SGD with Momentum**, **RMSprop**, **Adam**
- **Early stopping** (monitor=val_loss/val_acc/train_loss/acc, patience, min_delta, restore_best)
- **History recording** and **plotting** (`plot_history`) for loss/accuracy (train + val)
- CLI flags in `train_on_dataset.py` for the above

## Quick Start
```bash
# Iris with Adam, early stopping, and saved plots
python train_on_dataset.py --dataset iris   --optimizer adam --lr 0.001   --early_stop --patience 20 --monitor val_loss --restore_best   --record_history --plot_curves --plot_prefix iris_run
```

## Plotting programmatically
```python
from Neural_Nets import plot_history, MLP
# history = mlp.fit(..., record_history=True, X_val=..., y_val=...)
plot_history(history, prefix="training", save=True)
```

## Files
- `Neural_Nets.py` — networks, optimizers, early stopping, history + `plot_history`
- `train_on_dataset.py` — CLI trainer (datasets/CSV) with all flags
- `main.py` — small Iris demo (can be extended to pass optimizer/early-stop args)
- `demo_training_*.png` — example plots generated here
```
Generated: Wed Oct 22 15:44:51 2025
