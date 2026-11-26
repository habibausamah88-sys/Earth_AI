# ─────────────────────────────────────────────────────────────────────────────
# EarthLens AI — Baselines (trivial + simple learned)
# Owner: Member 2 — Baseline & Analysis Engineer
# ─────────────────────────────────────────────────────────────────────────────
"""The two mandatory benchmarks from AIE231 spec 3.3, measured BEFORE any deep
network. Every later model is reported as an improvement over these numbers.

  1. trivial baseline  : majority-class prediction
  2. learned baseline  : single-hidden-layer MLP on flattened pixels

Both are evaluated on the SAME validation split used for model selection, and
results go to experiments/baseline_results.csv.

Run from the repo root:
    python -m src.baselines --data-dir data/EuroSAT [--max-per-class 250] [--epochs 30]
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from tensorflow import keras

from configs import config as cfg
from src import data as data_mod


# ─────────────────────────────────────────────────────────────────────────────
# 1) Trivial baseline: always predict the majority class of the TRAIN split
# ─────────────────────────────────────────────────────────────────────────────
def majority_class_baseline(y_train: np.ndarray, y_eval: np.ndarray,
                            class_names: list[str]) -> dict:
    majority = int(np.bincount(y_train).argmax())
    preds = np.full_like(y_eval, majority)
    acc = float((preds == y_eval).mean())
    recall = {}
    for idx, name in enumerate(class_names):
        mask = y_eval == idx
        recall[name] = float((preds[mask] == idx).mean()) if mask.any() else 0.0
    return {
        "baseline": "majority_class",
        "predicted_class": class_names[majority],
        "accuracy": round(acc, 4),
        "macro_recall": round(float(np.mean(list(recall.values()))), 4),
        "per_class_recall": recall,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2) Simple learned baseline: single-hidden-layer MLP (spec 3.3 option)
# ─────────────────────────────────────────────────────────────────────────────
def build_baseline_mlp(hidden_units: int = 256, dropout: float = 0.30) -> keras.Model:
    """Flattened pixels -> Dense(256, ReLU) -> Dropout -> softmax(10).

    A Rescaling layer INSIDE the model means predict.py feeds it raw 0-255
    pixels exactly like Model A (one preprocessing path, no duplication).
    """
    inputs = keras.Input(shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 3), name="input_0_255")
    x = keras.layers.Rescaling(1.0 / 255.0, name="rescale_0_1")(inputs)
    x = keras.layers.Flatten(name="flatten_64x64x3")(x)          # 12,288 features
    x = keras.layers.Dense(hidden_units, activation="relu", name="dense_hidden")(x)
    x = keras.layers.Dropout(dropout, name="drop_hidden")(x)
    outputs = keras.layers.Dense(cfg.NUM_CLASSES, activation="softmax", name="softmax_10")(x)
    return keras.Model(inputs, outputs, name="Baseline_MLP_1hidden")


def mlp_baseline(splits: dict, epochs: int = 30, batch_size: int = 64,
                 seed: int = cfg.SEED) -> dict:
    keras.utils.set_random_seed(seed)
    model = build_baseline_mlp(cfg.BASELINE_MLP["hidden_units"], cfg.BASELINE_MLP["dropout"])
    model.compile(optimizer=keras.optimizers.Adam(cfg.BASELINE_MLP["learning_rate"]),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    t0 = time.time()
    model.fit(splits["train_ds"], validation_data=splits["val_ds"],
              epochs=epochs, class_weight=splits["class_weights"], verbose=2)
    minutes = (time.time() - t0) / 60.0

    y_val = np.array(splits["y_val"])
    probs = model.predict(splits["val_ds"], verbose=0)
    preds = probs.argmax(axis=1)
    acc = float((preds == y_val).mean())
    macro_recall = float(np.mean([np.mean(preds[y_val == c] == c)
                                  if (y_val == c).any() else 0.0
                                  for c in range(cfg.NUM_CLASSES)]))
    return {
        "baseline": "mlp_1hidden_256",
        "accuracy": round(acc, 4),
        "macro_recall": round(macro_recall, 4),
        "train_minutes": round(minutes, 2),
        "params": int(model.count_params()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Runner — writes experiments/baseline_results.csv
# ─────────────────────────────────────────────────────────────────────────────
def run_all_baselines(data_dir: str, epochs: int = 30, batch_size: int = 64,
                      max_per_class: int | None = None, seed: int = cfg.SEED) -> dict:
    """Measure both baselines on identical splits and persist the comparison."""
    splits = data_mod.get_splits(data_dir, batch_size=batch_size,
                                 augment=False,           # baselines stay clean
                                 preprocessing="raw", max_per_class=max_per_class, seed=seed)
    print(f"[data] train/val/test = {len(splits['y_train'])}/{len(splits['y_val'])}"
          f"/{len(splits['y_test'])} images (leak-free)")

    maj = majority_class_baseline(splits["y_train"], splits["y_val"], splits["class_names"])
    print(f"[baseline] majority-class -> accuracy={maj['accuracy']} "
          f"(always predicts {maj['predicted_class']})")

    from src.train import run_training
    result = run_training('baseline_mlp', data_dir, epochs=epochs, batch_size=batch_size, seed=seed,
                          augment=False, max_per_class=max_per_class, author='Baseline engineer',
                          config_tag='smoke' if max_per_class else 'baseline')
    import json
    recall = json.loads((cfg.RUNS_DIR / f"{result['run_name']}_val_recall.json").read_text())
    mlp = dict(baseline='mlp_1hidden_256', accuracy=result['val_accuracy'], macro_recall=float(np.mean(list(recall.values()))),
               train_minutes=result['minutes'], params=result['params'])
    print(f"[baseline] MLP 1-hidden-layer -> accuracy={mlp['accuracy']} "
          f"macro_recall={mlp['macro_recall']} ({mlp['train_minutes']} min)")

    out = cfg.EXPERIMENTS_DIR
    out.mkdir(exist_ok=True)
    csv_path = out / "baseline_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["baseline", "val_accuracy", "val_macro_recall", "train_minutes",
                    "params", "notes"])
        w.writerow([maj["baseline"], maj["accuracy"], maj["macro_recall"], "",
                    "", f"always predicts {maj['predicted_class']}"])
        w.writerow([mlp["baseline"], mlp["accuracy"], mlp["macro_recall"],
                    mlp["train_minutes"], mlp["params"], "Dense256+Dropout0.3"])
    print(f"[baseline] comparison saved -> {csv_path}")
    return {"majority": maj, "mlp": mlp}


def main() -> None:
    ap = argparse.ArgumentParser(description="Baselines (Member 2)")
    ap.add_argument("--data-dir", default=str(cfg.DATA_DIR))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="cap images per class for quick runs")
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    args = ap.parse_args()
    run_all_baselines(args.data_dir, args.epochs, args.batch_size,
                      args.max_per_class, args.seed)


if __name__ == "__main__":
    main()
