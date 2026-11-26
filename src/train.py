# ─────────────────────────────────────────────────────────────────────────────
# EarthLens AI — Unified training entry point
# Owner: Member 2 — Baseline & Analysis Engineer (writes it for Model A,
#        then Member 3 trains B/C and Member 4 runs the studies through it)
# ─────────────────────────────────────────────────────────────────────────────
"""One command trains every model; one experiment log row is appended per run
(AIE231 spec 3.5 "log every run" + spec 6.1 "training entry point").

TEST DISCIPLINE (spec: any sign the test set was used for tuning costs marks):
this script NEVER touches the test split — model selection uses validation only.
Test evaluation happens once, at the very end, via src/evaluate.py.

Examples (repo root):
    python -m src.train --model model_a --author "Member2"
    python -m src.train --model model_b --author "Mariam"
    python -m src.train --model model_c --author "Mariam" --lr 1e-4
    python -m src.train --model model_a --smoke            # 2-min sanity run
    python -m src.train --model model_a --no-augment       # ablation arm
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs import config as cfg
from src import data as data_mod
from src.baselines import build_baseline_mlp
from src.models import MODEL_BUILDERS, build_model_a, compile_model, fine_tune_from_b, hand_parameter_report


# ─────────────────────────────────────────────────────────────────────────────
# Model registry: builder + preprocessing mode per model key
# ─────────────────────────────────────────────────────────────────────────────
def build_model(key: str, pretrained: bool = True, **overrides):
    if key == "baseline_mlp":
        return build_baseline_mlp(cfg.BASELINE_MLP["hidden_units"],
                                  cfg.BASELINE_MLP["dropout"])
    if key == "model_a":
        return build_model_a(**overrides)
    if key not in MODEL_BUILDERS:
        raise ValueError(f"unknown model {key!r}")
    return MODEL_BUILDERS[key](pretrained=pretrained)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment log (experiments/experiment_log.csv) — shared by all members
# ─────────────────────────────────────────────────────────────────────────────
LOG_COLUMNS = [
    "run_id", "date", "author", "model", "config_tag", "lr", "batch_size",
    "optimizer", "epochs", "augment", "seed", "params", "train_time_min",
    "val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1", "notes",
    "train_accuracy", "train_loss", "val_loss", "dropout", "dense_units", "batch_norm", "class_weights", "dataset_signature", "max_per_class", "parent_run", "pretrained", "epochs_completed",
]


def append_experiment_row(row: dict, log_path: Path | None = None) -> None:
    log_path = Path(log_path or cfg.EXPERIMENTS_DIR / "experiment_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_path.exists() or log_path.stat().st_size == 0
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in LOG_COLUMNS})


def read_experiment_log(log_path: Path | None = None):
    log_path = Path(log_path or cfg.EXPERIMENTS_DIR / "experiment_log.csv")
    if not log_path.exists():
        return []
    with open(log_path, newline="") as f:
        return list(csv.DictReader(f))


# ─────────────────────────────────────────────────────────────────────────────
# Learning curves (saved next to the model so every run has its evidence)
# ─────────────────────────────────────────────────────────────────────────────
def save_curves(history, run_name: str) -> Path:
    out = cfg.RUNS_DIR / f"{run_name}_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("epoch"); axes[0].grid(alpha=0.3)
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set_title("Loss"); axes[1].set_xlabel("epoch"); axes[1].grid(alpha=0.3)
    for ax in axes:
        ax.legend()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def save_manifest_entry(run_name: str, model_key: str, file_name: str,
                        val_accuracy: float, **metadata) -> None:
    """models/manifest.json maps each saved model -> preprocessing + class names,
    so predict.py / evaluate.py / the demo can load it without any hard-coding."""
    cfg.MODELS_DIR.mkdir(exist_ok=True)
    manifest_path = cfg.MODELS_DIR / "manifest.json"
    entries = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    entries = [e for e in entries if e["run_name"] != run_name]  # overwrite same run
    entries.append({
        "run_name": run_name,
        "model_file": file_name,
        "model_key": model_key,
        "preprocessing": cfg.PREPROCESSING[model_key],
        "class_names": cfg.CLASS_NAMES,
        "val_accuracy": val_accuracy,
        **metadata,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
    })
    manifest_path.write_text(json.dumps(entries, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# The one training function everyone uses
# ─────────────────────────────────────────────────────────────────────────────
def macro_f1(y_true, probs) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, probs.argmax(axis=1), average="macro"))


def run_training(
    model_key: str,
    data_dir: str = str(cfg.DATA_DIR),
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    optimizer: str = "adam",
    augment: bool | None = None,
    seed: int = cfg.SEED,
    run_name: str | None = None,
    author: str = "unknown",
    max_per_class: int | None = None,
    pretrained: bool = True,
    config_tag: str = "reference",
    notes: str = "",
    dropout: float | None = None,
    dense_units: int | None = None,
    batch_norm: bool = True,
    class_weights: bool = True,
    parent_run: str | None = None,
) -> dict:
    """Full training run: data -> model -> callbacks -> fit -> validation metrics
    -> saved model + manifest + curves + experiment-log row."""
    import getpass
    import numpy as np
    import tensorflow as tf

    epochs = epochs if epochs is not None else cfg.TRAINING["epochs"]
    batch_size = batch_size or cfg.TRAINING["batch_size"]
    augment = cfg.TRAINING["augment"] if augment is None else augment
    if lr is None:
        default_lr = {"model_a": cfg.MODEL_A["learning_rate"],
                      "model_b": cfg.MODEL_B["learning_rate"],
                      "model_c": cfg.MODEL_C["learning_rate"],
                      "baseline_mlp": cfg.BASELINE_MLP["learning_rate"]}
        lr = default_lr[model_key]

    if max_per_class is not None and max_per_class < 20:
        raise ValueError("Use at least 20 images per class for a stratified smoke split.")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_name = run_name or f"{stamp}_{model_key}_{config_tag}"
    cfg.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[train] run={run_name}  model={model_key}  lr={lr}  batch={batch_size}  "
          f"optimizer={optimizer}  augment={augment}  seed={seed}")

    # 1) reproducibility
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    # 2) leak-free splits + pipelines (Member 1's module)
    preprocessing = cfg.PREPROCESSING[model_key]
    split = data_mod.get_splits(data_dir, batch_size=batch_size, augment=augment,
                                preprocessing=preprocessing, max_per_class=max_per_class,
                                seed=seed)
    print(f"[data] train={len(split['y_train'])} val={len(split['y_val'])} "
          f"test={len(split['y_test'])} (test NOT used here)")

    # 3) model
    if model_key == 'model_a':
        model = build_model_a(dropout=cfg.MODEL_A['dropout'] if dropout is None else dropout,
                              dense_units=cfg.MODEL_A['dense_units'] if dense_units is None else dense_units,
                              batch_norm=batch_norm)
    elif model_key == 'model_c':
        entries = json.loads((cfg.MODELS_DIR / 'manifest.json').read_text()) if (cfg.MODELS_DIR / 'manifest.json').exists() else []
        candidates = [e for e in entries if e['model_key'] == 'model_b' and e.get('dataset_signature') == split['dataset_signature'] and e.get('eligible') == (max_per_class is None and config_tag != 'smoke' and pretrained) and (cfg.MODELS_DIR / e['model_file']).exists()]
        if parent_run:
            candidates = [e for e in candidates if e['run_name'] == parent_run]
        if not candidates:
            raise ValueError('Train Model B on the same dataset/smoke mode before Model C; optionally pass --parent-run.')
        parent = max(candidates, key=lambda e: e['val_macro_f1'])
        parent_run = parent['run_name']
        model = fine_tune_from_b(cfg.MODELS_DIR / parent['model_file'])
    else:
        model = build_model(model_key, pretrained=pretrained)
    compile_model(model, lr, optimizer)
    n_params = int(model.count_params())
    print(f"[model] {model.name}  params={n_params:,}")

    # 4) callbacks: early stopping on validation loss (mandatory, spec 3.4)
    class TimeBudget(tf.keras.callbacks.Callback):
        """Stop at a batch boundary before a run consumes excessive free-GPU time."""
        def on_train_begin(self, logs=None):
            self.started = time.monotonic()
        def on_train_batch_end(self, batch, logs=None):
            if time.monotonic() - self.started > 27 * 60:
                self.model.stop_training = True
    callbacks = [
        TimeBudget(),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.TRAINING["patience"],
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.CSVLogger(cfg.RUNS_DIR / f"{run_name}_history.csv"),
    ]

    # 5) fit with class weights (imbalance handling, spec 3.2)
    t0 = time.time()
    history = model.fit(
        split["train_ds"], validation_data=split["val_ds"], epochs=epochs,
        class_weight=split["class_weights"] if class_weights else None, callbacks=callbacks, verbose=2,
    )
    minutes = (time.time() - t0) / 60.0

    # 6) validation metrics (model selection happens ONLY here)
    y_val = np.array(split["y_val"])
    val_probs = model.predict(split["val_ds"], verbose=0)
    val_acc = float((val_probs.argmax(axis=1) == y_val).mean())
    val_f1 = macro_f1(y_val, val_probs)
    print(f"[result] val_accuracy={val_acc:.4f}  val_macro_f1={val_f1:.4f}  ({minutes:.1f} min)")

    # 7) artifacts: model, manifest, curves, log row
    cfg.MODELS_DIR.mkdir(exist_ok=True)
    file_name = f"{run_name}.keras"
    model.save(cfg.MODELS_DIR / file_name)
    eligible = max_per_class is None and config_tag != 'smoke' and (pretrained or model_key in ('model_a', 'baseline_mlp'))
    metadata = dict(val_macro_f1=val_f1, eligible=eligible, dataset_signature=split['dataset_signature'],
                    max_per_class=max_per_class, seed=seed, parent_run=parent_run, config_tag=config_tag,
                    lr=lr, batch_size=batch_size, optimizer=optimizer, augment=augment, epochs=epochs,
                    dropout=dropout, dense_units=dense_units, batch_norm=batch_norm, class_weights=class_weights, pretrained=pretrained)
    save_manifest_entry(run_name, model_key, file_name, val_acc, **metadata)
    from sklearn.metrics import recall_score
    recall = recall_score(y_val, val_probs.argmax(axis=1), labels=list(range(cfg.NUM_CLASSES)), average=None, zero_division=0)
    (cfg.RUNS_DIR / f'{run_name}_val_recall.json').write_text(json.dumps(dict(zip(cfg.CLASS_NAMES, recall.tolist())), indent=2))
    with open(cfg.RUNS_DIR / f'{run_name}_parameters.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['layer','manual_params','keras_params','output_shape']); w.writerows(hand_parameter_report(model))
    with open(cfg.RUNS_DIR / f'{run_name}_summary.txt', 'w') as f:
        model.summary(print_fn=lambda line: f.write(line + '\n'), expand_nested=True)
    try:
        tf.keras.utils.plot_model(model, to_file=str(cfg.RUNS_DIR / f'{run_name}_architecture.png'), show_shapes=True, show_layer_names=True, expand_nested=False)
    except Exception as exc:
        print(f'[diagram] optional Graphviz diagram unavailable: {exc}')
    curves = save_curves(history, run_name)
    print(f"[saved] {cfg.MODELS_DIR / file_name}\n[saved] {curves}")

    append_experiment_row({
        "run_id": run_name, "date": stamp, "author": author or getpass.getuser(),
        "model": model_key, "config_tag": config_tag, "lr": lr,
        "batch_size": batch_size, "optimizer": optimizer, "epochs": epochs,
        "augment": int(augment), "seed": seed, "params": n_params,
        "train_time_min": round(minutes, 2),
        "val_accuracy": round(val_acc, 4), "val_macro_f1": round(val_f1, 4),
        "test_accuracy": "", "test_macro_f1": "", "notes": notes,
        "train_accuracy": history.history['accuracy'][-1], "train_loss": history.history['loss'][-1],
        "val_loss": float(model.evaluate(split['val_ds'], verbose=0)[0]),
        "dropout": dropout, "dense_units": dense_units, "batch_norm": batch_norm, "class_weights": class_weights,
        "dataset_signature": split['dataset_signature'], "max_per_class": max_per_class, "parent_run": parent_run,
        "pretrained": pretrained, "epochs_completed": len(history.history['loss']),
    })
    print(f"[log] row appended -> {cfg.EXPERIMENTS_DIR / 'experiment_log.csv'}")

    return {"run_name": run_name, "val_accuracy": val_acc, "val_macro_f1": val_f1,
            "params": n_params, "minutes": minutes, "model_file": file_name}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Train one model (AIE231 project)")
    ap.add_argument("--model", required=True,
                    choices=["baseline_mlp", "model_a", "model_b", "model_c"])
    ap.add_argument("--data-dir", default=str(cfg.DATA_DIR))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--optimizer", default="adam",
                    choices=["adam", "sgd_momentum", "sgd"])
    ap.add_argument("--no-augment", action="store_true", help="ablation arm")
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    ap.add_argument("--author", default=None, help="your GitHub handle")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--config-tag", default="reference")
    ap.add_argument("--notes", default="")
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="cap per class (quick experiments)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to check the pipeline end-to-end")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="skip ImageNet weights download (offline smoke test)")
    ap.add_argument('--dropout', type=float)
    ap.add_argument('--dense-units', type=int)
    ap.add_argument('--no-batch-norm', action='store_true')
    ap.add_argument('--no-class-weights', action='store_true')
    ap.add_argument('--parent-run')
    args = ap.parse_args()

    kwargs = {}
    if args.smoke:
        kwargs.update(epochs=2, max_per_class=60, config_tag="smoke",
                      batch_size=32)
        print("[smoke] pipeline sanity check — results are meaningless")
    run_training(args.model, args.data_dir, epochs=kwargs.pop("epochs", args.epochs),
                 batch_size=kwargs.pop("batch_size", args.batch_size),
                 lr=args.lr, optimizer=args.optimizer,
                 augment=(not args.no_augment), seed=args.seed, run_name=args.run_name,
                 author=args.author, max_per_class=kwargs.pop("max_per_class", args.max_per_class),
                 pretrained=(not args.no_pretrained),
                 config_tag=kwargs.pop("config_tag", args.config_tag),
                 notes=args.notes, dropout=args.dropout, dense_units=args.dense_units, batch_norm=not args.no_batch_norm, class_weights=not args.no_class_weights, parent_run=args.parent_run)


if __name__ == "__main__":
    main()
