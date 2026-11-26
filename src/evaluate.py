# ─────────────────────────────────────────────────────────────────────────────
# EarthLens AI — Final evaluation & error analysis
# Owner: Member 4 — Experimentation Engineer
# ─────────────────────────────────────────────────────────────────────────────
"""The ONE place where the held-out test set is touched (AIE231 spec 3.6:
test set is used once, at the end). Produces the full metric set:

  accuracy, confusion matrix, per-class precision/recall/F1, macro-F1,
  ROC-AUC (one-vs-rest) and the failure-case analysis (>= 20 worst errors).

Outputs (all under experiments/final_evaluation/):
  metrics.json, confusion_matrix.png, roc_curves.png, per_class_report.csv,
  worst_errors.csv, worst_errors.png

Examples (repo root):
    python -m src.evaluate --model models/<run>.keras
    python -m src.evaluate --latest                     # best entry in manifest
    python -m src.evaluate --latest --smoke             # pipeline check
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, roc_auc_score, roc_curve)

from configs import config as cfg
from src import data as data_mod

OUT_DIR = cfg.EXPERIMENTS_DIR / "final_evaluation"


# ─────────────────────────────────────────────────────────────────────────────
# Model loading through the manifest (no hard-coded paths)
# ─────────────────────────────────────────────────────────────────────────────
def load_model_entry(model_path: str | None = None, run_name: str | None = None):
    manifest_path = cfg.MODELS_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("models/manifest.json not found — train a model first "
                                "(python -m src.train --model model_a)")
    entries = json.loads(manifest_path.read_text())
    entries = [e for e in entries if (cfg.MODELS_DIR / e['model_file']).exists()]
    if run_name or model_path:
        matches = [e for e in entries if e['run_name'] == run_name] if run_name else [e for e in entries if (cfg.MODELS_DIR / e['model_file']).resolve() == Path(model_path).resolve()]
        if not matches:
            raise ValueError('Requested model is not registered in manifest.')
        entry = matches[0]
    else:
        eligible = [e for e in entries if e.get('eligible') and e['model_key'] != 'baseline_mlp']
        if not eligible:
            raise ValueError('No full neural model. For smoke inference explicitly pass --model models/<run>.keras.')
        final = cfg.EXPERIMENTS_DIR / 'final_seeds.json'
        if final.exists():
            name = json.loads(final.read_text())['deployment_run']
            entry = next(e for e in eligible if e['run_name'] == name)
        else:
            entry = max(eligible, key=lambda e: e['val_macro_f1'])
    model = tf.keras.models.load_model(cfg.MODELS_DIR / entry['model_file'], compile=False)
    return model, entry


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, test_ds, y_test: np.ndarray, class_names: list[str]) -> dict:
    probs = model.predict(test_ds, verbose=0)
    return metrics_from_probs(y_test, probs, class_names)


def metrics_from_probs(y_test, probs, class_names):
    preds = probs.argmax(axis=1)
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    cm = confusion_matrix(y_test, preds, labels=list(range(len(class_names))))
    try:
        roc_auc = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = float("nan")
    return {"probs": probs, "preds": preds, "accuracy": float(acc),
            "macro_f1": float(macro_f1), "confusion_matrix": cm,
            "roc_auc_ovr_macro": float(roc_auc)}


def save_metrics(evals: dict, class_names: list[str], model_name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {"model": model_name, "test_accuracy": evals["accuracy"],
               "test_macro_f1": evals["macro_f1"],
               "test_roc_auc_ovr_macro": evals["roc_auc_ovr_macro"]}
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[metrics] accuracy={metrics['test_accuracy']:.4f}  "
          f"macro_f1={metrics['test_macro_f1']:.4f}  "
          f"roc_auc(ovr)={metrics['test_roc_auc_ovr_macro']:.4f}")


def save_per_class_report(y_test, preds, class_names) -> None:
    report = classification_report(y_test, preds, target_names=class_names,
                                   digits=4, output_dict=True, labels=list(range(len(class_names))), zero_division=0)
    with open(OUT_DIR / "per_class_report.csv", "w") as f:
        f.write("class,precision,recall,f1_score,support\n")
        for name in class_names:
            r = report[name]
            f.write(f"{name},{r['precision']:.4f},{r['recall']:.4f},"
                    f"{r['f1-score']:.4f},{int(r['support'])}\n")
        f.write(f"macro avg,{report['macro avg']['precision']:.4f},"
                f"{report['macro avg']['recall']:.4f},"
                f"{report['macro avg']['f1-score']:.4f},{int(report['macro avg']['support'])}\n")
    print(f"[saved] {OUT_DIR / 'per_class_report.csv'}")


def save_confusion_figure(cm, class_names) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 7.2), constrained_layout=True)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix — held-out test set")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)
    print(f"[saved] {OUT_DIR / 'confusion_matrix.png'}")


def save_roc_curves(y_test, probs, class_names) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.4), constrained_layout=True)
    y_onehot = np.eye(len(class_names))[y_test]
    for idx, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_onehot[:, idx], probs[:, idx])
        ax.plot(fpr, tpr, lw=1.6, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC — one-vs-rest (test set)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.savefig(OUT_DIR / "roc_curves.png", dpi=200)
    plt.close(fig)
    print(f"[saved] {OUT_DIR / 'roc_curves.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Error analysis: >= 20 worst misclassified samples, grouped by failure type
# ─────────────────────────────────────────────────────────────────────────────
def error_analysis(test_paths, y_test, probs, class_names, top_n: int = 20) -> None:
    preds = probs.argmax(axis=1)
    wrong = np.where(preds != y_test)[0]
    # "worst" = model was least confident about the TRUE class
    true_prob = probs[np.arange(len(y_test)), y_test]
    order = wrong[np.argsort(true_prob[wrong])][:top_n]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "worst_errors.csv", "w") as f:
        f.write("path,true_class,pred_class,true_prob,pred_prob\n")
        for i in order:
            f.write(f"{test_paths[i]},{class_names[y_test[i]]},"
                    f"{class_names[preds[i]]},{true_prob[i]:.4f},{probs[i, preds[i]]:.4f}\n")
    print(f"[errors] {len(wrong)} misclassified of {len(y_test)} "
          f"({100 * len(wrong) / len(y_test):.1f}%); worst {len(order)} -> "
          f"{OUT_DIR / 'worst_errors.csv'}")

    # failure categories: the confusion pairs that dominate
    pairs: dict[tuple[str, str], int] = {}
    for i in wrong:
        key = (class_names[y_test[i]], class_names[preds[i]])
        pairs[key] = pairs.get(key, 0) + 1
    top_pairs = sorted(pairs.items(), key=lambda kv: -kv[1])[:5]
    print("[errors] most common confusions (true -> pred):")
    for (t, p), n in top_pairs:
        print(f"    {t:<22} -> {p:<22} x{n}")
    (OUT_DIR / "confusion_pairs.json").write_text(json.dumps([{"true": t, "predicted": p, "count": n} for (t,p), n in sorted(pairs.items(), key=lambda kv: -kv[1])], indent=2))
    if top_pairs:
        t, p = top_pairs[0][0], top_pairs[0][1]
        print(f"[errors] proposed fix for '{t}' vs '{p}': add targeted augmentation "
              f"(rotation/contrast) for these two classes, raise their class weight, "
              f"and inspect training tiles for label noise. Retrain and re-check "
              f"per-class recall before touching the test set again.")

    # visual grid of the worst errors
    n = len(order)
    if n == 0:
        return
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.5 * rows),
                             constrained_layout=True)
    axes = np.asarray(axes).reshape(rows, cols)
    for k, i in enumerate(order):
        img = tf.io.decode_image(tf.io.read_file(test_paths[i]), channels=3, expand_animations=False)
        ax = axes[k // cols, k % cols]
        ax.imshow(img.numpy())
        ax.set_title(f"true: {class_names[y_test[i]]}\npred: {class_names[preds[i]]} "
                     f"({probs[i, preds[i]]:.2f})", fontsize=7)
        ax.axis("off")
    for k in range(n, rows * cols):
        axes[k // cols, k % cols].axis("off")
    fig.savefig(OUT_DIR / "worst_errors.png", dpi=160)
    plt.close(fig)
    print(f"[saved] {OUT_DIR / 'worst_errors.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global OUT_DIR
    ap = argparse.ArgumentParser(description='Final test evaluation; run only after validation selection is frozen')
    ap.add_argument('--model')
    ap.add_argument('--latest', action='store_true')
    ap.add_argument('--data-dir', default=str(cfg.DATA_DIR))
    ap.add_argument('--errors', type=int, default=20)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--all-models', action='store_true', help='also evaluate best validation configuration of A/B/C')
    args = ap.parse_args()
    model, entry = load_model_entry(args.model, None)
    if not args.smoke and not entry.get('eligible'):
        ap.error('A smoke/capped model is not eligible for final test evaluation.')
    if args.smoke and entry.get('eligible'):
        ap.error('Smoke evaluation requires an explicitly selected smoke model.')
    paths, labels, names = data_mod.scan_dataset(args.data_dir, entry.get('max_per_class'))
    signature = data_mod.dataset_signature(paths)
    if signature != entry.get('dataset_signature'):
        raise ValueError('Dataset differs from training dataset; refusing mismatched evaluation.')
    splits = data_mod.stratified_split(paths, labels, cfg.SEED)
    data_mod.leakage_check(splits)
    test_paths, y_test = splits['test']
    root = cfg.EXPERIMENTS_DIR / ('smoke_evaluation' if args.smoke else 'final_evaluation')
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'comparison.json').exists() and not args.smoke:
        raise RuntimeError('Final evaluation already recorded. Do not use test results to tune; keep the frozen report.')
    entries = json.loads((cfg.MODELS_DIR / 'manifest.json').read_text())
    entries = [e for e in entries if e.get('dataset_signature') == signature and e.get('eligible') == entry.get('eligible') and (cfg.MODELS_DIR / e['model_file']).exists()]
    chosen = [('final', entry)]
    if not args.smoke:
        baseline = [e for e in entries if e['model_key'] == 'baseline_mlp']
        if not baseline:
            raise ValueError('Train the full learned baseline before final evaluation.')
        chosen.append(('baseline_mlp', max(baseline, key=lambda e: e['val_macro_f1'])))
        if args.all_models:
            for key in ('model_a','model_b','model_c'):
                candidates = [e for e in entries if e['model_key'] == key and e.get('config_tag') not in ('final_seed','seed_parent')]
                if candidates:
                    chosen.append((key, max(candidates, key=lambda e: e['val_macro_f1'])))
    results = []
    def report(label, run, metrics):
        global OUT_DIR
        OUT_DIR = root / label
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        save_metrics(metrics, names, run)
        save_per_class_report(y_test, metrics['preds'], names)
        save_confusion_figure(metrics['confusion_matrix'], names)
        save_roc_curves(y_test, metrics['probs'], names)
        error_analysis(test_paths, y_test, metrics['probs'], names, top_n=max(20, args.errors))
        results.append(dict(label=label, run=run, accuracy=metrics['accuracy'], macro_f1=metrics['macro_f1'], roc_auc=metrics['roc_auc_ovr_macro']))
    if not args.smoke:
        majority = int(np.bincount(splits['train'][1]).argmax())
        probs = np.tile(np.eye(len(names))[majority], (len(y_test),1))
        report('majority', 'majority_class', metrics_from_probs(y_test, probs, names))
    cache = {}
    for label, e in chosen:
        if e['run_name'] not in cache:
            current = model if e['run_name'] == entry['run_name'] else tf.keras.models.load_model(cfg.MODELS_DIR / e['model_file'], compile=False)
            ds = data_mod.make_dataset(test_paths, y_test, training=False, batch_size=64, preprocessing=e['preprocessing'])
            cache[e['run_name']] = evaluate_model(current, ds, y_test, names)
            del current
        report(label, e['run_name'], cache[e['run_name']])
    (root / 'comparison.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
