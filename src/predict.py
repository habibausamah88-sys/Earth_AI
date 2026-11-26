# ─────────────────────────────────────────────────────────────────────────────
# EarthLens AI — Inference pipeline on raw input
# Owner: Member 5 — Evaluation & Deployment Engineer
# ─────────────────────────────────────────────────────────────────────────────
"""Single entry point (AIE231 spec 3.7): takes a RAW image file, applies the
SAME preprocessing used in training (imported from src.data — no duplication),
and returns the prediction with its confidence, per-class probabilities,
inference time per sample and model size on disk.

Examples (repo root):
    python -m src.predict --model models/<run>.keras --image path/to/tile.jpg
    python -m src.predict --latest --image data/EuroSAT/River/River_10.jpg
    python -m src.predict --latest --dir some_folder_with_images
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from configs import config as cfg
from src import data as data_mod
from src.evaluate import load_model_entry


def model_size_mb(model_path: Path) -> float:
    return model_path.stat().st_size / 1e6


def predict_image(model, entry: dict, image_path: str | Path) -> dict:
    """Raw image file -> prediction dict (label, confidence, probabilities, timing).

    The exact same preprocessing branch as training:
      "raw"      -> feed 0-255 float (Rescaling layer lives INSIDE the model)
      "resnet50" -> src.data.preprocess_resnet (same function as the pipeline)
    """
    img = data_mod.decode_and_resize(str(image_path))          # float32, 0-255, 64x64
    x = img[tf.newaxis, ...]                                   # add batch dimension
    if entry["preprocessing"] == "resnet50":
        x = data_mod.preprocess_resnet(x)

    t0 = time.time()
    probs = model.predict(x, verbose=0)[0]
    ms = (time.time() - t0) * 1000.0

    names = entry["class_names"]
    top = int(np.argmax(probs))
    return {
        "image": str(image_path),
        "label": names[top],
        "confidence": round(float(probs[top]), 4),
        "probabilities": {names[i]: round(float(p), 4) for i, p in enumerate(probs)},
        "inference_ms": round(ms, 1),
        "model_size_mb": round(model_size_mb(cfg.MODELS_DIR / entry["model_file"]), 2),
        "model": entry["run_name"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Inference on raw images (Member 5)")
    ap.add_argument("--model", default=None, help="path to a saved .keras model")
    ap.add_argument("--latest", action="store_true", help="best model in manifest")
    ap.add_argument("--image", default=None, help="single image file")
    ap.add_argument("--dir", default=None, help="folder of images (up to 10 shown)")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = ap.parse_args()

    model, entry = load_model_entry(args.model, None)
    targets: list[Path] = []
    if args.image:
        targets = [Path(args.image)]
    elif args.dir:
        d = Path(args.dir)
        targets = sorted(p for p in d.iterdir()
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:10]
    if not targets:
        ap.error("give --image <file> or --dir <folder>")

    results = [predict_image(model, entry, t) for t in targets]
    for r in results:
        if args.json:
            print(json.dumps(r, indent=2))
        else:
            print(f"\n{r['image']}")
            print(f"  prediction : {r['label']}  (confidence {r['confidence']:.1%})")
            top3 = sorted(r["probabilities"].items(), key=lambda kv: -kv[1])[:3]
            print(f"  top-3      : " + ", ".join(f"{k} {v:.1%}" for k, v in top3))
            print(f"  inference  : {r['inference_ms']} ms  |  model size {r['model_size_mb']} MB")


if __name__ == "__main__":
    main()
