# ─────────────────────────────────────────────────────────────────────────────
# EarthLens AI — Data module (loading, audit, leak-free splitting, pipelines)
# Owner: Member 1 — Data Engineer
# ─────────────────────────────────────────────────────────────────────────────
"""Everything that touches raw images lives here so that training and
inference share ONE preprocessing implementation (AIE231 spec 6.1 "no
duplicated preprocessing between training and inference").

Milestone 1+2 deliverables owned by Member 1:
  * audit / leak-free stratified train-val-test split (70 / 15 / 15)
  * class-imbalance measurement + class weights
  * reusable tf.data pipelines with training-only augmentation

Run the audit from the repo root:
    python -m src.data --audit --data-dir data/EuroSAT
"""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from configs import config as cfg

AUTOTUNE = tf.data.AUTOTUNE


# ─────────────────────────────────────────────────────────────────────────────
# Scanning and auditing (Milestone 1)
# ─────────────────────────────────────────────────────────────────────────────
def class_names(data_dir: str | Path) -> list[str]:
    """Return the class folder names sorted alphabetically.

    Label index = alphabetical position. Sorted order is deterministic, so the
    same folder always yields the same label on every machine and at inference.
    """
    d = Path(data_dir)
    names = sorted(p.name for p in d.iterdir() if p.is_dir())
    if names != cfg.CLASS_NAMES:
        raise ValueError(
            f"Expected {cfg.NUM_CLASSES} class folders in {d}, found {len(names)}: {names}. "
            "Run `python data/download_data.py` first."
        )
    return names


def scan_dataset(
    data_dir: str | Path, max_per_class: int | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Scan class folders -> (file paths, integer labels, class names)."""
    names = class_names(data_dir)
    records = _clean_records(str(Path(data_dir).resolve()))
    paths, labels = [], []
    for idx, cname in enumerate(names):
        files = [path for path, label, _ in records if label == idx]
        if max_per_class is not None:
            files = files[:max_per_class]
        paths.extend(files)
        labels.extend([idx] * len(files))
    return np.array(paths), np.array(labels), names


from functools import lru_cache
from PIL import Image

@lru_cache(maxsize=2)
def _inventory(data_dir: str):
    """Hash decoded RGB pixels, reject unreadable files and record label conflicts."""
    groups, corrupt = {}, []
    for idx, cname in enumerate(class_names(data_dir)):
        for f in sorted((Path(data_dir) / cname).iterdir()):
            if not f.is_file() or f.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            try:
                with Image.open(f) as im:
                    rgb = im.convert('RGB')
                    digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            except (OSError, ValueError):
                corrupt.append(str(f)); continue
            groups.setdefault(digest, []).append((str(f), idx, digest))
    return groups, corrupt

@lru_cache(maxsize=2)
def _clean_records(data_dir: str):
    groups, _ = _inventory(data_dir)
    # Keep one copy when labels agree; discard all copies if labels conflict.
    return sorted(v[0] for v in groups.values() if len({r[1] for r in v}) == 1)


def dataset_signature(paths):
    """Fingerprint clean content and labels, independent of machine paths."""
    if len(paths) == 0:
        raise ValueError('Empty dataset')
    root = str(Path(paths[0]).resolve().parents[1])
    lookup = {p: (label, digest) for p, label, digest in _clean_records(root)}
    values = [(str(Path(p).relative_to(Path(p).parents[1])), *lookup[str(Path(p).resolve())]) for p in paths]
    return hashlib.sha256(repr(values).encode()).hexdigest()


def file_md5(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(chunk), b""):
            h.update(part)
    return h.hexdigest()


def find_duplicates(data_dir: str | Path, limit_per_class: int | None = None):
    groups, _ = _inventory(str(Path(data_dir).resolve()))
    return [[r[0] for r in v] for v in groups.values() if len(v) > 1]


def audit_dataset(data_dir: str | Path, max_per_class: int | None = None) -> dict:
    """Full audit used in the EDA notebook and by ``--audit``: counts, imbalance,
    duplicate check and image-size check on a sample."""
    paths, labels, names = scan_dataset(data_dir, max_per_class)
    counts = Counter(labels.tolist())
    dist = {names[i]: counts.get(i, 0) for i in range(len(names))}
    total = int(labels.size)
    ratio = (max(dist.values()) / min(dist.values())) if total and min(dist.values()) else 0.0

    # image-size sanity check on a small sample
    sizes: Counter = Counter()
    for p in paths[:: max(1, total // 200)]:
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        sizes[tuple(int(d) for d in img.shape)] += 1

    return {
        "total_images": total,
        "class_distribution": dist,
        "imbalance_ratio_max_over_min": round(float(ratio), 2),
        "sampled_image_shapes": {str(k): v for k, v in sizes.items()},
        "n_duplicates_in_sample": len(find_duplicates(data_dir)), # legacy notebook key; now full scan
        "duplicate_groups_full_scan": len(find_duplicates(data_dir)),
        "corrupt_files": len(_inventory(str(Path(data_dir).resolve()))[1]),
        "conflicting_label_groups": sum(len({r[1] for r in v}) > 1 for v in _inventory(str(Path(data_dir).resolve()))[0].values()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Splitting and leakage guards (Milestones 1+2)
# ─────────────────────────────────────────────────────────────────────────────
def stratified_split(
    paths: np.ndarray, labels: np.ndarray, seed: int = cfg.SEED
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Stratified 70 / 15 / 15 split (AIE231 spec 3.2), no leakage.

    RGB release has no scene-group metadata available here. Exact duplicates are
    removed, but geographic overlap and near-duplicates remain limitations.
    """
    test_frac = cfg.TEST_SPLIT
    val_frac_of_rest = cfg.VAL_SPLIT / (cfg.TRAIN_SPLIT + cfg.VAL_SPLIT)

    p_trv, p_te, y_trv, y_te = train_test_split(
        paths, labels, test_size=test_frac, stratify=labels, random_state=seed
    )
    p_tr, p_va, y_tr, y_va = train_test_split(
        p_trv, y_trv, test_size=val_frac_of_rest, stratify=y_trv, random_state=seed
    )
    return {
        "train": (p_tr, y_tr),
        "val": (p_va, y_va),
        "test": (p_te, y_te),
    }


def leakage_check(splits: dict[str, tuple[np.ndarray, np.ndarray]]) -> bool:
    """Assert the three splits share zero file paths (hard leakage guard)."""
    all_paths = [p for v in splits.values() for p in v[0]]
    root = str(Path(all_paths[0]).resolve().parents[1])
    groups, _ = _inventory(root)
    hashes = {p: digest for digest, rows in groups.items() for p, _, _ in rows}
    sets = {k: {hashes[str(Path(p).resolve())] for p in v[0]} for k, v in splits.items()}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    ok = all(sets[a].isdisjoint(sets[b]) for a, b in pairs)
    if not ok:
        raise RuntimeError("LEAKAGE DETECTED: a file appears in more than one split!")
    return ok


def class_distribution(y: np.ndarray, n_classes: int = cfg.NUM_CLASSES) -> dict[int, int]:
    return dict(sorted(Counter(y.tolist()).items()))


def compute_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """Balanced class weights (AIE231 spec 3.2 'class imbalance')."""
    classes = np.arange(cfg.NUM_CLASSES)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing + augmentation (Milestones 1+2 / 3)
# ─────────────────────────────────────────────────────────────────────────────
def build_augmenter(enabled: bool = True, seed: int = cfg.SEED) -> keras.Sequential | None:
    """Training-only augmentation (4 ops >= the 2 required by spec 3.2).

    NEVER used on validation/test: it is applied only to the training dataset
    inside :func:`make_dataset` when ``training=True``.
    """
    if not enabled:
        return None
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
            keras.layers.RandomRotation(0.15, seed=seed + 1),
            keras.layers.RandomZoom(0.15, seed=seed + 2),
            keras.layers.RandomContrast(0.20, seed=seed + 3),
        ],
        name="augmentation",
    )


def preprocess_resnet(images: tf.Tensor) -> tf.Tensor:
    """Single source of truth for ResNet50 input preprocessing (Models B and C).

    Used by the tf.data pipeline during training AND by src/predict.py at
    inference so no preprocessing is ever duplicated.
    """
    return tf.keras.applications.resnet50.preprocess_input(images)


def decode_and_resize(path: tf.Tensor, img_size: int = cfg.IMG_SIZE) -> tf.Tensor:
    """Read a JPEG/TIFF tile -> float32 RGB image scaled 0-255, resized."""
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, (img_size, img_size))
    return tf.cast(img, tf.float32)


def make_dataset(
    paths: np.ndarray,
    labels: np.ndarray,
    training: bool,
    batch_size: int = 64,
    img_size: int = cfg.IMG_SIZE,
    augment: bool = False,
    preprocessing: str = "raw",
    seed: int = cfg.SEED,
) -> tf.data.Dataset:
    """Reusable tf.data pipeline.

    Order matters: decode -> cache -> (shuffle -> augment) -> preprocess -> batch
    -> prefetch. Augmentation runs only when ``training`` AND ``augment`` are set.
    ``preprocessing`` is "raw" (Model A / MLP keep a Rescaling layer inside the
    model) or "resnet50" (pipeline applies preprocess_input, see above).
    """
    ds = tf.data.Dataset.from_tensor_slices((paths, tf.one_hot(labels, cfg.NUM_CLASSES)))
    ds = ds.map(lambda p, y: (decode_and_resize(p, img_size), y), num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.cache()
        ds = ds.shuffle(min(len(paths), 4096), seed=seed, reshuffle_each_iteration=True)
        aug = build_augmenter(augment, seed)
        if aug is not None:
            ds = ds.map(
                lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE
            )
    else:
        ds = ds.cache()

    if preprocessing == "resnet50":
        ds = ds.map(lambda x, y: (preprocess_resnet(x), y), num_parallel_calls=AUTOTUNE)
    elif preprocessing != "raw":
        raise ValueError(f"unknown preprocessing mode: {preprocessing}")

    ds = ds.batch(batch_size)
    return ds.prefetch(AUTOTUNE)


def get_splits(
    data_dir: str | Path,
    batch_size: int = 64,
    augment: bool = True,
    preprocessing: str = "raw",
    max_per_class: int | None = None,
    seed: int = cfg.SEED,
) -> dict:
    """One-call helper: scan -> split -> leakage check -> tf.data datasets.

    Returns dict with ``train_ds``/``val_ds``/``test_ds``, raw label arrays
    (for baselines and metrics), class names and class weights.
    """
    paths, labels, names = scan_dataset(data_dir, max_per_class)
    splits = stratified_split(paths, labels, cfg.SEED)
    leakage_check(splits)
    y = {k: v[1] for k, v in splits.items()}

    ds = {
        k: make_dataset(
            splits[k][0], y[k], training=(k == "train"),
            batch_size=batch_size, augment=(k == "train" and augment),
            preprocessing=preprocessing, seed=seed,
        )
        for k in ("train", "val", "test")
    }
    return {
        "train_ds": ds["train"], "val_ds": ds["val"], "test_ds": ds["test"],
        "train_paths": splits["train"][0], "val_paths": splits["val"][0],
        "test_paths": splits["test"][0],
        "y_train": y["train"], "y_val": y["val"], "y_test": y["test"],
        "class_names": names,
        "dataset_signature": dataset_signature(paths),
        "class_weights": compute_class_weights(y["train"]),
        "train_counts": class_distribution(y["train"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI audit
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Audit EuroSAT splits (Member 1)")
    ap.add_argument("--data-dir", default=str(cfg.DATA_DIR))
    ap.add_argument("--audit", action="store_true", help="full audit report")
    ap.add_argument("--split-report", action="store_true", help="show split sizes")
    ap.add_argument("--max-per-class", type=int, default=None)
    args = ap.parse_args()

    if args.audit:
        rep = audit_dataset(args.data_dir, args.max_per_class)
        print("== Dataset audit ==")
        for k, v in rep.items():
            print(f"{k:>28}: {v}")

    if args.split_report:
        paths, labels, names = scan_dataset(args.data_dir, args.max_per_class)
        splits = stratified_split(paths, labels)
        leakage_check(splits)
        print("== Split report (leak-free) ==")
        for k, (p, y) in splits.items():
            print(f"{k:>6}: {len(p):>6} images | {class_distribution(y)}")


if __name__ == "__main__":
    main()
