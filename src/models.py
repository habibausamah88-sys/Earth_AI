# ─────────────────────────────────────────────────────────────────────────────
# EarthLens AI — Neural network architectures (Models A, B, C)
# Owner: Member 3 (Mariam) — Tech Lead & Model Architect
# ─────────────────────────────────────────────────────────────────────────────
"""All architectures live in this module (AIE231 spec 6.1 `src/models.py`).

  Model A  — custom CNN built from scratch (inputs 0-255; Rescaling inside model)
  Model B  — pretrained ResNet50, fully frozen base (inputs preprocessed by src.data)
  Model C  — ResNet50 fine-tuned, last block unfrozen, small learning rate

Also contains the two spec-mandated extras:
  * hand parameter count report, checked against model.summary()   (spec 3.4)
  * a manual tf.GradientTape training step demo                    (spec 3.4)

CLI (from the repo root):
    python -m src.models --summary model_a
    python -m src.models --summary model_b --hand-count
    python -m src.models --tape-demo --data-dir data/EuroSAT --smoke
"""
from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras

from configs import config as cfg

IMG_SHAPE = (cfg.IMG_SIZE, cfg.IMG_SIZE, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Model A — custom CNN from scratch
# ─────────────────────────────────────────────────────────────────────────────
def build_model_a(
    conv_blocks: list[tuple[int, int]] | None = None,
    dense_units: int = 128,
    dropout: float = 0.40,
    batch_norm: bool = True,
) -> keras.Model:
    """Custom CNN: 3 conv blocks (32-64-128) + BN + dropout + GAP head.

    Justification (kept here for the report): small 64x64 inputs favour 3x3
    kernels with ReLU; BatchNorm stabilises training; GlobalAveragePooling
    keeps the head light; Dropout fights overfitting on 27k images; softmax +
    categorical cross-entropy is the standard multi-class output (spec 3.4).
    """
    conv_blocks = conv_blocks or cfg.MODEL_A["conv_blocks"]
    inputs = keras.Input(shape=IMG_SHAPE, name="input_0_255")
    x = keras.layers.Rescaling(1.0 / 255.0, name="rescale_0_1")(inputs)

    for block_id, (filters, n_layers) in enumerate(conv_blocks, start=1):
        for layer_id in range(1, n_layers + 1):
            x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=not batch_norm, kernel_initializer="he_normal",
                                    name=f"conv_b{block_id}.{layer_id}")(x)
            if batch_norm:
                x = keras.layers.BatchNormalization(name=f"bn_b{block_id}.{layer_id}")(x)
            x = keras.layers.ReLU(name=f"relu_b{block_id}.{layer_id}")(x)
        x = keras.layers.MaxPooling2D(name=f"pool_b{block_id}")(x)
        x = keras.layers.Dropout(dropout / 2, name=f"drop_b{block_id}")(x)

    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dense(dense_units, activation="relu", name="dense_head")(x)
    x = keras.layers.Dropout(dropout, name="drop_head")(x)
    outputs = keras.layers.Dense(cfg.NUM_CLASSES, activation="softmax", name="softmax_10")(x)
    return keras.Model(inputs, outputs, name="ModelA_CustomCNN")


# ─────────────────────────────────────────────────────────────────────────────
# Models B and C — pretrained ResNet50 (ImageNet, allowed for the image track)
# ─────────────────────────────────────────────────────────────────────────────
def build_resnet_backbone(pretrained: bool = True, unfreeze_from: str | None = None) -> keras.Model:
    """ResNet50 backbone without its top classifier.

    ``pretrained=True`` downloads ImageNet weights (needs internet; pass False
    only for offline smoke tests). ``unfreeze_from="conv5"`` unfreezes the last
    stage for fine-tuning while BatchNorm layers stay frozen (standard practice
    on small images: 64x64 differs a lot from ImageNet's 224x224, but the early
    edge/texture filters remain useful).
    """
    base = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=IMG_SHAPE,
        pooling=None,
    )
    base.trainable = unfreeze_from is not None
    if unfreeze_from is not None:
        for layer in base.layers:
            layer.trainable = layer.name.startswith(unfreeze_from) and not isinstance(
                layer, keras.layers.BatchNormalization
            )
    return base


def build_resnet_model(
    unfreeze_from: str | None,
    dense_units: int = 128,
    dropout: float = 0.30,
    pretrained: bool = True,
) -> keras.Model:
    """Shared builder for Models B (frozen) and C (fine-tuned).

    NOTE: the model receives inputs already transformed by
    ``src.data.preprocess_resnet`` (caffe-style BGR mean subtraction) inside the
    tf.data pipeline and predict.py — the preprocessing is NOT duplicated here.
    """
    base = build_resnet_backbone(pretrained=pretrained, unfreeze_from=unfreeze_from)
    inputs = keras.Input(shape=IMG_SHAPE, name="input_preprocessed")
    x = base(inputs, training=False)  # training=False keeps frozen BN statistics
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dense(dense_units, activation="relu", name="dense_head")(x)
    x = keras.layers.Dropout(dropout, name="drop_head")(x)
    outputs = keras.layers.Dense(cfg.NUM_CLASSES, activation="softmax", name="softmax_10")(x)
    name = "ModelC_ResNet50_FineTuned" if unfreeze_from else "ModelB_ResNet50_Frozen"
    return keras.Model(inputs, outputs, name=name)


def build_model_b(pretrained: bool = True) -> keras.Model:
    cfgb = cfg.MODEL_B
    return build_resnet_model(None, cfgb["dense_units"], cfgb["dropout"], pretrained)


def build_model_c(pretrained: bool = True) -> keras.Model:
    cfgc = cfg.MODEL_C
    return build_resnet_model(cfgc["unfreeze_from"], cfgc["dense_units"], cfgc["dropout"], pretrained)


MODEL_BUILDERS = {
    "model_a": build_model_a,
    "model_b": build_model_b,
    "model_c": build_model_c,
}


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer factory (spec 3.4: compare SGD, SGD+momentum, Adam on our data)
# ─────────────────────────────────────────────────────────────────────────────
def build_optimizer(name: str, learning_rate: float) -> keras.optimizers.Optimizer:
    if name == "adam":
        return keras.optimizers.Adam(learning_rate)
    if name == "sgd_momentum":
        return keras.optimizers.SGD(learning_rate, momentum=0.9, nesterov=True)
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate)
    raise ValueError(f"unknown optimizer {name!r} (use adam / sgd_momentum / sgd)")


def compile_model(model: keras.Model, learning_rate: float, optimizer: str = "adam") -> None:
    model.compile(
        optimizer=build_optimizer(optimizer, learning_rate),
        loss="categorical_crossentropy",   # pipeline emits one-hot labels
        metrics=["accuracy"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hand parameter count (spec 3.4 — every number must match model.summary())
# ─────────────────────────────────────────────────────────────────────────────
def _hand_layer_params(layer: keras.layers.Layer) -> int | None:
    """Closed-form parameter count for the layer types we use (spec 3.4 formulas)."""
    if isinstance(layer, keras.Model):
        return sum(_hand_layer_params(child) for child in layer.layers)
    if isinstance(layer, keras.layers.Conv2D):
        c_in = int(layer.input.shape[-1]); kh, kw = layer.kernel_size
        return (kh * kw * c_in // layer.groups + int(layer.use_bias)) * layer.filters
    if isinstance(layer, keras.layers.Dense):
        return (int(layer.input.shape[-1]) + int(layer.use_bias)) * layer.units
    if isinstance(layer, keras.layers.BatchNormalization):
        c = int(layer.input.shape[-1])
        return c * (2 + int(layer.scale) + int(layer.center))
    if layer.weights:
        raise ValueError(f'No manual formula for {layer.name}: {type(layer).__name__}')
    return 0


def hand_parameter_report(model):
    """Calculate every leaf layer independently; compare with Keras, include shapes."""
    rows = []
    def walk(net, prefix=''):
        for layer in net.layers:
            if isinstance(layer, keras.Model):
                walk(layer, prefix + layer.name + '/')
                continue
            hand, actual = _hand_layer_params(layer), int(layer.count_params())
            shape = str(layer.output.shape) if hasattr(layer.output, 'shape') else str(layer.output)
            assert hand == actual, f'{layer.name}: {hand} != {actual}'
            rows.append((prefix + layer.name, hand, actual, shape))
    walk(model)
    assert sum(r[1] for r in rows) == model.count_params()
    for name, hand, actual, shape in rows:
        print(f'{name:45} {shape:24} hand={hand:10,} actual={actual:10,}')
    return rows


def fine_tune_from_b(path):
    """Restore trained frozen classifier, then unfreeze the final residual stage."""
    model = keras.models.load_model(path, compile=False)
    base = next(layer for layer in model.layers if isinstance(layer, keras.Model))
    base.trainable = True
    for layer in base.layers:
        layer.trainable = layer.name.startswith('conv5') and not isinstance(layer, keras.layers.BatchNormalization)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Manual GradientTape training step (spec 3.4, done once for Model A)
# ─────────────────────────────────────────────────────────────────────────────
def manual_training_step_demo(
    data_dir: str,
    steps: int = 15,
    max_per_class: int = 40,
    batch_size: int = 32,
) -> None:
    """One manual training loop with tf.GradientTape: forward -> loss -> grads -> update.

    Shows the loss decreasing, proving understanding of what model.fit() does.
    """
    from src import data as data_mod  # local import avoids a circular import

    split = data_mod.get_splits(
        data_dir, batch_size=batch_size, augment=True, preprocessing="raw",
        max_per_class=max_per_class,
    )
    model = build_model_a(dropout=0.2)
    loss_fn = keras.losses.CategoricalCrossentropy()
    optimizer = build_optimizer("adam", 1e-3)

    step, epoch_losses = 0, []
    x, y = next(iter(split["train_ds"]))
    for _ in range(steps):
        if step >= steps:
            break
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        epoch_losses.append(float(loss))
        print(f"  step {step + 1:>3}/{steps}  loss={float(loss):.4f}")
        step += 1

    from pathlib import Path
    cfg.EXPERIMENTS_DIR.mkdir(exist_ok=True)
    np.savetxt(cfg.EXPERIMENTS_DIR / "gradient_tape_losses.csv", epoch_losses, delimiter=",", header="loss", comments="")
    first, last = np.mean(epoch_losses[: max(1, steps // 3)]), np.mean(epoch_losses[-max(1, steps // 3):])
    print(f"  mean loss first third={first:.4f}  last third={last:.4f}  ->  "
          + ("decreasing ✔" if last < first else "not decreasing ✘ (increase steps)"))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="EarthLens AI architectures (Mariam)")
    ap.add_argument("--summary", choices=["model_a", "model_b", "model_c"])
    ap.add_argument("--hand-count", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true", help="offline smoke test")
    ap.add_argument("--tape-demo", action="store_true")
    ap.add_argument("--data-dir", default=str(cfg.DATA_DIR))
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.summary:
        model = build_model_a() if args.summary == "model_a" else MODEL_BUILDERS[args.summary](pretrained=not args.no_pretrained)
        model.summary(line_length=96)
        if args.hand_count:
            print("\n== Hand parameter count vs summary ==")
            hand_parameter_report(model)
    if args.tape_demo:
        manual_training_step_demo(
            args.data_dir, steps=args.steps, max_per_class=20 if args.smoke else 40
        )


if __name__ == "__main__":
    main()
