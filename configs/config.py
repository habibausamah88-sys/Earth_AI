# ─────────────────────────────────────────────────────────────────────────────
# EarthLens AI — Central configuration
# Owner: Member 3 (Mariam) — Tech Lead & Model Architect
# ─────────────────────────────────────────────────────────────────────────────
"""Single source of truth for every hyper-parameter, path and split ratio used
in the project. No other module may hard-code a path, a class name or a
hyper-parameter: they all import from here (see AIE231 spec 6.1 "configs/").
"""
from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── Paths (anchored to the repository root — no CWD dependence) ─────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "EuroSAT"     # never committed to git (see .gitignore)
MODELS_DIR = ROOT / "models"
EXPERIMENTS_DIR = ROOT / "experiments"
RUNS_DIR = EXPERIMENTS_DIR / "runs"
NOTEBOOKS_DIR = ROOT / "notebooks"
APP_DIR = ROOT / "app"

# ── Dataset (EuroSAT RGB, Sentinel-2, 64x64 px, 10 classes) ─────────────────
# Official Zenodo record (Helber et al., 2019), CC BY 4.0 licence.
EUROSAT_ZIP_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
EUROSAT_ZIP_URL_FALLBACK = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"
EUROSAT_CITATION = (
    "Helber, P., Bischke, B., Dengel, A., Borth, D. (2019). "
    "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and "
    "Land Cover Classification. IEEE Journal of Selected Topics in Applied "
    "Earth Observations and Remote Sensing (JSTARS). "
    "Dataset: https://zenodo.org/records/7711810 (CC BY 4.0)"
)

IMG_SIZE = 64
NUM_CLASSES = 10
# Label index = alphabetical position of the class folder (must stay consistent
# with src/data.class_names(), which derives the same order from the disk).
CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# ── Splits (70 / 15 / 15, stratified — AIE231 spec 3.2) ──────────────────────
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ── Model A: custom CNN built from scratch (trained by Member 2) ─────────────
MODEL_A = dict(
    key="model_a",
    conv_blocks=[(32, 2), (64, 2), (128, 1)],   # (filters, conv layers in block)
    dense_units=128,
    dropout=0.40,
    batch_norm=True,
    learning_rate=1e-3,
)

# ── Model B: pretrained ResNet50, frozen base (trained by Member 3 / Mariam) ─
MODEL_B = dict(
    key="model_b",
    backbone="ResNet50",
    pretrained=True,            # ImageNet weights — allowed by spec (image track)
    unfreeze_from=None,         # None = fully frozen base
    dense_units=128,
    dropout=0.30,
    learning_rate=1e-3,
)

# ── Model C: ResNet50 fine-tuned, last block unfrozen (Member 3 / Mariam) ────
MODEL_C = dict(
    key="model_c",
    backbone="ResNet50",
    pretrained=True,
    unfreeze_from="conv5",      # unfreeze layers whose name starts with conv5
    dense_units=128,
    dropout=0.30,
    learning_rate=1e-4,         # small LR is mandatory when fine-tuning
)

# ── Simple learned baseline: single-hidden-layer MLP (Member 2) ──────────────
BASELINE_MLP = dict(
    key="baseline_mlp",
    hidden_units=256,
    dropout=0.30,
    learning_rate=1e-3,
)

# ── Default training settings (AIE231 spec 3.4) ──────────────────────────────
TRAINING = dict(
    batch_size=64,
    epochs=12,
    patience=4,                 # early stopping on val_loss, restore best weights
    augment=True,               # training-only augmentation (ablated in Milestone 4)
)

# ── Experiment grid (Member 4, one factor at a time — spec 3.5) ──────────────
TUNING = dict(
    learning_rate=[1e-4, 1e-3, 1e-2],
    batch_size=[32, 64, 128],
    optimizer=["adam", "sgd_momentum", "sgd"],
    architecture_factor=dict(   # depth/units factor for Model A
        reference_units=128,
        variants=[64, 256],
    ),
    final_seeds=[42, 123, 2024],  # 3-seed mean +/- std for the final config
)

# ── Which preprocessing each model family expects at inference time ─────────
# "raw"     -> model starts with a Rescaling(1/255) layer (Model A, baseline)
# "resnet50"-> inputs are 0-255 float, tf.keras.applications.resnet50.preprocess_input
#              is applied in the data pipeline AND by src/predict.py (single source)
PREPROCESSING = {
    "baseline_mlp": "raw",
    "model_a": "raw",
    "model_b": "resnet50",
    "model_c": "resnet50",
}
