# ─────────────────────────────────────────────────────────────────────────────
# EarthLens AI — Live demo (Streamlit)
# Owner: Member 5 — Evaluation & Deployment Engineer
# ─────────────────────────────────────────────────────────────────────────────
"""AIE231 spec 3.7: a small live demo on top of the inference pipeline.

Run from the repository root:
    streamlit run app/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]           # repo root on sys.path
sys.path.insert(0, str(ROOT))

from src.evaluate import load_model_entry            # noqa: E402
from src.predict import predict_image                # noqa: E402

st.set_page_config(page_title="EarthLens AI", page_icon="🛰️", layout="centered")

st.title("🛰️ EarthLens AI — Satellite Image Classifier")
st.caption("EuroSAT RGB (Sentinel-2, 64×64) — 10 land-use classes · AIE231 course project")


@st.cache_resource
def get_model(run_name: str | None):
    model, entry = load_model_entry(None, run_name)
    return model, entry


# ── Model selector from models/manifest.json ────────────────────────────────
manifest_path = ROOT / "models" / "manifest.json"
if not manifest_path.exists():
    st.warning("No trained model found. Train one first:\n\n"
               "`python -m src.train --model model_a --author YourGitHubHandle`")
    st.stop()

entries = [e for e in json.loads(manifest_path.read_text()) if (ROOT / "models" / e["model_file"]).exists()]
if not entries:
    st.warning("No saved models found."); st.stop()
entries.sort(key=lambda e: (e.get("eligible", False), e.get("val_macro_f1", 0)), reverse=True)
labels = [f"{e['run_name']}  (val_F1 {e['val_macro_f1']:.3f})" for e in entries]
choice = st.selectbox("Choose a saved model", labels, index=0)
model, entry = get_model(entries[labels.index(choice)]["run_name"])
st.info(f"Architecture: **{entry['model_key']}** · preprocessing: "
        f"`{entry['preprocessing']}` · size: "
        f"{(ROOT / 'models' / entry['model_file']).stat().st_size / 1e6:.1f} MB")

# ── Upload ───────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a satellite tile (JPG/PNG, any size — it is resized to 64×64)",
    type=["jpg", "jpeg", "png"])

if uploaded is not None:
    temp_path = ROOT / "data" / "_tmp_upload.jpg"
    temp_path.parent.mkdir(exist_ok=True)
    temp_path.write_bytes(uploaded.getvalue())

    st.image(uploaded.getvalue(), width=260, caption=uploaded.name)
    with st.spinner("Running inference ..."):
        result = predict_image(model, entry, temp_path)

    st.success(f"**{result['label']}** — confidence {result['confidence']:.1%}")
    top3 = sorted(result["probabilities"].items(), key=lambda kv: -kv[1])[:3]
    cols = st.columns(3)
    for col, (name, p) in zip(cols, top3):
        col.metric(name, f"{p:.1%}")

    st.subheader("Per-class probabilities")
    import pandas as pd
    st.bar_chart(pd.Series(result["probabilities"], name="probability"))

    st.caption(f"Inference: {result['inference_ms']} ms per sample · "
               f"model: {result['model']}")
    temp_path.unlink(missing_ok=True)
else:
    st.caption("Try tiles from `data/EuroSAT/<ClassName>/` — e.g. a River or a "
               "Highway image, to inspect the prediction.")
