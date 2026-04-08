"""
app.py — Algospeak Classifier demo

Streamlit UI for the dual BERTweet model.
Type a social media post and see the predicted class + confidence scores.

Usage:
    uv run streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "poc" / "src"))

import yaml
import torch
import numpy as np
import emoji
import streamlit as st
from transformers import AutoTokenizer

from inference import load_unsupervised_encoder, classify_text

BASE_DIR = Path(__file__).parent

CLASS_COLORS = {
    "Allowed":            "green",
    "Offensive Language": "red",
    "Mature Content":     "orange",
    "Algospeak":          "violet",
}


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    with open(BASE_DIR / "poc" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_unsupervised_encoder(
        BASE_DIR / cfg["checkpoint_dir"] / "best_model.pt", cfg, device
    )
    prototypes = np.load(BASE_DIR / cfg["results_dir"] / "prototypes.npy")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)
    return encoder, prototypes, tokenizer, cfg, device


# ─────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────

st.title("Algospeak Classifier")
st.caption("Dual BERTweet model · type a social media post to classify it.")

text = st.text_area("Post text", height=120, placeholder="Type something here...")

if st.button("Classify", type="primary") and text.strip():
    encoder, prototypes, tokenizer, cfg, device = load_model()
    result = classify_text(text, encoder, prototypes, tokenizer, cfg["max_length"], device)

    label = result["predicted_label"]
    color = CLASS_COLORS[label]

    st.markdown(f"## :{color}[{label}]")
    st.divider()

    st.write("**Confidence scores:**")
    for name, score in sorted(result["scores"].items(), key=lambda x: -x[1]):
        st.progress(float(score), text=f"{name}: {score:.1%}")
