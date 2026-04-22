"""
experiments/three_class/app.py

Local Streamlit demo for the 3-class algospeak classifier.
Classes: Allowed / Harmful Content / Algospeak

Usage:
    uv run streamlit run experiments/three_class/app.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "poc" / "src"))

import yaml
import torch
import numpy as np
import emoji
import streamlit as st
from transformers import AutoTokenizer

from inference import load_unsupervised_encoder, classify_text

EXP_DIR       = Path(__file__).parent
CONFIG_PATH   = EXP_DIR / "config.yaml"
CHECKPOINT    = EXP_DIR / "checkpoints" / "best_model.pt"
PROTOTYPES    = EXP_DIR / "results" / "prototypes.npy"

CLASS_COLORS = {
    "Allowed":         "green",
    "Harmful Content": "red",
    "Algospeak":       "violet",
}

CLASS_DESCRIPTIONS = {
    "Allowed":         "This post does not appear to violate content guidelines.",
    "Harmful Content": "This post contains offensive language or mature content.",
    "Algospeak":       "This post uses coded language to evade content filters.",
}


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # Override class_labels keys to ints for inference.py compatibility
    if "class_labels" in cfg:
        cfg["class_labels"] = {int(k): v for k, v in cfg["class_labels"].items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not CHECKPOINT.exists():
        st.error(f"Checkpoint not found: {CHECKPOINT}")
        st.stop()
    if not PROTOTYPES.exists():
        st.error(f"Prototypes not found: {PROTOTYPES}")
        st.stop()

    encoder    = load_unsupervised_encoder(str(CHECKPOINT), cfg, device)
    prototypes = np.load(str(PROTOTYPES))
    tokenizer  = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)
    return encoder, prototypes, tokenizer, cfg, device


# ─────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Algospeak Classifier (3-class)", page_icon="🔍", layout="centered")
st.title("Algospeak Classifier")
st.caption("3-class model · Allowed / Harmful Content / Algospeak · Dual BERTweet")

text = st.text_area("Enter a social media post:", height=120,
                    placeholder="e.g. 'gonna unalive myself lol'")

col1, col2 = st.columns([1, 4])
with col1:
    classify_btn = st.button("Classify", type="primary", use_container_width=True)

if classify_btn and text.strip():
    encoder, prototypes, tokenizer, cfg, device = load_model()
    class_names = [cfg["class_labels"][i] for i in range(cfg["num_classes"])]
    result = classify_text(text, encoder, prototypes, tokenizer, cfg["max_length"], device,
                           class_names=class_names)

    label = result["predicted_label"]
    color = CLASS_COLORS.get(label, "blue")
    desc  = CLASS_DESCRIPTIONS.get(label, "")

    st.markdown(f"## :{color}[{label}]")
    st.caption(desc)
    st.divider()

    st.write("**Similarity scores to each class prototype:**")
    for name, score in sorted(result["scores"].items(), key=lambda x: -x[1]):
        bar_color = CLASS_COLORS.get(name, "blue")
        st.progress(float(score), text=f"{name}  —  {score:.3f}")

elif classify_btn and not text.strip():
    st.warning("Please enter some text first.")

st.divider()
with st.expander("About this model"):
    st.markdown("""
**Architecture:** Dual BERTweet (vinai/bertweet-base) with supervised InfoNCE loss
**Training:** 22,295 posts — 7,400/class (Allowed), ~7,447/class (Harmful Content + Algospeak)
**Classes:**
- **Allowed** — posts that don't violate content guidelines
- **Harmful Content** — offensive language, slurs, mature/explicit content (original classes 1+2 combined)
- **Algospeak** — posts using coded language to evade filters (e.g. "unalive" for suicide, "corn" for porn)

**Test accuracy:** 89.2% · **Macro F1:** 89.3% · **Mean AUC:** 97.7%
    """)
