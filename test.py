import streamlit as st
import onnxruntime as ort
import numpy as np
import joblib
import torch
from transformers import RobertaTokenizerFast

# ==========================================================
# PATH CONFIGURATION
# ==========================================================
ONNX_MODEL_PATH = r"E:\data\all i have\plant model\news text classification bert\robertaroberta_text_classifier (1).onnx"
LABEL_ENCODER_PATH = r"E:\data\all i have\plant model\news text classification bert\model main\label_encoder (2).pkl"

# ==========================================================
# LOAD COMPONENTS (CACHED)
# ==========================================================
@st.cache_resource
def load_components():
    session = ort.InferenceSession(
        ONNX_MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return session, tokenizer, label_encoder

session, tokenizer, le = load_components()

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================
def predict_text(text):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    ort_inputs = {
        "input_ids": encoded["input_ids"].numpy(),
        "attention_mask": encoded["attention_mask"].numpy()
    }

    outputs = session.run(None, ort_inputs)
    logits = outputs[0]

    probs = torch.softmax(torch.tensor(logits), dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    label = le.inverse_transform([pred_idx])[0]
    confidence = probs[0][pred_idx].item()

    return label, confidence

# ==========================================================
# STREAMLIT UI
# ==========================================================
st.set_page_config(page_title="News Text Classifier", layout="centered")

st.title("News Text Classification")
st.markdown(
    """
    ✔ Fast inference  
    ✔ No training  
    ✔ Production-ready  
    """
)

text_input = st.text_area(
    "Enter text to classify",
    height=160,
    placeholder="Paste or type news text here..."
)

if st.button("🔍 Predict Label"):
    if text_input.strip():
        label, confidence = predict_text(text_input)

        st.success(f"**Predicted Label:** {label}")
        st.info(f"**Confidence:** {confidence:.2f}")
    else:
        st.warning("Please enter some text.")

