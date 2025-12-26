# 📰 News Text Classification (RoBERTa + ONNX)

This repository contains a **minimal, production-focused NLP pipeline** for news text classification using **RoBERTa**, converted to **ONNX** for fast inference and tested via a lightweight app/script.

---
## 🎥 Demo (GitHub Auto-Play)

▶ Click the video below to see .


## 1️⃣ Training File

**Purpose:**  
Train a RoBERTa-based multi-class text classifier using labeled news data.

**Key Steps:**
- Load dataset (`Information`, `Subcategory`)
- Encode labels using `LabelEncoder`
- Stratified train/validation split
- Tokenization with `RobertaTokenizerFast`
- Fine-tune `roberta-base`
- Save:
  - Trained model
  - Tokenizer
  - Label encoder

**Output:**
- Trained RoBERTa model
- `label_encoder.pkl`

---

## 2️⃣ Convert to ONNX File

**Purpose:**  
Optimize the trained RoBERTa model for **fast CPU inference** using ONNX.

**Key Steps:**
- Load trained model config and tokenizer
- Load model weights from `model.safetensors`
- Create dummy input for tracing
- Export model using `torch.onnx.export`

**Output:**
- `roberta_text_classifier.onnx`

**Benefits:**
- Faster inference
- Lower latency
- Production-ready deployment

---

## 3️⃣ Testing File

**Purpose:**  
Run inference on new text using the ONNX model.

**Key Steps:**
- Load ONNX model using `onnxruntime`
- Tokenize input text
- Run inference
- Apply softmax to get confidence score
- Decode predicted label using `LabelEncoder`

**Output:**
- Predicted class label
- Confidence score

---

## ✅ Notes

- Supports **English & Hinglish** text classification
- Can be extended to **any text classification task** if labeled data is available
- Suitable for deployment using **Streamlit, FastAPI, or Flask**

---

**Author:** Hardik Sood  
*MSc Data Science | NLP | AI*
