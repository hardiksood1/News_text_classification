import os
import torch
from transformers import RobertaTokenizerFast, RobertaConfig, RobertaForSequenceClassification
from safetensors.torch import load_file

# Path to model directory
model_dir = "/workspace/mha/mha/roberta_text_classifier"

# Load tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)

# Load config
config = RobertaConfig.from_pretrained(model_dir)

# Initialize model architecture
model = RobertaForSequenceClassification(config)

# Load weights from .safetensors
state_dict = load_file(os.path.join(model_dir, "model.safetensors"))
model.load_state_dict(state_dict)
model.eval()

# Dummy input for ONNX tracing
sample_text = "This is a sample sentence for ONNX export."
inputs = tokenizer(sample_text, return_tensors="pt")

# ONNX export path
onnx_path = os.path.join(model_dir, "roberta_text_classifier.onnx")

# Export the model to ONNX
torch.onnx.export(
    model,                                           # model
    (inputs["input_ids"], inputs["attention_mask"]), # inputs
    onnx_path,                                       # output path
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "logits": {0: "batch_size"}
    },
    opset_version=15,
    do_constant_folding=True
)

print(f"ONNX model exported to: {onnx_path}")
