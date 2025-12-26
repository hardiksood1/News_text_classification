# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # +
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import joblib

# 1. Load your data
df = pd.read_excel('./merged_output.xlsx')  # Ensure columns: 'Information', 'Subcategory'

# 2. Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['Subcategory'])

# Print class mapping (Optional, for reference)
label2id = {label: idx for idx, label in enumerate(le.classes_)}
id2label = {idx: label for label, idx in label2id.items()}
print("Label mapping:", label2id)

# 3. Stratified train/validation split
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42
)

# 4. Tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

def tokenize(batch):
    return tokenizer(batch['Information'], padding='max_length', truncation=True, max_length=128)

# 5. Create Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df[['Information', 'label']])
val_dataset = Dataset.from_pandas(val_df[['Information', 'label']])

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# Set format for PyTorch
columns = ['input_ids', 'attention_mask', 'label']
train_dataset.set_format(type='torch', columns=columns)
val_dataset.set_format(type='torch', columns=columns)

# ✅ Add this check before training
num_labels = df['label'].nunique()
assert train_dataset['label'].max().item() < num_labels, "Some labels exceed num_labels!"
assert train_dataset['label'].min().item() >= 0, "Some labels are negative!"

# 6. Load RoBERTa model with correct number of labels
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=num_labels,  # ✅ Dynamically set
    id2label=id2label,
    label2id=label2id
).to('cuda')


# 7. Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=15,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    logging_dir='./logs',
    logging_steps=25,
    report_to="none"
)


# 8. Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = (preds == labels).mean()
    return {'accuracy': accuracy}

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 10. Train
trainer.train()

import os

# Create the directory if it doesn't exist
save_dir = './roberta_text_classifier'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
# Now safely save your files
joblib.dump(le, os.path.join(save_dir, 'label_encoder.pkl'))
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
# 11. Save model and tokenizer
# joblib.dump(le, '.roberta_text_classifier/label_encoder.pkl')
# model.save_pretrained('.roberta_text_classifier')
# tokenizer.save_pretrained('.roberta_text_classifier')

# 12. Load model and tokenizer for inference
model = RobertaForSequenceClassification.from_pretrained('./roberta_text_classifier')
tokenizer = RobertaTokenizerFast.from_pretrained('./roberta_text_classifier')

# 13. Test with input text
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        pred_subcategory = le.inverse_transform([pred_label])[0]
    return pred_subcategory, probs[0][pred_label].item()

# Interactive prediction loop
while True:
    test_text = input("Enter Information to classify (or type 'exitexit' to quit): ")
    if test_text.lower() == 'exitexit':
        print("Exiting...")
        break
    predicted_subcategory, confidence = predict(test_text)
    print(f"Predicted Subcategory: {predicted_subcategory} (confidence: {confidence:.2f})\n")


