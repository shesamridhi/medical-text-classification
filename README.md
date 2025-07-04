
# Medical Text Classification using BERT

This project uses a fine-tuned BERT model to classify clinical notes into diagnosis categories.

## 🧠 Model
- `bert-base-uncased` (HuggingFace Transformers)
- Dataset: Medical Transcription (or similar)
- Fine-tuned on classification task

## 💡 Features
- Tokenizes and trains BERT on medical text
- Outputs classification report
- Can be extended with Streamlit or REST API

## 🚀 How to Run
```bash
pip install -r requirements.txt
python train.py
