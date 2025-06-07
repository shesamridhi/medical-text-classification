import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from sklearn.metrics import classification_report

class MedicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)

# Load sample data
df = pd.read_csv("data/sample.csv")  # columns: text, label
texts = df['text'].tolist()
labels = df['label'].astype(int).tolist()

# Split
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Dataset and DataLoader
train_ds = MedicalDataset(train_texts, train_labels, tokenizer)
val_ds = MedicalDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_ds, batch_size=8)
val_loader = DataLoader(val_ds, batch_size=8)

# Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(set(labels)))
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training Loop
for epoch in range(3):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1} complete.")

# Save model
model.save_pretrained("models/saved_model/")
tokenizer.save_pretrained("models/saved_model/")

# Evaluation
model.eval()
preds, true = [], []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)
        outputs = model(input_ids, attention_mask=attention_mask)
        preds += torch.argmax(outputs.logits, axis=1).cpu().tolist()
        true += labels.cpu().tolist()

print(classification_report(true, preds))
