from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "model_manifest.csv"

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


df = pd.read_csv(MANIFEST_PATH)

report_df = (
    df[["report_id", "split", "cardiomegaly", "indication"]]
    .drop_duplicates("report_id")
    .copy()
)

report_df["indication"] = report_df["indication"].fillna("")

train_df = report_df[report_df["split"] == "train"].copy()
val_df = report_df[report_df["split"] == "val"].copy()

print("Train reports:", len(train_df))
print("Validation reports:", len(val_df))

class IndicationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=64):
        self.texts = dataframe["indication"].tolist()
        self.labels = dataframe["cardiomegaly"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

train_dataset = IndicationDataset(train_df, tokenizer)
val_dataset = IndicationDataset(val_df, tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

model = model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,
)

NUM_EPOCHS = 1
best_val_auc = 0.0
patience = 2
epochs_without_improvement = 0

for epoch in range(NUM_EPOCHS):

    # ----- TRAIN -----
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

        # ----- VALIDATION -----
    model.eval()

    val_probs = []
    val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            probs = torch.softmax(outputs.logits, dim=1)[:, 1]

            val_probs.extend(probs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_auc = roc_auc_score(val_labels, val_probs)
    val_pr_auc = average_precision_score(val_labels, val_probs)

    print(
        f"Epoch {epoch + 1}: "
        f"Train Loss={train_loss:.4f}, "
        f"Val AUROC={val_auc:.4f}, "
        f"Val PR-AUC={val_pr_auc:.4f}"
    )
    
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            "results/best_text_finetune.pt"
        )

        print("Saved new best model.")

    else:
        epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping.")
            break
