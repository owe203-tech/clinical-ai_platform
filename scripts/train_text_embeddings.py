from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score


MANIFEST_PATH = Path("data/processed/model_manifest.csv")

df = pd.read_csv(MANIFEST_PATH)

report_df = df[
    ["report_id", "split", "cardiomegaly", "indication"]
].drop_duplicates("report_id").copy()

report_df["indication"] = report_df["indication"].fillna("")

train_df = report_df[report_df["split"] == "train"].copy()
val_df = report_df[report_df["split"] == "val"].copy()

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder = AutoModel.from_pretrained(MODEL_NAME)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

encoder = encoder.to(device)
encoder.eval()

def encode_texts(texts, batch_size=32):
    embeddings = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size].tolist()

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            outputs = encoder(**inputs)

            batch_embeddings = outputs.last_hidden_state[:, 0, :]

            embeddings.append(
                batch_embeddings.cpu().numpy()
            )

    return np.vstack(embeddings)

X_train = encode_texts(train_df["indication"])
X_val = encode_texts(val_df["indication"])

y_train = train_df["cardiomegaly"].astype(int)
y_val = val_df["cardiomegaly"].astype(int)

clf = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

clf.fit(X_train, y_train)

val_probs = clf.predict_proba(X_val)[:, 1]

val_auc = roc_auc_score(y_val, val_probs)
val_pr_auc = average_precision_score(y_val, val_probs)

print(f"Validation AUROC: {val_auc:.4f}")
print(f"Validation PR-AUC: {val_pr_auc:.4f}")