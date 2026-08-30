from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


# -----------------------------
# Paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "model_manifest.csv"
)
# -----------------------------
# Load manifest
# -----------------------------

df = pd.read_csv(MANIFEST_PATH)


# -----------------------------
# One row per report
# -----------------------------

report_df = df[
    [
        "report_id",
        "split",
        "cardiomegaly",
        "indication",
    ]
].drop_duplicates(
    subset="report_id"
).copy()


# Missing indication -> empty string
report_df["indication"] = (
    report_df["indication"]
    .fillna("")
)


# -----------------------------
# Split
# -----------------------------

train_df = report_df[
    report_df["split"] == "train"
].copy()

val_df = report_df[
    report_df["split"] == "val"
].copy()

test_df = report_df[
    report_df["split"] == "test"
].copy()


print(f"Train reports: {len(train_df)}")
print(f"Validation reports: {len(val_df)}")
print(f"Test reports: {len(test_df)}")


# -----------------------------
# Text / labels
# -----------------------------

X_train_text = train_df["indication"]
X_val_text = val_df["indication"]

y_train = train_df["cardiomegaly"].astype(int)
y_val = val_df["cardiomegaly"].astype(int)


# -----------------------------
# TF-IDF
# -----------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=10000,
)

X_train = vectorizer.fit_transform(
    X_train_text
)

X_val = vectorizer.transform(
    X_val_text
)

print(f"TF-IDF features: {X_train.shape[1]}")


# -----------------------------
# Logistic regression
# -----------------------------

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

model.fit(
    X_train,
    y_train
)


# -----------------------------
# Validation predictions
# -----------------------------

val_probs = model.predict_proba(
    X_val
)[:, 1]


# -----------------------------
# Discrimination metrics
# -----------------------------

val_auc = roc_auc_score(
    y_val,
    val_probs
)

val_pr_auc = average_precision_score(
    y_val,
    val_probs
)


print("\nIndication-only TF-IDF baseline")
print(f"Validation AUROC: {val_auc:.4f}")
print(f"Validation PR-AUC: {val_pr_auc:.4f}")

MANIFEST_PATH = Path(
    "data/processed/model_manifest.csv"
)

df = pd.read_csv(MANIFEST_PATH)

frontal_counts = (
    df.groupby("report_id")
    .size()
)

print(frontal_counts.value_counts().sort_index())

print(
    "\nReports with multiple frontal images:",
    (frontal_counts > 1).sum()
)

print(
    "Percentage:",
    (frontal_counts > 1).mean() * 100
)

image_pred_df = pd.DataFrame({
    "report_id": val_df["report_id"].values,
    "image_prob": val_probs
})

report_image_probs = (
    image_pred_df
    .groupby("report_id", as_index=False)["image_prob"]
    .mean()
)
val_probs = model.predict_proba(X_val)[:, 1]

train_probs = model.predict_proba(
    X_train
)[:, 1]

train_text_pred_df = pd.DataFrame({
    "report_id": train_df["report_id"].values,
    "text_prob": train_probs,
    "cardiomegaly": y_train.values,
})

text_pred_df = pd.DataFrame({
    "report_id": val_df["report_id"].values,
    "text_prob": val_probs,
    "cardiomegaly": y_val.values,
})

OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

train_text_pred_df.to_csv(
    OUTPUT_DIR / "train_text_predictions.csv",
    index=False,
)

print("Saved training text predictions.")

text_pred_df.to_csv(
    OUTPUT_DIR / "val_text_predictions.csv",
    index=False,
)

print("Saved validation text predictions.")



train_text_pred_df = pd.DataFrame({
    "report_id": train_df["report_id"].values,
    "text_prob": train_probs,
    "cardiomegaly": y_train.values,
})

train_text_pred_df.to_csv(
    "results/train_text_predictions.csv",
    index=False,
)

