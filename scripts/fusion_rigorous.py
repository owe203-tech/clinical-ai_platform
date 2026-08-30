from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


TRAIN_IMAGE_PATH = Path(
    "results/train_image_predictions.csv"
)

TRAIN_TEXT_PATH = Path(
    "results/train_text_predictions.csv"
)

VAL_IMAGE_PATH = Path(
    "results/val_image_predictions.csv"
)

VAL_TEXT_PATH = Path(
    "results/val_text_predictions.csv"
)


# -----------------------------
# Load training predictions
# -----------------------------

train_image_df = pd.read_csv(TRAIN_IMAGE_PATH)
train_text_df = pd.read_csv(TRAIN_TEXT_PATH)

train_fusion_df = train_image_df.merge(
    train_text_df[
        ["report_id", "text_prob"]
    ],
    on="report_id",
    how="inner",
)


# -----------------------------
# Load validation predictions
# -----------------------------

val_image_df = pd.read_csv(VAL_IMAGE_PATH)
val_text_df = pd.read_csv(VAL_TEXT_PATH)

val_fusion_df = val_image_df.merge(
    val_text_df[
        ["report_id", "text_prob"]
    ],
    on="report_id",
    how="inner",
)


# -----------------------------
# Train fusion model
# -----------------------------

X_train = train_fusion_df[
    ["image_prob", "text_prob"]
]

y_train = train_fusion_df[
    "cardiomegaly"
].astype(int)

fusion_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
)

fusion_model.fit(
    X_train,
    y_train,
)


# -----------------------------
# Evaluate on validation
# -----------------------------

X_val = val_fusion_df[
    ["image_prob", "text_prob"]
]

y_val = val_fusion_df[
    "cardiomegaly"
].astype(int)

val_fusion_probs = fusion_model.predict_proba(
    X_val
)[:, 1]

val_auc = roc_auc_score(
    y_val,
    val_fusion_probs
)

val_pr_auc = average_precision_score(
    y_val,
    val_fusion_probs
)


print("Rigorous learned fusion")
print(f"Validation AUROC: {val_auc:.4f}")
print(f"Validation PR-AUC: {val_pr_auc:.4f}")

print("\nLearned coefficients:")
print(
    pd.Series(
        fusion_model.coef_[0],
        index=X_train.columns,
    )
)

print(
    f"Intercept: "
    f"{fusion_model.intercept_[0]:.4f}"
)