from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


# -----------------------------
# Paths
# -----------------------------

IMAGE_PRED_PATH = Path(
    "results/val_image_predictions.csv"
)

TEXT_PRED_PATH = Path(
    "results/val_text_predictions.csv"
)


# -----------------------------
# Load validation predictions
# -----------------------------

image_df = pd.read_csv(IMAGE_PRED_PATH)
text_df = pd.read_csv(TEXT_PRED_PATH)


# -----------------------------
# Merge by report_id
# -----------------------------

fusion_df = image_df.merge(
    text_df[
        [
            "report_id",
            "text_prob",
        ]
    ],
    on="report_id",
    how="inner",
)


# -----------------------------
# Inputs and labels
# -----------------------------

X = fusion_df[
    [
        "image_prob",
        "text_prob",
    ]
]

y = fusion_df["cardiomegaly"].astype(int)


# -----------------------------
# Learned fusion model
# -----------------------------

fusion_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
)

fusion_model.fit(
    X,
    y
)


# -----------------------------
# Validation probabilities
# -----------------------------

fusion_probs = fusion_model.predict_proba(
    X
)[:, 1]


# -----------------------------
# Metrics
# -----------------------------

auroc = roc_auc_score(
    y,
    fusion_probs
)

pr_auc = average_precision_score(
    y,
    fusion_probs
)

print("Learned logistic fusion")
print(f"Validation AUROC: {auroc:.4f}")
print(f"Validation PR-AUC: {pr_auc:.4f}")

print("\nLearned coefficients:")
print(
    pd.Series(
        fusion_model.coef_[0],
        index=X.columns,
    )
)

print(f"Intercept: {fusion_model.intercept_[0]:.4f}")