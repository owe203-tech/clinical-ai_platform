from pathlib import Path

import numpy as np
import pandas as pd

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
# Load predictions
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

print(fusion_df.head())
print(f"Fusion reports: {len(fusion_df)}")


# -----------------------------
# Check labels
# -----------------------------

print(
    fusion_df["cardiomegaly"].value_counts()
)


# -----------------------------
# Test fusion weights
# -----------------------------

results = []

for alpha in np.arange(0.0, 1.01, 0.1):

    fusion_prob = (
        alpha * fusion_df["image_prob"]
        +
        (1 - alpha) * fusion_df["text_prob"]
    )

    auroc = roc_auc_score(
        fusion_df["cardiomegaly"],
        fusion_prob,
    )

    pr_auc = average_precision_score(
        fusion_df["cardiomegaly"],
        fusion_prob,
    )

    results.append({
        "image_weight": alpha,
        "text_weight": 1 - alpha,
        "auroc": auroc,
        "pr_auc": pr_auc,
    })


results_df = pd.DataFrame(results)

print("\nFusion results:")
print(results_df)


# -----------------------------
# Best validation fusion
# -----------------------------

best_row = results_df.sort_values(
    "auroc",
    ascending=False,
).iloc[0]

print("\nBest fusion by validation AUROC:")
print(best_row)

print(
    results_df[
        results_df["image_weight"].isin(
            [0.0, 0.8, 1.0]
        )
    ]
)