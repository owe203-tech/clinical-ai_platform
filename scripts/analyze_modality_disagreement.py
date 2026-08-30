from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

# -------------------------
# 1. LOAD VALIDATION PROBS
# -------------------------

image_df = pd.read_csv(RESULTS_DIR / "val_image_predictions.csv")
text_df = pd.read_csv(RESULTS_DIR / "val_text_predictions.csv")

print("Image reports:", len(image_df))
print("Text reports:", len(text_df))


# -------------------------
# 2. MERGE BY REPORT
# -------------------------

df = image_df.merge(
    text_df[["report_id", "text_prob"]],
    on="report_id",
    how="inner",
    validate="one_to_one",
)

print("Merged reports:", len(df))

# Safety check: one row per report
assert df["report_id"].is_unique

# Make sure labels are binary
assert set(df["cardiomegaly"].unique()).issubset({0, 1})


# -------------------------
# 3. CREATE FUSION SCORE
# -------------------------

# Validation-selected simple fusion from our previous experiment
ALPHA = 0.8

df["fusion_prob"] = (
    ALPHA * df["image_prob"]
    + (1 - ALPHA) * df["text_prob"]
)


# -------------------------
# 4. VERIFY METRICS
# -------------------------

y = df["cardiomegaly"].values

for name in ["image_prob", "text_prob", "fusion_prob"]:
    auc = roc_auc_score(y, df[name])
    pr_auc = average_precision_score(y, df[name])

    print(
        f"{name:12s} "
        f"AUROC={auc:.4f} "
        f"PR-AUC={pr_auc:.4f}"
    )


# -------------------------
# 5. MEASURE DISAGREEMENT
# -------------------------

# Absolute difference between modality probabilities.
# Large value = image and text disagree strongly.
df["modality_gap"] = np.abs(
    df["image_prob"] - df["text_prob"]
)

print("\nProbability disagreement:")
print(df["modality_gap"].describe())


# -------------------------
# 6. DOES TEXT MOVE THE
#    PREDICTION THE RIGHT WAY?
# -------------------------

# Distance from the true binary label.
# Smaller = probability moved closer to the truth.

df["image_error"] = np.abs(
    df["cardiomegaly"] - df["image_prob"]
)

df["fusion_error"] = np.abs(
    df["cardiomegaly"] - df["fusion_prob"]
)

df["fusion_improvement"] = (
    df["image_error"] - df["fusion_error"]
)

# > 0: fusion moved closer to truth
# < 0: fusion moved farther from truth

EPS = 1e-12

df["fusion_effect"] = np.select(
    [
        df["fusion_improvement"] > EPS,
        df["fusion_improvement"] < -EPS,
    ],
    [
        "helped",
        "hurt",
    ],
    default="unchanged",
)

print("\nFusion effect counts:")
print(df["fusion_effect"].value_counts())

print("\nFusion effect proportions:")
print(df["fusion_effect"].value_counts(normalize=True))


# -------------------------
# 7. ANALYZE POSITIVES VS
#    NEGATIVES SEPARATELY
# -------------------------

print("\nFusion effect by true label:")

effect_by_label = pd.crosstab(
    df["cardiomegaly"],
    df["fusion_effect"],
    normalize="index",
)

print(effect_by_label)


# -------------------------
# 8. STRONGEST HELP / HARM
# -------------------------

columns = [
    "report_id",
    "cardiomegaly",
    "image_prob",
    "text_prob",
    "fusion_prob",
    "modality_gap",
    "fusion_improvement",
]

print("\nTop 10 cases where text helped fusion:")
print(
    df.nlargest(10, "fusion_improvement")[columns]
    .to_string(index=False)
)

print("\nTop 10 cases where text hurt fusion:")
print(
    df.nsmallest(10, "fusion_improvement")[columns]
    .to_string(index=False)
)


# -------------------------
# 9. SAVE CASE-LEVEL RESULTS
# -------------------------

output_path = RESULTS_DIR / "val_modality_disagreement.csv"

df.sort_values(
    "modality_gap",
    ascending=False,
).to_csv(output_path, index=False)

print(f"\nSaved analysis to: {output_path}")
