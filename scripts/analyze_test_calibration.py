from pathlib import Path


import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
INPUT_PATH = RESULTS_DIR / "final_test_predictions.csv"

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD FROZEN TEST PREDICTIONS
# ============================================================

df = pd.read_csv(INPUT_PATH)

print(f"Test reports: {len(df)}")

y = df["cardiomegaly"].astype(int).values

models = {
    "image": df["image_prob"].values,
    "text": df["text_prob"].values,
    "fusion": df["fusion_prob"].values,
}

print(
    f"Positive prevalence: {y.mean():.4f}"
)


# ============================================================
# 1. PROBABILITY QUALITY
# ============================================================

calibration_summary = []

for name, probs in models.items():

    brier = brier_score_loss(
        y,
        probs,
    )

    ll = log_loss(
        y,
        probs,
    )

    calibration_summary.append({
        "model": name,
        "brier_score": brier,
        "log_loss": ll,
        "mean_predicted_probability":
            probs.mean(),
        "actual_prevalence":
            y.mean(),
    })


calibration_summary_df = pd.DataFrame(
    calibration_summary
)

print("\nCALIBRATION SUMMARY")
print(
    calibration_summary_df.to_string(
        index=False
    )
)


calibration_summary_df.to_csv(
    OUTPUT_DIR /
    "test_calibration_summary.csv",
    index=False,
)


# ============================================================
# 2. CALIBRATION BINS
# ============================================================

calibration_rows = []

for name, probs in models.items():

    fraction_positive, mean_predicted = (
        calibration_curve(
            y,
            probs,
            n_bins=10,
            strategy="quantile",
        )
    )

    for bin_index, (
        predicted,
        observed,
    ) in enumerate(
        zip(
            mean_predicted,
            fraction_positive,
        ),
        start=1,
    ):

        calibration_rows.append({
            "model": name,
            "bin": bin_index,
            "mean_predicted_probability":
                predicted,
            "observed_positive_fraction":
                observed,
        })


calibration_bins_df = pd.DataFrame(
    calibration_rows
)

calibration_bins_df.to_csv(
    OUTPUT_DIR /
    "test_calibration_bins.csv",
    index=False,
)

print(
    "\nSaved calibration bins."
)


# ============================================================
# 3. OPERATING POINTS
# ============================================================

# IMPORTANT:
# These thresholds are for CHARACTERIZATION ONLY.
#
# We are NOT selecting the best test threshold.
# We simply describe how the frozen probabilities behave
# at several predefined operating points.

thresholds = [
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
]

operating_rows = []

for name, probs in models.items():

    for threshold in thresholds:

        preds = (
            probs >= threshold
        ).astype(int)

        tn, fp, fn, tp = (
            confusion_matrix(
                y,
                preds,
            ).ravel()
        )

        sensitivity = recall_score(
            y,
            preds,
            zero_division=0,
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else np.nan
        )

        precision = precision_score(
            y,
            preds,
            zero_division=0,
        )

        f1 = f1_score(
            y,
            preds,
            zero_division=0,
        )

        operating_rows.append({
            "model": name,
            "threshold": threshold,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1": f1,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        })


operating_df = pd.DataFrame(
    operating_rows
)

operating_df.to_csv(
    OUTPUT_DIR /
    "test_operating_points.csv",
    index=False,
)


print("\nOPERATING POINTS")

print(
    operating_df.to_string(
        index=False
    )
)


# ============================================================
# 4. LOCKED IMAGE THRESHOLD
# ============================================================

# 0.06 was selected on VALIDATION,
# so evaluating it on test is legitimate.

LOCKED_IMAGE_THRESHOLD = 0.06

image_probs = models["image"]

preds = (
    image_probs
    >= LOCKED_IMAGE_THRESHOLD
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y,
    preds,
).ravel()

sensitivity = recall_score(
    y,
    preds,
)

specificity = (
    tn / (tn + fp)
)

precision = precision_score(
    y,
    preds,
    zero_division=0,
)

f1 = f1_score(
    y,
    preds,
    zero_division=0,
)


print(
    "\nLOCKED REPORT-LEVEL IMAGE "
    "THRESHOLD CHARACTERIZATION"
)

print(
    f"Threshold:   "
    f"{LOCKED_IMAGE_THRESHOLD:.2f}"
)

print(
    f"Sensitivity: {sensitivity:.4f}"
)

print(
    f"Specificity: {specificity:.4f}"
)

print(
    f"Precision:   {precision:.4f}"
)

print(
    f"F1:          {f1:.4f}"
)

print(
    f"TN={tn} FP={fp} "
    f"FN={fn} TP={tp}"
)


print("\nSaved:")
print(
    "results/test_calibration_summary.csv"
)
print(
    "results/test_calibration_bins.csv"
)
print(
    "results/test_operating_points.csv"
)