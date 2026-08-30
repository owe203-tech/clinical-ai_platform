from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
ALPHA = 0.8

# -------------------------
# 1. LOAD VALIDATION SCORES
# -------------------------

image_df = pd.read_csv(RESULTS_DIR / "val_image_predictions.csv")
text_df = pd.read_csv(RESULTS_DIR / "val_text_predictions.csv")

df = image_df.merge(
    text_df[["report_id", "text_prob"]],
    on="report_id",
    how="inner",
    validate="one_to_one",
)

df["fusion_prob"] = (
    ALPHA * df["image_prob"]
    + (1 - ALPHA) * df["text_prob"]
)

# -------------------------
# 2. SPLIT POS / NEG
# -------------------------

pos = df[df["cardiomegaly"] == 1].copy()
neg = df[df["cardiomegaly"] == 0].copy()

print("Positive reports:", len(pos))
print("Negative reports:", len(neg))
print("Total pos-neg pairs:", len(pos) * len(neg))

# -------------------------
# 3. COUNT PAIRWISE RANKINGS
# -------------------------

repaired = []
broken = []
unchanged_correct = 0
unchanged_wrong = 0

for _, p in pos.iterrows():
    for _, n in neg.iterrows():

        image_correct = p["image_prob"] > n["image_prob"]
        fusion_correct = p["fusion_prob"] > n["fusion_prob"]

        # Ignore exact ties for simplicity
        if p["image_prob"] == n["image_prob"]:
            continue
        if p["fusion_prob"] == n["fusion_prob"]:
            continue

        if not image_correct and fusion_correct:
            repaired.append({
                "positive_report_id": p["report_id"],
                "negative_report_id": n["report_id"],
                "positive_image_prob": p["image_prob"],
                "negative_image_prob": n["image_prob"],
                "positive_text_prob": p["text_prob"],
                "negative_text_prob": n["text_prob"],
                "positive_fusion_prob": p["fusion_prob"],
                "negative_fusion_prob": n["fusion_prob"],
                "image_margin": p["image_prob"] - n["image_prob"],
                "fusion_margin": p["fusion_prob"] - n["fusion_prob"],
            })

        elif image_correct and not fusion_correct:
            broken.append({
                "positive_report_id": p["report_id"],
                "negative_report_id": n["report_id"],
                "positive_image_prob": p["image_prob"],
                "negative_image_prob": n["image_prob"],
                "positive_text_prob": p["text_prob"],
                "negative_text_prob": n["text_prob"],
                "positive_fusion_prob": p["fusion_prob"],
                "negative_fusion_prob": n["fusion_prob"],
                "image_margin": p["image_prob"] - n["image_prob"],
                "fusion_margin": p["fusion_prob"] - n["fusion_prob"],
            })

        elif image_correct and fusion_correct:
            unchanged_correct += 1

        else:
            unchanged_wrong += 1

repaired_df = pd.DataFrame(repaired)
broken_df = pd.DataFrame(broken)

# -------------------------
# 4. SUMMARY
# -------------------------

print("\nPairwise ranking changes:")
print("Repaired by fusion:", len(repaired_df))
print("Broken by fusion:", len(broken_df))
print("Stayed correct:", unchanged_correct)
print("Stayed wrong:", unchanged_wrong)

net_gain = len(repaired_df) - len(broken_df)

print("\nNet repaired pairs:", net_gain)

# -------------------------
# 5. TOP REPAIRS / BREAKS
# -------------------------

if not repaired_df.empty:
    repaired_df["margin_gain"] = (
        repaired_df["fusion_margin"]
        - repaired_df["image_margin"]
    )

    print("\nTop repaired rankings:")
    print(
        repaired_df.nlargest(10, "margin_gain")
        .to_string(index=False)
    )

if not broken_df.empty:
    broken_df["margin_loss"] = (
        broken_df["image_margin"]
        - broken_df["fusion_margin"]
    )

    print("\nTop broken rankings:")
    print(
        broken_df.nlargest(10, "margin_loss")
        .to_string(index=False)
    )

# -------------------------
# 6. SAVE
# -------------------------

repaired_df.to_csv(
    RESULTS_DIR / "val_pairwise_rankings_repaired.csv",
    index=False
)

broken_df.to_csv(
    RESULTS_DIR / "val_pairwise_rankings_broken.csv",
    index=False
)

print("\nSaved ranking analysis.")