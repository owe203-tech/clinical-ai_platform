from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# LOCKED CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "model_manifest.csv"
CHECKPOINT_PATH = (
    PROJECT_ROOT / "best_resnet18_finetuned_cardiomegaly.pt"
)

OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Selected using validation data.
# DO NOT change after looking at test results.
FUSION_ALPHA = 0.80

# Image-level operating threshold selected on validation.
# Kept here only for image-level threshold evaluation.
IMAGE_THRESHOLD = 0.06


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ============================================================
# LOAD MANIFEST
# ============================================================

df = pd.read_csv(MANIFEST_PATH)

train_image_df = df[
    df["split"] == "train"
].copy()

test_image_df = df[
    df["split"] == "test"
].copy()

print(f"Train image rows: {len(train_image_df)}")
print(f"Test image rows:  {len(test_image_df)}")


# ============================================================
# IMAGE DATASET
# ============================================================

class ChestXrayDataset(Dataset):

    def __init__(
        self,
        dataframe,
        images_dir,
        transform=None,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        image_id = row["image_id"]
        label = int(row["cardiomegaly"])

        image_path = (
            self.images_dir / f"{image_id}.png"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ============================================================
# LOCKED IMAGE PREPROCESSING
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


test_dataset = ChestXrayDataset(
    test_image_df,
    IMAGES_DIR,
    transform=image_transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
)


# ============================================================
# LOAD LOCKED IMAGE MODEL
# ============================================================

image_model = resnet18(weights=None)

num_features = image_model.fc.in_features

image_model.fc = nn.Linear(
    num_features,
    1,
)

image_model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )
)

image_model = image_model.to(device)
image_model.eval()

print(
    "Loaded locked unweighted "
    "fine-tuned ResNet-18."
)


# ============================================================
# IMAGE PREDICTIONS
# ============================================================

def collect_predictions(
    model,
    loader,
    device,
):

    model.eval()

    labels = []
    probabilities = []

    with torch.no_grad():

        for images, batch_labels in loader:

            images = images.to(device)

            logits = model(images)

            probs = torch.sigmoid(
                logits
            ).cpu().numpy().ravel()

            probabilities.extend(probs)

            labels.extend(
                batch_labels.numpy().ravel()
            )

    return (
        np.asarray(labels),
        np.asarray(probabilities),
    )


test_image_labels, test_image_probs = (
    collect_predictions(
        image_model,
        test_loader,
        device,
    )
)

assert (
    len(test_image_probs)
    == len(test_image_df)
)

print(
    f"Generated {len(test_image_probs)} "
    "test image predictions."
)


# ============================================================
# IMAGE-LEVEL RESULTS
# ============================================================

image_level_auroc = roc_auc_score(
    test_image_labels,
    test_image_probs,
)

image_level_pr_auc = average_precision_score(
    test_image_labels,
    test_image_probs,
)

print("\nIMAGE-LEVEL TEST PERFORMANCE")

print(
    f"AUROC:  {image_level_auroc:.4f}"
)

print(
    f"PR-AUC: {image_level_pr_auc:.4f}"
)


# ============================================================
# LOCKED IMAGE THRESHOLD
# ============================================================

image_predictions = (
    test_image_probs >= IMAGE_THRESHOLD
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    test_image_labels,
    image_predictions,
).ravel()

image_sensitivity = recall_score(
    test_image_labels,
    image_predictions,
)

image_specificity = (
    tn / (tn + fp)
)

image_precision = precision_score(
    test_image_labels,
    image_predictions,
    zero_division=0,
)

image_f1 = f1_score(
    test_image_labels,
    image_predictions,
    zero_division=0,
)

print(
    f"\nLocked image threshold: "
    f"{IMAGE_THRESHOLD:.2f}"
)

print(
    f"Sensitivity: {image_sensitivity:.4f}"
)

print(
    f"Specificity: {image_specificity:.4f}"
)

print(
    f"Precision:   {image_precision:.4f}"
)

print(
    f"F1:          {image_f1:.4f}"
)

print(
    f"Confusion matrix: "
    f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
)


# ============================================================
# IMAGE -> REPORT LEVEL
# ============================================================

image_prediction_df = pd.DataFrame({
    "report_id":
        test_image_df["report_id"].values,

    "image_prob":
        test_image_probs,

    "cardiomegaly":
        test_image_df[
            "cardiomegaly"
        ].astype(int).values,
})


test_report_image = (
    image_prediction_df
    .groupby(
        "report_id",
        as_index=False,
    )
    .agg({
        "image_prob": "mean",
        "cardiomegaly": "first",
    })
)


test_report_image.to_csv(
    OUTPUT_DIR /
    "test_image_predictions.csv",
    index=False,
)

print(
    f"\nReport-level image predictions: "
    f"{len(test_report_image)}"
)


# ============================================================
# TEXT DATA
# ============================================================

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

report_df["indication"] = (
    report_df["indication"]
    .fillna("")
)

train_text_df = report_df[
    report_df["split"] == "train"
].copy()

test_text_df = report_df[
    report_df["split"] == "test"
].copy()

print(
    f"\nTrain text reports: "
    f"{len(train_text_df)}"
)

print(
    f"Test text reports:  "
    f"{len(test_text_df)}"
)


# ============================================================
# LOCKED TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=10000,
)

X_train_text = (
    train_text_df["indication"]
)

X_test_text = (
    test_text_df["indication"]
)

y_train_text = (
    train_text_df[
        "cardiomegaly"
    ].astype(int)
)

y_test_text = (
    test_text_df[
        "cardiomegaly"
    ].astype(int)
)


X_train = vectorizer.fit_transform(
    X_train_text
)

# IMPORTANT:
# test is transformed only.
# We NEVER fit TF-IDF on test.
X_test = vectorizer.transform(
    X_test_text
)

print(
    f"TF-IDF features: "
    f"{X_train.shape[1]}"
)


# ============================================================
# LOCKED LOGISTIC REGRESSION
# ============================================================

text_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

text_model.fit(
    X_train,
    y_train_text,
)

test_text_probs = (
    text_model.predict_proba(
        X_test
    )[:, 1]
)


text_auroc = roc_auc_score(
    y_test_text,
    test_text_probs,
)

text_pr_auc = average_precision_score(
    y_test_text,
    test_text_probs,
)


print("\nTEXT-ONLY TEST PERFORMANCE")

print(
    f"AUROC:  {text_auroc:.4f}"
)

print(
    f"PR-AUC: {text_pr_auc:.4f}"
)


test_text_predictions = pd.DataFrame({
    "report_id":
        test_text_df["report_id"].values,

    "text_prob":
        test_text_probs,

    "cardiomegaly":
        y_test_text.values,
})


test_text_predictions.to_csv(
    OUTPUT_DIR /
    "test_text_predictions.csv",
    index=False,
)


# ============================================================
# MERGE MODALITIES
# ============================================================

final_df = test_report_image.merge(
    test_text_predictions[
        [
            "report_id",
            "text_prob",
            "cardiomegaly",
        ]
    ],
    on="report_id",
    how="inner",
    validate="one_to_one",
    suffixes=(
        "_image",
        "_text",
    ),
)


# Verify image/text labels agree
assert (
    final_df["cardiomegaly_image"]
    == final_df["cardiomegaly_text"]
).all()


final_df["cardiomegaly"] = (
    final_df[
        "cardiomegaly_image"
    ].astype(int)
)


final_df = final_df.drop(
    columns=[
        "cardiomegaly_image",
        "cardiomegaly_text",
    ]
)


print(
    f"\nMerged multimodal test reports: "
    f"{len(final_df)}"
)


# ============================================================
# LOCKED 80/20 FUSION
# ============================================================

final_df["fusion_prob"] = (
    FUSION_ALPHA
    * final_df["image_prob"]

    + (1.0 - FUSION_ALPHA)
    * final_df["text_prob"]
)


# ============================================================
# FINAL REPORT-LEVEL TEST METRICS
# ============================================================

y_test = (
    final_df[
        "cardiomegaly"
    ].values
)


def discrimination_metrics(
    y_true,
    probabilities,
):

    return {
        "auroc":
            roc_auc_score(
                y_true,
                probabilities,
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                probabilities,
            ),
    }


image_metrics = discrimination_metrics(
    y_test,
    final_df["image_prob"],
)

text_metrics = discrimination_metrics(
    y_test,
    final_df["text_prob"],
)

fusion_metrics = discrimination_metrics(
    y_test,
    final_df["fusion_prob"],
)


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame([
    {
        "model": "image_only",
        **image_metrics,
    },
    {
        "model": "text_only_tfidf",
        **text_metrics,
    },
    {
        "model": "fusion_80_image_20_text",
        **fusion_metrics,
    },
])


print(
    "\n================================"
)

print(
    "FINAL UNTOUCHED TEST RESULTS"
)

print(
    "================================"
)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# DELTAS
# ============================================================

auroc_delta = (
    fusion_metrics["auroc"]
    - image_metrics["auroc"]
)

pr_auc_delta = (
    fusion_metrics["pr_auc"]
    - image_metrics["pr_auc"]
)


print(
    "\nFusion vs image-only:"
)

print(
    f"AUROC delta:  "
    f"{auroc_delta:+.4f}"
)

print(
    f"PR-AUC delta: "
    f"{pr_auc_delta:+.4f}"
)


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_DIR /
    "final_test_metrics.csv",
    index=False,
)

final_df.to_csv(
    OUTPUT_DIR /
    "final_test_predictions.csv",
    index=False,
)


print("\nSaved:")

print(
    "results/test_image_predictions.csv"
)

print(
    "results/test_text_predictions.csv"
)

print(
    "results/final_test_predictions.csv"
)

print(
    "results/final_test_metrics.csv"
)