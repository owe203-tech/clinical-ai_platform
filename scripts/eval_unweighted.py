from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
# -----------------------------
# Paths
# -----------------------------

IMAGES_DIR = Path("data/raw/images")
MANIFEST_PATH = Path("data/processed/model_manifest.csv")
CHECKPOINT_PATH = Path(
    "best_resnet18_finetuned_cardiomegaly.pt"
)


# -----------------------------
# Load validation split
# -----------------------------

model_df = pd.read_csv(MANIFEST_PATH)

train_df = model_df[
    model_df["split"] == "train"
].copy()

val_df = model_df[
    model_df["split"] == "val"
].copy()


# -----------------------------
# Dataset
# -----------------------------

class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, images_dir, transform=None):
        self.dataframe = dataframe
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        image_id = row["image_id"]
        label = row["cardiomegaly"]

        image_path = self.images_dir / f"{image_id}.png"

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# -----------------------------
# Preprocessing
# -----------------------------

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# -----------------------------
# Validation DataLoader
# -----------------------------

val_dataset = ChestXrayDataset(
    val_df,
    IMAGES_DIR,
    transform=image_transform,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
)


# -----------------------------
# Device
# -----------------------------

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using device: {device}")


# -----------------------------
# Recreate ResNet architecture
# -----------------------------

model = resnet18(weights=None)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)


# -----------------------------
# Load best fine-tuned checkpoint
# -----------------------------

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Unweighted fine-tuned checkpoint loaded.")
def evaluate_model(model, val_loader, device, threshold=0.5):
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()

            all_probs.extend(probs)
            all_labels.extend(labels.numpy().ravel())

    predictions = [
        1 if prob >= threshold else 0
        for prob in all_probs
    ]

    tn, fp, fn, tp = confusion_matrix(
        all_labels,
        predictions
    ).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    precision = precision_score(
        all_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        predictions
    )

    f1 = f1_score(
        all_labels,
        predictions
    )

    auroc = roc_auc_score(
        all_labels,
        all_probs
    )

    pr_auc = average_precision_score(
        all_labels,
        all_probs
    )

    return {
        "auroc": auroc,
        "pr_auc": pr_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
    }

unweighted_model = resnet18(weights=None)

num_features = unweighted_model.fc.in_features
unweighted_model.fc = nn.Linear(num_features, 1)

unweighted_model.load_state_dict(
    torch.load(
        "best_resnet18_finetuned_cardiomegaly.pt",
        map_location=device
    )
)

unweighted_model = unweighted_model.to(device)

thresholds = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
]

for threshold in thresholds:
    results = evaluate_model(
        model,
        val_loader,
        device,
        threshold=threshold,
    )

    print(f"\nThreshold: {threshold:.2f}")
    print(f"Sensitivity: {results['sensitivity']:.3f}")
    print(f"Specificity: {results['specificity']:.3f}")
    print(f"Precision:   {results['precision']:.3f}")
    print(f"F1:          {results['f1']:.3f}")

print("Unweighted:")
print(results)

def collect_predictions(model, loader, device):
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            all_labels.extend(
                labels.numpy().ravel()
            )

            all_probs.extend(
                probs.cpu().numpy().ravel()
            )

    return all_labels, all_probs

all_labels, all_probs = collect_predictions(
    model,
    val_loader,
    device
)
threshold_results = []

for threshold in np.arange(0.01, 0.51, 0.01):
    predictions = [
        1 if prob >= threshold else 0
        for prob in all_probs
    ]

    tn, fp, fn, tp = confusion_matrix(
        all_labels,
        predictions
    ).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    precision = precision_score(
        all_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        predictions,
        zero_division=0
    )

    threshold_results.append({
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    })

threshold_df = pd.DataFrame(threshold_results)

eligible = threshold_df[
    threshold_df["sensitivity"] >= 0.75
]

best_threshold = eligible.sort_values(
    "specificity",
    ascending=False
).iloc[0]

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

threshold_df.to_csv(
    OUTPUT_DIR / "unweighted_thresholds.csv",
    index=False
)

print(best_threshold)

summary_df = pd.DataFrame([
    {
        "model": "unweighted_finetuned_resnet18",
        "best_val_auroc": 0.8811,
        "pr_auc": 0.3944,
        "selected_threshold": 0.06,
        "sensitivity": 0.755,
        "specificity": 0.872,
        "precision": 0.356,
        "f1": 0.484,
    },
    {
        "model": "weighted_finetuned_resnet18",
        "best_val_auroc": 0.8668,
        "pr_auc": 0.3871,
        "selected_threshold": 0.49,
        "sensitivity": 0.776,
        "specificity": 0.807,
        "precision": 0.273,
        "f1": 0.404,
    },
    {
        "model": "regularized_finetuned_resnet18",
        "best_val_auroc": 0.8570,
        "pr_auc": None,
        "selected_threshold": None,
        "sensitivity": None,
        "specificity": None,
        "precision": None,
        "f1": None,
    },
])

summary_df.to_csv(
    "results/image_experiment_summary.csv",
    index=False
)

image_pred_df = pd.DataFrame({
    "report_id": val_df["report_id"].values,
    "image_prob": all_probs,
    "cardiomegaly": val_df["cardiomegaly"].astype(int).values,
})

report_image_probs = (
    image_pred_df
    .groupby("report_id", as_index=False)
    .agg({
        "image_prob": "mean",
        "cardiomegaly": "first",
    })
)

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

report_image_probs.to_csv(
    OUTPUT_DIR / "val_image_predictions.csv",
    index=False,
)

print("Saved validation image predictions.")

print(len(all_probs))
print(len(val_df))

print(len(report_image_probs))

train_eval_dataset = ChestXrayDataset(
    train_df,
    IMAGES_DIR,
    transform=image_transform,
)

train_eval_loader = DataLoader(
    train_eval_dataset,
    batch_size=32,
    shuffle=False,
)

train_labels, train_probs = collect_predictions(
    model,
    train_eval_loader,
    device,
)

train_image_pred_df = pd.DataFrame({
    "report_id": train_df["report_id"].values,
    "image_prob": train_probs,
    "cardiomegaly": train_df["cardiomegaly"].astype(int).values,
})

train_report_image_probs = (
    train_image_pred_df
    .groupby("report_id", as_index=False)
    .agg({
        "image_prob": "mean",
        "cardiomegaly": "first",
    })
)
train_report_image_probs.to_csv(
    "results/train_image_predictions.csv",
    index=False,
)
