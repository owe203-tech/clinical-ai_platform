from pathlib import Path

import numpy as np
import pandas as pd
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
    "best_resnet18_weighted_finetuned_cardiomegaly.pt"
)


# -----------------------------
# Load validation split
# -----------------------------

model_df = pd.read_csv(MANIFEST_PATH)

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
# Recreate model architecture
# -----------------------------

model = resnet18(weights=None)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)


# -----------------------------
# Load weighted checkpoint
# -----------------------------

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Weighted fine-tuned checkpoint loaded.")


# -----------------------------
# Collect probabilities
# -----------------------------

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


# -----------------------------
# Threshold-independent metrics
# -----------------------------

auroc = roc_auc_score(
    all_labels,
    all_probs
)

pr_auc = average_precision_score(
    all_labels,
    all_probs
)

print(f"\nValidation AUROC: {auroc:.4f}")
print(f"Validation PR-AUC: {pr_auc:.4f}")


# -----------------------------
# Evaluate threshold 0.50
# -----------------------------

threshold = 0.50

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
    predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    predictions,
    zero_division=0
)

print("\nWeighted model at threshold 0.50:")
print(f"Sensitivity: {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"Precision:   {precision:.3f}")
print(f"Recall:      {recall:.3f}")
print(f"F1:          {f1:.3f}")

print({
    "tn": tn,
    "fp": fp,
    "fn": fn,
    "tp": tp,
})


# -----------------------------
# Threshold search
# -----------------------------

threshold_results = []

for threshold in np.arange(0.01, 1.00, 0.01):
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


threshold_df = pd.DataFrame(
    threshold_results
)


# -----------------------------
# Select operating threshold
# Rule:
# sensitivity >= 0.75,
# then maximize specificity
# -----------------------------

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
    OUTPUT_DIR / "weighted_thresholds.csv",
    index=False
)
print(
    "\nBest weighted threshold "
    "with sensitivity >= 0.75:"
)

print(best_threshold)