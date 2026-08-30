from pathlib import Path

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_auc_score


IMAGES_DIR = Path("data/raw/images")
MANIFEST_PATH = Path("data/processed/model_manifest.csv")

model_df = pd.read_csv(MANIFEST_PATH)

train_df = model_df[model_df["split"] == "train"].copy()
val_df = model_df[model_df["split"] == "val"].copy()
test_df = model_df[model_df["split"] == "test"].copy()

# -----------------------------
# PyTorch Dataset
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
# Image preprocessing
# -----------------------------

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

train_dataset = ChestXrayDataset(
    train_df,
    IMAGES_DIR,
    transform=image_transform,
)

val_dataset = ChestXrayDataset(
    val_df,
    IMAGES_DIR,
    transform=image_transform,
)

test_dataset = ChestXrayDataset(
    test_df,
    IMAGES_DIR,
    transform=image_transform,
)


# -----------------------------
# DataLoaders
# -----------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
)


# -----------------------------
# ResNet-18 baseline
# -----------------------------

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)


# -----------------------------
# Loss, optimizer, device
# -----------------------------

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=0.001,
)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using device: {device}")

model = model.to(device)


# -----------------------------
# Training, Validation epochs
# -----------------------------

num_epochs = 10

best_val_auc = 0.0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")

    # Training
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_running_loss = 0.0

    all_val_labels = []
    all_val_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            val_running_loss += loss.item()

            probs = torch.sigmoid(logits)

            all_val_labels.extend(labels.cpu().numpy())
            all_val_probs.extend(probs.cpu().numpy())

    average_val_loss = val_running_loss / len(val_loader)

    val_auc = roc_auc_score(
        all_val_labels,
        all_val_probs
    )

    print(f"Train loss: {average_train_loss:.4f}")
    print(f"Validation loss: {average_val_loss:.4f}")
    print(f"Validation AUROC: {val_auc:.4f}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc

        torch.save(
            model.state_dict(),
            "best_resnet18_cardiomegaly.pt"
        )

        print(
            f"Saved new best model with AUROC: "
            f"{best_val_auc:.4f}"
        )

model.load_state_dict(
    torch.load(
        "best_resnet18_cardiomegaly.pt",
        map_location=device
    )
)

model = model.to(device)

