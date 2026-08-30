from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_auc_score



# -----------------------------
# Paths
# -----------------------------

REPORTS_DIR = Path("data/raw/reports")
IMAGES_DIR = Path("data/raw/images")
VIEW_LABELS_DIR = Path("data/raw/view_labels")

xml_files = list(REPORTS_DIR.rglob("*.xml"))


# -----------------------------
# Parse reports
# -----------------------------

def parse_report(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    report_id = root.find("uId").attrib["id"]

    findings_element = root.find(".//AbstractText[@Label='FINDINGS']")
    indication_element = root.find(".//AbstractText[@Label='INDICATION']")
    impression_element = root.find(".//AbstractText[@Label='IMPRESSION']")

    image_ids = []

    for image_element in root.findall(".//parentImage"):
        image_ids.append(image_element.attrib["id"])

    return {
        "report_id": report_id,
        "findings": findings_element.text if findings_element is not None else None,
        "indication": indication_element.text if indication_element is not None else None,
        "impression": impression_element.text if impression_element is not None else None,
        "image_ids": image_ids,
    }


all_reports_data = []

for xml_file in xml_files:
    all_reports_data.append(parse_report(xml_file))

report_df = pd.DataFrame(all_reports_data)


# -----------------------------
# View labels
# -----------------------------

frontal_labels = pd.read_csv(
    VIEW_LABELS_DIR / "frontal_final.csv",
    header=None,
    names=["image_id"],
)

lateral_labels = pd.read_csv(
    VIEW_LABELS_DIR / "lateral_final.csv",
    header=None,
    names=["image_id"],
)

frontal_labels["image_id"] = "CXR" + frontal_labels["image_id"]
lateral_labels["image_id"] = "CXR" + lateral_labels["image_id"]

frontal_image_ids = set(frontal_labels["image_id"])
lateral_image_ids = set(lateral_labels["image_id"])


# -----------------------------
# Build image-level DataFrame
# -----------------------------

image_rows = []

for report in all_reports_data:
    for image_id in report["image_ids"]:

        if image_id in frontal_image_ids:
            view = "frontal"
        elif image_id in lateral_image_ids:
            view = "lateral"
        else:
            view = "unknown"

        image_rows.append({
            "report_id": report["report_id"],
            "image_id": image_id,
            "view": view,
        })

image_df = pd.DataFrame(image_rows)


# -----------------------------
# MeSH labels
# -----------------------------

report_mesh_data = []

for xml_file in xml_files:
    tree = ET.parse(xml_file)
    root = tree.getroot()

    report_id = root.find("uId").attrib["id"]
    mesh_terms = set()

    for major in root.findall(".//major"):
        if major.text is not None:
            normalized_term = major.text.split("/")[0]
            mesh_terms.add(normalized_term)

    report_mesh_data.append({
        "report_id": report_id,
        "mesh_terms": mesh_terms,
    })

report_mesh_df = pd.DataFrame(report_mesh_data)

report_mesh_df["cardiomegaly"] = report_mesh_df["mesh_terms"].apply(
    lambda terms: "Cardiomegaly" in terms
)


# -----------------------------
# Modeling DataFrame
# -----------------------------

combined_df = image_df.merge(
    report_mesh_df,
    on="report_id",
    how="left",
)

frontal_df = combined_df[
    combined_df["view"] == "frontal"
].copy()


# -----------------------------
# Train / validation / test split
# -----------------------------

report_split_df = frontal_df[
    ["report_id", "cardiomegaly"]
].drop_duplicates()

train_reports, temp_reports = train_test_split(
    report_split_df,
    test_size=0.30,
    stratify=report_split_df["cardiomegaly"],
    random_state=42,
)

val_reports, test_reports = train_test_split(
    temp_reports,
    test_size=0.50,
    stratify=temp_reports["cardiomegaly"],
    random_state=42,
)

train_reports = train_reports.copy()
val_reports = val_reports.copy()
test_reports = test_reports.copy()

train_reports["split"] = "train"
val_reports["split"] = "val"
test_reports["split"] = "test"

split_df = pd.concat(
    [train_reports, val_reports, test_reports]
)

model_df = frontal_df.merge(
    split_df[["report_id", "split"]],
    on="report_id",
    how="left",
)

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

