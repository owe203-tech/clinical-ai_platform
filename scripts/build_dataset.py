from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import pandas as pd



# -----------------------------
# Paths
# -----------------------------

REPORTS_DIR = PROJECT_ROOT / "data" / "raw" / "reports"
VIEW_LABELS_DIR = PROJECT_ROOT / "data" / "raw" / "view_labels"

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
combined_df = combined_df.merge(
    report_df[
        [
            "report_id",
            "indication",
            "findings",
            "impression",
        ]
    ],
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

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "model_manifest.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

model_df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved manifest to: {OUTPUT_PATH}")
print(model_df.shape)
print(model_df["split"].value_counts())
