from pathlib import Path
import pandas as pd


MANIFEST_PATH = Path("data/processed/model_manifest.csv")

df = pd.read_csv(MANIFEST_PATH)


# One row per report for text analysis
report_df = df[
    [
        "report_id",
        "split",
        "cardiomegaly",
        "indication",
        "findings",
        "impression",
    ]
].drop_duplicates(subset="report_id")


print(f"Reports: {len(report_df)}")

print("\nMissing text:")
print(
    report_df[
        ["indication", "findings", "impression"]
    ].isna().sum()
)


# -----------------------------
# Indication examples
# -----------------------------

print("\nExample indications:")

examples = report_df[
    "indication"
].dropna().sample(
    min(20, report_df["indication"].notna().sum()),
    random_state=42,
)

for text in examples:
    print(f"- {text}")


# -----------------------------
# Potential target leakage
# -----------------------------

indication_text = (
    report_df["indication"]
    .fillna("")
    .str.lower()
)

leakage_terms = [
    "cardiomegaly",
    "cardiac enlargement",
    "enlarged heart",
    "enlarged cardiac",
]

contains_target_language = indication_text.apply(
    lambda text: any(
        term in text
        for term in leakage_terms
    )
)

print("\nIndications containing target-related language:")
print(contains_target_language.sum())

print(
    f"Percentage: "
    f"{contains_target_language.mean() * 100:.2f}%"
)


print("\nExamples containing target-related language:")

for text in report_df.loc[
    contains_target_language,
    "indication"
].head(20):
    print(f"- {text}")