from pathlib import Path

import pytest



PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "model_manifest.csv"

if not MANIFEST_PATH.exists():
    pytest.skip(
        "Local model manifest not available.",
        allow_module_level=True,
    )

import pandas as pd

def load_manifest():
    assert MANIFEST_PATH.exists(), (
        f"Manifest not found at {MANIFEST_PATH}. "
        "Run scripts/build_dataset.py first."
    )
    return pd.read_csv(MANIFEST_PATH)


def test_required_columns_exist():
    df = load_manifest()

    required = {"report_id", "split", "cardiomegaly"}
    missing = required - set(df.columns)

    assert not missing, f"Missing required columns: {missing}"


def test_no_missing_report_ids_or_splits():
    df = load_manifest()

    assert df["report_id"].notna().all(), "Found missing report_id values."
    assert df["split"].notna().all(), "Found missing split values."


def test_only_expected_splits_exist():
    df = load_manifest()

    expected = {"train", "val", "test"}
    actual = set(df["split"].unique())

    assert actual == expected, (
        f"Expected splits {expected}, but found {actual}"
    )


def test_reports_do_not_cross_splits():
    df = load_manifest()

    splits_per_report = df.groupby("report_id")["split"].nunique()
    leaking_reports = splits_per_report[splits_per_report > 1]

    assert leaking_reports.empty, (
        "Data leakage detected: some report_ids appear in multiple splits:\n"
        f"{leaking_reports}"
    )


def test_labels_are_consistent_within_report():
    df = load_manifest()

    labels_per_report = df.groupby("report_id")["cardiomegaly"].nunique()
    inconsistent = labels_per_report[labels_per_report > 1]

    assert inconsistent.empty, (
        "Some reports have conflicting cardiomegaly labels:\n"
        f"{inconsistent}"
    )