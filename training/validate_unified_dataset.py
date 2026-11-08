
from pathlib import Path

import pandas as pd

from training.adapters.pad_ufes_adapter import PADUFESAdapter


DATASET_DIR = Path(
    r"C:\Users\shiva\Downloads\DiagnoAI_Research_Starter\PAD-UFES-20"
)

CSV_PATH = Path("data/processed/unified_dataset.csv")


def main():
    print("Starting dataset quality checks...\n")

    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

    dataset = pd.read_csv(CSV_PATH)

    # Reuse the adapter's existing schema and image validation.
    adapter = PADUFESAdapter(DATASET_DIR)
    dataset = adapter.validate(dataset)

    print("1. Dataset structure")
    print("Records:", len(dataset))
    print("Columns:", len(dataset.columns))

    print("\n2. Missing values")
    print(dataset.isnull().sum())

    print("\n3. Duplicate image paths")
    duplicate_images = dataset["image_path"].duplicated().sum()
    print("Duplicates:", duplicate_images)

    print("\n4. Diagnosis distribution")
    print(dataset["diagnosis"].value_counts())

    print("\n5. Patient information")
    print("Unique patients:", dataset["patient_id"].nunique())
    print("Missing patient IDs:", dataset["patient_id"].isnull().sum())

    print("\n6. Unknown symptom information")

    known_columns = [
        column
        for column in dataset.columns
        if column.endswith("_known")
    ]

    for column in known_columns:
        unknown_count = (dataset[column] == 0).sum()
        print(f"{column}: {unknown_count} unknown records")

    print("\n7. Records per dataset")
    print(dataset["dataset_source"].value_counts())

    print("\nQuality checks completed.")
    print("Schema and image validation: PASSED")

    if duplicate_images:
        print("WARNING: Duplicate image paths detected.")

    if dataset["patient_id"].isnull().any():
        print("WARNING: Some records have no patient ID.")


if __name__ == "__main__":
    main()
