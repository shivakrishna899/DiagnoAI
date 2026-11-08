
from pathlib import Path

import pandas as pd

from training.adapters.pad_ufes_adapter import PADUFESAdapter


# Dataset location
DATASET_DIR = Path(
    r"C:\Users\shiva\Downloads\DiagnoAI_Research_Starter\PAD-UFES-20"
)

# Output location
OUTPUT_PATH = Path("data/processed/unified_dataset.csv")


def main():
    print("Starting unified dataset preparation...")

    # Initialize the PAD-UFES-20 adapter
    adapter = PADUFESAdapter(DATASET_DIR)

    # Load and validate standardized records
    dataset = adapter.load_and_validate()

    # Create output directory if it doesn't exist
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save standardized dataset
    dataset.to_csv(OUTPUT_PATH, index=False)

    print("\nDataset preparation completed.")
    print(f"Dataset source: {adapter.dataset_name}")
    print(f"Total records: {len(dataset)}")
    print(f"Total columns: {len(dataset.columns)}")
    print(f"Unique patients: {dataset['patient_id'].nunique()}")

    print("\nDiagnosis distribution:")
    print(dataset["diagnosis"].value_counts())

    print(f"\nSaved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
