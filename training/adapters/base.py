
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


# Shared symptom definitions used across all datasets.
SYMPTOM_NAMES = [
    "itching",
    "growth",
    "pain",
    "appearance_change",
    "bleeding",
    "elevation",
]

# Every dataset must produce these columns.
STANDARD_COLUMNS = [
    "image_path",
    "diagnosis",
    "patient_id",
    "dataset_source",
]

# Each symptom has two columns:
# Its value (0 or 1) and whether that value is known (0 or 1).
SYMPTOM_COLUMNS = [
    column
    for symptom in SYMPTOM_NAMES
    for column in (symptom, f"{symptom}_known")
]

REQUIRED_COLUMNS = STANDARD_COLUMNS + SYMPTOM_COLUMNS


class DatasetAdapter(ABC):
    """
    Base interface for all DiagnoAI dataset adapters.
    """

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Return the name of the dataset."""
        pass

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Read the original dataset and return records
        using DiagnoAI's standardized schema.
        """
        pass

    def validate(self, records: pd.DataFrame) -> pd.DataFrame:
        """
        Validate standardized dataset records.
        """

        if records.empty:
            raise ValueError("Dataset contains no records.")

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in records.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        records = records[REQUIRED_COLUMNS].copy()

        # Essential fields cannot be missing.
        essential_columns = [
            "image_path",
            "diagnosis",
            "dataset_source",
        ]

        if records[essential_columns].isnull().any().any():
            raise ValueError(
                "Image path, diagnosis and dataset source "
                "cannot contain missing values."
            )

        # Patient IDs can be missing for datasets that do not
        # provide patient-level identifiers.

        records["image_path"] = records["image_path"].astype(str)
        records["diagnosis"] = records["diagnosis"].astype(str)
        records["dataset_source"] = records["dataset_source"].astype(str)

        # Check that symptom values and known indicators are binary.
        for column in SYMPTOM_COLUMNS:
            records[column] = pd.to_numeric(
                records[column], errors="raise"
            )

            if not records[column].isin([0, 1]).all():
                raise ValueError(
                    f"Column '{column}' must contain only 0 or 1."
                )

        # Unknown symptoms must use 0 as their placeholder value.
        for symptom in SYMPTOM_NAMES:
            unknown_mask = records[f"{symptom}_known"] == 0

            if not (records.loc[unknown_mask, symptom] == 0).all():
                raise ValueError(
                    f"Unknown '{symptom}' values must use 0 "
                    "as their placeholder."
                )

        # Verify that the referenced image files exist.
        missing_images = [
            path
            for path in records["image_path"]
            if not Path(path).is_file()
        ]

        if missing_images:
            raise FileNotFoundError(
                f"{len(missing_images)} image files were not found. "
                f"Example: {missing_images[0]}"
            )

        return records

    def load_and_validate(self) -> pd.DataFrame:
        """Load records and validate the standardized output."""
        records = self.load()
        return self.validate(records)
