from training.label_mapping import map_diagnosis
from pathlib import Path

import pandas as pd

from training.adapters.base import (
    DatasetAdapter,
    SYMPTOM_NAMES,
)


class PADUFESAdapter(DatasetAdapter):
    """
    Adapter for the PAD-UFES-20 dataset.

    Reads original metadata and images, then converts
    them into DiagnoAI's standardized dataset format.
    """

    SYMPTOM_MAPPING = {
        "itching": "itch",
        "growth": "grew",
        "pain": "hurt",
        "appearance_change": "changed",
        "bleeding": "bleed",
        "elevation": "elevation",
    }

    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.metadata_path = self.dataset_dir / "metadata.csv"

        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}"
            )

    @property
    def dataset_name(self) -> str:
        return "PAD-UFES-20"

    @staticmethod
    def encode_symptom(value):
        """
        Convert source symptom values into:
        (symptom_value, symptom_known)

        True  -> (1, 1)
        False -> (0, 1)
        UNK   -> (0, 0)
        """

        if pd.isna(value):
            return 0, 0

        normalized = str(value).strip().lower()

        if normalized in {"true", "1", "yes"}:
            return 1, 1

        if normalized in {"false", "0", "no"}:
            return 0, 1

        if normalized in {"unk", "unknown", "nan", ""}:
            return 0, 0

        raise ValueError(
            f"Unexpected symptom value: {value!r}"
        )

    def discover_images(self) -> dict[str, str]:
        """
        Discover image files recursively and map filenames
        to their absolute paths.
        """

        extensions = {".jpg", ".jpeg", ".png"}

        image_map = {}

        for path in self.dataset_dir.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in extensions:
                continue

            if path.name in image_map:
                raise ValueError(
                    f"Duplicate image filename found: {path.name}"
                )

            image_map[path.name] = str(path.resolve())

        return image_map

    def load(self) -> pd.DataFrame:
        """
        Load PAD-UFES-20 and return standardized records.
        """

        metadata = pd.read_csv(self.metadata_path)

        required_source_columns = [
            "img_id",
            "diagnostic",
            "patient_id",
            *self.SYMPTOM_MAPPING.values(),
        ]

        missing_columns = [
            column
            for column in required_source_columns
            if column not in metadata.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing PAD-UFES-20 metadata columns: "
                f"{missing_columns}"
            )

        image_map = self.discover_images()

        metadata["img_id"] = metadata["img_id"].astype(str)

        metadata["image_path"] = metadata["img_id"].map(image_map)

        # Keep only records whose images are available locally.
        records = metadata.dropna(subset=["image_path"]).copy()

        records["diagnosis"] = records["diagnostic"].apply(
            map_diagnosis
        )

        records["dataset_source"] = self.dataset_name

        standardized = pd.DataFrame()

        standardized["image_path"] = records["image_path"]
        standardized["diagnosis"] = records["diagnosis"].astype(str)
        standardized["patient_id"] = records["patient_id"]
        standardized["dataset_source"] = records["dataset_source"]

        # Encode all six symptoms.
        for symptom in SYMPTOM_NAMES:
            source_column = self.SYMPTOM_MAPPING[symptom]

            encoded = records[source_column].apply(
                self.encode_symptom
            )

            standardized[symptom] = encoded.apply(lambda x: x[0])
            standardized[f"{symptom}_known"] = encoded.apply(
                lambda x: x[1]
            )

        return standardized
