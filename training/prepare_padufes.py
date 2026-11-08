from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from diagnoai.constants import SYMPTOM_NAMES

SOURCE_SYMPTOM_MAP = {
    "itching": "itch",
    "growth": "grew",
    "pain": "hurt",
    "appearance_change": "changed",
    "bleeding": "bleed",
    "elevation": "elevation",
}
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def normalize_binary(value: object) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return int(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "1.0", "true", "t", "yes", "y"}:
        return 1
    if normalized in {"0", "0.0", "false", "f", "no", "n"}:
        return 0
    return None


def build_image_index(images_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in images_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            index.setdefault(path.name.lower(), path)
    return index


def locate_image(row: pd.Series, images_root: Path, image_index: dict[str, Path]) -> Path | None:
    candidates: list[str] = []
    for column in ("img_path", "img_id", "image_id", "image_path"):
        if column in row.index and pd.notna(row[column]):
            value = str(row[column]).strip()
            if value:
                candidates.append(value)
    for candidate in candidates:
        raw = Path(candidate)
        direct = raw if raw.is_absolute() else images_root / raw
        if direct.is_file():
            return direct
        basename = raw.name
        indexed = image_index.get(basename.lower())
        if indexed:
            return indexed
        if not raw.suffix:
            for extension in ALLOWED_EXTENSIONS:
                indexed = image_index.get(f"{basename}{extension}".lower())
                if indexed:
                    return indexed
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Map PAD-UFES-20 metadata to DiagnoAI's paired training CSV schema.")
    parser.add_argument("--metadata", required=True, help="Path to the PAD-UFES-20 metadata CSV")
    parser.add_argument("--images-root", required=True, help="Root folder containing extracted image files")
    parser.add_argument("--out", default="data/padufes_paired.csv", help="Output CSV path")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    images_root = Path(args.images_root)
    if not metadata_path.is_file():
        parser.error(f"Metadata CSV not found: {metadata_path}")
    if not images_root.is_dir():
        parser.error(f"Image folder not found: {images_root}")

    frame = pd.read_csv(metadata_path)
    required = {"diagnostic", "patient_id", *SOURCE_SYMPTOM_MAP.values()}
    missing = sorted(required - set(frame.columns))
    if missing:
        parser.error(f"Metadata is missing required PAD-UFES-20 fields: {', '.join(missing)}")
    image_index = build_image_index(images_root)
    if not image_index:
        parser.error(f"No supported images found under {images_root}")

    rows: list[dict[str, object]] = []
    no_image = 0
    incomplete_symptoms = 0
    unsupported_labels = 0
    # Some copies label Bowen's disease as BOW. This scaffold maps BOW to SCC,
    # consistent with dataset summaries that group it under SCC/SCC in situ.
    label_map = {"BOW": "SCC", "BOD": "SCC"}
    allowed_diagnoses = {"ACK", "BCC", "MEL", "NEV", "SCC", "SEK"}

    for _, row in frame.iterrows():
        image_path = locate_image(row, images_root, image_index)
        if image_path is None:
            no_image += 1
            continue
        symptoms: dict[str, int] = {}
        invalid_symptom = False
        for model_name, source_name in SOURCE_SYMPTOM_MAP.items():
            value = normalize_binary(row[source_name])
            if value is None:
                invalid_symptom = True
                break
            symptoms[model_name] = value
        if invalid_symptom:
            incomplete_symptoms += 1
            continue
        diagnosis = str(row["diagnostic"]).strip().upper()
        diagnosis = label_map.get(diagnosis, diagnosis)
        if diagnosis not in allowed_diagnoses:
            unsupported_labels += 1
            continue
        try:
            relative_path = image_path.relative_to(images_root).as_posix()
        except ValueError:
            relative_path = str(image_path)
        output_row: dict[str, object] = {
            "image_path": relative_path,
            "diagnosis": diagnosis,
            "patient_id": str(row["patient_id"]).strip(),
        }
        output_row.update(symptoms)
        rows.append(output_row)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows, columns=["image_path", "diagnosis", "patient_id", *SYMPTOM_NAMES])
    result.to_csv(output, index=False)
    print(f"Input metadata rows: {len(frame)}")
    print(f"Paired usable rows: {len(result)}")
    print(f"Skipped rows, image not found: {no_image}")
    print(f"Skipped rows, missing/invalid symptom field: {incomplete_symptoms}")
    print(f"Skipped rows, unsupported diagnosis label: {unsupported_labels}")
    print("Diagnosis counts:")
    if not result.empty:
        print(result["diagnosis"].value_counts().to_string())
    print(f"Saved: {output}")
    if result.empty:
        raise SystemExit("No usable rows were produced. Check the image folder and metadata column names.")


if __name__ == "__main__":
    main()
