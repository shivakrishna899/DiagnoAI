from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from diagnoai.constants import DEFAULT_IMAGE_SIZE, SYMPTOM_FEATURE_NAMES
from diagnoai.model import DiagnoAIFusionModel


class ValidationAuditDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_root: Path, class_to_index: dict[str, int]):
        self.frame = frame.reset_index(drop=True)
        self.image_root = image_root
        self.class_to_index = class_to_index
        self.transform = transforms.Compose([
            transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = Path(str(row["image_path"]))
        if not path.is_absolute():
            path = self.image_root / path
        with Image.open(path) as raw:
            image = self.transform(raw.convert("RGB"))
        symptoms = torch.tensor([float(row[name]) for name in SYMPTOM_FEATURE_NAMES], dtype=torch.float32)
        target = self.class_to_index[str(row["diagnosis"])]
        return image, symptoms, target, index


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    required = {"model_state_dict", "classes", "symptom_feature_names", "temperature"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")
    if checkpoint["symptom_feature_names"] != list(SYMPTOM_FEATURE_NAMES):
        raise ValueError("Checkpoint symptom feature order differs from current constants.")
    return checkpoint


def load_validation_frame(csv_path: Path, manifest_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path).reset_index(drop=True)
    required = {"image_path", "diagnosis", "patient_id", *SYMPTOM_FEATURE_NAMES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset CSV is missing columns: {sorted(missing)}")
    if frame["image_path"].astype(str).duplicated().any():
        raise ValueError("Dataset contains duplicate image_path values.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != 1 or not isinstance(manifest.get("records"), list):
        raise ValueError("Unsupported or invalid split manifest.")

    by_path = {}
    for record in manifest["records"]:
        path = str(record["image_path"])
        if path in by_path:
            raise ValueError(f"Duplicate image_path in manifest: {path}")
        by_path[path] = record

    paths = frame["image_path"].astype(str).tolist()
    if set(paths) != set(by_path):
        raise ValueError("Split manifest does not match the current dataset CSV.")

    validation_indices = []
    patient_sets = {"train": set(), "validation": set(), "test": set()}
    for record in manifest["records"]:
        split = str(record["split"])
        if split not in patient_sets:
            raise ValueError(f"Unexpected split name: {split}")
        patient_sets[split].add(str(record["patient_id"]))

    for i, row in frame.iterrows():
        record = by_path[str(row["image_path"])]
        if str(record["patient_id"]) != str(row["patient_id"]):
            raise ValueError(f"Patient ID mismatch: {row['image_path']}")
        if str(record["diagnosis"]) != str(row["diagnosis"]):
            raise ValueError(f"Diagnosis mismatch: {row['image_path']}")
        if record["split"] == "validation":
            validation_indices.append(i)

    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = patient_sets[first] & patient_sets[second]
        if overlap:
            raise ValueError(f"Patient leakage between {first} and {second}: {len(overlap)} patients.")

    validation = frame.iloc[validation_indices].copy().reset_index(drop=True)
    if validation.empty:
        raise ValueError("Validation split is empty.")
    return validation


def main():
    parser = argparse.ArgumentParser(description="Audit DiagnoAI E1 on validation data only.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--out-dir", default="models/e1_validation_audit")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")

    csv_path = Path(args.csv).resolve()
    image_root = Path(args.image_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    manifest_path = Path(args.split_manifest).resolve()
    output_dir = Path(args.out_dir).resolve()
    for path in (csv_path, checkpoint_path, manifest_path):
        if not path.is_file():
            parser.error(f"File does not exist: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    checkpoint = load_checkpoint(checkpoint_path, device)
    classes = list(checkpoint["classes"])
    class_to_index = {name: i for i, name in enumerate(classes)}
    temperature = float(checkpoint["temperature"])
    if temperature <= 0:
        raise ValueError("Checkpoint temperature must be positive.")

    validation_frame = load_validation_frame(csv_path, manifest_path)
    unknown = set(validation_frame["diagnosis"].astype(str)) - set(classes)
    if unknown:
        raise ValueError(f"Unknown validation classes: {sorted(unknown)}")

    loader = DataLoader(
        ValidationAuditDataset(validation_frame, image_root, class_to_index),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = DiagnoAIFusionModel(
        num_classes=len(classes),
        num_symptoms=len(SYMPTOM_FEATURE_NAMES),
        pretrained_image=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logits_list, labels_list, indices_list = [], [], []
    with torch.no_grad():
        for images, symptoms, labels, indices in loader:
            output = model(images.to(device), symptoms.to(device))
            logits_list.append(output.cpu().numpy())
            labels_list.append(labels.numpy())
            indices_list.append(indices.numpy())

    logits = np.concatenate(logits_list)
    labels = np.concatenate(labels_list)
    indices = np.concatenate(indices_list)
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    matrix = confusion_matrix(labels, predictions, labels=list(range(len(classes))))
    report = classification_report(
        labels, predictions, labels=list(range(len(classes))),
        target_names=classes, output_dict=True, zero_division=0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_frame = pd.DataFrame(matrix, index=classes, columns=classes)
    matrix_frame.index.name = "actual_class"
    matrix_path = output_dir / "validation_confusion_matrix.csv"
    matrix_frame.to_csv(matrix_path)

    report_path = output_dir / "validation_classification_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    rows = []
    for i, dataset_index in enumerate(indices):
        source = validation_frame.iloc[int(dataset_index)]
        actual, predicted = int(labels[i]), int(predictions[i])
        item = {
            "image_path": str(source["image_path"]),
            "patient_id": str(source["patient_id"]),
            "actual_class": classes[actual],
            "predicted_class": classes[predicted],
            "correct": bool(actual == predicted),
            "confidence": float(confidence[i]),
        }
        for class_index, class_name in enumerate(classes):
            item[f"probability_{class_name}"] = float(probabilities[i, class_index])
        rows.append(item)

    predictions_frame = pd.DataFrame(rows)
    predictions_path = output_dir / "validation_predictions.csv"
    predictions_frame.to_csv(predictions_path, index=False)

    actual_counts = validation_frame["diagnosis"].value_counts().reindex(classes, fill_value=0)
    predicted_counts = pd.Series([classes[i] for i in predictions]).value_counts().reindex(classes, fill_value=0)

    print("\\nValidation audit completed. Test split was not evaluated.")
    print(f"Validation records: {len(validation_frame)}")
    print(f"Temperature: {temperature:.4f}")
    print("\\nActual vs predicted counts:")
    for name in classes:
        print(f"{name}: actual={int(actual_counts[name])}, predicted={int(predicted_counts[name])}")
    print("\\nConfusion matrix (rows=actual, columns=predicted):")
    print(matrix_frame.to_string())
    print("\\nPer-class validation results:")
    for name in classes:
        values = report[name]
        print(
            f"{name}: precision={values['precision']:.4f}, "
            f"recall={values['recall']:.4f}, f1={values['f1-score']:.4f}, "
            f"support={int(values['support'])}"
        )

    incorrect = predictions_frame.loc[~predictions_frame["correct"]]
    print(f"\\nMisclassified validation records: {len(incorrect)}")
    print(f"Misclassified SCC validation records: {int((incorrect['actual_class'] == 'SCC').sum())}")
    print(f"Misclassified MEL validation records: {int((incorrect['actual_class'] == 'MEL').sum())}")
    print("\\nSaved audit files:")
    print(matrix_path)
    print(report_path)
    print(predictions_path)


if __name__ == "__main__":
    main()
