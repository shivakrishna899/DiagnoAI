from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from diagnoai.constants import DEFAULT_IMAGE_SIZE, SYMPTOM_NAMES, SYMPTOM_FEATURE_NAMES
from diagnoai.model import DiagnoAIFusionModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PairedImageSymptomsDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: Path,
        class_to_index: dict[str, int],
        augment: bool,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_root = image_root
        self.class_to_index = class_to_index

        image_transforms = [
            transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))
        ]
        if augment:
            image_transforms.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ])
        image_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])
        self.transform = transforms.Compose(image_transforms)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        path = Path(str(row["image_path"]))
        if not path.is_absolute():
            path = self.image_root / path

        with Image.open(path) as raw:
            image = self.transform(raw.convert("RGB"))

        symptoms = torch.tensor(
            [float(row[name]) for name in SYMPTOM_FEATURE_NAMES],
            dtype=torch.float32,
        )
        target = torch.tensor(
            self.class_to_index[str(row["diagnosis"])],
            dtype=torch.long,
        )
        return image, symptoms, target


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit one temperature using validation logits only."""
    if logits.ndim != 2 or labels.ndim != 1:
        raise ValueError("Invalid logits or labels shape.")
    if logits.shape[0] != labels.shape[0] or logits.shape[0] == 0:
        raise ValueError("Logits and labels must have matching non-empty rows.")

    log_temperature = torch.zeros(
        1, requires_grad=True, device=logits.device
    )
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.05, max_iter=50
    )
    criterion = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(
        torch.exp(log_temperature)
        .clamp(0.05, 20.0)
        .detach()
        .cpu()
        .item()
    )


def evaluate_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for images, symptoms, labels in loader:
            logits = model(images.to(device), symptoms.to(device))
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())

    if not all_logits:
        raise ValueError("Cannot evaluate an empty data split.")

    return (
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_labels, axis=0),
    )


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Calculate dataset-level weighted cross-entropy on a split."""
    model.eval()
    total_loss = 0.0
    total_weight = 0.0

    with torch.no_grad():
        for images, symptoms, labels in loader:
            images = images.to(device)
            symptoms = symptoms.to(device)
            labels = labels.to(device)

            logits = model(images, symptoms)

            # Sum weighted losses, then divide by the sum of target weights.
            # This avoids bias from averaging batches of different sizes.
            loss_sum = nn.functional.cross_entropy(
                logits,
                labels,
                weight=criterion.weight,
                reduction="sum",
            )
            total_loss += float(loss_sum.item())

            if criterion.weight is None:
                total_weight += float(labels.numel())
            else:
                total_weight += float(
                    criterion.weight[labels].sum().item()
                )

    if total_weight <= 0:
        raise ValueError("Cannot calculate validation loss for an empty split.")

    return total_loss / total_weight


def save_loss_history(
    history: list[dict[str, float]],
    output_path: Path,
) -> tuple[Path, Path | None]:
    """Save epoch losses as CSV and, when available, a PNG curve."""
    csv_path = output_path.with_suffix(".loss.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "training_loss",
                "validation_loss",
                "validation_accuracy",
                "validation_macro_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    png_path = output_path.with_suffix(".loss.png")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return csv_path, None

    epochs = [int(row["epoch"]) for row in history]
    train_losses = [row["training_loss"] for row in history]
    val_losses = [row["validation_loss"] for row in history]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, train_losses, marker="o", label="Training loss")
    axis.plot(epochs, val_losses, marker="o", label="Validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Weighted cross-entropy loss")
    axis.set_title("DiagnoAI E2: Training vs Validation Loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(png_path, dpi=160)
    plt.close(figure)

    return csv_path, png_path


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 10,
) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0

    for index in range(bins):
        if index == bins - 1:
            mask = (
                (confidence >= edges[index])
                & (confidence <= edges[index + 1])
            )
        else:
            mask = (
                (confidence >= edges[index])
                & (confidence < edges[index + 1])
            )
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean())
                - float(confidence[mask].mean())
            )
    return float(ece)


def compute_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    temperature: float = 1.0,
) -> dict[str, Any]:
    class_indices = list(range(len(classes)))
    scaled_logits = logits / max(float(temperature), 1e-3)
    scaled_logits -= scaled_logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(scaled_logits)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    one_hot = np.eye(len(classes), dtype=np.float64)[labels]

    nll = float(
        log_loss(labels, probabilities, labels=class_indices)
    )
    return {
        "rows": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(
            f1_score(
                labels,
                predicted,
                labels=class_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "negative_log_likelihood": nll,
        "brier_score_multiclass_sum": float(
            np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
        ),
        "top_label_ece_10_bins": expected_calibration_error(
            probabilities, labels
        ),
        "class_report": classification_report(
            labels,
            predicted,
            labels=class_indices,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
    }


def _group_holdout(
    indices: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    holdout_fraction: float,
    seed: int,
    classes: list[int],
    candidates: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Find a patient-disjoint split with all classes in both partitions."""
    if not 0.05 <= holdout_fraction <= 0.40:
        raise ValueError("Holdout fraction must be between 0.05 and 0.40.")

    local_y = y[indices]
    local_groups = groups[indices]
    splitter = GroupShuffleSplit(
        n_splits=candidates,
        test_size=holdout_fraction,
        random_state=seed,
    )
    overall = np.array([(local_y == cls).mean() for cls in classes])
    best: tuple[float, np.ndarray, np.ndarray] | None = None

    for local_train, local_holdout in splitter.split(
        indices, local_y, local_groups
    ):
        train_indices = indices[local_train]
        holdout_indices = indices[local_holdout]
        train_y = y[train_indices]
        holdout_y = y[holdout_indices]

        if any(not np.any(train_y == cls) for cls in classes):
            continue
        if any(not np.any(holdout_y == cls) for cls in classes):
            continue

        train_dist = np.array([(train_y == cls).mean() for cls in classes])
        holdout_dist = np.array([(holdout_y == cls).mean() for cls in classes])
        size_penalty = abs(
            len(holdout_indices) / len(indices) - holdout_fraction
        )
        score = (
            float(np.abs(train_dist - overall).mean())
            + float(np.abs(holdout_dist - overall).mean())
            + size_penalty
        )
        if best is None or score < best[0]:
            best = (score, train_indices, holdout_indices)

    if best is None:
        raise ValueError(
            "Could not create a patient-disjoint split containing every "
            "class in both partitions. Review rare-class patient counts "
            "or adjust split sizes."
        )
    return best[1], best[2]


def validate_split_indices(
    frame: pd.DataFrame,
    y: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    split_arrays = {
        "train": np.asarray(train_indices, dtype=np.int64),
        "validation": np.asarray(val_indices, dtype=np.int64),
        "test": np.asarray(test_indices, dtype=np.int64),
    }
    combined = np.concatenate(list(split_arrays.values()))

    if len(combined) != len(frame):
        raise ValueError("Split manifest does not cover every dataset row.")
    if len(np.unique(combined)) != len(frame):
        raise ValueError("Split manifest contains duplicate row assignments.")
    if set(combined.tolist()) != set(range(len(frame))):
        raise ValueError("Split manifest contains invalid or missing row indices.")

    groups = frame["patient_id"].astype(str).str.strip().to_numpy()
    classes = set(np.unique(y).tolist())
    group_sets: dict[str, set[str]] = {}

    for split_name, indices in split_arrays.items():
        if len(indices) == 0:
            raise ValueError(f"The {split_name} split is empty.")
        missing = classes - set(y[indices].tolist())
        if missing:
            raise ValueError(
                f"{split_name} split is missing classes: {sorted(missing)}"
            )
        group_sets[split_name] = set(groups[indices])

    names = list(group_sets)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            overlap = group_sets[first] & group_sets[second]
            if overlap:
                raise ValueError(
                    f"Patient leakage between {first} and {second}: "
                    f"{len(overlap)} overlapping patients."
                )


def create_splits(
    frame: pd.DataFrame,
    y: np.ndarray,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if (
        val_size <= 0
        or test_size <= 0
        or val_size >= 0.4
        or test_size >= 0.4
        or val_size + test_size >= 0.7
    ):
        raise ValueError(
            "Validation and test sizes must be between 0 and 0.4, "
            "and their sum must be below 0.7."
        )
    if "patient_id" not in frame.columns:
        raise ValueError("patient_id column is mandatory.")
    if frame["patient_id"].isna().any():
        raise ValueError("patient_id contains missing values.")

    groups = frame["patient_id"].astype(str).str.strip().to_numpy()
    if np.any(groups == ""):
        raise ValueError("patient_id contains empty values.")
    if len(np.unique(groups)) < 5:
        raise ValueError("At least five unique patients are required.")

    all_indices = np.arange(len(frame))
    classes = sorted(np.unique(y).tolist())

    # test_size is a fraction of the full dataset.
    # val_size is a fraction of the remaining train + validation records.
    trainval_indices, test_indices = _group_holdout(
        all_indices, y, groups, test_size, seed, classes
    )
    train_indices, val_indices = _group_holdout(
        trainval_indices, y, groups, val_size, seed + 1, classes
    )

    validate_split_indices(
        frame, y, train_indices, val_indices, test_indices
    )
    return (
        train_indices,
        val_indices,
        test_indices,
        "patient-disjoint grouped holdout",
    )


def save_split_manifest(
    path: Path,
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
    val_size: float,
    test_size: float,
) -> None:
    assignment: dict[int, str] = {}
    for split_name, indices in (
        ("train", train_indices),
        ("validation", val_indices),
        ("test", test_indices),
    ):
        for index in indices:
            assignment[int(index)] = split_name

    records = []
    for index, row in frame.reset_index(drop=True).iterrows():
        records.append({
            "image_path": str(row["image_path"]),
            "patient_id": str(row["patient_id"]),
            "diagnosis": str(row["diagnosis"]),
            "split": assignment[index],
        })

    payload = {
        "manifest_version": 1,
        "split_strategy": "patient-disjoint grouped holdout",
        "seed": int(seed),
        "validation_fraction_of_trainval": float(val_size),
        "test_fraction_of_full_dataset": float(test_size),
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split_manifest(
    path: Path,
    frame: pd.DataFrame,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise ValueError(f"Split manifest does not exist: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("manifest_version") != 1:
        raise ValueError("Unsupported split manifest version.")

    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Split manifest is missing its records list.")

    current = frame.reset_index(drop=True)
    current_paths = current["image_path"].astype(str).tolist()
    if len(set(current_paths)) != len(current_paths):
        raise ValueError("Current dataset contains duplicate image_path values.")

    manifest_by_path: dict[str, dict[str, str]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid record in split manifest.")
        required = {"image_path", "patient_id", "diagnosis", "split"}
        if not required.issubset(record):
            raise ValueError("A split manifest record is missing required fields.")

        image_path = str(record["image_path"])
        if image_path in manifest_by_path:
            raise ValueError(f"Duplicate image_path in split manifest: {image_path}")
        manifest_by_path[image_path] = record

    if set(manifest_by_path) != set(current_paths):
        missing = set(current_paths) - set(manifest_by_path)
        extra = set(manifest_by_path) - set(current_paths)
        raise ValueError(
            "Split manifest does not match the current dataset. "
            f"Missing records: {len(missing)}, extra records: {len(extra)}."
        )

    split_indices: dict[str, list[int]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for index, row in current.iterrows():
        record = manifest_by_path[str(row["image_path"])]
        if str(record["patient_id"]) != str(row["patient_id"]):
            raise ValueError(f"Patient ID changed for image: {row['image_path']}")
        if str(record["diagnosis"]) != str(row["diagnosis"]):
            raise ValueError(f"Diagnosis changed for image: {row['image_path']}")

        split_name = str(record["split"])
        if split_name not in split_indices:
            raise ValueError(f"Invalid split name '{split_name}' in manifest.")
        split_indices[split_name].append(index)

    train_indices = np.asarray(split_indices["train"], dtype=np.int64)
    val_indices = np.asarray(split_indices["validation"], dtype=np.int64)
    test_indices = np.asarray(split_indices["test"], dtype=np.int64)

    validate_split_indices(
        current, y, train_indices, val_indices, test_indices
    )
    return train_indices, val_indices, test_indices


def validate_training_data(
    frame: pd.DataFrame,
    parser: argparse.ArgumentParser,
) -> pd.DataFrame:
    required = {
        "image_path",
        "diagnosis",
        "patient_id",
        *SYMPTOM_FEATURE_NAMES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        parser.error(
            f"CSV is missing required columns: {', '.join(missing)}"
        )
    if frame.empty or frame["diagnosis"].nunique() < 2:
        parser.error(
            "Provide a non-empty dataset with at least two diagnosis classes."
        )

    essential = ["image_path", "diagnosis", "patient_id"]
    if frame[essential].isna().any().any():
        parser.error(
            "image_path, diagnosis and patient_id cannot contain missing values."
        )

    for column in essential:
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            parser.error(f"{column} contains empty values.")

    if frame["image_path"].duplicated().any():
        count = int(frame["image_path"].duplicated().sum())
        parser.error(f"image_path contains {count} duplicate entries.")

    for name in SYMPTOM_FEATURE_NAMES:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
        if frame[name].isna().any():
            parser.error(
                f"Symptom feature '{name}' contains missing or invalid values."
            )
        if not frame[name].isin([0, 1]).all():
            parser.error(f"Symptom feature '{name}' must contain only 0 or 1.")
        frame[name] = frame[name].astype(float)

    for symptom in SYMPTOM_NAMES:
        invalid_unknown = (
            (frame[f"{symptom}_known"] == 0)
            & (frame[symptom] != 0)
        )
        if invalid_unknown.any():
            parser.error(
                f"Unknown symptom '{symptom}' must have value 0."
            )
    return frame


def resolve_image_path(raw_path: str, image_root: Path) -> Path:
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = image_root / path
    return path


def validate_image_paths(
    frame: pd.DataFrame,
    image_root: Path,
    parser: argparse.ArgumentParser,
) -> None:
    missing_paths = []
    for raw_path in frame["image_path"]:
        path = resolve_image_path(str(raw_path), image_root)
        if not path.is_file():
            missing_paths.append(str(path))

    if missing_paths:
        parser.error(
            f"{len(missing_paths)} image files are missing. "
            f"Examples: {missing_paths[:5]}"
        )


def validate_image_integrity(
    frame: pd.DataFrame,
    image_root: Path,
    parser: argparse.ArgumentParser,
) -> None:
    """Decode-check all images before training to catch corrupt files early."""
    invalid_images = []
    for raw_path in frame["image_path"]:
        path = resolve_image_path(str(raw_path), image_root)
        try:
            with Image.open(path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError):
            invalid_images.append(str(path))

    if invalid_images:
        parser.error(
            f"{len(invalid_images)} image files are unreadable or corrupted. "
            f"Examples: {invalid_images[:5]}"
        )


def print_split_report(
    frame: pd.DataFrame,
    classes: list[str],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    total = len(frame)
    print("\nSplit sizes:")
    for name, indices in (
        ("Training", train_indices),
        ("Validation", val_indices),
        ("Testing", test_indices),
    ):
        subset = frame.iloc[indices]
        percentage = len(indices) / total * 100
        print(
            f"{name}: {len(indices)} records ({percentage:.2f}%), "
            f"{subset['patient_id'].nunique()} patients"
        )

    print("\nClass distribution:")
    for name, indices in (
        ("Training", train_indices),
        ("Validation", val_indices),
        ("Testing", test_indices),
    ):
        counts = (
            frame.iloc[indices]["diagnosis"]
            .value_counts()
            .reindex(classes, fill_value=0)
        )
        print(f"{name}: {counts.to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DiagnoAI E2 using effective-number class weighting."
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--out", default="models/diagnoai.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--beta", type=float, default=0.999,
        help="Beta for effective-number class weighting (0 <= beta < 1).",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Initialize image encoder using ImageNet pretrained weights.",
    )
    parser.add_argument(
        "--split-manifest",
        type=str,
        default=None,
        help=(
            "Existing split manifest to reuse. If omitted, a new split is "
            "generated and saved beside the model checkpoint."
        ),
    )

    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("Epochs, batch size and learning rate must be positive.")
    if args.val_size <= 0 or args.test_size <= 0:
        parser.error("Validation and test sizes must be positive.")
    if not 0.0 <= args.beta < 1.0:
        parser.error("Beta must be >= 0 and < 1.")

    set_seed(args.seed)

    csv_path = Path(args.csv).resolve()
    if not csv_path.is_file():
        parser.error(f"Dataset CSV does not exist: {csv_path}")

    frame = pd.read_csv(csv_path)
    frame = validate_training_data(frame, parser)
    image_root = Path(args.image_root).resolve()

    validate_image_paths(frame, image_root, parser)
    print("Image path check passed.")
    print("Checking image integrity. This may take a few minutes...")
    validate_image_integrity(frame, image_root, parser)
    print("Image integrity check passed.")

    classes = sorted(frame["diagnosis"].unique().tolist())
    class_to_index = {name: index for index, name in enumerate(classes)}
    y = frame["diagnosis"].map(class_to_index).to_numpy(dtype=np.int64)

    out_path = Path(args.out).resolve()
    manifest_path = (
        Path(args.split_manifest).resolve()
        if args.split_manifest
        else out_path.with_suffix(".splits.json")
    )

    if args.split_manifest:
        try:
            train_indices, val_indices, test_indices = load_split_manifest(
                manifest_path, frame, y
            )
        except (ValueError, json.JSONDecodeError) as exc:
            parser.error(f"Could not load split manifest: {exc}")
        split_name = "patient-disjoint grouped holdout (reused manifest)"
        print(f"Reusing split manifest: {manifest_path}")
    else:
        try:
            (
                train_indices,
                val_indices,
                test_indices,
                split_name,
            ) = create_splits(
                frame, y, args.val_size, args.test_size, args.seed
            )
        except ValueError as exc:
            parser.error(str(exc))

        save_split_manifest(
            manifest_path,
            frame,
            train_indices,
            val_indices,
            test_indices,
            args.seed,
            args.val_size,
            args.test_size,
        )
        print(f"Saved split manifest: {manifest_path}")

    train_frame = frame.iloc[train_indices].copy()
    val_frame = frame.iloc[val_indices].copy()
    test_frame = frame.iloc[test_indices].copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Classes: {classes}")
    print(f"Split strategy: {split_name}")
    print_split_report(
        frame, classes, train_indices, val_indices, test_indices
    )

    train_set = PairedImageSymptomsDataset(
        train_frame, image_root, class_to_index, augment=True
    )
    val_set = PairedImageSymptomsDataset(
        val_frame, image_root, class_to_index, augment=False
    )
    test_set = PairedImageSymptomsDataset(
        test_frame, image_root, class_to_index, augment=False
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    print("\nInitializing model...")
    model = DiagnoAIFusionModel(
        num_classes=len(classes),
        num_symptoms=len(SYMPTOM_FEATURE_NAMES),
        pretrained_image=args.pretrained,
    ).to(device)

    class_counts = np.bincount(
        train_frame["diagnosis"].map(class_to_index).to_numpy(),
        minlength=len(classes),
    )
    # Class-balanced effective-number weighting (Cui et al., CVPR 2019).
    # E_n = (1 - beta^n) / (1 - beta), so the weight is proportional to
    # (1 - beta) / (1 - beta^n). Normalize weights to mean 1 for a stable
    # overall loss scale. This changes loss emphasis, not sample diversity.
    beta = float(args.beta)
    effective_weights = (1.0 - beta) / (
        1.0 - np.power(beta, np.maximum(class_counts, 1))
    )
    class_weights = effective_weights / effective_weights.mean()

    print("\nTraining class weights:")
    print({
        class_name: round(float(class_weights[index]), 4)
        for index, class_name in enumerate(classes)
    })

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights, dtype=torch.float32, device=device
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_macro_f1 = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    best_validation_loss = float("inf")
    best_validation_loss_epoch = 0
    epoch_history: list[dict[str, float]] = []

    print("\nStarting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        seen = 0

        for images, symptoms, labels in train_loader:
            images = images.to(device)
            symptoms = symptoms.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images, symptoms)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()

            batch_weight = (
                float(criterion.weight[labels].sum().item())
                if criterion.weight is not None
                else float(labels.size(0))
            )
            total_loss += float(loss.item()) * batch_weight
            seen += batch_weight

        val_logits, val_labels = evaluate_logits(
            model, val_loader, device
        )
        val_metrics = compute_metrics(val_logits, val_labels, classes)
        average_loss = total_loss / max(seen, 1)
        validation_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
            device,
        )
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_history.append({
            "epoch": float(epoch + 1),
            "training_loss": float(average_loss),
            "validation_loss": float(validation_loss),
            "validation_accuracy": float(val_metrics["accuracy"]),
            "validation_macro_f1": float(val_metrics["macro_f1"]),
        })

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"| train_loss={average_loss:.4f} "
            f"| val_loss={validation_loss:.4f} "
            f"| val_accuracy={val_metrics['accuracy']:.4f} "
            f"| val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"| lr={current_lr:.6g}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_validation_loss_epoch = epoch + 1

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        scheduler.step()

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint.")

    model.load_state_dict(best_state)
    model.to(device).eval()

    # Temperature calibration uses validation data only.
    val_logits, val_labels = evaluate_logits(model, val_loader, device)
    temperature = fit_temperature(
        torch.tensor(val_logits, dtype=torch.float32, device=device),
        torch.tensor(val_labels, dtype=torch.long, device=device),
    )
    validation_metrics = compute_metrics(
        val_logits, val_labels, classes, temperature
    )

    # E2 intentionally does not evaluate the held-out test set.
    # The test split is preserved in the original manifest for a later,
    # final evaluation after all experiment decisions are frozen.

    loss_csv_path, loss_png_path = save_loss_history(
        epoch_history,
        out_path,
    )

    split_summary = {
        "strategy": split_name,
        "seed": args.seed,
        "train_rows": len(train_set),
        "validation_rows": len(val_set),
        "test_rows": len(test_set),
        "train_patients": int(train_frame["patient_id"].nunique()),
        "validation_patients": int(val_frame["patient_id"].nunique()),
        "test_patients": int(test_frame["patient_id"].nunique()),
        "train_class_counts": {
            str(k): int(v)
            for k, v in train_frame["diagnosis"].value_counts().items()
        },
        "validation_class_counts": {
            str(k): int(v)
            for k, v in val_frame["diagnosis"].value_counts().items()
        },
        "test_class_counts": {
            str(k): int(v)
            for k, v in test_frame["diagnosis"].value_counts().items()
        },
        "manifest_path": str(manifest_path),
    }

    checkpoint = {
        "model_state_dict": best_state,
        "classes": classes,
        "symptom_names": list(SYMPTOM_NAMES),
        "symptom_feature_names": list(SYMPTOM_FEATURE_NAMES),
        "temperature": temperature,
        "validation_metrics": validation_metrics,
        "split_summary": split_summary,
        "image_size": DEFAULT_IMAGE_SIZE,
        "model_architecture": "resnet18_late_fusion_v1",
        "best_epoch": best_epoch,
        "best_epoch_selection_metric": "validation_macro_f1",
        "best_validation_loss": best_validation_loss,
        "best_validation_loss_epoch": best_validation_loss_epoch,
        "epoch_history": epoch_history,
        "seed": args.seed,
        "pretrained_image": args.pretrained,
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR",
            "loss": "effective-number class-weighted CrossEntropyLoss",
            "class_weighting": "effective_number",
            "beta": beta,
            "class_weights": {
                str(class_name): float(class_weights[index])
                for index, class_name in enumerate(classes)
            },
            "gradient_clip_norm": 1.0,
            "selection_metric": "validation_macro_f1",
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_path)

    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_payload = {
        "experiment": "E2",
        "model_architecture": "resnet18_late_fusion_v1",
        "best_epoch": best_epoch,
        "best_epoch_selection_metric": "validation_macro_f1",
        "best_validation_loss": best_validation_loss,
        "best_validation_loss_epoch": best_validation_loss_epoch,
        "epoch_history": epoch_history,
        "loss_history_csv": str(loss_csv_path),
        "loss_curve_png": str(loss_png_path) if loss_png_path else None,
        "temperature_fitted_on": "validation only",
        "final_test_used_for_model_selection_or_calibration": False,
        "temperature": temperature,
        "split_summary": split_summary,
        "validation_metrics": validation_metrics,
        "held_out_test_evaluated": False,
        "class_weighting": "effective_number",
        "beta": beta,
        "class_weights": {
            str(class_name): float(class_weights[index])
            for index, class_name in enumerate(classes)
        },
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\nTraining completed.")
    print(f"Best epoch: {best_epoch}")
    print(f"Saved checkpoint: {out_path}")
    print(f"Saved evaluation report: {metrics_path}")
    print(f"Saved split manifest: {manifest_path}")
    print(f"Saved loss history: {loss_csv_path}")
    if loss_png_path is not None:
        print(f"Saved loss curve: {loss_png_path}")
    else:
        print("Matplotlib not installed, loss curve PNG was skipped.")
    print(
        f"Lowest validation loss: {best_validation_loss:.4f} "
        f"(epoch {best_validation_loss_epoch})"
    )
    print(f"Validation temperature: {temperature:.4f}")
    print("Held-out test set was NOT evaluated in E2.")
    print(
        "These are dataset-specific research metrics, "
        "not evidence of clinical performance."
    )


if __name__ == "__main__":
    main()
