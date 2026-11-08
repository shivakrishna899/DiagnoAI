from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from diagnoai.constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_ENTROPY_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MARGIN_THRESHOLD,
    SYMPTOM_NAMES,
    SYMPTOM_FEATURE_NAMES,
)
from diagnoai.explainability import grad_cam_overlay, symptom_sensitivity
from diagnoai.model import DiagnoAIFusionModel


class ModelNotReadyError(RuntimeError):
    pass


class Predictor:
    def __init__(
        self,
        checkpoint_path: str | Path = "models/diagnoai.pt",
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model: DiagnoAIFusionModel | None = None

        self.classes: list[str] = []
        self.symptom_names: list[str] = list(SYMPTOM_NAMES)
        self.symptom_feature_names: list[str] = list(SYMPTOM_FEATURE_NAMES)

        self.temperature = 1.0
        self.validation_metrics: dict[str, Any] = {}

        self.transform = transforms.Compose([
            transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

        self.load_error: str | None = None
        self._load()

    @property
    def ready(self) -> bool:
        return self.model is not None

    def _load(self) -> None:
        if not self.checkpoint_path.is_file():
            self.load_error = (
                f"Trained checkpoint not found: {self.checkpoint_path}. "
                "Train a model before requesting predictions."
            )
            return

        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )

            self.classes = list(checkpoint["classes"])

            # New checkpoints must explicitly store the 12 input feature names.
            checkpoint_features = checkpoint.get("symptom_feature_names")

            if checkpoint_features is None:
                raise ValueError(
                    "Checkpoint does not contain symptom_feature_names. "
                    "Retrain the model using the updated 12-feature pipeline."
                )

            self.symptom_feature_names = list(checkpoint_features)

            if self.symptom_feature_names != SYMPTOM_FEATURE_NAMES:
                raise ValueError(
                    "Checkpoint symptom feature order does not match "
                    "the current application configuration."
                )

            self.symptom_names = list(SYMPTOM_NAMES)

            self.temperature = float(
                checkpoint.get("temperature", 1.0)
            )

            self.validation_metrics = dict(
                checkpoint.get("validation_metrics", {})
            )

            self.model = DiagnoAIFusionModel(
                num_classes=len(self.classes),
                num_symptoms=len(self.symptom_feature_names),
                pretrained_image=False,
            )

            self.model.load_state_dict(checkpoint["model_state_dict"])

            self.model.to(self.device)
            self.model.eval()

            self.load_error = None

        except Exception as exc:
            self.model = None
            self.load_error = f"Could not load checkpoint: {exc}"

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return (
            self.transform(image.convert("RGB"))
            .unsqueeze(0)
            .to(self.device)
        )

    @staticmethod
    def encode_symptoms(
        symptom_states: dict[str, str] | None,
    ) -> list[float]:
        """
        Convert six three-state symptom inputs into 12 model features.

        yes     -> [1, 1]
        no      -> [0, 1]
        unknown -> [0, 0]

        Missing symptoms default to unknown.
        """

        if symptom_states is None:
            symptom_states = {}

        if not isinstance(symptom_states, dict):
            raise TypeError(
                "symptom_states must be a dictionary."
            )

        invalid_names = sorted(
            set(symptom_states.keys()) - set(SYMPTOM_NAMES)
        )

        if invalid_names:
            raise ValueError(
                f"Unsupported symptom names: {', '.join(invalid_names)}"
            )

        allowed_states = {"yes", "no", "unknown"}

        feature_vector: list[float] = []

        for symptom in SYMPTOM_NAMES:
            state = str(
                symptom_states.get(symptom, "unknown")
            ).strip().lower()

            if state not in allowed_states:
                raise ValueError(
                    f"Invalid state '{state}' for symptom '{symptom}'. "
                    "Allowed values are: yes, no, unknown."
                )

            if state == "yes":
                feature_vector.extend([1.0, 1.0])

            elif state == "no":
                feature_vector.extend([0.0, 1.0])

            else:
                feature_vector.extend([0.0, 0.0])

        if len(feature_vector) != len(SYMPTOM_FEATURE_NAMES):
            raise RuntimeError(
                "Generated symptom vector has an unexpected feature count."
            )

        return feature_vector

    @staticmethod
    def _uncertainty(
        probabilities: np.ndarray,
    ) -> dict[str, Any]:

        probs = np.clip(
            probabilities.astype(np.float64),
            1e-12,
            1.0,
        )

        probs = probs / probs.sum()

        sorted_probs = np.sort(probs)[::-1]

        top_confidence = float(sorted_probs[0])

        margin = (
            float(sorted_probs[0] - sorted_probs[1])
            if len(sorted_probs) > 1
            else top_confidence
        )

        entropy = float(
            -(probs * np.log(probs)).sum()
            / np.log(max(len(probs), 2))
        )

        return {
            "top_confidence": round(top_confidence, 4),
            "top_two_margin": round(margin, 4),
            "normalized_entropy": round(entropy, 4),
            "uncertain": bool(
                top_confidence < DEFAULT_CONFIDENCE_THRESHOLD
                or margin < DEFAULT_MARGIN_THRESHOLD
                or entropy > DEFAULT_ENTROPY_THRESHOLD
            ),
        }

    def predict(
        self,
        image: Image.Image,
        symptom_states: dict[str, str] | None = None,
        top_k: int = 3,
        include_explanations: bool = True,
    ) -> dict[str, Any]:

        if self.model is None:
            raise ModelNotReadyError(
                self.load_error or "Model is not ready."
            )

        try:
            image = image.convert("RGB")
        except Exception as exc:
            raise ValueError(
                "Image could not be decoded."
            ) from exc

        # Convert user symptom responses into the 12-feature representation.
        symptom_vector = self.encode_symptoms(symptom_states)

        image_tensor = self.preprocess(image)

        symptom_tensor = torch.tensor(
            [symptom_vector],
            dtype=torch.float32,
            device=self.device,
        )

        self.model.eval()

        with torch.no_grad():
            logits = self.model(
                image_tensor,
                symptom_tensor,
            )[0]

            probabilities = torch.softmax(
                logits / max(self.temperature, 1e-3),
                dim=0,
            ).cpu().numpy()

        order = np.argsort(probabilities)[::-1][
            :max(1, min(int(top_k), len(self.classes)))
        ]

        uncertainty = self._uncertainty(probabilities)

        predictions = [
            {
                "rank": rank,
                "label": self.classes[int(index)],
                "probability": round(
                    float(probabilities[index]),
                    4,
                ),
            }
            for rank, index in enumerate(order, start=1)
        ]

        result: dict[str, Any] = {
            "predictions": predictions,
            "uncertainty": uncertainty,
            "temperature_scaled": True,
            "validation_metrics": self.validation_metrics,
            "disclaimer": (
                "Research prototype only. Not a diagnosis. "
                "Consult a qualified healthcare professional."
            ),
        }

        if uncertainty["uncertain"]:
            result["warning"] = (
                "The model output is uncertain. "
                "Do not interpret the top-ranked class as a diagnosis."
            )

        if include_explanations:
            target_index = int(order[0])

            result["grad_cam_png_base64"] = grad_cam_overlay(
                self.model,
                image_tensor,
                symptom_tensor,
                image,
                target_index,
            )

            result["symptom_sensitivity"] = symptom_sensitivity(
                model=self.model,
                image_tensor=image_tensor,
                symptom_values=symptom_vector,
                symptom_names=self.symptom_names,
                class_index=target_index,
                temperature=self.temperature,
            )

        return result


def image_from_bytes(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()

        return Image.open(
            io.BytesIO(data)
        ).convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            "Uploaded file is not a valid readable image."
        ) from exc