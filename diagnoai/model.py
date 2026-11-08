from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class DiagnoAIFusionModel(nn.Module):
    """Image + structured-symptom fusion model for one fixed label set."""

    def __init__(
        self,
        num_classes: int,
        num_symptoms: int,
        pretrained_image: bool = False,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained_image else None
        image_model = resnet18(weights=weights)
        image_model.fc = nn.Identity()
        self.image_encoder = image_model

        self.symptom_encoder = nn.Sequential(
            nn.Linear(num_symptoms, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(512 + 32, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor, symptoms: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width]")
        if symptoms.ndim != 2:
            raise ValueError("symptoms must have shape [batch, num_symptoms]")
        image_features = self.image_encoder(images)
        symptom_features = self.symptom_encoder(symptoms.float())
        fused = torch.cat((image_features, symptom_features), dim=1)
        return self.fusion_head(fused)
