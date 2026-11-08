from __future__ import annotations
from diagnoai.constants import DEFAULT_IMAGE_SIZE
import base64
import io
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def grad_cam_overlay(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    symptom_tensor: torch.Tensor,
    original: Image.Image,
    class_index: int,
) -> str:
    """
    Return a base64 PNG Grad-CAM overlay for a ResNet-18-backed model.

    The heatmap explains image regions contributing to the selected
    class score, conditioned on the supplied symptom inputs.
    """

    activations: list[torch.Tensor] = []

    target_layer = model.image_encoder.layer4[-1]

    hook = target_layer.register_forward_hook(
        lambda _module, _inputs, output: activations.append(output)
    )

    try:
        model.zero_grad(set_to_none=True)
        model.eval()

        image_tensor = (
            image_tensor.detach()
            .clone()
            .requires_grad_(True)
        )

        logits = model(
            image_tensor,
            symptom_tensor,
        )

        if not activations:
            raise RuntimeError(
                "Grad-CAM activation was not captured."
            )

        gradients = torch.autograd.grad(
            logits[0, class_index],
            activations[0],
            retain_graph=False,
        )[0]

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = torch.relu(
            (weights * activations[0]).sum(
                dim=1,
                keepdim=True,
            )
        )

        cam = F.interpolate(
            cam,
            size=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        cam = cam.detach().cpu().numpy()

        cam = (
            (cam - cam.min())
            / (cam.max() - cam.min() + 1e-8)
        )

        red = np.clip(1.5 * cam, 0, 1)

        green = np.clip(
            1.5 - np.abs(2 * cam - 1) * 1.5,
            0,
            1,
        )

        blue = np.clip(1.0 - 1.5 * cam, 0, 1)

        heat = (
            np.stack((red, green, blue), axis=-1) * 255
        ).astype(np.uint8)

        heat_img = Image.fromarray(
            heat,
            mode="RGB",
        ).resize(original.size)

        base = original.convert("RGB")

        overlay = Image.blend(
            base,
            heat_img,
            alpha=0.38,
        )

        buffer = io.BytesIO()

        overlay.save(
            buffer,
            format="PNG",
        )

        return base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")

    finally:
        hook.remove()


def symptom_sensitivity(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    symptom_values: Iterable[float],
    symptom_names: list[str],
    class_index: int,
    temperature: float,
    top_n: int = 5,
) -> list[dict[str, float | str]]:
    """
    Estimate sensitivity by changing each reported-present symptom
    from yes to no while preserving its known indicator.

    Positive score_drop means the selected class probability fell
    after that change.

    These are model sensitivities, not causal or medical explanations.
    """

    values = [float(value) for value in symptom_values]

    expected_feature_count = len(symptom_names) * 2

    if len(values) != expected_feature_count:
        raise ValueError(
            f"Expected {expected_feature_count} symptom features, "
            f"received {len(values)}."
        )

    device = next(model.parameters()).device

    symptoms = torch.tensor(
        [values],
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():

        baseline = torch.softmax(
            model(image_tensor, symptoms)
            / max(temperature, 1e-3),
            dim=1,
        )[0, class_index].item()

        changes = []

        for index, symptom_name in enumerate(symptom_names):

            value_index = index * 2
            known_index = value_index + 1

            symptom_value = values[value_index]
            known_indicator = values[known_index]

            # Only evaluate symptoms explicitly reported as present.
            if symptom_value != 1.0 or known_indicator != 1.0:
                continue

            altered = symptoms.clone()

            # Change yes to no.
            # Keep the known indicator at 1 because the symptom
            # is still known, just changed from present to absent.
            altered[0, value_index] = 0.0

            score = torch.softmax(
                model(image_tensor, altered)
                / max(temperature, 1e-3),
                dim=1,
            )[0, class_index].item()

            changes.append({
                "symptom": symptom_name,
                "score_drop": round(
                    baseline - score,
                    5,
                ),
            })

    return sorted(
        changes,
        key=lambda item: float(item["score_drop"]),
        reverse=True,
    )[:top_n]