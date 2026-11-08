import pytest


torch = pytest.importorskip("torch")
from diagnoai.model import DiagnoAIFusionModel


def test_fusion_model_output_shape():
    model = DiagnoAIFusionModel(num_classes=4, num_symptoms=6, pretrained_image=False)
    images = torch.randn(2, 3, 224, 224)
    symptoms = torch.zeros(2, 6)
    output = model(images, symptoms)
    assert output.shape == (2, 4)


def test_model_rejects_bad_symptom_shape():
    model = DiagnoAIFusionModel(num_classes=3, num_symptoms=6, pretrained_image=False)
    images = torch.randn(1, 3, 224, 224)
    with pytest.raises(ValueError):
        model(images, torch.zeros(1, 6, 1))
