import pytest

np = pytest.importorskip("numpy")
from diagnoai.inference import Predictor


def test_uncertainty_flags_flat_distribution():
    result = Predictor._uncertainty(np.array([0.25, 0.25, 0.25, 0.25]))
    assert result["uncertain"] is True
    assert result["normalized_entropy"] == pytest.approx(1.0)


def test_uncertainty_for_clear_distribution():
    result = Predictor._uncertainty(np.array([0.94, 0.02, 0.02, 0.02]))
    assert result["top_confidence"] == pytest.approx(0.94, abs=0.001)
    assert result["uncertain"] is False
