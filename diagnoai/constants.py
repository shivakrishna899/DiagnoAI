"""Shared symptom names aligned to the PAD-UFES-20 clinical fields."""

# Maps to the source columns: itch, grew, hurt, changed, bleed, elevation.
SYMPTOM_NAMES = [
    "itching",
    "growth",
    "pain",
    "appearance_change",
    "bleeding",
    "elevation",
]

DEFAULT_IMAGE_SIZE = 224
DEFAULT_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_MARGIN_THRESHOLD = 0.10
DEFAULT_ENTROPY_THRESHOLD = 0.72


# Indicators showing whether each symptom is known.
SYMPTOM_KNOWN_NAMES = [
    f"{name}_known"
    for name in SYMPTOM_NAMES
]

# Final 12-feature input order.
# Each symptom is followed by its known indicator.
SYMPTOM_FEATURE_NAMES = [
    feature
    for symptom in SYMPTOM_NAMES
    for feature in (symptom, f"{symptom}_known")
]
