
"""
Central diagnosis label mapping for DiagnoAI.

All supported datasets should map their original diagnosis
labels to this shared taxonomy.
"""

LABEL_ALIASES = {
    "BCC": {
        "bcc",
        "basal cell carcinoma",
        "basal-cell carcinoma",
    },
    "ACK": {
        "ack",
        "actinic keratosis",
        "actinic keratosis and intraepithelial carcinoma",
    },
    "SCC": {
        "scc",
        "squamous cell carcinoma",
    },
    "NEV": {
        "nev",
        "nevus",
        "melanocytic nevus",
    },
    "SEK": {
        "sek",
        "seborrheic keratosis",
    },
    "MEL": {
        "mel",
        "melanoma",
    },
}


def normalize_label(label: str) -> str:
    """Normalize text before matching a diagnosis."""

    return " ".join(str(label).strip().lower().split())


def map_diagnosis(label: str) -> str:
    """
    Convert an original diagnosis into DiagnoAI's
    canonical diagnosis label.

    Raises an error if the diagnosis is not recognized.
    """

    normalized = normalize_label(label)

    if not normalized:
        raise ValueError("Diagnosis label cannot be empty.")

    for canonical_label, aliases in LABEL_ALIASES.items():
        normalized_aliases = {
            normalize_label(alias) for alias in aliases
        }

        if normalized == canonical_label.lower():
            return canonical_label

        if normalized in normalized_aliases:
            return canonical_label

    raise ValueError(
        f"Unrecognized diagnosis label: {label!r}. "
        "Add a verified mapping before using this diagnosis."
    )


def get_supported_labels() -> list[str]:
    """Return all canonical diagnosis labels."""

    return list(LABEL_ALIASES.keys())
