import numpy as np
import pandas as pd

from training.train import create_splits


def test_grouped_splits_keep_patients_disjoint():
    rows = []
    for label in ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK"):
        for patient_index in range(12):
            patient = f"{label}-{patient_index}"
            for image_index in range(2):
                rows.append({"patient_id": patient, "diagnosis": label})
    frame = pd.DataFrame(rows)
    labels = pd.Categorical(frame["diagnosis"], categories=sorted(frame["diagnosis"].unique())).codes

    train_idx, val_idx, test_idx, split_name = create_splits(frame, labels, val_size=0.2, test_size=0.2, seed=13)

    train_patients = set(frame.iloc[train_idx]["patient_id"])
    val_patients = set(frame.iloc[val_idx]["patient_id"])
    test_patients = set(frame.iloc[test_idx]["patient_id"])
    assert not train_patients.intersection(val_patients)
    assert not train_patients.intersection(test_patients)
    assert not val_patients.intersection(test_patients)
    assert set(labels[train_idx]) == set(labels)
    assert "patient-disjoint" in split_name
