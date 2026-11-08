
import numpy as np
import pandas as pd

from training.train import create_splits


CSV_PATH = "data/processed/unified_dataset.csv"

def main():
    df = pd.read_csv(CSV_PATH)

    classes = sorted(df["diagnosis"].unique().tolist())
    class_to_index = {
        name: index for index, name in enumerate(classes)
    }

    y = df["diagnosis"].map(class_to_index).to_numpy(
        dtype=np.int64
    )

    train_idx, val_idx, test_idx, strategy = create_splits(
        frame=df,
        y=y,
        val_size=0.2,
        test_size=0.2,
        seed=42,
    )

    splits = {
        "Training": df.iloc[train_idx],
        "Validation": df.iloc[val_idx],
        "Testing": df.iloc[test_idx],
    }

    print("\nSPLIT STRATEGY")
    print(strategy)

    print("\nSPLIT SIZES")
    for name, data in splits.items():
        print(
            f"{name}: {len(data)} records "
            f"({len(data) / len(df) * 100:.2f}%)"
        )

    print("\nCLASS DISTRIBUTION")
    for name, data in splits.items():
        counts = data["diagnosis"].value_counts().reindex(
            classes, fill_value=0
        )
        print(f"\n{name}:")
        print(counts.to_string())

        missing = counts[counts == 0].index.tolist()
        if missing:
            print("WARNING: Missing classes:", missing)

    print("\nPATIENT LEAKAGE CHECK")

    patient_groups = {
        name: set(data["patient_id"].dropna().astype(str))
        for name, data in splits.items()
    }

    pairs = [
        ("Training", "Validation"),
        ("Training", "Testing"),
        ("Validation", "Testing"),
    ]

    leakage_found = False

    for first, second in pairs:
        overlap = patient_groups[first] & patient_groups[second]

        print(f"{first} vs {second}: {len(overlap)} overlapping patients")

        if overlap:
            leakage_found = True

    if leakage_found:
        print("FAILED: Patient leakage detected.")
    else:
        print("PASSED: No patient overlap detected.")

    print("\nSPLIT VERIFICATION COMPLETED")


if __name__ == "__main__":
    main()

