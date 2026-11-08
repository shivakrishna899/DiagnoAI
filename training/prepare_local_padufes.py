
from pathlib import Path
import pandas as pd

# Dataset location on your computer
dataset_dir = Path(
    r"C:\Users\shiva\Downloads\DiagnoAI_Research_Starter\PAD-UFES-20"
)
metadata_path = dataset_dir / "metadata.csv"

# Find all available images recursively
image_extensions = {".jpg", ".jpeg", ".png"}
image_map = {
    path.name: str(path.resolve())
    for path in dataset_dir.rglob("*")
    if path.is_file() and path.suffix.lower() in image_extensions
}

print(f"Images discovered: {len(image_map)}")

# Read dataset metadata
df = pd.read_csv(metadata_path)
df["img_id"] = df["img_id"].astype(str)

# Keep only records with images that exist
df["image_path"] = df["img_id"].map(image_map)
df = df.dropna(subset=["image_path"]).copy()

# Convert original dataset columns to DiagnoAI column names
df = df.rename(
    columns={
        "diagnostic": "diagnosis",
        "itch": "itching",
        "grew": "growth",
        "hurt": "pain",
        "changed": "appearance_change",
        "bleed": "bleeding",
    }
)

# Keep the fields required by DiagnoAI
columns = [
    "image_path",
    "diagnosis",
    "patient_id",
    "itching",
    "growth",
    "pain",
    "appearance_change",
    "bleeding",
    "elevation",
]

training_df = df[columns].copy()

# Save a separate CSV; preserve the original template
output_path = Path("data/padufes_train.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
training_df.to_csv(output_path, index=False)

print(f"Matched training records: {len(training_df)}")
print(f"Unique patients: {training_df['patient_id'].nunique()}")
print(f"Missing values by column:\n{training_df.isna().sum()}")
print(f"Diagnosis counts:\n{training_df['diagnosis'].value_counts()}")
print(f"Saved to: {output_path.resolve()}")
