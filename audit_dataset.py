
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "prompt_injection_dataset.csv"

df = pd.read_csv(DATA)

for column in ["category", "technique"]:
    if column not in df.columns:
        print(f"\nMissing column: {column}")
        continue

    print(f"\n{'=' * 20} {column.upper()} {'=' * 20}")

    table = pd.crosstab(
        df[column].fillna("unknown"),
        df["label"].str.lower().str.strip(),
        margins=True,
    )

    print(table.to_string())

print("\nDataset split by label:")
print(pd.crosstab(df["split"], df["label"]).to_string())
