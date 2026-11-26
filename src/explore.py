
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

# Locate the original dataset
files = [
    p for p in ROOT.rglob("*.jsonl")
    if not any(
        part in {".venv", ".git", "summaries", "evaluation"}
        for part in p.relative_to(ROOT).parts
    )
]

if len(files) != 1:
    print("Dataset candidates:", files)
    raise SystemExit("Please identify the original dataset.")

dataset_path = files[0]

# Load JSONL records
with dataset_path.open("r", encoding="utf-8") as f:
    records = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("Dataset path:", dataset_path)
print("Total company records:", len(records))

# Inspect top-level fields
field_counts = Counter()

for record in records:
    field_counts.update(record.keys())

print("\nAVAILABLE FIELDS")

for field, count in field_counts.items():
    print(f"{field}: {count}/{len(records)} records")

# Check duplicate registration numbers
registration_numbers = [
    record.get("registrationNumber")
    for record in records
]

valid_ids = [
    value for value in registration_numbers
    if value is not None
]

print("\nDATA QUALITY")
print("Missing registration numbers:",
      registration_numbers.count(None))
print("Unique registration numbers:",
      len(set(valid_ids)))
print("Duplicate registration numbers:",
      len(valid_ids) - len(set(valid_ids)))

# Inspect the first company
if records:
    print("\nFIRST COMPANY RECORD")
    print("=" * 60)

    print(json.dumps(
        records[0],
        indent=2,
        ensure_ascii=False
    ))
