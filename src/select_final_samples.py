import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPLIT_PATH = ROOT / "evaluation" / "sample_split.json"

OUTPUT_PATH = ROOT / "evaluation" / "final_evaluation_split.json"

with SPLIT_PATH.open("r", encoding="utf-8") as f:
    splits = json.load(f)

additional = splits["additional"]

rng = random.Random(42)

final_ids = rng.sample(
    additional,
    min(10, len(additional))
)

result = {
    "randomSeed": 42,
    "sampleSize": len(final_ids),
    "evaluation": final_ids
}

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(result, f, indent=4)

print("=" * 60)
print("FINAL EVALUATION SAMPLE")
print("=" * 60)

print("Selected companies:", len(final_ids))

for company_id in final_ids:
    print(company_id)

print("\nOutput:", OUTPUT_PATH)