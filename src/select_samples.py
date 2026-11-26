
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROFILE_PATH = ROOT / "evaluation" / "dataset_profile.csv"
OUTPUT_PATH = ROOT / "evaluation" / "sample_split.json"

SEED = 42
random.seed(SEED)

with PROFILE_PATH.open("r", encoding="utf-8") as f:
    records = list(csv.DictReader(f))

random.shuffle(records)

# Important scenarios to include in evaluation
scenarios = {
    "Winding up": lambda r:
        r["status"].lower() == "winding up",

    "Missing financial data": lambda r:
        r["financialYearsAvailable"] == "0",

    "One financial year": lambda r:
        r["financialYearsAvailable"] == "1",

    "Two financial years": lambda r:
        r["financialYearsAvailable"] == "2",

    "Negative equity": lambda r:
        r["negativeEquity"] == "True",

    "Modified audit opinion": lambda r:
        r["latestAuditOpinion"] not in ("Unqualified", ""),

    "Litigation": lambda r:
        int(r["litigationCaseCount"] or 0) > 0,

    "Latest-year loss": lambda r:
        r["hasLoss"] == "True",

    "Negative free cash flow": lambda r:
        r["negativeFreeCashFlow"] == "True",

    "Declining revenue": lambda r:
        r["decliningRevenue"] == "True",

    "DSCR below 1": lambda r:
        r["dscrBelowOne"] == "True",

    "Majority shareholder": lambda r:
        r["concentratedOwnership"] == "True",

    "No majority shareholder": lambda r:
        r["concentratedOwnership"] == "False",

    "No recorded litigation": lambda r:
        r["litigationCaseCount"] == "0"
}

evaluation = []
selected_ids = set()

# Ensure scenario coverage
for scenario, condition in scenarios.items():

    candidates = [
        r for r in records
        if condition(r)
    ]

    already_covered = any(
        condition(r) for r in evaluation
    )

    if already_covered:
        continue

    available = [
        r for r in candidates
        if r["registrationNumber"] not in selected_ids
    ]

    if available:
        selected = random.choice(available)
        evaluation.append(selected)
        selected_ids.add(
            selected["registrationNumber"]
        )

# Fill evaluation set to 15 companies
remaining = [
    r for r in records
    if r["registrationNumber"] not in selected_ids
]

evaluation.extend(
    random.sample(remaining, 15 - len(evaluation))
)

evaluation_ids = {
    r["registrationNumber"] for r in evaluation
}

# Select development set from remaining companies
remaining = [
    r for r in records
    if r["registrationNumber"] not in evaluation_ids
]

development = random.sample(remaining, 5)

development_ids = {
    r["registrationNumber"] for r in development
}

additional = [
    r for r in remaining
    if r["registrationNumber"] not in development_ids
]

split = {
    "randomSeed": SEED,
    "development": sorted(development_ids),
    "evaluation": sorted(evaluation_ids),
    "additional": sorted(
        r["registrationNumber"] for r in additional
    )
}

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(split, f, indent=2)

print("=" * 60)
print("DATASET SAMPLE SELECTION")
print("=" * 60)

print("Random seed:", SEED)
print("Development companies:", len(development))
print("Evaluation companies:", len(evaluation))
print("Additional companies:", len(additional))

print("\nEVALUATION SCENARIO COVERAGE")
print("-" * 60)

for scenario, condition in scenarios.items():

    count = sum(
        condition(r) for r in evaluation
    )

    print(f"{scenario}: {count}")

print("\nDEVELOPMENT COMPANY IDs")
for company_id in split["development"]:
    print(company_id)

print("\nEVALUATION COMPANY IDs")
for company_id in split["evaluation"]:
    print(company_id)

print("\nSample split saved to:")
print(OUTPUT_PATH)
