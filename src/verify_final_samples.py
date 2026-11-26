
import json
from pathlib import Path


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

SPLIT_PATH = (
    ROOT / "evaluation"
    / "final_evaluation_split.json"
)

SUMMARY_PATHS = {
    "V1": ROOT / "summaries" / "v1_baseline.jsonl",
    "V2": ROOT / "summaries" / "v2_improved.jsonl",
    "V3": ROOT / "summaries" / "v3_concise.jsonl"
}


# ==========================================
# 2. LOAD FINAL EVALUATION SAMPLE
# ==========================================

with SPLIT_PATH.open(
    "r", encoding="utf-8"
) as f:

    split = json.load(f)

expected_ids = set(split["evaluation"])

if not expected_ids:
    raise ValueError("Final evaluation sample is empty.")

print("=" * 60)
print("FINAL EVALUATION VERIFICATION")
print("=" * 60)

print("Expected companies:", len(expected_ids))

all_passed = True


# ==========================================
# 3. VERIFY EACH VERSION
# ==========================================

for version, path in SUMMARY_PATHS.items():

    print("\n" + "-" * 60)
    print("Version:", version)

    if not path.exists():

        print("FAILED: Summary file not found.")

        all_passed = False

        continue

    summaries = {}

    duplicates = []

    empty_summaries = []

    with path.open("r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            record = json.loads(line)

            company_id = record["registrationNumber"]

            if company_id in summaries:

                duplicates.append(company_id)

            summary = record.get("summary")

            if not isinstance(summary, str):

                empty_summaries.append(company_id)

                summary = ""

            elif not summary.strip():

                empty_summaries.append(company_id)

            summaries[company_id] = summary

    missing = expected_ids - set(summaries)

    selected = [

        summaries[company_id]

        for company_id in expected_ids

        if company_id in summaries
    ]

    word_counts = [

        len(summary.split())

        for summary in selected
    ]

    compliant = sum(

        100 <= count <= 150

        for count in word_counts
    )

    print("Total saved summaries:", len(summaries))

    print("Final evaluation summaries:", len(selected))

    print("Missing companies:", len(missing))

    print("Duplicate records:", len(duplicates))

    print("Empty summaries:", len(empty_summaries))

    if word_counts:

        print(
            "Average word count:",
            round(
                sum(word_counts) / len(word_counts),
                2
            )
        )

        print(
            "Word limit compliance:",
            round(
                compliant / len(word_counts) * 100,
                2
            ),
            "%"
        )

    if missing:

        print("Missing IDs:", sorted(missing))

    if duplicates:

        print("Duplicate IDs:", duplicates)

    if empty_summaries:

        print("Empty IDs:", empty_summaries)

    if missing or duplicates or empty_summaries:

        print("Verification: FAILED")

        all_passed = False

    else:

        print("Verification: PASSED")


# ==========================================
# 4. FINAL VERIFICATION RESULT
# ==========================================

print("\n" + "=" * 60)

if all_passed:

    print("ALL THREE VERSIONS ARE READY FOR EVALUATION.")

else:

    print("PLEASE RESOLVE THE IDENTIFIED ISSUES.")

print("=" * 60)

if not all_passed:
    raise SystemExit(1)
