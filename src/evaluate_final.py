
import csv
import json

from pathlib import Path
from statistics import mean, median


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

EVALUATION_DIR = ROOT / "evaluation"

DATA_PATH = ROOT / "data" / "companies.jsonl"

SPLIT_PATH = (
    EVALUATION_DIR / "final_evaluation_split.json"
)

RULE_PATH = (
    EVALUATION_DIR / "final_rule_based_scores.csv"
)

OUTPUT_PATH = (
    EVALUATION_DIR / "final_comparison.csv"
)

FULL_DATASET_PATH = (
    EVALUATION_DIR / "full_dataset_comparison.csv"
)

VERSIONS = {
    "V1": {
        "name": "Baseline",
        "summary": ROOT / "summaries" / "v1_baseline.jsonl",
        "log": EVALUATION_DIR / "v1_generation_log.csv"
    },
    "V2": {
        "name": "Improved",
        "summary": ROOT / "summaries" / "v2_improved.jsonl",
        "log": EVALUATION_DIR / "v2_generation_log.csv"
    },
    "V3": {
        "name": "Concise",
        "summary": ROOT / "summaries" / "v3_concise.jsonl",
        "log": EVALUATION_DIR / "v3_generation_log.csv"
    }
}


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def load_csv(path):

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:

        return list(csv.DictReader(f))


def load_jsonl(path):

    records = {}

    with path.open("r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            record = json.loads(line)

            company_id = record["registrationNumber"]

            if company_id in records:

                raise ValueError(
                    f"Duplicate company in {path.name}: "
                    f"{company_id}"
                )

            records[company_id] = record

    return records


def percentage(numerator, denominator):

    if denominator == 0:
        return None

    return round(
        numerator / denominator * 100,
        2
    )


def total(rows, field):

    return sum(
        int(row[field])
        for row in rows
        if row.get(field) not in ("", None)
    )


def average(rows, field):

    values = [
        row.get(field)
        for row in rows
    ]

    if not values:
        return None

    if any(
        value in ("", None)
        for value in values
    ):
        return None

    return round(
        mean(float(value) for value in values),
        2
    )


def is_true(value):

    return str(value).strip().lower() in (
        "true",
        "1",
        "yes"
    )


def validate_ids(rows, expected_ids, label):

    actual_ids = [
        row["registrationNumber"]
        for row in rows
    ]

    if len(actual_ids) != len(set(actual_ids)):

        raise ValueError(
            f"Duplicate records in {label}"
        )

    if set(actual_ids) != set(expected_ids):

        missing = set(expected_ids) - set(actual_ids)

        unexpected = set(actual_ids) - set(expected_ids)

        raise ValueError(
            f"{label}: Missing={missing}, "
            f"Unexpected={unexpected}"
        )


def write_csv(path, rows):

    if not rows:

        raise ValueError(
            f"No results available for {path.name}"
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()

        writer.writerows(rows)


# ==========================================
# 3. LOAD DATA
# ==========================================

companies = load_jsonl(DATA_PATH)

all_ids = set(companies.keys())

with SPLIT_PATH.open(
    "r", encoding="utf-8"
) as f:

    evaluation_ids = json.load(f)["evaluation"]

if len(evaluation_ids) != len(set(evaluation_ids)):

    raise ValueError(
        "Duplicate final evaluation IDs."
    )

evaluation_ids = set(evaluation_ids)

if not evaluation_ids:

    raise ValueError(
        "Final evaluation sample is empty."
    )

if not evaluation_ids.issubset(all_ids):

    raise ValueError(
        "Unknown final evaluation company IDs."
    )

rule_scores = load_csv(RULE_PATH)

expected_pairs = {
    (company_id, version)
    for company_id in evaluation_ids
    for version in VERSIONS
}

actual_pairs = [
    (
        row["registrationNumber"],
        row["version"]
    )
    for row in rule_scores
]

if (
    len(actual_pairs) != len(set(actual_pairs))
    or set(actual_pairs) != expected_pairs
):

    raise ValueError(
        "Final rule-based evaluation records "
        "do not match the expected sample."
    )

comparison = []

full_dataset_comparison = []

generation_models = set()


# ==========================================
# 4. EVALUATE EACH VERSION
# ==========================================

for version, config in VERSIONS.items():

    print("\n" + "=" * 60)

    print(
        f"PROCESSING {version}: {config['name']}"
    )

    print("=" * 60)

    summaries = load_jsonl(config["summary"])

    # --------------------------------------
    # Validate all generated summaries
    # --------------------------------------

    if set(summaries.keys()) != all_ids:

        missing = all_ids - set(summaries.keys())

        unexpected = set(summaries.keys()) - all_ids

        raise ValueError(
            f"{version}: Missing={missing}, "
            f"Unexpected={unexpected}"
        )

    for company_id, record in summaries.items():

        summary = record.get("summary")

        if (
            not isinstance(summary, str)
            or not summary.strip()
        ):

            raise ValueError(
                f"Empty summary: "
                f"{version} {company_id}"
            )

    # --------------------------------------
    # Load final rule-based results
    # --------------------------------------

    rules = [
        row
        for row in rule_scores
        if row["version"] == version
    ]

    validate_ids(
        rules,
        evaluation_ids,
        f"{version} rule-based scores"
    )

    # --------------------------------------
    # Load generation logs
    # --------------------------------------

    logs = load_csv(config["log"])

    successful_logs = [
        row
        for row in logs
        if row["status"].lower() == "success"
    ]

    validate_ids(
        successful_logs,
        all_ids,
        f"{version} full generation logs"
    )

    models = {
        row["returnedModel"]
        for row in successful_logs
        if row.get("returnedModel")
    }

    if len(models) != 1:

        raise ValueError(
            f"{version} uses multiple or unknown "
            f"generation models: {models}"
        )

    generation_models.update(models)

    model = next(iter(models))

    final_logs = [
        row
        for row in successful_logs
        if row["registrationNumber"]
        in evaluation_ids
    ]

    validate_ids(
        final_logs,
        evaluation_ids,
        f"{version} final generation logs"
    )

    # --------------------------------------
    # Verify recorded word counts
    # --------------------------------------

    for row in successful_logs:

        company_id = row["registrationNumber"]

        actual_words = len(
            summaries[company_id]["summary"].split()
        )

        logged_words = int(row["wordCount"])

        if actual_words != logged_words:

            raise ValueError(
                f"Word count mismatch: "
                f"{version} {company_id}"
            )

    for row in rules:

        company_id = row["registrationNumber"]

        actual_words = len(
            summaries[company_id]["summary"].split()
        )

        if actual_words != int(row["wordCount"]):

            raise ValueError(
                f"Rule score word count mismatch: "
                f"{version} {company_id}"
            )

    # --------------------------------------
    # Final sample metrics
    # --------------------------------------

    applicable_checks = total(
        rules, "applicableChecks"
    )

    matched_checks = total(
        rules, "matchedChecks"
    )

    applicable_numerical = total(
        rules, "applicableNumericalChecks"
    )

    matched_numerical = total(
        rules, "matchedNumericalChecks"
    )

    compliant = sum(
        is_true(row["withinWordLimit"])
        for row in rules
    )

    latencies = [
        float(row["latencySeconds"])
        for row in final_logs
    ]

    api_requests = sum(
        int(row.get("apiRequests") or 1)
        for row in final_logs
    )

    result = {

        "version": version,

        "versionName": config["name"],

        "generationModel": model,

        "evaluatedCompanies": len(rules),

        "applicableChecks": applicable_checks,

        "matchedChecks": matched_checks,

        "keyFactMatchRatePct": percentage(
            matched_checks,
            applicable_checks
        ),

        "applicableNumericalChecks":
            applicable_numerical,

        "matchedNumericalChecks":
            matched_numerical,

        "numericalMatchRatePct": percentage(
            matched_numerical,
            applicable_numerical
        ),

        "averageWordCount": average(
            rules, "wordCount"
        ),

        "wordLimitCompliancePct": percentage(
            compliant,
            len(rules)
        ),

        "averageLatencySeconds": average(
            final_logs,
            "latencySeconds"
        ),

        "medianLatencySeconds": round(
            median(latencies),
            2
        ),

        "averageInputTokens": average(
            final_logs,
            "inputTokens"
        ),

        "averageOutputTokens": average(
            final_logs,
            "outputTokens"
        ),

        "totalApiRequests": api_requests,

        "averageApiRequests": round(
            api_requests / len(final_logs),
            2
        )
    }

    comparison.append(result)

    # --------------------------------------
    # Full dataset metrics
    # --------------------------------------

    word_counts = [

        len(record["summary"].split())

        for record in summaries.values()
    ]

    compliant_count = sum(
        100 <= count <= 150
        for count in word_counts
    )

    full_requests = sum(
        int(row.get("apiRequests") or 1)
        for row in successful_logs
    )

    full_result = {

        "version": version,

        "versionName": config["name"],

        "generationModel": model,

        "totalCompanies": len(all_ids),

        "completedSummaries": len(summaries),

        "completionRatePct": percentage(
            len(summaries),
            len(all_ids)
        ),

        "averageWordCount": round(
            mean(word_counts),
            2
        ),

        "minimumWordCount": min(word_counts),

        "maximumWordCount": max(word_counts),

        "compliantSummaries": compliant_count,

        "nonCompliantSummaries":
            len(word_counts) - compliant_count,

        "wordLimitCompliancePct": percentage(
            compliant_count,
            len(word_counts)
        ),

        "averageLatencySeconds": average(
            successful_logs,
            "latencySeconds"
        ),

        "averageInputTokens": average(
            successful_logs,
            "inputTokens"
        ),

        "averageOutputTokens": average(
            successful_logs,
            "outputTokens"
        ),

        "totalApiRequests": full_requests
    }

    full_dataset_comparison.append(full_result)

    print("Total summaries:", len(summaries))

    print("Final evaluation sample:", len(rules))

    print("Generation model:", model)

    print("Validation: PASSED")


# ==========================================
# 5. CHECK MODEL CONSISTENCY
# ==========================================

if len(generation_models) != 1:

    raise ValueError(
        "Different generation models were used "
        "across versions. Comparison is confounded."
    )


# ==========================================
# 6. EXPORT RESULTS
# ==========================================

write_csv(
    OUTPUT_PATH,
    comparison
)

write_csv(
    FULL_DATASET_PATH,
    full_dataset_comparison
)


# ==========================================
# 7. PRINT COMPARISON TABLE
# ==========================================

def print_comparison(rows, metrics):

    print(
        f"{'Metric':<35}"
        f"{'V1':>14}"
        f"{'V2':>14}"
        f"{'V3':>14}"
    )

    print("-" * 77)

    for label, field in metrics:

        values = []

        for row in rows:

            value = row.get(field)

            if value is None:

                values.append("N/A")

            elif isinstance(value, float):

                values.append(f"{value:.2f}")

            else:

                values.append(str(value))

        print(
            f"{label:<35}"
            f"{values[0]:>14}"
            f"{values[1]:>14}"
            f"{values[2]:>14}"
        )


print("\n" + "=" * 77)

print("FINAL HELD-OUT SAMPLE COMPARISON")

print("=" * 77)

print_comparison(

    comparison,

    [
        ("Evaluated companies", "evaluatedCompanies"),

        ("Key fact match (%)", "keyFactMatchRatePct"),

        ("Numerical match (%)", "numericalMatchRatePct"),

        ("Average word count", "averageWordCount"),

        ("Word limit compliance (%)", "wordLimitCompliancePct"),

        ("Average latency (s)", "averageLatencySeconds"),

        ("Median latency (s)", "medianLatencySeconds"),

        ("Average input tokens", "averageInputTokens"),

        ("Average output tokens", "averageOutputTokens"),

        ("Total API requests", "totalApiRequests"),

        ("Average API requests", "averageApiRequests")
    ]
)


print("\n" + "=" * 77)

print("FULL DATASET COMPARISON")

print("=" * 77)

print_comparison(

    full_dataset_comparison,

    [
        ("Completed summaries", "completedSummaries"),

        ("Completion rate (%)", "completionRatePct"),

        ("Average word count", "averageWordCount"),

        ("Minimum word count", "minimumWordCount"),

        ("Maximum word count", "maximumWordCount"),

        ("Compliant summaries", "compliantSummaries"),

        ("Non-compliant summaries", "nonCompliantSummaries"),

        ("Word limit compliance (%)", "wordLimitCompliancePct"),

        ("Average latency (s)", "averageLatencySeconds"),

        ("Average input tokens", "averageInputTokens"),

        ("Average output tokens", "averageOutputTokens"),

        ("Total API requests", "totalApiRequests")
    ]
)


# ==========================================
# 8. EVALUATION LIMITATIONS
# ==========================================

print("\n" + "=" * 77)

print("EVALUATION NOTES")

print("=" * 77)

print(
    "1. Key fact matching is a heuristic coverage "
    "measure, not verified factual accuracy."
)

print(
    "2. Numerical matching measures whether "
    "selected source values appear in summaries."
)

print(
    "3. Final held-out evaluation uses "
    f"{len(evaluation_ids)} companies."
)

print(
    "4. Full dataset metrics measure generation "
    "completion, length and efficiency."
)

print(
    "5. LLM judge results are excluded due to "
    "unreliable and incomplete responses."
)

print(
    "6. V3 latency and token consumption "
    "include revision requests where recorded."
)

print(
    "7. API request totals represent recorded "
    "successful-generation workflows, not "
    "all historical failed requests."
)

print("\nFinal comparison:", OUTPUT_PATH)

print("Full dataset comparison:", FULL_DATASET_PATH)

print("\n" + "=" * 77)

print("FINAL COMPARISON COMPLETED")

print("=" * 77)
