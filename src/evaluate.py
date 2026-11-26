
import csv
import json

from pathlib import Path
from statistics import mean, median


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

EVAL_DIR = ROOT / "evaluation"

RULE_PATH = EVAL_DIR / "rule_based_scores.csv"

JUDGE_PATH = EVAL_DIR / "llm_judge_scores.csv"

SPLIT_PATH = EVAL_DIR / "sample_split.json"

OUTPUT_PATH = EVAL_DIR / "comparison.csv"

LOG_PATHS = {
    "V1": EVAL_DIR / "v1_generation_log.csv",
    "V2": EVAL_DIR / "v2_generation_log.csv"
}


# ==========================================
# 2. LOAD EVALUATION DATA
# ==========================================

def load_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file missing: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig"
    ) as f:

        return list(csv.DictReader(f))


rule_scores = load_csv(RULE_PATH)

judge_scores = load_csv(JUDGE_PATH)

with SPLIT_PATH.open("r", encoding="utf-8") as f:

    evaluation_ids = set(
        json.load(f)["evaluation"]
    )


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def percentage(numerator, denominator):

    if denominator == 0:
        return None

    return round(
        numerator / denominator * 100,
        2
    )


def total(rows, field):

    return sum(
        float(row[field])
        for row in rows
    )


def average(rows, field):

    values = [
        float(row[field])
        for row in rows
        if row.get(field, "").strip()
    ]

    if not values or len(values) != len(rows):
        return None

    return round(mean(values), 2)


def validate_coverage(rows, version, source):

    ids = [
        row["registrationNumber"]
        for row in rows
    ]

    if (
        set(ids) != evaluation_ids
        or len(ids) != len(evaluation_ids)
    ):

        raise ValueError(
            f"Incomplete or duplicate records "
            f"in {source} for {version}."
        )


def load_generation_logs(version):

    rows = load_csv(
        LOG_PATHS[version]
    )

    successful = {}

    for row in rows:

        company_id = row["registrationNumber"]

        if (
            row["status"] == "success"
            and company_id in evaluation_ids
        ):

            if company_id in successful:

                raise ValueError(
                    f"Duplicate successful generation: "
                    f"{version} {company_id}"
                )

            successful[company_id] = row

    if set(successful) != evaluation_ids:

        raise ValueError(
            f"Incomplete generation logs for {version}"
        )

    return list(successful.values())


# ==========================================
# 4. VALIDATE JUDGE MODEL
# ==========================================

judge_models = {
    row["judgeModel"]
    for row in judge_scores
    if row["registrationNumber"] in evaluation_ids
}

if len(judge_models) != 1 or "" in judge_models:

    raise ValueError(
        "Inconsistent judge models detected."
    )


# ==========================================
# 5. CALCULATE FINAL METRICS
# ==========================================

results = []

generation_models = {}

for version in ["V1", "V2"]:

    rules = [
        row for row in rule_scores
        if row["version"] == version
    ]

    judges = [
        row for row in judge_scores
        if row["version"] == version
    ]

    validate_coverage(
        rules,
        version,
        "rule-based scores"
    )

    validate_coverage(
        judges,
        version,
        "LLM judge scores"
    )

    logs = load_generation_logs(version)

    generation_models[version] = {
        row["returnedModel"]
        for row in logs
    }

    result = {

        "version": version,

        "evaluatedCompanies": len(rules),

        # ----------------------------------
        # Rule-based metrics
        # ----------------------------------

        "keyFactMatchRatePct": percentage(
            total(rules, "matchedChecks"),
            total(rules, "applicableChecks")
        ),

        "numericalMatchRatePct": percentage(
            total(
                rules,
                "matchedNumericalChecks"
            ),
            total(
                rules,
                "applicableNumericalChecks"
            )
        ),

        "averageWordCount": average(
            rules,
            "wordCount"
        ),

        "wordLimitCompliancePct": percentage(
            sum(
                row["withinWordLimit"].lower()
                in ("true", "1")
                for row in rules
            ),
            len(rules)
        ),

        # ----------------------------------
        # LLM judge metrics
        # ----------------------------------

        "estimatedFactualAccuracyPct": percentage(
            total(judges, "correctClaims"),
            total(judges, "totalClaims")
        ),

        "estimatedIncorrectClaimRatePct": percentage(
            total(judges, "incorrectClaims"),
            total(judges, "totalClaims")
        ),

        "estimatedHallucinationRatePct": percentage(
            total(judges, "unsupportedClaims"),
            total(judges, "totalClaims")
        ),

        "estimatedMaterialFactCoveragePct": percentage(
            total(
                judges,
                "materialFactsIncluded"
            ),
            total(
                judges,
                "materialFactsAvailable"
            )
        ),

        "estimatedCriticalErrorRatePct": percentage(
            total(judges, "criticalError"),
            len(judges)
        ),

        "averageUsefulness": average(
            judges,
            "usefulness"
        ),

        "averageReadability": average(
            judges,
            "readability"
        ),

        "averageConciseness": average(
            judges,
            "conciseness"
        ),

        # ----------------------------------
        # Generation performance metrics
        # ----------------------------------

        "averageLatencySeconds": average(
            logs,
            "latencySeconds"
        ),

        "medianLatencySeconds": round(
            median(
                float(row["latencySeconds"])
                for row in logs
            ),
            2
        ),

        "averageInputTokens": average(
            logs,
            "inputTokens"
        ),

        "averageOutputTokens": average(
            logs,
            "outputTokens"
        )
    }

    results.append(result)


# ==========================================
# 6. VALIDATE GENERATION MODEL CONSISTENCY
# ==========================================

if (
    generation_models["V1"]
    != generation_models["V2"]

    or len(generation_models["V1"]) != 1

    or "" in generation_models["V1"]
):

    raise ValueError(
        "Generation model inconsistency detected."
    )


# ==========================================
# 7. VALIDATE SHARED MATERIAL FACT COUNTS
# ==========================================

material_counts = {}

for row in judge_scores:

    company_id = row["registrationNumber"]

    if company_id not in evaluation_ids:
        continue

    count = int(row["materialFactsAvailable"])

    if (
        company_id in material_counts
        and material_counts[company_id] != count
    ):

        raise ValueError(
            f"Inconsistent material facts: {company_id}"
        )

    material_counts[company_id] = count


# ==========================================
# 8. SAVE FINAL COMPARISON
# ==========================================

with OUTPUT_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()

    writer.writerows(results)


# ==========================================
# 9. DISPLAY FINAL RESULTS
# ==========================================

print("=" * 85)

print("FINAL PERFORMANCE COMPARISON")

print("=" * 85)

print(
    "Generation model:",
    next(iter(generation_models["V1"]))
)

print(
    "Judge model:",
    next(iter(judge_models))
)

print(
    f"\n{'Metric':<43}"
    f"{'V1':>12}"
    f"{'V2':>12}"
    f"{'Change':>12}"
)

print("-" * 85)

for metric in results[0]:

    if metric == "version":
        continue

    v1 = results[0][metric]

    v2 = results[1][metric]

    change = (
        round(v2 - v1, 2)
        if v1 is not None and v2 is not None
        else "N/A"
    )

    print(
        f"{metric:<43}"
        f"{str(v1):>12}"
        f"{str(v2):>12}"
        f"{str(change):>12}"
    )

print("\nComparison saved to:")

print(OUTPUT_PATH)

print(
    "\nNote: LLM judge metrics are estimates "
    "and have not been independently human-validated."
)
