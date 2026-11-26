
import json
import csv
import re

from pathlib import Path
from statistics import mean


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "companies.jsonl"

SPLIT_PATH = (
    ROOT / "evaluation"
    / "final_evaluation_split.json"
)

OUTPUT_PATH = (
    ROOT / "evaluation"
    / "final_rule_based_scores.csv"
)

SUMMARY_PATHS = {
    "V1": ROOT / "summaries" / "v1_baseline.jsonl",
    "V2": ROOT / "summaries" / "v2_improved.jsonl",
    "V3": ROOT / "summaries" / "v3_concise.jsonl"
}


# ==========================================
# 2. LOAD DATA
# ==========================================

def load_jsonl(path):

    records = {}

    with path.open("r", encoding="utf-8") as f:

        for line in f:

            if line.strip():

                item = json.loads(line)

                company_id = item["registrationNumber"]

                if company_id in records:

                    raise ValueError(
                        f"Duplicate company: {company_id}"
                    )

                records[company_id] = item

    return records


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def safe_dict(value):

    return value if isinstance(value, dict) else {}


def is_number(value):

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


def normalise(text):

    return re.sub(
        r"\s+",
        " ",
        str(text).lower()
    ).strip()


def contains_text(summary, value):

    if value is None or value == "":
        return None

    pattern = (
        r"(?<!\w)"
        + re.escape(normalise(value))
        + r"(?!\w)"
    )

    return bool(
        re.search(pattern, normalise(summary))
    )


# ==========================================
# 4. EXTRACT NUMERICAL EXPRESSIONS
# ==========================================

NUMBER_PATTERN = re.compile(
    r"(?<![\w.])"
    r"(?P<sign>[-−])?\s*"
    r"(?P<currency>RM|MYR)?\s*"
    r"(?P<sign2>[-−])?\s*"
    r"(?P<number>\d[\d,]*(?:\.\d+)?)"
    r"\s*(?P<unit>billion|million|thousand|bn|mil|[bmk%x])?"
    r"(?![\w.])",
    re.IGNORECASE
)


def extract_numbers(text):

    numbers = []

    for match in NUMBER_PATTERN.finditer(text):

        raw = match.group("number").replace(",", "")

        value = float(raw)

        unit = (match.group("unit") or "").lower()

        currency = match.group("currency")

        multipliers = {
            "billion": 1_000_000_000,
            "bn": 1_000_000_000,
            "b": 1_000_000_000,
            "million": 1_000_000,
            "mil": 1_000_000,
            "m": 1_000_000,
            "thousand": 1_000,
            "k": 1_000
        }

        multiplier = multipliers.get(unit, 1)

        value *= multiplier

        before = text[
            max(0, match.start() - 30):match.start()
        ]

        negative = bool(
            match.group("sign")
            or match.group("sign2")
            or re.search(
                r"\b(?:negative|loss(?:\s+of)?)\s*$",
                before,
                re.IGNORECASE
            )
        )

        if negative:
            value = -value

        decimals = (
            len(raw.split(".")[1])
            if "." in raw else 0
        )

        rounding_tolerance = (
            0.5 * multiplier * 10 ** (-decimals)
        )

        numbers.append({
            "value": value,
            "unit": unit,
            "currency": currency,
            "tolerance": rounding_tolerance
        })

    return numbers


def match_number(numbers, expected, kind):

    if not is_number(expected):
        return None

    for item in numbers:

        if kind == "percentage":

            if item["unit"] != "%":
                continue

        elif kind == "ratio":

            if item["unit"] != "x":
                continue

        elif kind == "money":

            if not item["currency"]:
                continue

            if item["unit"] in ("%", "x"):
                continue

        tolerance = max(
            item["tolerance"],
            0.000001
        )

        if abs(item["value"] - expected) <= tolerance:

            return True

    return False


# ==========================================
# 5. MISSING FINANCIAL DATA CHECK
# ==========================================

def acknowledges_missing_financials(summary):

    text = normalise(summary)

    patterns = [
        r"no financial (?:data|records|statements|information)",

        r"financial (?:data|records|statements|information)"
        r".{0,35}(?:unavailable|not available|not provided)",

        r"absence of financial",

        r"financial information is missing",

        r"no financial years"
    ]

    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


# ==========================================
# 6. EVALUATE INDIVIDUAL SUMMARY
# ==========================================

def evaluate_summary(record, summary, version):

    profile = safe_dict(
        record.get("profile")
    )

    financials = safe_dict(
        record.get("financials")
    )

    relationships = safe_dict(
        record.get("relationships")
    )

    years = financials.get("years") or []

    years = [
        year for year in years
        if isinstance(year, dict)
        and is_number(year.get("financialYear"))
    ]

    years.sort(
        key=lambda year: year["financialYear"]
    )

    latest = years[-1] if years else {}

    statements = safe_dict(
        latest.get("statements")
    )

    ratios = safe_dict(
        latest.get("ratiosAndMetrics")
    )

    profitability = safe_dict(
        ratios.get("profitability")
    )

    liquidity = safe_dict(
        ratios.get("liquidityAndCashFlow")
    )

    capital = safe_dict(
        ratios.get("capitalStructure")
    )

    audit = safe_dict(
        latest.get("audit")
    )

    shareholders = (
        relationships.get("shareholders") or []
    )

    percentages = [
        shareholder["percentage"]
        for shareholder in shareholders
        if isinstance(shareholder, dict)
        and is_number(
            shareholder.get("percentage")
        )
    ]

    largest_shareholder = (
        max(percentages)
        if percentages else None
    )

    numbers = extract_numbers(summary)

    checks = {}

    # --------------------------------------
    # Company profile
    # --------------------------------------

    checks["companyNameMatch"] = contains_text(
        summary,
        profile.get("companyName")
    )

    checks["companyStatusMatch"] = contains_text(
        summary,
        profile.get("status")
    )

    checks["industryMatch"] = contains_text(
        summary,
        profile.get("industry")
    )

    # --------------------------------------
    # Financial figures
    # --------------------------------------

    checks["revenueMatch"] = match_number(
        numbers,
        statements.get("revenue"),
        "money"
    )

    checks["profitMatch"] = match_number(
        numbers,
        statements.get("profitAfterTax"),
        "money"
    )

    checks["freeCashFlowMatch"] = match_number(
        numbers,
        liquidity.get("freeCashFlow"),
        "money"
    )

    checks["borrowingsMatch"] = match_number(
        numbers,
        statements.get("borrowings"),
        "money"
    )

    margin = profitability.get(
        "netProfitMargin"
    )

    checks["netProfitMarginMatch"] = match_number(
        numbers,
        margin * 100 if is_number(margin) else None,
        "percentage"
    )

    checks["dscrMatch"] = match_number(
        numbers,
        capital.get("debtServiceCoverageRatio"),
        "ratio"
    )

    # --------------------------------------
    # Ownership
    # --------------------------------------

    checks["majorityShareholderMatch"] = match_number(
        numbers,
        largest_shareholder
        if largest_shareholder is not None
        and largest_shareholder > 50
        else None,
        "percentage"
    )

    # --------------------------------------
    # Audit
    # --------------------------------------

    checks["auditOpinionMatch"] = contains_text(
        summary,
        audit.get("opinion")
    )

    # --------------------------------------
    # Missing financial information
    # --------------------------------------

    checks["missingFinancialDataAcknowledged"] = (
        acknowledges_missing_financials(summary)
        if not years else None
    )

    # ======================================
    # 7. CALCULATE SCORES
    # ======================================

    applicable = [
        value for value in checks.values()
        if value is not None
    ]

    matched = sum(
        value is True for value in applicable
    )

    numerical_fields = [
        "revenueMatch",
        "profitMatch",
        "freeCashFlowMatch",
        "borrowingsMatch",
        "netProfitMarginMatch",
        "dscrMatch",
        "majorityShareholderMatch"
    ]

    numerical_checks = [
        checks[field]
        for field in numerical_fields
        if checks[field] is not None
    ]

    numerical_matched = sum(
        value is True
        for value in numerical_checks
    )

    word_count = len(summary.split())

    result = {
        "registrationNumber": record[
            "registrationNumber"
        ],

        "version": version,

        "applicableChecks": len(applicable),

        "matchedChecks": matched,

        "keyFactMatchRatePct": round(
            matched / len(applicable) * 100,
            2
        ) if applicable else None,

        "applicableNumericalChecks": len(
            numerical_checks
        ),

        "matchedNumericalChecks": numerical_matched,

        "numericalMatchRatePct": round(
            numerical_matched
            / len(numerical_checks) * 100,
            2
        ) if numerical_checks else None,

        "wordCount": word_count,

        "withinWordLimit": (
            100 <= word_count <= 150
        )
    }

    result.update(checks)

    return result


# ==========================================
# 8. PERCENTAGE HELPER
# ==========================================

def percentage(numerator, denominator):

    return (
        round(numerator / denominator * 100, 2)
        if denominator else None
    )


# ==========================================
# 9. MAIN EVALUATION
# ==========================================

def main():

    companies = load_jsonl(DATA_PATH)

    with SPLIT_PATH.open(
        "r", encoding="utf-8"
    ) as f:

        evaluation_ids = json.load(f)["evaluation"]

    if not evaluation_ids:

        raise ValueError(
            "Final evaluation sample is empty."
        )

    if len(evaluation_ids) != len(set(evaluation_ids)):

        raise ValueError(
            "Duplicate evaluation company IDs."
        )

    unknown_ids = set(evaluation_ids) - set(companies)

    if unknown_ids:

        raise ValueError(
            f"Unknown company IDs: {sorted(unknown_ids)}"
        )

    print("=" * 60)

    print("FINAL THREE-VERSION RULE-BASED EVALUATION")

    print("=" * 60)

    print("Companies:", len(evaluation_ids))

    print("Versions:", len(SUMMARY_PATHS))

    results = []

    # --------------------------------------
    # Evaluate V1, V2 and V3
    # --------------------------------------

    for version, path in SUMMARY_PATHS.items():

        summaries = load_jsonl(path)

        missing = [
            company_id
            for company_id in evaluation_ids
            if company_id not in summaries
        ]

        if missing:

            raise ValueError(
                f"{version} missing summaries: {missing}"
            )

        print("\n" + "=" * 60)

        print("Evaluating:", version)

        print("=" * 60)

        for company_id in evaluation_ids:

            record = companies[company_id]

            summary = summaries[company_id]["summary"]

            if (
                not isinstance(summary, str)
                or not summary.strip()
            ):

                raise ValueError(
                    f"Empty or invalid summary: "
                    f"{version} - {company_id}"
                )

            result = evaluate_summary(
                record,
                summary,
                version
            )

            results.append(result)

            print(
                company_id,
                "| Fact matching:",
                result["keyFactMatchRatePct"],
                "%",
                "| Numerical matching:",
                result["numericalMatchRatePct"],
                "%",
                "| Words:",
                result["wordCount"]
            )

    # ======================================
    # 10. SAVE EVALUATION RESULTS
    # ======================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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

    # ======================================
    # 11. DISPLAY PERFORMANCE COMPARISON
    # ======================================

    print("\n" + "=" * 60)

    print("FINAL RULE-BASED EVALUATION RESULTS")

    print("=" * 60)

    comparison = []

    for version in SUMMARY_PATHS:

        rows = [
            row for row in results
            if row["version"] == version
        ]

        fact_match = percentage(
            sum(
                row["matchedChecks"]
                for row in rows
            ),
            sum(
                row["applicableChecks"]
                for row in rows
            )
        )

        numerical_match = percentage(
            sum(
                row["matchedNumericalChecks"]
                for row in rows
            ),
            sum(
                row["applicableNumericalChecks"]
                for row in rows
            )
        )

        word_compliance = percentage(
            sum(
                row["withinWordLimit"]
                for row in rows
            ),
            len(rows)
        )

        average_words = round(
            mean(
                row["wordCount"]
                for row in rows
            ),
            2
        )

        print("\nVERSION:", version)

        print(
            "Companies evaluated:",
            len(rows)
        )

        print(
            "Key fact matching rate:",
            fact_match,
            "%"
        )

        print(
            "Numerical matching rate:",
            numerical_match,
            "%"
        )

        print(
            "Average word count:",
            average_words
        )

        print(
            "Word limit compliance:",
            word_compliance,
            "%"
        )

        comparison.append({
            "version": version,
            "factMatch": fact_match,
            "numericalMatch": numerical_match,
            "averageWords": average_words,
            "wordCompliance": word_compliance
        })

    # ======================================
    # 12. FINAL COMPARISON TABLE
    # ======================================

    print("\n" + "=" * 75)

    print("THREE-VERSION COMPARISON")

    print("=" * 75)

    print(
        f"{'Version':<10}"
        f"{'Fact Match':<16}"
        f"{'Numerical':<16}"
        f"{'Avg Words':<16}"
        f"{'Compliance':<16}"
    )

    print("-" * 75)

    def format_pct(value):

        return (
            f"{value:.2f}%"
            if value is not None
            else "N/A"
        )

    for row in comparison:

        print(
            f"{row['version']:<10}"
            f"{format_pct(row['factMatch']):<16}"
            f"{format_pct(row['numericalMatch']):<16}"
            f"{row['averageWords']:<16.2f}"
            f"{format_pct(row['wordCompliance']):<16}"
        )

    print("-" * 75)

    print(
        "\nTotal evaluated companies:",
        len(evaluation_ids)
    )

    print(
        "Total evaluated summaries:",
        len(results)
    )

    print("\nResults saved to:")

    print(OUTPUT_PATH)

    print("\n" + "=" * 60)

    print("FINAL RULE-BASED EVALUATION COMPLETED")

    print("=" * 60)


if __name__ == "__main__":

    main()
