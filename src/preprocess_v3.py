
import json

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "companies.jsonl"

OUTPUT_PATH = (
    ROOT / "evaluation" / "preprocessed_v3.jsonl"
)


# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def safe_dict(value):

    return value if isinstance(value, dict) else {}


def safe_list(value):

    return value if isinstance(value, list) else []


def is_number(value):

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


def percentage_change(current, previous):

    if (
        not is_number(current)
        or not is_number(previous)
        or previous <= 0
    ):
        return None

    return round(
        (current - previous) / previous * 100,
        2
    )


def remove_empty(value):

    if isinstance(value, dict):

        return {
            key: remove_empty(item)
            for key, item in value.items()
            if item is not None
            and item != ""
            and item != []
            and item != {}
        }

    if isinstance(value, list):

        return [
            remove_empty(item)
            for item in value
        ]

    return value


# ==========================================
# 2. COMPANY PROFILE
# ==========================================

def extract_profile(record):

    profile = safe_dict(
        record.get("profile")
    )

    return {
        "companyName": profile.get("companyName"),
        "status": profile.get("status"),
        "industry": profile.get("industry"),
        "state": profile.get("state"),
        "natureOfBusiness": profile.get(
            "natureOfBusiness"
        ),
        "incorporationDate": profile.get(
            "incorporationDate"
        )
    }


# ==========================================
# 3. OWNERSHIP INFORMATION
# ==========================================

def extract_ownership(record):

    relationships = safe_dict(
        record.get("relationships")
    )

    shareholders = [
        item
        for item in safe_list(
            relationships.get("shareholders")
        )
        if isinstance(item, dict)
    ]

    directors = safe_list(
        relationships.get("directors")
    )

    valid_shareholders = [
        shareholder
        for shareholder in shareholders
        if is_number(
            shareholder.get("percentage")
        )
    ]

    largest = (
        max(
            valid_shareholders,
            key=lambda x: x["percentage"]
        )
        if valid_shareholders
        else {}
    )

    percentage = largest.get("percentage")

    return {

        "directorCount": (
            len(directors)
            if "directors" in relationships
            else None
        ),

        "shareholderCount": (
            len(shareholders)
            if "shareholders" in relationships
            else None
        ),

        "largestShareholder": largest.get("name"),

        "largestShareholderPct": percentage,

        "majorityOwnership": (
            percentage > 50
            if is_number(percentage)
            else None
        )
    }


# ==========================================
# 4. FINANCIAL METRICS
# ==========================================

def extract_year_metrics(year):

    statements = safe_dict(
        year.get("statements")
    )

    ratios = safe_dict(
        year.get("ratiosAndMetrics")
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
        year.get("audit")
    )

    margin = profitability.get(
        "netProfitMargin"
    )

    return {

        "financialYear": year.get(
            "financialYear"
        ),

        "revenueRM": statements.get(
            "revenue"
        ),

        "profitAfterTaxRM": statements.get(
            "profitAfterTax"
        ),

        "freeCashFlowRM": liquidity.get(
            "freeCashFlow"
        ),

        "borrowingsRM": statements.get(
            "borrowings"
        ),

        "totalEquityRM": statements.get(
            "totalEquity"
        ),

        "netProfitMarginPct": (
            round(margin * 100, 2)
            if is_number(margin)
            else None
        ),

        "quickRatio": liquidity.get(
            "quickRatio"
        ),

        "debtServiceCoverageRatio": capital.get(
            "debtServiceCoverageRatio"
        ),

        "auditOpinion": audit.get(
            "opinion"
        )
    }


def extract_financials(record):

    financials = safe_dict(
        record.get("financials")
    )

    years = [
        year
        for year in safe_list(
            financials.get("years")
        )
        if isinstance(year, dict)
    ]

    years.sort(
        key=lambda year: str(
            year.get("financialYear", "")
        )
    )

    if not years:

        return {
            "financialYearsAvailable": 0,
            "dataLimitation": (
                "No financial statements available."
            )
        }

    metrics = [
        extract_year_metrics(year)
        for year in years
    ]

    latest = metrics[-1]

    result = {
        "financialYearsAvailable": len(metrics),
        "latestFinancials": latest
    }

    # Compare the two most recent reported years.

    if len(metrics) >= 2:

        previous = metrics[-2]

        current_year = latest["financialYear"]

        previous_year = previous["financialYear"]

        try:

            consecutive = (
                int(current_year)
                - int(previous_year)
                == 1
            )

        except (ValueError, TypeError):

            consecutive = False

        result["financialComparison"] = {

            "fromYear": previous_year,

            "toYear": current_year,

            "consecutiveYears": consecutive,

            "previousRevenueRM": previous[
                "revenueRM"
            ],

            "previousProfitAfterTaxRM": previous[
                "profitAfterTaxRM"
            ],

            "revenueChangePct": percentage_change(
                latest["revenueRM"],
                previous["revenueRM"]
            ),

            "profitChangePct": percentage_change(
                latest["profitAfterTaxRM"],
                previous["profitAfterTaxRM"]
            )
        }

    else:

        result["dataLimitation"] = (
            "Only one financial year is available. "
            "Financial trends cannot be established."
        )

    fcf_values = [
        year["freeCashFlowRM"]
        for year in metrics
        if is_number(
            year["freeCashFlowRM"]
        )
    ]

    result["negativeFCFYears"] = sum(
        value < 0
        for value in fcf_values
    )

    result["yearsWithFCFData"] = len(
        fcf_values
    )

    # Retain unusual historical audit opinions.

    result["historicalAuditExceptions"] = [

        {
            "financialYear": year["financialYear"],
            "auditOpinion": year["auditOpinion"]
        }

        for year in metrics[:-1]

        if year["auditOpinion"]

        and str(
            year["auditOpinion"]
        ).strip().lower() != "unqualified"
    ]

    return result


# ==========================================
# 5. LITIGATION
# ==========================================

def extract_litigation(record):

    litigation = safe_dict(
        record.get("litigation")
    )

    if not isinstance(
        litigation.get("cases"),
        list
    ):

        return {
            "dataAvailable": False,
            "dataLimitation": (
                "Litigation information unavailable."
            )
        }

    cases = [
        case
        for case in litigation["cases"]
        if isinstance(case, dict)
    ]

    role_counts = {}

    status_counts = {}

    pending_cases = []

    for case in cases:

        role = str(
            case.get("role") or "Unknown"
        ).strip().lower()

        status = str(
            case.get("status") or "Unknown"
        ).strip().lower()

        role_counts[role] = (
            role_counts.get(role, 0) + 1
        )

        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

        if status in (
            "pending",
            "ongoing",
            "active",
            "open"
        ):

            pending_cases.append({

                "role": role,

                "status": status,

                "caseType": case.get(
                    "caseType"
                ),

                "claimAmountRM": case.get(
                    "claimAmount"
                )
            })

    return {

        "dataAvailable": True,

        "totalRecordedCases": len(cases),

        "roleCounts": role_counts,

        "statusCounts": status_counts,

        "pendingCases": pending_cases
    }


# ==========================================
# 6. MATERIAL FINANCIAL INDICATORS
# ==========================================

def identify_material_indicators(
    profile,
    financials,
    litigation
):

    indicators = []

    status = str(
        profile.get("status") or ""
    ).lower()

    if "winding up" in status:

        indicators.append(
            "Company status is winding up."
        )

    latest = safe_dict(
        financials.get("latestFinancials")
    )

    comparison = safe_dict(
        financials.get("financialComparison")
    )

    profit = latest.get(
        "profitAfterTaxRM"
    )

    equity = latest.get(
        "totalEquityRM"
    )

    fcf = latest.get(
        "freeCashFlowRM"
    )

    dscr = latest.get(
        "debtServiceCoverageRatio"
    )

    revenue_change = comparison.get(
        "revenueChangePct"
    )

    if is_number(profit) and profit < 0:

        indicators.append(
            "Latest reported financial year "
            "has a net loss."
        )

    if is_number(equity) and equity < 0:

        indicators.append(
            "Latest reported equity is negative."
        )

    if is_number(fcf) and fcf < 0:

        indicators.append(
            "Latest free cash flow is negative."
        )

    if is_number(dscr) and dscr < 1:

        indicators.append(
            "Reported debt service coverage "
            "ratio is below 1."
        )

    if (
        is_number(revenue_change)
        and revenue_change < 0
    ):

        indicators.append(
            "Revenue declined between the two "
            "most recent reported financial years."
        )

    audit = str(
        latest.get("auditOpinion") or ""
    ).strip().lower()

    if audit and audit != "unqualified":

        indicators.append(
            "Latest audit opinion requires attention: "
            + latest["auditOpinion"]
        )

    pending_defendant = [

        case

        for case in litigation.get(
            "pendingCases", []
        )

        if case.get("role") == "defendant"
    ]

    if pending_defendant:

        indicators.append(
            "Company has recorded pending "
            "defendant litigation."
        )

    return indicators


# ==========================================
# 7. COMPLETE V3 PREPROCESSING
# ==========================================

def preprocess_company_v3(record):

    profile = extract_profile(record)

    ownership = extract_ownership(record)

    financials = extract_financials(record)

    litigation = extract_litigation(record)

    indicators = identify_material_indicators(
        profile,
        financials,
        litigation
    )

    result = {

        "registrationNumber": record[
            "registrationNumber"
        ],

        "companyProfile": profile,

        "financialInformation": financials,

        "ownership": ownership,

        "litigation": litigation,

        "materialIndicators": indicators
    }

    return remove_empty(result)


# ==========================================
# 8. EXPORT PREPROCESSED DATA
# ==========================================

def main():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    count = 0

    with DATA_PATH.open(
        "r",
        encoding="utf-8"
    ) as source:

        with OUTPUT_PATH.open(
            "w",
            encoding="utf-8"
        ) as output:

            for line in source:

                if not line.strip():
                    continue

                record = json.loads(line)

                processed = preprocess_company_v3(
                    record
                )

                output.write(
                    json.dumps(
                        processed,
                        ensure_ascii=False
                    ) + "\n"
                )

                count += 1

    print("=" * 60)

    print("V3 PREPROCESSING COMPLETED")

    print("=" * 60)

    print("Companies processed:", count)

    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":

    main()
