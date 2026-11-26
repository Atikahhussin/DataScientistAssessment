
import json
from pathlib import Path
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]


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

    if not is_number(current):
        return None

    if not is_number(previous) or previous <= 0:
        return None

    return round(
        (current - previous) / previous * 100,
        2
    )


def safe_ratio(numerator, denominator):

    if not is_number(numerator):
        return None

    if not is_number(denominator) or denominator <= 0:
        return None

    return round(numerator / denominator, 2)


def preprocess_company(record):

    profile = safe_dict(record.get("profile"))
    relationships = safe_dict(record.get("relationships"))
    financials = safe_dict(record.get("financials"))
    litigation = safe_dict(record.get("litigation"))

    # ----------------------------------
    # 1. Company profile
    # ----------------------------------

    company_profile = {
        "companyName": profile.get("companyName"),
        "status": profile.get("status"),
        "companyType": profile.get("companyType"),
        "incorporationDate": profile.get(
            "incorporationDate"
        ),
        "industry": profile.get("industry"),
        "state": profile.get("state"),
        "natureOfBusiness": profile.get(
            "natureOfBusiness"
        ),
        "shareCapital": profile.get("shareCapital"),
        "formerNames": profile.get("formerNames")
    }

    # ----------------------------------
    # 2. Ownership and governance
    # ----------------------------------

    directors = relationships.get("directors")
    shareholders = relationships.get("shareholders")

    valid_shareholders = [
        s for s in safe_list(shareholders)
        if isinstance(s, dict)
        and is_number(s.get("percentage"))
    ]

    largest_shareholder = max(
        valid_shareholders,
        key=lambda s: s["percentage"],
        default=None
    )

    ownership = {
        "directorCount": (
            len(directors)
            if isinstance(directors, list)
            else None
        ),
        "shareholderCount": (
            len(shareholders)
            if isinstance(shareholders, list)
            else None
        ),
        "largestShareholder": largest_shareholder,
        "majorityOwnership": (
            largest_shareholder["percentage"] > 50
            if largest_shareholder
            else None
        ),
        "directors": directors,
        "shareholders": shareholders
    }

    # ----------------------------------
    # 3. Financial feature engineering
    # ----------------------------------

    years = [
        y for y in safe_list(financials.get("years"))
        if isinstance(y, dict)
        and is_number(y.get("financialYear"))
    ]

    years.sort(key=lambda y: y["financialYear"])

    financial_history = []

    for year in years:

        statements = safe_dict(year.get("statements"))

        ratios = safe_dict(year.get("ratiosAndMetrics"))

        profitability = safe_dict(
            ratios.get("profitability")
        )

        liquidity = safe_dict(
            ratios.get("liquidityAndCashFlow")
        )

        capital = safe_dict(
            ratios.get("capitalStructure")
        )

        revenue = statements.get("revenue")
        profit = statements.get("profitAfterTax")

        margin = profitability.get("netProfitMargin")

        financial_history.append({
            "financialYear": year.get("financialYear"),
            "financialYearEnd": year.get(
                "financialYearEnd"
            ),
            "revenue": revenue,
            "profitAfterTax": profit,
            "netProfitMarginPct": (
                round(margin * 100, 2)
                if is_number(margin)
                else None
            ),
            "operatingProfit": statements.get(
                "operatingProfit"
            ),
            "cash": statements.get("cash"),
            "borrowings": statements.get("borrowings"),
            "totalAssets": statements.get("totalAssets"),
            "totalLiabilities": statements.get(
                "totalLiabilities"
            ),
            "totalEquity": statements.get("totalEquity"),
            "operatingCashFlow": statements.get(
                "operatingCashFlow"
            ),
            "freeCashFlow": liquidity.get("freeCashFlow"),
            "quickRatio": liquidity.get("quickRatio"),
            "currentRatio": safe_ratio(
                statements.get("currentAssets"),
                statements.get("currentLiabilities")
            ),
            "debtServiceCoverageRatio": capital.get(
                "debtServiceCoverageRatio"
            ),
            "interestCoverageRatio": capital.get(
                "interestCoverageRatio"
            ),
            "auditOpinion": safe_dict(
                year.get("audit")
            ).get("opinion")
        })

    latest = (
        financial_history[-1]
        if financial_history
        else {}
    )

    previous = (
        financial_history[-2]
        if len(financial_history) >= 2
        else {}
    )

    earliest = (
        financial_history[0]
        if financial_history
        else {}
    )

    revenue_growth = percentage_change(
        latest.get("revenue"),
        previous.get("revenue")
    )

    profit_growth = percentage_change(
        latest.get("profitAfterTax"),
        previous.get("profitAfterTax")
    )

    borrowings_growth = percentage_change(
        latest.get("borrowings"),
        previous.get("borrowings")
    )

    margin_change = None

    if (
        is_number(latest.get("netProfitMarginPct"))
        and is_number(previous.get("netProfitMarginPct"))
    ):
        margin_change = round(
            latest["netProfitMarginPct"]
            - previous["netProfitMarginPct"],
            2
        )

    fcf_values = [
        y["freeCashFlow"]
        for y in financial_history
        if is_number(y.get("freeCashFlow"))
    ]

    negative_fcf_years = sum(
        value < 0 for value in fcf_values
    )

    financial_summary = {
        "financialYearsAvailable": len(financial_history),
        "financialHistory": financial_history,
        "latestFinancialYear": latest.get("financialYear"),
        "previousFinancialYear": previous.get(
            "financialYear"
        ),
        "latestFinancialPosition": latest or None,
        "revenueChangePct": revenue_growth,
        "profitChangePct": profit_growth,
        "borrowingsChangePct": borrowings_growth,
        "netProfitMarginChangePercentagePoints": (
            margin_change
        ),
        "totalRevenueChangePct": (
            percentage_change(
                latest.get("revenue"),
                earliest.get("revenue")
            )
            if len(financial_history) >= 2
            else None
        ),
        "negativeFreeCashFlowYears": negative_fcf_years,
        "freeCashFlowYearsAvailable": len(fcf_values)
    }

    # ----------------------------------
    # 4. Litigation analysis
    # ----------------------------------

    raw_cases = litigation.get("cases")

    cases = [
        c for c in safe_list(raw_cases)
        if isinstance(c, dict)
    ]

    role_counts = Counter(
        str(c.get("role") or "Unknown").lower()
        for c in cases
    )

    status_counts = Counter(
        str(c.get("status") or "Unknown").lower()
        for c in cases
    )

    pending_cases = [
        c for c in cases
        if str(c.get("status")).lower() == "pending"
    ]

    pending_defendant_cases = [
        c for c in pending_cases
        if str(c.get("role")).lower() == "defendant"
    ]

    pending_plaintiff_cases = [
        c for c in pending_cases
        if str(c.get("role")).lower() == "plaintiff"
    ]

    def total_claim_amount(case_list):

        amounts = [
            c.get("claimAmount")
            for c in case_list
        ]

        if any(not is_number(a) for a in amounts):
            return None

        return sum(amounts)

    litigation_summary = {
        "litigationDataAvailable": isinstance(
            raw_cases, list
        ),
        "totalCases": (
            len(cases)
            if isinstance(raw_cases, list)
            else None
        ),
        "roleCounts": dict(role_counts),
        "statusCounts": dict(status_counts),
        "pendingDefendantCases": (
            len(pending_defendant_cases)
            if isinstance(raw_cases, list)
            else None
        ),
        "pendingPlaintiffCases": (
            len(pending_plaintiff_cases)
            if isinstance(raw_cases, list)
            else None
        ),
        "pendingDefendantClaimAmount": (
            total_claim_amount(pending_defendant_cases)
            if isinstance(raw_cases, list)
            else None
        ),
        "pendingPlaintiffClaimAmount": (
            total_claim_amount(pending_plaintiff_cases)
            if isinstance(raw_cases, list)
            else None
        ),
        "cases": raw_cases
    }

    # ----------------------------------
    # 5. Rule-based review indicators
    # ----------------------------------

    risk_indicators = []
    positive_indicators = []
    data_limitations = []

    if str(profile.get("status")).lower() == "winding up":
        risk_indicators.append(
            "Company status is winding up."
        )

    if not financial_history:

        data_limitations.append(
            "No financial statements are available."
        )

    elif len(financial_history) == 1:

        data_limitations.append(
            "Only one financial year is available; "
            "financial trends cannot be established."
        )

    if revenue_growth is not None:

        if revenue_growth < 0:
            risk_indicators.append(
                f"Revenue decreased by "
                f"{abs(revenue_growth)}% "
                "between the latest two available years."
            )

        elif revenue_growth > 0:
            positive_indicators.append(
                f"Revenue increased by {revenue_growth}% "
                "between the latest two available years."
            )

    profit = latest.get("profitAfterTax")

    if is_number(profit):

        if profit < 0:
            risk_indicators.append(
                "Latest financial year recorded a net loss."
            )

        elif profit > 0:
            positive_indicators.append(
                "Latest financial year recorded a net profit."
            )

    equity = latest.get("totalEquity")

    if is_number(equity) and equity < 0:
        risk_indicators.append(
            "Latest financial year recorded negative equity."
        )

    fcf = latest.get("freeCashFlow")

    if is_number(fcf) and fcf < 0:
        risk_indicators.append(
            "Latest financial year recorded negative "
            "free cash flow."
        )

    if (
        len(financial_history) >= 2
        and negative_fcf_years == len(financial_history)
    ):
        risk_indicators.append(
            "Free cash flow was negative across "
            "all available financial years."
        )

    dscr = latest.get("debtServiceCoverageRatio")

    if is_number(dscr) and dscr < 1:
        risk_indicators.append(
            f"Debt service coverage ratio is {dscr}x, "
            "below 1x."
        )

    if ownership["majorityOwnership"]:

        percentage = largest_shareholder["percentage"]

        risk_indicators.append(
            f"Ownership is concentrated, with the largest "
            f"shareholder holding {percentage}%."
        )

    if pending_defendant_cases:
        risk_indicators.append(
            f"{len(pending_defendant_cases)} pending "
            "litigation case(s) as defendant."
        )

    if isinstance(raw_cases, list) and not cases:
        positive_indicators.append(
            "No litigation cases are recorded "
            "in the supplied data."
        )

    if not isinstance(raw_cases, list):
        data_limitations.append(
            "Litigation information is unavailable."
        )

    audit_opinion = latest.get("auditOpinion")

    if audit_opinion:

        if audit_opinion.lower() == "unqualified":

            positive_indicators.append(
                "Latest recorded audit opinion is unqualified."
            )

        else:

            risk_indicators.append(
                f"Latest audit opinion: {audit_opinion}."
            )

    # ----------------------------------
    # 6. Final structured output
    # ----------------------------------

    return {
        "registrationNumber": record.get(
            "registrationNumber"
        ),
        "profile": company_profile,
        "ownershipAndGovernance": ownership,
        "financialAnalysis": financial_summary,
        "litigationAnalysis": litigation_summary,
        "potentialReviewIndicators": risk_indicators,
        "positiveIndicators": positive_indicators,
        "dataLimitations": data_limitations
    }


# Test preprocessing independently
if __name__ == "__main__":

    data_path = ROOT / "data" / "companies.jsonl"

    with data_path.open("r", encoding="utf-8") as f:

        records = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    processed = [
        preprocess_company(record)
        for record in records
    ]

    output_path = (
        ROOT / "evaluation" / "preprocessed_companies.jsonl"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open("w", encoding="utf-8") as f:

        for record in processed:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    print("=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)

    print("Total processed companies:", len(processed))

    print("\nFIRST PREPROCESSED COMPANY")

    print(json.dumps(
        processed[0],
        indent=2,
        ensure_ascii=False
    ))

    print("\nOutput saved to:")
    print(output_path)
