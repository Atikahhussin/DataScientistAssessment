
import json
import csv
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "companies.jsonl"
OUTPUT_PATH = ROOT / "evaluation" / "dataset_profile.csv"

with DATA_PATH.open("r", encoding="utf-8") as f:
    records = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

status_counts = Counter()
industry_counts = Counter()
audit_counts = Counter()
financial_year_counts = Counter()
missing_counts = Counter()
litigation_status_counts = Counter()

companies_with_litigation = 0
companies_with_losses = 0
companies_with_negative_fcf = 0
companies_with_negative_equity = 0
companies_with_concentrated_ownership = 0
companies_with_declining_revenue = 0
companies_with_dscr_below_one = 0

litigation_examples = []
results = []


def safe_dict(value):
    return value if isinstance(value, dict) else {}


def safe_list(value):
    return value if isinstance(value, list) else []


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


for record in records:

    profile = safe_dict(record.get("profile"))
    relationships = safe_dict(record.get("relationships"))
    financials = safe_dict(record.get("financials"))
    litigation = safe_dict(record.get("litigation"))

    status = profile.get("status")
    industry = profile.get("industry")

    status_counts[status or "Missing"] += 1
    industry_counts[industry or "Missing"] += 1

    years = safe_list(financials.get("years"))

    years = [
        year for year in years
        if isinstance(year, dict)
        and is_number(year.get("financialYear"))
    ]

    years.sort(key=lambda x: x["financialYear"])

    financial_year_counts[len(years)] += 1

    for year in years:
        audit = safe_dict(year.get("audit"))
        audit_counts[audit.get("opinion") or "Missing"] += 1

    latest = years[-1] if years else {}
    previous = years[-2] if len(years) >= 2 else {}

    statements = safe_dict(latest.get("statements"))
    previous_statements = safe_dict(previous.get("statements"))

    ratios = safe_dict(latest.get("ratiosAndMetrics"))
    liquidity = safe_dict(ratios.get("liquidityAndCashFlow"))
    capital = safe_dict(ratios.get("capitalStructure"))

    revenue = statements.get("revenue")
    previous_revenue = previous_statements.get("revenue")
    profit = statements.get("profitAfterTax")
    equity = statements.get("totalEquity")
    fcf = liquidity.get("freeCashFlow")
    dscr = capital.get("debtServiceCoverageRatio")

    revenue_growth = None

    if (
        is_number(revenue)
        and is_number(previous_revenue)
        and previous_revenue > 0
    ):
        revenue_growth = round(
            (revenue - previous_revenue)
            / previous_revenue * 100,
            2
        )

    cases = litigation.get("cases")
    case_count = len(cases) if isinstance(cases, list) else None

    if case_count is None:
        litigation_status_counts["Missing"] += 1
    elif case_count > 0:
        litigation_status_counts["Cases recorded"] += 1
        companies_with_litigation += 1

        if len(litigation_examples) < 2:
            litigation_examples.append({
                "registrationNumber": record.get(
                    "registrationNumber"
                ),
                "cases": cases
            })
    else:
        litigation_status_counts["No recorded cases"] += 1

    shareholders = safe_list(
        relationships.get("shareholders")
    )

    percentages = [
        s["percentage"]
        for s in shareholders
        if isinstance(s, dict)
        and is_number(s.get("percentage"))
    ]

    largest_shareholder = (
        max(percentages) if percentages else None
    )

    has_loss = (
        profit < 0 if is_number(profit) else None
    )

    negative_fcf = (
        fcf < 0 if is_number(fcf) else None
    )

    negative_equity = (
        equity < 0 if is_number(equity) else None
    )

    concentrated_ownership = (
        largest_shareholder > 50
        if is_number(largest_shareholder)
        else None
    )

    declining_revenue = (
        revenue_growth < 0
        if revenue_growth is not None
        else None
    )

    low_dscr = (
        dscr < 1 if is_number(dscr) else None
    )

    companies_with_losses += has_loss is True
    companies_with_negative_fcf += negative_fcf is True
    companies_with_negative_equity += negative_equity is True
    companies_with_concentrated_ownership += (
        concentrated_ownership is True
    )
    companies_with_declining_revenue += (
        declining_revenue is True
    )
    companies_with_dscr_below_one += low_dscr is True

    important_fields = {
        "status": status,
        "industry": industry,
        "latestRevenue": revenue,
        "latestProfit": profit,
        "latestEquity": equity,
        "latestFreeCashFlow": fcf,
        "latestDSCR": dscr,
        "litigationCaseCount": case_count,
        "largestShareholder": largest_shareholder
    }

    for field, value in important_fields.items():
        if value is None or value == "":
            missing_counts[field] += 1

    results.append({
        "registrationNumber": record.get("registrationNumber"),
        "companyName": profile.get("companyName"),
        "status": status,
        "industry": industry,
        "financialYearsAvailable": len(years),
        "latestFinancialYear": latest.get("financialYear"),
        "latestRevenue": revenue,
        "revenueGrowthPct": revenue_growth,
        "latestProfitAfterTax": profit,
        "latestEquity": equity,
        "latestFreeCashFlow": fcf,
        "latestDSCR": dscr,
        "largestShareholderPct": largest_shareholder,
        "litigationCaseCount": case_count,
        "latestAuditOpinion": safe_dict(
            latest.get("audit")
        ).get("opinion"),
        "hasLoss": has_loss,
        "negativeFreeCashFlow": negative_fcf,
        "negativeEquity": negative_equity,
        "concentratedOwnership": concentrated_ownership,
        "decliningRevenue": declining_revenue,
        "dscrBelowOne": low_dscr
    })


def print_counter(title, counter):
    print(f"\n{title}")
    print("-" * 50)

    if not counter:
        print("None")

    for key, count in counter.most_common():
        print(f"{key}: {count}")


print("=" * 60)
print("EXTENDED DATASET PROFILE")
print("=" * 60)

print("Total companies:", len(records))

print_counter("COMPANY STATUS", status_counts)
print_counter("INDUSTRY DISTRIBUTION", industry_counts)
print_counter("FINANCIAL YEARS AVAILABLE", financial_year_counts)
print_counter("AUDIT OPINIONS (ALL YEARS)", audit_counts)
print_counter("LITIGATION STATUS", litigation_status_counts)
print_counter("MISSING IMPORTANT FIELDS", missing_counts)

print("\nPOTENTIAL RISK INDICATORS")
print("-" * 50)

print("Companies with litigation:",
      companies_with_litigation)

print("Companies with latest-year losses:",
      companies_with_losses)

print("Companies with negative free cash flow:",
      companies_with_negative_fcf)

print("Companies with negative equity:",
      companies_with_negative_equity)

print("Companies with concentrated ownership:",
      companies_with_concentrated_ownership)

print("Companies with declining revenue:",
      companies_with_declining_revenue)

print("Companies with DSCR below 1:",
      companies_with_dscr_below_one)

print("\nLITIGATION EXAMPLES")
print(json.dumps(
    litigation_examples,
    indent=2,
    ensure_ascii=False
))

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with OUTPUT_PATH.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)

print("\nDataset profile saved to:")
print(OUTPUT_PATH)
