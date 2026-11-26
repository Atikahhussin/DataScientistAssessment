
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

COMPARISON_PATH = (
    ROOT / "evaluation" / "comparison.csv"
)

RULE_PATH = (
    ROOT / "evaluation" / "rule_based_scores.csv"
)

OUTPUT_DIR = (
    ROOT / "evaluation" / "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

if not COMPARISON_PATH.exists():

    raise FileNotFoundError(
        "Run src/evaluate.py first."
    )


# ==========================================
# 2. LOAD DATA
# ==========================================

df = pd.read_csv(COMPARISON_PATH)

df = df.set_index("version")

df = df.loc[["V1", "V2"]]

rules = pd.read_csv(RULE_PATH)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "figure.dpi": 120
})

VERSION_LABELS = [
    "V1 Baseline",
    "V2 Improved"
]

COLORS = [
    "#94A3B8",
    "#2563EB"
]


# ==========================================
# 3. HELPER FUNCTION
# ==========================================

def save_chart(filename):

    path = OUTPUT_DIR / filename

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print("Saved:", path.name)


def grouped_chart(
    metrics,
    labels,
    title,
    ylabel,
    filename,
    ylim=None,
    suffix=""
):

    values_v1 = df.loc["V1", metrics].astype(float)

    values_v2 = df.loc["V2", metrics].astype(float)

    if (
        values_v1.isna().any()
        or values_v2.isna().any()
    ):

        print(
            "Skipped chart due to missing values:",
            filename
        )

        return

    x = np.arange(len(metrics))

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(max(8, len(metrics) * 2.5), 5.5)
    )

    bars_v1 = ax.bar(
        x - width / 2,
        values_v1,
        width,
        label=VERSION_LABELS[0],
        color=COLORS[0]
    )

    bars_v2 = ax.bar(
        x + width / 2,
        values_v2,
        width,
        label=VERSION_LABELS[1],
        color=COLORS[1]
    )

    for bars in [bars_v1, bars_v2]:

        for bar in bars:

            height = bar.get_height()

            ax.annotate(
                f"{height:.2f}{suffix}",
                (
                    bar.get_x()
                    + bar.get_width() / 2,
                    height
                ),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10
            )

    ax.set_title(
        title,
        fontweight="bold",
        pad=18
    )

    ax.set_ylabel(ylabel)

    ax.set_xticks(x)

    ax.set_xticklabels(labels)

    if ylim:
        ax.set_ylim(*ylim)
    else:
        ax.margins(y=0.20)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        frameon=False
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    save_chart(filename)


# ==========================================
# 4. CHART 1: RULE-BASED PERFORMANCE
# ==========================================

grouped_chart(

    metrics=[
        "keyFactMatchRatePct",
        "numericalMatchRatePct"
    ],

    labels=[
        "Key Fact Matching",
        "Numerical Matching"
    ],

    title="Rule-Based Evaluation Performance",

    ylabel="Matching Rate (%)",

    filename="01_rule_based_performance.png",

    ylim=(0, 110),

    suffix="%"
)


# ==========================================
# 5. CHART 2: ESTIMATED ACCURACY & COVERAGE
# ==========================================

grouped_chart(

    metrics=[
        "estimatedFactualAccuracyPct",
        "estimatedMaterialFactCoveragePct"
    ],

    labels=[
        "Factual Accuracy",
        "Material Fact Coverage"
    ],

    title="LLM Judge: Estimated Summary Quality",

    ylabel="Estimated Score (%)",

    filename="02_accuracy_and_coverage.png",

    ylim=(0, 110),

    suffix="%"
)


# ==========================================
# 6. CHART 3: ESTIMATED ERROR RATES
# ==========================================

grouped_chart(

    metrics=[
        "estimatedIncorrectClaimRatePct",
        "estimatedHallucinationRatePct",
        "estimatedCriticalErrorRatePct"
    ],

    labels=[
        "Incorrect Claims",
        "Unsupported Claims",
        "Critical Errors"
    ],

    title="LLM Judge: Estimated Error Rates",

    ylabel="Estimated Error Rate (%)",

    filename="03_error_rates.png",

    ylim=(
        0,
        max(
            10,
            df[
                [
                    "estimatedIncorrectClaimRatePct",
                    "estimatedHallucinationRatePct",
                    "estimatedCriticalErrorRatePct"
                ]
            ].max().max() * 1.25
        )
    ),

    suffix="%"
)


# ==========================================
# 7. CHART 4: SUBJECTIVE QUALITY
# ==========================================

grouped_chart(

    metrics=[
        "averageUsefulness",
        "averageReadability",
        "averageConciseness"
    ],

    labels=[
        "Usefulness",
        "Readability",
        "Conciseness"
    ],

    title="LLM Judge: Qualitative Ratings",

    ylabel="Average Rating (out of 5)",

    filename="04_qualitative_ratings.png",

    ylim=(0, 5.8)
)


# ==========================================
# 8. CHART 5: SUMMARY LENGTH
# ==========================================

fig, ax = plt.subplots(
    figsize=(8, 5.5)
)

values = df["averageWordCount"].values

ax.axhspan(
    100,
    150,
    color="#16A34A",
    alpha=0.12,
    label="Target: 100-150 words"
)

bars = ax.bar(
    VERSION_LABELS,
    values,
    color=COLORS,
    width=0.5
)

for bar in bars:

    height = bar.get_height()

    ax.annotate(
        f"{height:.2f}",
        (
            bar.get_x() + bar.get_width() / 2,
            height
        ),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center"
    )

ax.set_title(
    "Average Summary Length",
    fontweight="bold",
    pad=15
)

ax.set_ylabel("Word Count")

ax.set_ylim(
    0,
    max(values.max() * 1.2, 180)
)

ax.legend(frameon=False)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)

save_chart("05_summary_length.png")


# ==========================================
# 9. CHART 6: GENERATION LATENCY
# ==========================================

grouped_chart(

    metrics=[
        "averageLatencySeconds",
        "medianLatencySeconds"
    ],

    labels=[
        "Mean Latency",
        "Median Latency"
    ],

    title="Summary Generation Latency",

    ylabel="Processing Time (Seconds)",

    filename="06_generation_latency.png"
)


# ==========================================
# 10. CHART 7: TOKEN USAGE
# ==========================================

grouped_chart(

    metrics=[
        "averageInputTokens",
        "averageOutputTokens"
    ],

    labels=[
        "Input Tokens",
        "Output Tokens"
    ],

    title="Average Token Consumption",

    ylabel="Tokens per Summary",

    filename="07_token_consumption.png"
)


# ==========================================
# 11. OPTIONAL CHART: COMPANY-LEVEL RESULTS
# ==========================================

pivot = rules.pivot(
    index="registrationNumber",
    columns="version",
    values="keyFactMatchRatePct"
)

pivot = pivot.sort_index()

fig, ax = plt.subplots(
    figsize=(10, 9)
)

y = np.arange(len(pivot))

height = 0.35

ax.barh(
    y - height / 2,
    pivot["V1"],
    height,
    color=COLORS[0],
    label=VERSION_LABELS[0]
)

ax.barh(
    y + height / 2,
    pivot["V2"],
    height,
    color=COLORS[1],
    label=VERSION_LABELS[1]
)

ax.set_yticks(y)

ax.set_yticklabels(pivot.index)

ax.invert_yaxis()

ax.set_xlim(0, 105)

ax.set_xlabel("Key Fact Matching Rate (%)")

ax.set_title(
    "Company-Level Performance Comparison",
    fontweight="bold",
    pad=15
)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=2,
    frameon=False
)

ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.3
)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)

save_chart("08_company_level_comparison.png")


# ==========================================
# 12. FINAL REPORT
# ==========================================

print("\n" + "=" * 60)

print("VISUALISATION COMPLETED")

print("=" * 60)

print("Figures directory:", OUTPUT_DIR)

print(
    "PNG files available:",
    len(list(OUTPUT_DIR.glob("*.png")))
)
