
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

EVALUATION_DIR = ROOT / "evaluation"

FINAL_PATH = (
    EVALUATION_DIR / "final_comparison.csv"
)

FULL_PATH = (
    EVALUATION_DIR / "full_dataset_comparison.csv"
)

RULE_PATH = (
    EVALUATION_DIR / "final_rule_based_scores.csv"
)

OUTPUT_DIR = (
    EVALUATION_DIR / "final_figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VERSIONS = ["V1", "V2", "V3"]

VERSION_LABELS = [
    "V1 Baseline",
    "V2 Improved",
    "V3 Concise"
]

COLORS = [
    "#94A3B8",
    "#3B82F6",
    "#16A34A"
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False
})


# ==========================================
# 2. LOAD RESULTS
# ==========================================

final = pd.read_csv(FINAL_PATH)

full = pd.read_csv(FULL_PATH)

rules = pd.read_csv(RULE_PATH)

final = final.set_index("version").loc[VERSIONS]

full = full.set_index("version").loc[VERSIONS]

print("=" * 60)

print("FINAL RESULT VISUALISATION")

print("=" * 60)


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def save_figure(fig, filename):

    path = OUTPUT_DIR / filename

    fig.tight_layout()

    fig.savefig(
        path,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print("Saved:", filename)


def add_labels(ax, bars, decimals=2, suffix=""):

    for bar in bars:

        height = bar.get_height()

        if not np.isfinite(height):

            continue

        ax.annotate(
            f"{height:.{decimals}f}{suffix}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10
        )


def style_axis(ax):

    ax.set_axisbelow(True)

    ax.grid(
        axis="y",
        alpha=0.2,
        linestyle="--"
    )


def single_bar(
    dataframe,
    field,
    title,
    ylabel,
    filename,
    decimals=2,
    suffix="",
    upper_limit=None
):

    values = dataframe[field].to_numpy()

    if not np.isfinite(values).all():

        print(
            f"Skipped {filename}: missing values."
        )

        return

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    bars = ax.bar(
        VERSION_LABELS,
        values,
        color=COLORS,
        width=0.55
    )

    ax.set_title(
        title,
        fontweight="bold",
        pad=18
    )

    ax.set_ylabel(ylabel)

    style_axis(ax)

    add_labels(
        ax,
        bars,
        decimals,
        suffix
    )

    if upper_limit is not None:

        ax.set_ylim(0, upper_limit)

    else:

        ax.set_ylim(
            0,
            max(values) * 1.2
        )

    save_figure(
        fig,
        filename
    )


def grouped_bar(
    dataframe,
    fields,
    labels,
    title,
    ylabel,
    filename,
    decimals=2,
    suffix="",
    upper_limit=None
):

    if dataframe[fields].isna().any().any():

        print(
            f"Skipped {filename}: missing values."
        )

        return

    fig, ax = plt.subplots(
        figsize=(10, 5.5)
    )

    x = np.arange(len(fields))

    width = 0.23

    for index, version in enumerate(VERSIONS):

        values = [

            dataframe.loc[version, field]

            for field in fields
        ]

        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=VERSION_LABELS[index],
            color=COLORS[index]
        )

        add_labels(
            ax,
            bars,
            decimals,
            suffix
        )

    ax.set_title(
        title,
        fontweight="bold",
        pad=18
    )

    ax.set_ylabel(ylabel)

    ax.set_xticks(x)

    ax.set_xticklabels(labels)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False
    )

    style_axis(ax)

    if upper_limit is not None:

        ax.set_ylim(0, upper_limit)

    else:

        ax.set_ylim(
            0,
            dataframe[fields].max().max() * 1.2
        )

    save_figure(
        fig,
        filename
    )


# ==========================================
# 4. KEY FACT AND NUMERICAL MATCHING
# ==========================================

grouped_bar(
    dataframe=final,

    fields=[
        "keyFactMatchRatePct",
        "numericalMatchRatePct"
    ],

    labels=[
        "Key fact matching",
        "Numerical matching"
    ],

    title=(
        "Rule-Based Information Coverage\n"
        "Held-Out Sample (10 Companies)"
    ),

    ylabel="Matching rate (%)",

    filename="01_information_coverage.png",

    suffix="%",

    upper_limit=100
)


# ==========================================
# 5. HELD-OUT WORD COUNT
# ==========================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

bars = ax.bar(
    VERSION_LABELS,
    final["averageWordCount"],
    color=COLORS,
    width=0.55
)

ax.axhspan(
    100,
    150,
    color="#16A34A",
    alpha=0.12,
    label="Target: 100–150 words"
)

ax.axhline(
    150,
    color="#16A34A",
    linestyle="--",
    linewidth=1.2
)

ax.set_title(
    "Average Summary Length\n"
    "Held-Out Sample (10 Companies)",
    fontweight="bold",
    pad=18
)

ax.set_ylabel("Average word count")

ax.set_ylim(
    0,
    final["averageWordCount"].max() * 1.2
)

add_labels(ax, bars)

style_axis(ax)

ax.legend(frameon=False)

save_figure(
    fig,
    "02_heldout_summary_length.png"
)


# ==========================================
# 6. WORD LIMIT COMPLIANCE
# ==========================================

compliance = pd.DataFrame({

    "heldOut":
        final["wordLimitCompliancePct"],

    "fullDataset":
        full["wordLimitCompliancePct"]
})

grouped_bar(
    dataframe=compliance,

    fields=[
        "heldOut",
        "fullDataset"
    ],

    labels=[
        "Held-out sample (10)",
        "Full dataset (50)"
    ],

    title="Word Limit Compliance",

    ylabel="Compliant summaries (%)",

    filename="03_word_limit_compliance.png",

    decimals=0,

    suffix="%",

    upper_limit=110
)


# ==========================================
# 7. FULL DATASET SUMMARY LENGTH
# ==========================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

bars = ax.bar(
    VERSION_LABELS,
    full["averageWordCount"],
    color=COLORS,
    width=0.55
)

ax.axhspan(
    100,
    150,
    color="#16A34A",
    alpha=0.12,
    label="Target: 100–150 words"
)

ax.axhline(
    150,
    color="#16A34A",
    linestyle="--",
    linewidth=1.2
)

ax.set_title(
    "Average Summary Length\n"
    "Full Dataset (50 Companies)",
    fontweight="bold",
    pad=18
)

ax.set_ylabel("Average word count")

ax.set_ylim(
    0,
    full["averageWordCount"].max() * 1.2
)

add_labels(ax, bars)

style_axis(ax)

ax.legend(frameon=False)

save_figure(
    fig,
    "04_full_dataset_summary_length.png"
)


# ==========================================
# 8. GENERATION LATENCY
# ==========================================

latency = pd.DataFrame({

    "heldOut":
        final["averageLatencySeconds"],

    "fullDataset":
        full["averageLatencySeconds"]
})

grouped_bar(
    dataframe=latency,

    fields=[
        "heldOut",
        "fullDataset"
    ],

    labels=[
        "Held-out sample (10)",
        "Full dataset (50)"
    ],

    title="Average Generation Latency",

    ylabel="Seconds per summary",

    filename="05_generation_latency.png"
)


# ==========================================
# 9. FULL DATASET TOKEN CONSUMPTION
# ==========================================

grouped_bar(
    dataframe=full,

    fields=[
        "averageInputTokens",
        "averageOutputTokens"
    ],

    labels=[
        "Input tokens",
        "Output tokens"
    ],

    title=(
        "Average Token Consumption\n"
        "Full Dataset (50 Companies)"
    ),

    ylabel="Average tokens per summary",

    filename="06_token_consumption.png",

    decimals=0
)


# ==========================================
# 10. TOTAL API REQUESTS
# ==========================================

single_bar(
    dataframe=full,

    field="totalApiRequests",

    title=(
        "Recorded API Requests\n"
        "Full Dataset (50 Companies)"
    ),

    ylabel="Number of API requests",

    filename="07_api_requests.png",

    decimals=0
)


# ==========================================
# 11. COMPANY-LEVEL KEY FACT MATCHING
# ==========================================

pivot = rules.pivot(
    index="registrationNumber",
    columns="version",
    values="keyFactMatchRatePct"
)

pivot = pivot[VERSIONS]

fig, ax = plt.subplots(
    figsize=(12, 6)
)

x = np.arange(len(pivot))

width = 0.25

for index, version in enumerate(VERSIONS):

    ax.bar(
        x + (index - 1) * width,
        pivot[version],
        width,
        color=COLORS[index],
        label=VERSION_LABELS[index]
    )

ax.set_title(
    "Company-Level Key Fact Matching\n"
    "Held-Out Sample",
    fontweight="bold",
    pad=18
)

ax.set_ylabel("Key fact matching rate (%)")

ax.set_xlabel("Company registration number")

ax.set_xticks(x)

ax.set_xticklabels(
    pivot.index,
    rotation=45,
    ha="right"
)

ax.set_ylim(0, 110)

ax.legend(
    ncol=3,
    frameon=False
)

style_axis(ax)

save_figure(
    fig,
    "08_company_level_comparison.png"
)


# ==========================================
# 12. INFORMATION COVERAGE VS LENGTH
# ==========================================

fig, ax = plt.subplots(
    figsize=(8, 5.5)
)

ax.axvspan(
    100,
    150,
    color="#16A34A",
    alpha=0.10,
    label="Target word range"
)

for index, version in enumerate(VERSIONS):

    words = final.loc[
        version,
        "averageWordCount"
    ]

    coverage = final.loc[
        version,
        "keyFactMatchRatePct"
    ]

    ax.scatter(
        words,
        coverage,
        color=COLORS[index],
        s=150,
        zorder=3
    )

    ax.annotate(
        VERSION_LABELS[index],
        (words, coverage),
        xytext=(8, 8),
        textcoords="offset points"
    )

ax.set_title(
    "Information Coverage vs. Summary Length\n"
    "Held-Out Sample",
    fontweight="bold",
    pad=18
)

ax.set_xlabel("Average word count")

ax.set_ylabel("Key fact matching rate (%)")

ax.set_xlim(
    75,
    final["averageWordCount"].max() + 100
)

ax.set_ylim(0, 100)

ax.legend(
    frameon=False,
    loc="lower right"
)

style_axis(ax)

save_figure(
    fig,
    "09_coverage_length_tradeoff.png"
)


# ==========================================
# 13. FULL DATASET COMPLIANCE COUNTS
# ==========================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

compliant = full["compliantSummaries"]

noncompliant = full["nonCompliantSummaries"]

bars1 = ax.bar(
    VERSION_LABELS,
    compliant,
    color="#16A34A",
    label="Within 100–150 words"
)

bars2 = ax.bar(
    VERSION_LABELS,
    noncompliant,
    bottom=compliant,
    color="#CBD5E1",
    label="Outside target"
)

for bars in (bars1, bars2):

    labels = [

        str(int(bar.get_height()))

        if bar.get_height() > 0

        else ""

        for bar in bars
    ]

    ax.bar_label(
        bars,
        labels=labels,
        label_type="center"
    )

ax.set_title(
    "Summary Length Compliance Counts\n"
    "Full Dataset",
    fontweight="bold",
    pad=18
)

ax.set_ylabel("Number of summaries")

ax.set_ylim(
    0,
    full["completedSummaries"].max() * 1.15
)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.1),
    ncol=2,
    frameon=False
)

style_axis(ax)

save_figure(
    fig,
    "10_compliance_counts.png"
)


# ==========================================
# 14. COMPLETION
# ==========================================

print("\n" + "=" * 60)

print("FINAL VISUALISATION COMPLETED")

print("=" * 60)

print("Output directory:", OUTPUT_DIR)

print(
    "Total PNG files:",
    len(list(OUTPUT_DIR.glob("*.png")))
)

print(
    "\nNote: Matching rates measure heuristic "
    "information coverage, not verified "
    "factual accuracy."
)

print(
    "Incomplete LLM judge results are excluded."
)

print(
    "V3 generation latency and token usage "
    "include recorded revision requests."
)
