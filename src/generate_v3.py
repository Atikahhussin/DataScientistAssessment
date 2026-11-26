
import os
import json
import csv
import time
import argparse

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError

from prompts import (
    V3_SYSTEM_PROMPT,
    V3_USER_PROMPT,
    V3_REVISION_PROMPT
)

from preprocess_v3 import preprocess_company_v3


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env", override=True)

DATA_PATH = ROOT / "data" / "companies.jsonl"

SPLIT_PATH = ROOT / "evaluation" / "sample_split.json"

FINAL_SPLIT_PATH = (
    ROOT / "evaluation" / "final_evaluation_split.json"
)

OUTPUT_PATH = ROOT / "summaries" / "v3_concise.jsonl"

LOG_PATH = ROOT / "evaluation" / "v3_generation_log.csv"

MODEL_ID = os.getenv("MODEL_ID")

API_KEY = os.getenv("OPENROUTER_API_KEY")

MIN_WORDS = 100
MAX_WORDS = 150
TARGET_WORDS = 125

if not MODEL_ID or not API_KEY:
    raise ValueError(
        "Missing MODEL_ID or OPENROUTER_API_KEY."
    )

if MODEL_ID == "openrouter/free":
    raise ValueError(
        "Select a fixed generation model."
    )

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

LOG_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# 2. COMMAND LINE ARGUMENTS
# ==========================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--split",
    choices=[
        "development",
        "evaluation",
        "additional",
        "final",
        "all"
    ],
    default="development"
)

args = parser.parse_args()


# ==========================================
# 3. API CLIENT
# ==========================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    timeout=120,
    max_retries=0
)


# ==========================================
# 4. LOAD DATA
# ==========================================

def load_jsonl(path):

    records = {}

    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            record = json.loads(line)

            company_id = record["registrationNumber"]

            if company_id in records:
                raise ValueError(
                    f"Duplicate record: {company_id}"
                )

            records[company_id] = record

    return records


companies = load_jsonl(DATA_PATH)

with SPLIT_PATH.open("r", encoding="utf-8") as f:
    splits = json.load(f)


# ==========================================
# 5. SELECT DATASET SPLIT
# ==========================================

if args.split == "all":

    selected_ids = list(companies.keys())

elif args.split == "final":

    with FINAL_SPLIT_PATH.open(
        "r", encoding="utf-8"
    ) as f:

        final_split = json.load(f)

    selected_ids = final_split["evaluation"]

else:

    selected_ids = splits[args.split]


existing = load_jsonl(OUTPUT_PATH)


# ==========================================
# 6. GENERATION LOG
# ==========================================

LOG_FIELDS = [
    "registrationNumber",
    "split",
    "requestedModel",
    "returnedModel",
    "latencySeconds",
    "inputTokens",
    "outputTokens",
    "wordCount",
    "initialWordCount",
    "revisionAttempts",
    "apiRequests",
    "withinWordLimit",
    "status",
    "error"
]


def write_log(row):

    write_header = (
        not LOG_PATH.exists()
        or LOG_PATH.stat().st_size == 0
    )

    with LOG_PATH.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=LOG_FIELDS
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)


# Check existing log compatibility.

if LOG_PATH.exists() and LOG_PATH.stat().st_size > 0:

    with LOG_PATH.open(
        "r", encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames != LOG_FIELDS:
            raise ValueError(
                "Existing V3 log has different columns."
            )

        previous_logs = list(reader)

    successful_models = {
        row["requestedModel"]
        for row in previous_logs
        if row.get("status", "").lower() == "success"
    }

    if successful_models and successful_models != {MODEL_ID}:

        raise ValueError(
            "Existing V3 summaries use a different model."
        )


# ==========================================
# 7. HELPER FUNCTIONS
# ==========================================

def count_words(summary):

    return len(summary.split())


def within_word_limit(summary):

    return (
        MIN_WORDS
        <= count_words(summary)
        <= MAX_WORDS
    )


def length_penalty(summary):

    words = count_words(summary)

    if words < MIN_WORDS:
        return MIN_WORDS - words

    if words > MAX_WORDS:
        return words - MAX_WORDS

    return 0


def request_summary(prompt):

    response = client.chat.completions.create(

        model=MODEL_ID,

        temperature=0,

        max_tokens=4096,

        extra_body={
            "reasoning": {
                "enabled": False
            }
        },

        messages=[
            {
                "role": "system",
                "content": V3_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    if not response.choices:
        raise ValueError(
            "No response choices returned."
        )

    choice = response.choices[0]

    summary = (
        choice.message.content or ""
    ).strip()

    if not summary:
        raise ValueError(
            "Empty summary returned."
        )

    if choice.finish_reason == "length":
        raise ValueError(
            "Summary was truncated."
        )

    usage = response.usage

    return {
        "summary": summary,

        "returnedModel": response.model,

        "inputTokens": (
            usage.prompt_tokens
            if usage else None
        ),

        "outputTokens": (
            usage.completion_tokens
            if usage else None
        )
    }


def add_tokens(first, second):

    if first is None or second is None:
        return None

    return first + second


# ==========================================
# 8. GENERATE V3 SUMMARIES
# ==========================================

print("=" * 60)
print("V3 CONCISE CREDIT-FOCUSED SUMMARISATION")
print("=" * 60)

print("Model:", MODEL_ID)
print("Dataset split:", args.split)
print("Selected companies:", len(selected_ids))
print("Previously completed:", len(existing))


for index, company_id in enumerate(
    selected_ids,
    start=1
):

    if company_id in existing:

        print(
            f"Skipping {company_id}: already generated"
        )

        continue

    print("\n" + "-" * 60)

    print(
        f"[{index}/{len(selected_ids)}] "
        f"Processing {company_id}"
    )

    start_time = time.perf_counter()

    log = {
        "registrationNumber": company_id,
        "split": args.split,
        "requestedModel": MODEL_ID,
        "returnedModel": "",
        "latencySeconds": "",
        "inputTokens": "",
        "outputTokens": "",
        "wordCount": "",
        "initialWordCount": "",
        "revisionAttempts": 0,
        "apiRequests": 0,
        "withinWordLimit": False,
        "status": "failed",
        "error": ""
    }

    stop_after_save = False

    try:

        record = companies[company_id]

        processed = preprocess_company_v3(
            record
        )

        company_data = json.dumps(
            processed,
            ensure_ascii=False
        )

        prompt = V3_USER_PROMPT.format(
            company_data=company_data
        )

        # ----------------------------------
        # Initial generation
        # ----------------------------------

        log["apiRequests"] += 1

        initial = request_summary(prompt)

        summary = initial["summary"]

        initial_words = count_words(summary)

        log["initialWordCount"] = initial_words

        log["returnedModel"] = initial[
            "returnedModel"
        ]

        log["inputTokens"] = initial[
            "inputTokens"
        ]

        log["outputTokens"] = initial[
            "outputTokens"
        ]

        print(
            "Initial word count:",
            initial_words
        )

        # ----------------------------------
        # Automatic length revision
        # ----------------------------------

        if not within_word_limit(summary):

            print(
                "Outside word limit. "
                "Attempting one revision..."
            )

            revision_prompt = (
                V3_REVISION_PROMPT.format(
                    company_data=company_data,
                    summary=summary
                )
            )

            log["revisionAttempts"] = 1

            time.sleep(4)

            try:

                log["apiRequests"] += 1

                revised = request_summary(
                    revision_prompt
                )

                revised_summary = revised[
                    "summary"
                ]

                revised_words = count_words(
                    revised_summary
                )

                log["inputTokens"] = add_tokens(
                    log["inputTokens"],
                    revised["inputTokens"]
                )

                log["outputTokens"] = add_tokens(
                    log["outputTokens"],
                    revised["outputTokens"]
                )

                print(
                    "Revised word count:",
                    revised_words
                )

                if (
                    revised["returnedModel"]
                    != initial["returnedModel"]
                ):

                    raise ValueError(
                        "Revision returned a different model."
                    )

                if (
                    length_penalty(revised_summary)
                    < length_penalty(summary)
                ):

                    summary = revised_summary

                    print(
                        "Using revised summary."
                    )

                else:

                    print(
                        "Keeping initial summary."
                    )

            except RateLimitError as e:

                print(
                    "Rate limit during revision. "
                    "Keeping initial summary."
                )

                log["error"] = str(e)

                stop_after_save = True

            except (APIError, ValueError) as e:

                print(
                    "Revision failed:",
                    e
                )

                log["error"] = str(e)

        # ----------------------------------
        # Final validation
        # ----------------------------------

        final_words = count_words(summary)

        compliant = within_word_limit(summary)

        log["wordCount"] = final_words

        log["withinWordLimit"] = compliant

        log["latencySeconds"] = round(
            time.perf_counter() - start_time,
            2
        )

        log["status"] = "success"

        # ----------------------------------
        # Save summary
        # ----------------------------------

        output_record = {
            "registrationNumber": company_id,
            "summary": summary
        }

        with OUTPUT_PATH.open(
            "a", encoding="utf-8"
        ) as f:

            f.write(
                json.dumps(
                    output_record,
                    ensure_ascii=False
                ) + "\n"
            )

        existing[company_id] = output_record

        write_log(log)

        print("SUCCESS")

        print(
            "Final word count:",
            final_words
        )

        print(
            "Word limit compliant:",
            compliant
        )

        print(
            "Latency:",
            log["latencySeconds"],
            "seconds"
        )

        print(
            "API requests:",
            log["apiRequests"]
        )

        if stop_after_save:
            break

    except RateLimitError as e:

        log["error"] = str(e)

        log["latencySeconds"] = round(
            time.perf_counter() - start_time,
            2
        )

        write_log(log)

        print("\nRate limit reached.")
        print("Error details:", e.body)

        print(
            "Retry after:",
            e.response.headers.get("retry-after")
        )

        print(
            "Run the same command later to resume."
        )

        break

    except (
        APIError,
        ValueError,
        KeyError,
        TypeError
    ) as e:

        log["error"] = str(e)

        log["latencySeconds"] = round(
            time.perf_counter() - start_time,
            2
        )

        write_log(log)

        print(
            "Generation failed:",
            e
        )

        break

    time.sleep(4)


# ==========================================
# 9. FINAL REPORT
# ==========================================

completed = sum(
    company_id in existing
    for company_id in selected_ids
)

selected_summaries = [
    existing[company_id]["summary"]
    for company_id in selected_ids
    if company_id in existing
]

print("\n" + "=" * 60)
print("V3 GENERATION FINISHED")
print("=" * 60)

print("Model:", MODEL_ID)
print("Dataset split:", args.split)
print("Selected companies:", len(selected_ids))
print("Completed in selected split:", completed)
print("Total saved summaries:", len(existing))

print(
    "Remaining in selected split:",
    len(selected_ids) - completed
)

if selected_summaries:

    average_words = sum(
        count_words(summary)
        for summary in selected_summaries
    ) / len(selected_summaries)

    compliance = sum(
        within_word_limit(summary)
        for summary in selected_summaries
    ) / len(selected_summaries) * 100

    print(
        "Average word count:",
        round(average_words, 2)
    )

    print(
        "Word limit compliance:",
        round(compliance, 2),
        "%"
    )

print("\nSummary output:")
print(OUTPUT_PATH)

print("\nGeneration log:")
print(LOG_PATH)
