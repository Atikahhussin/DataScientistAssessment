import os
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError

load_dotenv(override=True)

MODEL_ID = os.getenv("MODEL_ID")
API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    timeout=60,
    max_retries=0
)

print("Testing model:", MODEL_ID)
print("API key loaded:", bool(API_KEY))

try:
    response = client.chat.completions.create(
        model=MODEL_ID,
        temperature=0,
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": """
                Summarise the following company in 50 words.

                Company: ABC Trading Sdn. Bhd.
                Status: Active
                Revenue 2023: RM500,000
                Revenue 2024: RM600,000
                Net Profit 2024: RM50,000
                Litigation: No recorded cases.
                """
            }
        ]
    )

    print("\nAPI connection successful!")
    print("Model:", response.model)
    print("\nSummary:")
    print(response.choices[0].message.content)
    print("\nToken usage:", response.usage)

except RateLimitError as e:
    print("\nRate limit exceeded.")
    print("Error details:", e.body)
    print("Retry after:", e.response.headers.get("retry-after"))

except APIError as e:
    print("\nAPI error:", e)