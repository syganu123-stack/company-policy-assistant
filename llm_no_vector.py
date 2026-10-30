
import csv
import json
import os
import time
import requests

MODEL = "gemini-3-flash-preview"

def ask_llm(question):
    start = time.perf_counter()

    with open("company_policies.csv", "r",
              encoding="utf-8-sig") as file:
        policies = list(csv.DictReader(file))

    policy_data = json.dumps(policies, ensure_ascii=False)

    prompt = f"""
You are a company policy assistant.

Answer the employee's question using ONLY the
company policies provided below.

Rules:
1. Do not invent company policies.
2. Identify the relevant policy by its exact title.
3. If the database does not contain the answer,
   clearly state that the information is unavailable.
4. Return ONLY valid JSON with these fields:
   "answer" and "policy".

Company policies:
{policy_data}

Employee question:
{question}
"""

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{MODEL}:generateContent"
    )

    response = requests.post(
        url,
        headers={
            "x-goog-api-key": os.environ["GEMINI_API_KEY"],
            "Content-Type": "application/json"
        },
        json={
            "contents": [
                {"parts": [{"text": prompt}]}
            ],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json"
            }
        },
        timeout=90
    )

    response.raise_for_status()
    data = response.json()

    text = data["candidates"][0]["content"]["parts"][0]["text"]
    result = json.loads(text)

    usage = data.get("usageMetadata", {})
    elapsed = time.perf_counter() - start

    return {
        "answer": result.get("answer", ""),
        "policy": result.get("policy", "None"),
        "response_time": elapsed,
        "tokens": usage.get("totalTokenCount", 0),
        "input_tokens": usage.get("promptTokenCount", 0),
        "output_tokens": usage.get(
            "candidatesTokenCount", 0
        )
    }


if __name__ == "__main__":
    question = input("Ask a company policy question: ")
    result = ask_llm(question)

    print("\nAnswer:", result["answer"])
    print("Relevant policy:", result["policy"])
    print(
        "Response time:",
        round(result["response_time"], 4),
        "seconds"
    )
    print("Total tokens used:", result["tokens"])
    print("Input tokens:", result["input_tokens"])
    print("Output tokens:", result["output_tokens"])
