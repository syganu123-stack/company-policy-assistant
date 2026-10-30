
import csv
import json
import os
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL = "gemini-3-flash-preview"


def load_policies():
    with open("company_policies.csv", "r",
              encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


policies = load_policies()

documents = [
    (
        policy["title"] + " "
        + policy["title"] + " "
        + policy["department"] + " "
        + policy["category"] + " "
        + policy["policy_text"]
    )
    for policy in policies
]

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

policy_vectors = vectorizer.fit_transform(documents)


def ask_vector_llm(question):
    start = time.perf_counter()

    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(
        question_vector, policy_vectors
    )[0]

    top_indices = similarities.argsort()[::-1][:3]

    retrieved_policies = [
        policies[int(i)]
        for i in top_indices
        if similarities[int(i)] > 0
    ]

    if not retrieved_policies:
        return {
            "answer": "No matching policy found.",
            "policy": "None",
            "response_time": time.perf_counter() - start,
            "tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "retrieved_policies": []
        }

    policy_data = json.dumps(
        retrieved_policies,
        ensure_ascii=False
    )

    prompt = f"""
You are a company policy assistant.

Answer the employee question using ONLY the
retrieved company policies below.

Rules:
1. Never invent company policies.
2. Identify the relevant policy using its exact title.
3. If the retrieved policies do not contain the answer,
   say that the information is unavailable.
4. Return ONLY valid JSON containing:
   "answer" and "policy".

Retrieved company policies:
{policy_data}

Employee question:
{question}
"""

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{MODEL}:generateContent"
    )

    
    retry_strategy = Retry(
        total=2,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )

    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    try:
        response = session.post(
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
            timeout=30
        )
    finally:
        session.close()


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
        "output_tokens": usage.get("candidatesTokenCount", 0),
        "retrieved_policies": [
            p["title"] for p in retrieved_policies
        ]
    }


if __name__ == "__main__":
    question = input("Ask a company policy question: ")
    result = ask_vector_llm(question)

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
    print(
        "Retrieved policies:",
        result["retrieved_policies"]
    )
