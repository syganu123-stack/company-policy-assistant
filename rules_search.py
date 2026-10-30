
import csv
import re
import time

def load_policies():
    with open("company_policies.csv", "r", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))

def search_policy(question):
    start = time.perf_counter()
    policies = load_policies()

    stopwords = {
        "the", "a", "an", "is", "are", "do", "does",
        "i", "my", "me", "what", "when", "how",
        "can", "could", "about", "for", "to",
        "in", "of", "and", "on", "our", "we"
    }

    words = set(re.findall(r"\b\w+\b", question.lower()))
    words -= stopwords

    best_policy = None
    best_score = 0

    for policy in policies:
        title_words = set(re.findall(
            r"\b\w+\b", policy["title"].lower()
        ))
        text_words = set(re.findall(
            r"\b\w+\b", policy["policy_text"].lower()
        ))

        score = (
            3 * len(words & title_words)
            + len(words & text_words)
        )

        if score > best_score:
            best_score = score
            best_policy = policy

    elapsed = time.perf_counter() - start

    if best_policy is None:
        return {
            "answer": "No matching policy found.",
            "policy": "None",
            "response_time": elapsed,
            "tokens": 0
        }

    return {
        "answer": best_policy["policy_text"],
        "policy": best_policy["title"],
        "response_time": elapsed,
        "tokens": 0
    }

if __name__ == "__main__":
    question = input("Ask a company policy question: ")
    result = search_policy(question)

    print("\nAnswer:", result["answer"])
    print("Relevant policy:", result["policy"])
    print("Response time:", round(result["response_time"], 4), "seconds")
    print("LLM tokens used:", result["tokens"])
