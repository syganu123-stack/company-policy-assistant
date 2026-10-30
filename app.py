
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

from rules_search import search_policy
from llm_no_vector import ask_llm
from llm_vector import ask_vector_llm

app = FastAPI(
    title="Company Policy Assistant",
    description="Comparison of three policy search methods"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/compare")
def compare(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a policy question."
        )

    methods = {
        "rules_based": search_policy,
        "llm_no_vector": ask_llm,
        "llm_with_vector": ask_vector_llm
    }

    results = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            name: executor.submit(function, question)
            for name, function in methods.items()
        }

        for name, future in futures.items():
            try:
                results[name] = future.result(timeout=100)
                results[name]["status"] = "success"

            except Exception as error:
                results[name] = {
                    "status": "error",
                    "error": str(error),
                    "answer": "Method failed.",
                    "policy": "None",
                    "response_time": None,
                    "tokens": None
                }

    return {
        "question": question,
        "results": results
    }
