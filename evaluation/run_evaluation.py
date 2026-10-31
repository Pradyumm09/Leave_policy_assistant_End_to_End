from dotenv import load_dotenv

load_dotenv()

import requests

from langsmith import Client, evaluate

BACKEND_URL = "http://127.0.0.1:8000"

DATASET_NAME = "leave-policy-assistant-eval-v1"


def target(inputs: dict) -> dict:

    question = inputs["question"]

    response = requests.post(
        f"{BACKEND_URL}/query",
        json={
            "question": question
        },
        timeout=60,
    )

    response.raise_for_status()

    result = response.json()

    return {
        "answer": result.get("answer", ""),
        "intent": result.get("intent"),
        "confidence": result.get("confidence"),
        "sources": result.get("sources", []),
    }


def intent_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
):

    expected_intent = reference_outputs.get(
        "intent"
    )

    actual_intent = outputs.get(
        "intent"
    )

    return {
        "key": "intent_correct",
        "score": (
            1.0
            if expected_intent == actual_intent
            else 0.0
        ),
    }


def section_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
):

    expected_section = reference_outputs.get(
        "expected_section"
    )

    if not expected_section:

        return {
            "key": "section_match",
            "score": 1.0,
        }

    sources = outputs.get(
        "sources",
        []
    )

    sections = [
        source.get("section", "")
        for source in sources
    ]

    match = any(
        expected_section.lower()
        in section.lower()
        for section in sections
    )

    return {
        "key": "section_match",
        "score": (
            1.0
            if match
            else 0.0
        ),
    }


def main():

    client = Client()

    results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[
            intent_evaluator,
            section_evaluator,
        ],
        experiment_prefix="leave-policy-rag",
        metadata={
            "project": "Leave Policy Assistant",
            "version": "1.0.0",
            "retriever": "FAISS",
            "embedding": "BAAI/bge-small-en-v1.5",
            "llm": "Groq",
        },
    )

    print(results)


if __name__ == "__main__":
    main()