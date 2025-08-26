"""LLM-based Faithfulness example for Ragas.

What this measures
    Factual consistency of the `response` w.r.t. the `retrieved_contexts`.
    Extracts claims from the response and verifies each claim against the
    provided context to detect hallucinations.

When to use
    - RAG generation quality checks to prevent unsupported statements.
    - Debugging hallucinations during development.

Trade-offs
    - LLM judge cost and latency; detailed but more expensive than
      coarse proxies (e.g., ResponseGroundedness).

Implementation notes
    - Two-step process (claim extraction + verification).
    - Domain-adapt prompts or use specialized detectors for robustness.

Run
    uv run docs/ragas/examples/faithfulness.py

Requires
    OPENAI_API_KEY set. Uses LangChain wrappers.
"""

# %%
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness


def build_sample() -> SingleTurnSample:
    return SingleTurnSample(
        user_input="What is the tallest mountain on Earth?",
        response="Mount Everest is the tallest mountain on Earth.",
        retrieved_contexts=[
            "Mount Everest stands at 8,848.86 meters and is Earth's highest mountain.",
            "K2 is the second-highest mountain in the world.",
        ],
    )


def main() -> None:
    if os.getenv("OPENAI_API_KEY") is None:
        print("Skipping LLM example: OPENAI_API_KEY not set.")
        return
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini", temperature=0))
    metric = Faithfulness(llm=llm)
    sample = build_sample()
    try:
        score = metric.single_turn_score(sample)
        print("Faithfulness:", score)
    except Exception as e:
        print("Skipping LLM example due to error:", e)


if __name__ == "__main__":
    main()
