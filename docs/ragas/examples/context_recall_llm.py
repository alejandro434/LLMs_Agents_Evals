"""LLM-based Context Recall example for Ragas.

What this measures
    Completeness of retrieval: the share of factual claims in the
    `reference` that are supported by `retrieved_contexts`.

When to use
    - Auditing whether retrieval surfaces all necessary facts.
    - RAG pipeline debugging when answers lack critical details.

Trade-offs
    - Requires a judge LLM and a `reference` answer. Cost and latency
      apply; results depend on LLM quality.

Implementation notes
    - Decomposes reference into claims, then verifies support in contexts.
    - Consider domain-adapting prompts for better consistency.

Run
    uv run docs/ragas/examples/context_recall_llm.py

Requires
    OPENAI_API_KEY set. Uses LangChain wrappers.
"""

# %%
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall


def build_sample() -> SingleTurnSample:
    return SingleTurnSample(
        user_input="Who discovered penicillin?",
        response="Alexander Fleming discovered penicillin in 1928.",
        retrieved_contexts=[
            "Alexander Fleming discovered penicillin in 1928 at St. Mary's Hospital.",
            "Penicillin became widely used during World War II.",
        ],
        reference="Alexander Fleming discovered penicillin.",
    )


def main() -> None:
    if os.getenv("OPENAI_API_KEY") is None:
        print("Skipping LLM example: OPENAI_API_KEY not set.")
        return
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini", temperature=0))
    metric = LLMContextRecall(llm=llm)
    sample = build_sample()
    try:
        score = metric.single_turn_score(sample)
        print("LLMContextRecall:", score)
    except Exception as e:
        print("Skipping LLM example due to error:", e)


if __name__ == "__main__":
    main()
