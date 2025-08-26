"""NVIDIA ContextRelevance example for Ragas.

What this measures
    Relevance of `retrieved_contexts` to the `user_input` using a dual judge
    setup for stable scoring.

When to use
    - Monitoring retrieval quality at scale.
    - Comparing retriever configurations in A/B tests.

Trade-offs
    - Less explainable than token-level/claim-level reasoning.
    - Cheaper than multi-step precision/recall with references.

Implementation notes
    - Score is a normalized average from two independent judgments.

Run
    uv run docs/ragas/examples/context_relevance.py

Requires
    OPENAI_API_KEY set. Uses LangChain wrappers.
"""

# %%
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextRelevance


def build_sample() -> SingleTurnSample:
    return SingleTurnSample(
        user_input="Who was the first person to walk on the Moon?",
        response="Neil Armstrong was the first person to walk on the Moon.",
        retrieved_contexts=[
            "Neil Armstrong took the first steps on the lunar surface in 1969.",
            "Buzz Aldrin was the second person to walk on the Moon.",
            "The Amazon rainforest spans multiple countries in South America.",
        ],
    )


def main() -> None:
    if os.getenv("OPENAI_API_KEY") is None:
        print("Skipping LLM example: OPENAI_API_KEY not set.")
        return
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini", temperature=0))
    metric = ContextRelevance(llm=llm)
    sample = build_sample()
    score = metric.single_turn_score(sample)
    print("ContextRelevance:", score)


if __name__ == "__main__":
    main()
