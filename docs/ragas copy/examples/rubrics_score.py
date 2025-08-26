"""LLM-based RubricsScore example for Ragas.

What this measures
    A rubric-driven discrete score (e.g., 1–5) for flexible, domain-specific
    quality evaluation with natural-language descriptors for each level.

When to use
    - Planner/Reasoner outputs needing nuanced, human-like grading.
    - QA sign-off criteria encoded in a consistent automated evaluator.

Trade-offs
    - Requires careful rubric design; LLM cost applies.
    - Less granular than continuous scores but more interpretable.

Implementation notes
    - Provide clear, unambiguous descriptions for each score level.
    - Calibrate with a few gold examples if needed (prompt adaptation).

Run
    uv run docs/ragas/examples/rubrics_score.py
"""

# %%
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import RubricsScore


def build_sample() -> SingleTurnSample:
    return SingleTurnSample(
        user_input="Explain RAG in one sentence",
        response=(
            "RAG retrieves relevant documents and uses them to ground the model's"
            " generated answer."
        ),
    )


def main() -> None:
    if os.getenv("OPENAI_API_KEY") is None:
        print("Skipping LLM example: OPENAI_API_KEY not set.")
        return
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    metric = RubricsScore(
        llm=llm,
        rubrics={
            "score1_description": ("Completely inaccurate or irrelevant explanation"),
            "score2_description": ("Mostly inaccurate with little relevance"),
            "score3_description": ("Partially correct but missing key details"),
            "score4_description": ("Mostly correct and concise explanation"),
            "score5_description": ("Accurate, complete, and concise explanation"),
        },
    )
    sample = build_sample()
    score = metric.single_turn_score(sample)
    print("RubricsScore (1-5):", score)


if __name__ == "__main__":
    main()
