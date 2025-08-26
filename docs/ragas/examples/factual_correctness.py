"""LLM-based FactualCorrectness example for Ragas.

What this measures
    Claim-level agreement between `response` and `reference` using NLI-like
    comparison. Produces precision/recall/F1 of factual overlap.

When to use
    - End-to-end answer quality when a gold reference exists.
    - Debug what facts are missing or extra vs. the reference.

Trade-offs
    - Requires an LLM judge and a reference; cost applies.
    - More explainable than pure similarity but slower.

Implementation notes
    - Controls for `atomicity` and `coverage` to tune granularity.

Run
    uv run docs/ragas/examples/factual_correctness.py

Requires
    OPENAI_API_KEY set. Uses LangChain wrappers.
"""

# %%
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness


def build_sample() -> SingleTurnSample:
    return SingleTurnSample(
        user_input="Who wrote The Odyssey?",
        response="The Odyssey was written by Homer.",
        reference=(
            "Homer is traditionally credited as the author of The Iliad and The Odyssey."
        ),
    )


def main() -> None:
    if os.getenv("OPENAI_API_KEY") is None:
        print("Skipping LLM example: OPENAI_API_KEY not set.")
        return
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini", temperature=0))
    metric = FactualCorrectness(llm=llm)
    sample = build_sample()
    try:
        score = metric.single_turn_score(sample)
        print("FactualCorrectness:", score)
    except Exception as e:
        print("Skipping LLM example due to error:", e)


if __name__ == "__main__":
    main()
