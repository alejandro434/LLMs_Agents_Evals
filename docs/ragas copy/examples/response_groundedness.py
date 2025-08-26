"""NVIDIA ResponseGroundedness example for Ragas.

What this measures
    Degree to which `response` is supported by `retrieved_contexts`, using
    two independent LLM judges for robustness.

When to use
    - Production monitoring to detect hallucinations cheaply.
    - Faster alternative to detailed `Faithfulness` during CI.

Trade-offs
    - Less granular feedback than claim-level verification.
    - More robust and lower token usage vs. multi-step metrics.

Implementation notes
    - Dual-judge design reduces sensitivity to prompt variance.

Run
    uv run docs/ragas/examples/response_groundedness.py

Requires
    OPENAI_API_KEY set. Uses LangChain wrappers.
"""

# %%
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ResponseGroundedness


def build_sample() -> SingleTurnSample:
    return SingleTurnSample(
        user_input="Name a landmark located in Paris",
        response="The Eiffel Tower is located in Paris.",
        retrieved_contexts=[
            "The Eiffel Tower is a landmark in Paris, France.",
            "The Statue of Liberty is a landmark in New York City.",
        ],
    )


def main() -> None:
    if os.getenv("OPENAI_API_KEY") is None:
        print("Skipping LLM example: OPENAI_API_KEY not set.")
        return
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini", temperature=0))
    metric = ResponseGroundedness(llm=llm)
    sample = build_sample()
    score = metric.single_turn_score(sample)
    print("ResponseGroundedness:", score)


if __name__ == "__main__":
    main()
