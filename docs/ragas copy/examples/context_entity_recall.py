"""Context Entity Recall example for Ragas.

What this measures
    Recall of named entities present in the `reference` that also appear
    in the `retrieved_contexts`. Useful for entity-centric tasks.

When to use
    - Legal/medical/historical domains where entity coverage matters.
    - Validating that retrieval surfaces all critical actors/dates/terms.

Trade-offs
    - LLM-based extraction can be costlier and model-dependent.
    - Entity presence does not imply that the final answer used them.

Implementation notes
    - Requires an LLM to extract and compare entity sets.
    - Pair with `Faithfulness` for generation correctness.

Run
    uv run docs/ragas/examples/context_entity_recall.py
"""

# %%
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextEntityRecall


def build_sample() -> SingleTurnSample:
    """Create a sample where entities appear in reference and contexts."""
    return SingleTurnSample(
        user_input="Summarize the Apollo 11 mission",
        response=(
            "Apollo 11 landed on the Moon in 1969. Armstrong and Aldrin walked on"
            " the lunar surface while Collins orbited."
        ),
        retrieved_contexts=[
            "Apollo 11 was a 1969 mission; Neil Armstrong and Buzz Aldrin landed.",
            "Michael Collins remained in lunar orbit aboard the command module.",
        ],
        reference=(
            "Apollo 11 took place in 1969; crew: Neil Armstrong, Buzz Aldrin,"
            " and Michael Collins."
        ),
    )


def main() -> None:
    if os.getenv("OPENAI_API_KEY") is None:
        print("Skipping LLM example: OPENAI_API_KEY not set.")
        return
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini", temperature=0))
    metric = ContextEntityRecall(llm=llm)
    sample = build_sample()
    try:
        score = metric.single_turn_score(sample)
        print("ContextEntityRecall:", score)
    except Exception as e:
        print("Skipping LLM example due to error:", e)


if __name__ == "__main__":
    main()
